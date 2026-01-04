from propagation import propagate

import torch

def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    batch, height, width = depth_ref.shape
    
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack((x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(torch.inverse(extrinsics_src), extrinsics_ref),
                        torch.cat((xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(depth_src.view(batch, 1, height, width), torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2))

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(torch.inverse(extrinsics_ref), extrinsics_src),
                                torch.cat((xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, thre1=1, thre2=0.01):
    batch, height, width = depth_ref.shape
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device), indexing='ij')
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    inputs = [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = torch.logical_and(dist < thre1, relative_depth_diff < thre2)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff

def depth_propagation(ref_cam, ref_image, rendered_depth, src_cams, src_images, rendered_normal=None, patch_size=11, max_scale=2, radius_increment=2):
    depth_max = int(rendered_depth.max() + 1)
    depth_min = int(rendered_depth.min())

    images = list()
    intrinsics = list()
    poses = list()
    depth_intervals = list()

    images.append((ref_image * 255).permute((1, 2, 0)).to(torch.uint8))
    intrinsics.append(ref_cam.get_K()[:3,:3])
    poses.append(ref_cam.world_to_camera.transpose(0, 1))
    depth_interval = torch.tensor([depth_min, (depth_max-depth_min)/192.0, 192.0, depth_max])
    depth_intervals.append(depth_interval)
    
    depth = rendered_depth.unsqueeze(-1)
    if rendered_normal is None:
        normal = torch.empty(0, dtype=torch.float32)
    else:
        normal = rendered_normal
        if normal.shape[0] == 3:
            normal = normal.permute(1, 2, 0)

    for src_cam in src_cams:
        intrinsics.append(src_cam.get_K()[:3,:3])
        poses.append(src_cam.world_to_camera.transpose(0, 1))
    
    for src_image in src_images:
        images.append((src_image * 255).permute((1, 2, 0)).to(torch.uint8))
        depth_intervals.append(depth_interval)
        
    images = torch.stack(images)
    intrinsics = torch.stack(intrinsics)
    poses = torch.stack(poses)
    depth_intervals = torch.stack(depth_intervals).to(rendered_depth.device)

    results = propagate(images, intrinsics, poses, depth, normal, depth_intervals, patch_size, max_scale, radius_increment)
    propagated_depth = results[0].to(rendered_depth.device)
    propagated_normal = results[1:4].to(rendered_depth.device).permute(1, 2, 0)
    
    return propagated_depth, propagated_normal

    
def generate_edge_mask(propagated_depth, patch_size):
    # img gradient
    x_conv = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).float().cuda()
    y_conv = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).float().cuda()
    gradient_x = torch.abs(torch.nn.functional.conv2d(propagated_depth.unsqueeze(0).unsqueeze(0), x_conv, padding=1))
    gradient_y = torch.abs(torch.nn.functional.conv2d(propagated_depth.unsqueeze(0).unsqueeze(0), y_conv, padding=1))
    gradient = gradient_x + gradient_y

    # edge mask
    edge_mask = (gradient > 5).float()

    # dilation
    kernel = torch.ones(1, 1, patch_size, patch_size).float().cuda()
    dilated_mask = torch.nn.functional.conv2d(edge_mask, kernel, padding=(patch_size-1)//2)
    dilated_mask = torch.round(dilated_mask).squeeze().to(torch.bool)
    dilated_mask = ~dilated_mask

    return dilated_mask
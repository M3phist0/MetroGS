import os
import sys
import torch
import argparse
import numpy as np
import cv2
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
import torch.nn.functional as F

import json
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.dataparsers.estimated_mask_depth_colmap_dataparser import EstimatedDepthColmap
# from internal.dataparsers.estimated_depth_colmap_dataparser import EstimatedDepthColmap

def get_rays(camera, scale=1.0):
    W, H = int(camera.width/scale), int(camera.height/scale)
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    rays_d = torch.stack(
                [(ix-camera.cx/scale) / camera.fx * scale,
                (iy-camera.cy/scale) / camera.fy * scale,
                torch.ones_like(ix)], -1).float().cuda()
    return rays_d

def get_points_from_depth(fov_camera, depth, scale=1):
    st = int(max(int(scale/2)-1,0))
    depth_view = depth.squeeze()[st::scale,st::scale]
    rays_d = get_rays(fov_camera, scale=scale)
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1,3)
    R = fov_camera.R.cuda()
    T = fov_camera.T.cuda()
    pts = (pts-T)@R
    return pts

def depth_filter(camera, ref_depth, ref_w2c, src_depth, src_w2c, pixel_thred=8):
    '''
    ref_depth: [H, W] / ref_w2c: [4, 4]
    src_depth: [N, H, W] / src_w2c: [N, 4, 4]
    '''
    H, W = ref_depth.shape
    N = src_depth.shape[0]

    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to("cuda")
    
    pts = get_points_from_depth(camera, ref_depth)
    pts_in_src_cam = torch.matmul(src_w2c[:,None,:3,:3].expand(N,H*W,3,3).transpose(-1,-2), 
                        pts[None,:,:,None].expand(N,H*W,3,1))[...,0] + src_w2c[:,None,3,:3] # b, pts, 3
    pts_projections = torch.stack([pts_in_src_cam[...,0] * camera.fx / pts_in_src_cam[...,2] + camera.cx,
                        pts_in_src_cam[...,1] * camera.fy / pts_in_src_cam[...,2] + camera.cy], -1).float()
    d_mask = (pts_projections[..., 0] > 0) & (pts_projections[..., 0] < camera.width) &\
                    (pts_projections[..., 1] > 0) & (pts_projections[..., 1] < camera.height)
    
    # print(pts_in_src_cam[0].shape, pts_in_src_cam[0], pts_in_src_cam[0] / pts_in_src_cam[0][..., 2].unsqueeze(-1))
    
    pts_projections[..., 0] /= ((camera.width - 1) / 2)
    pts_projections[..., 1] /= ((camera.height - 1) / 2)
    pts_projections -= 1
    pts_projections = pts_projections.view(N, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(input=src_depth[:, None],
                                            grid=pts_projections,
                                            mode='bilinear',
                                            padding_mode='border',
                                            align_corners=True
                                            )[:,0,:,0]
    
    pts_in_src_cam[...,0] = pts_in_src_cam[...,0]/pts_in_src_cam[...,2]*map_z.squeeze()
    pts_in_src_cam[...,1] = pts_in_src_cam[...,1]/pts_in_src_cam[...,2]*map_z.squeeze()
    pts_in_src_cam[...,2] = map_z.squeeze()
    pts_ = (pts_in_src_cam - src_w2c[:,None,3,:3])
    pts_ = torch.matmul(src_w2c[:,None,:3,:3].expand(N,H*W,3,3), 
                        pts_[:,:,:,None].expand(N,H*W,3,1))[...,0]

    pts_in_view_cam = pts_ @ ref_w2c[:3,:3] + ref_w2c[None,None,3,:3]
    pts_projections = torch.stack(
                [pts_in_view_cam[...,0] * camera.fx / pts_in_view_cam[...,2] + camera.cx,
                pts_in_view_cam[...,1] * camera.fy / pts_in_view_cam[...,2] + camera.cy], -1).float()
    pixel_noise = torch.norm(pts_projections.reshape(N, H, W, 2) - pixels[None], dim=-1)

    d_mask_all = d_mask.reshape(N,H,W) & (pixel_noise < pixel_thred) & (pts_in_view_cam[...,2].reshape(N,H,W) > 0.0)
    d_mask_all = (d_mask_all.sum(0) >= 1)   

    return d_mask_all

def canny_edge_detection(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    otsu_threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 一般将 Otsu 阈值作为高阈值，低阈值设为高阈值的一半
    low_threshold = 0.5 * otsu_threshold
    high_threshold = otsu_threshold
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges

def find_good_corners(image, max_corners=2300, quality_level=0.01, min_distance=10):
    """
    使用 Shi-Tomasi 算法在灰度图上寻找最优角点。
    
    参数:
    image (numpy.ndarray): 输入的灰度图。
    max_corners (int): 最多返回的角点数。
    quality_level (float): 角点的最小质量水平，0到1之间的值。
    min_distance (int): 角点之间的最小欧氏距离。
    
    返回:
    numpy.ndarray: 找到的角点坐标数组，形状为 (N, 2)。
    """
    # Shi-Tomasi 算法要求输入是 float32 类型
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(image)
    
    # 寻找角点
    corners = cv2.goodFeaturesToTrack(gray_img, max_corners, quality_level, min_distance)
    
    # 创建一个与原图同样大小的空白二值化图片
    corner_map = np.zeros_like(gray_img)

    if corners is not None:
        # 遍历所有找到的角点，并在二值化图片上标记
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(corner_map, (int(x), int(y)), 1, 255, -1)
        
    return corner_map

def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)

    # return edge
    edge_uint8 = edge.cpu().numpy().astype(np.uint8) * 255
    return edge_uint8

def floyd_steinberg_dithering_pil(image: np.ndarray) -> np.ndarray:
    """
    使用 Pillow 库中的高效 Floyd-Steinberg 抖动算法。
    
    参数:
    image (numpy.ndarray): 输入图像，可以是灰度图或彩色图。
    
    返回:
    numpy.ndarray: 经过抖动处理的二值化图片，形状为 (H, W)，值域为 0 或 255。
    """
    # 检查图像维度，如果不是单通道，则转换为灰度图
    if image.ndim == 3:
        # Pillow 使用 RGB 格式，而 OpenCV 使用 BGR，需要转换
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # 1. 将 NumPy 数组转换为 PIL Image 对象
    pil_image = Image.fromarray(gray_img)

    # 2. 将图像转换为 1 位模式（二值化），并应用 Floyd-Steinberg 抖动
    # '1' 表示 1-bit pixel, black and white
    # dither=Image.Dither.FLOYDSTEINBERG 启用抖动
    dithered_pil_image = pil_image.convert(
        '1', dither=Image.Dither.FLOYDSTEINBERG
    )

    # 3. 将 PIL Image 对象转换回 NumPy 数组
    # 此时，值域为 0 或 255
    dithered_np_array = np.array(dithered_pil_image)

    dithered_np_array = dithered_np_array.astype(np.uint8) * 255
    
    return dithered_np_array

def delaunay_triangulation(edges, triangulated_img=None, ratio=0.1):
    """
    对边缘图片进行德劳内三角剖分，并保存和可视化结果。
    
    参数:
    edges (numpy.ndarray): Canny 边缘检测得到的二值化图片。
    save_path (str): 保存结果的图片路径。
    """
    all_points = np.argwhere(edges > 0)
    num_total_points = all_points.shape[0]

    tgt_num_points = int(num_total_points * ratio)

    if tgt_num_points >= num_total_points:
        points = all_points[:, ::-1]
    else:
        # 随机选择点的索引
        random_indices = np.random.choice(num_total_points, tgt_num_points, replace=False)
        points = all_points[random_indices, ::-1] # 转换为 (x, y) 坐标

    if points.shape[0] < 3:
        print("边缘点太少，无法进行三角剖分。")
        return

    rect = (0, 0, edges.shape[1], edges.shape[0])
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert(tuple(p.astype(float)))

    triangle_list = subdiv.getTriangleList()

    if triangulated_img is None:
        triangulated_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    
    RED = (0, 0, 255)

    print("number:", len(triangle_list))

    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        # 检查三角形的顶点是否在图片范围内
        if all(0 <= p[0] < edges.shape[1] and 0 <= p[1] < edges.shape[0] for p in [pt1, pt2, pt3]):
            cv2.line(triangulated_img, pt1, pt2, RED)
            cv2.line(triangulated_img, pt2, pt3, RED)
            cv2.line(triangulated_img, pt3, pt1, RED)
    
    return triangulated_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    # parser.add_argument("--invdepth_dir", "-i", type=str, default=None, required=True, help="directory of invdepthmap")
    parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')
    parser.add_argument("--downsample_factor", "-d", type=float, default=1)
    parser.add_argument("--eval_ratio", "-r", type=float, default=0.1)
    parser.add_argument("--eval_image_select_mode", "-m", type=str, default="ratio")
    parser.add_argument("--multi_view_num", type=int, default=2)
    parser.add_argument("--multi_view_max_angle", type=float, default=30)
    parser.add_argument("--multi_view_min_dis", type=float, default=0.01)
    parser.add_argument("--multi_view_max_dis", type=float, default=1.5)
    args = parser.parse_args()

    dataparser_config = EstimatedDepthColmap(
        down_sample_factor=args.downsample_factor,
        eval_image_select_mode=args.eval_image_select_mode,
        eval_ratio=args.eval_ratio,
        split_mode="experiment",
    )

    # load dataset
    dataparser_outputs = dataparser_config.instantiate(
        path=args.dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()

    os.makedirs("./tmp/", exist_ok=True)

    print("computing nearest_id")
    
    train_set = dataparser_outputs.train_set
    cameras = train_set.cameras
    
    world_view_transforms = []
    camera_centers = []
    center_rays = []
    
    for idx, camera in enumerate(cameras):
        world_view_transforms.append(camera.world_to_camera)
        camera_centers.append(camera.camera_center)
        R = camera.R.cuda()
        T = camera.T.cuda()
        center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
        center_ray = center_ray@R
        center_rays.append(center_ray)

    world_view_transforms = torch.stack(world_view_transforms)
    camera_centers = torch.stack(camera_centers)
    center_rays = torch.stack(center_rays)
    diss = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).cpu().numpy()
    tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1)
    angles = torch.arccos(tmp) * 180 / 3.14159
    angles = angles.cpu().numpy()

    os.makedirs("./filter/", exist_ok=True)
    for idx, camera in enumerate(cameras):
        print("name:", train_set.image_names[idx])

        sorted_indices = np.lexsort((angles[idx], diss[idx]))
        mask = (angles[idx][sorted_indices] < args.multi_view_max_angle) & \
                    (diss[idx][sorted_indices] > args.multi_view_min_dis) & \
                    (diss[idx][sorted_indices] < args.multi_view_max_dis)
        sorted_indices = sorted_indices[mask]
        multi_view_num = min(args.multi_view_num, len(sorted_indices))
        
        json_d = {'ref_name' : train_set.image_names[idx], 'src_name': []}
        ref_data = train_set.extra_data_processor(train_set.extra_data[idx])
        ref_extrinsic = train_set.cameras[idx].world_to_camera.cuda()

        src_data_list = []
        src_extrinsic_list = []
        for index in sorted_indices[:multi_view_num]:
            json_d["src_name"].append(train_set.image_names[index])
            src_data =  train_set.extra_data_processor(train_set.extra_data[index])
            if src_data is None:
                continue
            src_extrinsic = train_set.cameras[index].world_to_camera
            src_data_list.append(src_data)
            src_extrinsic_list.append(src_extrinsic)

            # depth = 1. / src_data[0].cpu().numpy()
            # depth = 1. / src_data.cpu().numpy()
            # depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            # depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
            # depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join("./filter/", train_set.image_names[index] + ".png"), depth_color)

        if ref_data is None:
            continue
        ref_depth = 1. / ref_data[0].cuda()


        ref_path = train_set.image_paths[idx]
        edges1 = canny_edge_detection(ref_path)
        ref_img = cv2.imread(ref_path)
        edges2 = floyd_steinberg_dithering_pil(ref_img)
        edges = edges1 & edges2
        # edges = find_good_corners(ref_img)
        # edges = find_good_orb(ref_img)
        triangles = delaunay_triangulation(edges, ref_img)
        depth_edges = depth_edge(ref_depth, rtol=0.01)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_dedge.png"), depth_edges)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_edge.png"), edges)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_tri.png"), triangles)

        ref_mask = ref_data[1]
        src_depth = torch.stack([1. / src_data[0].cuda() for src_data in src_data_list])
        # ref_depth = 1. / ref_data.cuda()
        # src_depth = torch.stack([1. / src_data.cuda() for src_data in src_data_list])
        src_extrinsic = torch.stack(src_extrinsic_list).cuda()
        
        filter_mask = depth_filter(camera, ref_depth, ref_extrinsic, src_depth, src_extrinsic)
        filter_mask = filter_mask.cpu().numpy()

        depth = ref_depth.cpu().numpy()
        depth = np.clip(depth, a_min=0.0, a_max=50.0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        depth_color[~filter_mask] = 0 
        # depth_color[~ref_mask] = 0 
        # cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_ref.png"), depth_color)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + ".png"), depth_color)

        break

    # with open(os.path.join(args.dataset_path, "multi_view.json"), 'w') as file:
    #     for idx, camera in enumerate(cameras):
    #         ref_depth, ref_mask = train_set.extra_data_processor(train_set.extra_data[idx])

    #         sorted_indices = np.lexsort((angles[idx], diss[idx]))
    #         mask = (angles[idx][sorted_indices] < args.multi_view_max_angle) & \
    #                     (diss[idx][sorted_indices] > args.multi_view_min_dis) & \
    #                     (diss[idx][sorted_indices] < args.multi_view_max_dis)
    #         sorted_indices = sorted_indices[mask]
    #         multi_view_num = min(args.multi_view_num, len(sorted_indices))
    #         json_d = {'ref_name' : train_set.image_names[idx], 'src_name': []}
    #         for index in sorted_indices[:multi_view_num]:
    #             json_d["src_name"].append(train_set.image_names[index])
    #         json_str = json.dumps(json_d, separators=(',', ':'))
    #         file.write(json_str)
    #         file.write('\n')
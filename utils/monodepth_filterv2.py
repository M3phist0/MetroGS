import os
import sys
import torch
import argparse
import numpy as np
import cv2
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from collections import defaultdict
from itertools import combinations
import pycolmap

import math
import json
import random
from PIL import Image
from numba import cuda
from plyfile import PlyData, PlyElement

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.dataparsers.estimated_mask_depth_colmap_dataparser import EstimatedDepthColmap

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

def depth_filter(camera, ref_depth, ref_w2c, src_depth, src_w2c, pixel_thred=2, thres_view=1):
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
    d_mask_all = (d_mask_all.sum(0) >= thres_view)   

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

    # print("number:", len(triangle_list))

    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        # 检查三角形的顶点是否在图片范围内
        if all(0 <= p[0] < edges.shape[1] and 0 <= p[1] < edges.shape[0] for p in [pt1, pt2, pt3]):
            cv2.line(triangulated_img, pt1, pt2, RED)
            cv2.line(triangulated_img, pt2, pt3, RED)
            cv2.line(triangulated_img, pt3, pt1, RED)
    
    return triangulated_img, triangle_list

def create_grid(triangle_list, img_shape, grid_size=32):
    h, w = img_shape[:2]
    
    # 计算网格尺寸
    grid_w = (w + grid_size - 1) // grid_size
    grid_h = (h + grid_size - 1) // grid_size
    
    # 使用列表的列表作为网格，每个单元格存储三角形索引
    grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
    
    # 将每个三角形映射到网格单元中
    for i, t in enumerate(triangle_list):
        # 计算三角形的AABB (Axis-Aligned Bounding Box)
        min_x = int(min(t[0], t[2], t[4]) / grid_size)
        min_y = int(min(t[1], t[3], t[5]) / grid_size)
        max_x = int(max(t[0], t[2], t[4]) / grid_size) + 1
        max_y = int(max(t[1], t[3], t[5]) / grid_size) + 1

        # 确保索引在网格范围内
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(grid_w, max_x)
        max_y = min(grid_h, max_y)

        # 遍历所有相交的网格单元，并添加三角形索引
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                grid[y][x].append(i)
                
    return grid, grid_w, grid_h

# 修改后的 CUDA Kernel，现在使用网格进行优化
@cuda.jit
def triangles_kernel(triangle_coords, output_image, grid_flat_offsets, grid_flat_indices, grid_w, grid_h, grid_size):
    x, y = cuda.grid(2)
    
    if x >= output_image.shape[1] or y >= output_image.shape[0]:
        return

    # 1. 确定当前像素点所在的网格单元
    cell_x = x // grid_size
    cell_y = y // grid_size

    if cell_x >= grid_w or cell_y >= grid_h:
        return

    start_index = grid_flat_offsets[cell_y, cell_x]
    
    if cell_x == grid_w - 1 and cell_y == grid_h - 1:
        end_index = len(grid_flat_indices)
    else:
        next_cell_x = cell_x + 1
        next_cell_y = cell_y
        if next_cell_x >= grid_w:
            next_cell_x = 0
            next_cell_y += 1
        end_index = grid_flat_offsets[next_cell_y, next_cell_x]
    
    output_image[y, x] = 0.0 # 初始化为0
    
    # 2. 只遍历这个网格单元中的三角形
    for k in range(start_index, end_index):
        i = grid_flat_indices[k]
        
        t = triangle_coords[i]
        
        t_x1, t_y1 = t[0], t[1]
        t_x2, t_y2 = t[2], t[3]
        t_x3, t_y3 = t[4], t[5]
        
        v0_x, v0_y = t_x3 - t_x1, t_y3 - t_y1
        v1_x, v1_y = t_x2 - t_x1, t_y2 - t_y1
        v2_x, v2_y = x - t_x1, y - t_y1

        dot00 = v0_x * v0_x + v0_y * v0_y
        dot01 = v0_x * v1_x + v0_y * v1_y
        dot02 = v0_x * v2_x + v0_y * v2_y
        dot11 = v1_x * v1_x + v1_y * v1_y
        dot12 = v1_x * v2_x + v1_y * v2_y

        inv_denom = dot00 * dot11 - dot01 * dot01

        if inv_denom == 0:
            continue
            
        inv_denom = 1.0 / inv_denom

        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        if u >= 0 and v >= 0 and (u + v) < 1:
            # 计算该三角形的包围盒
            min_x = min(t_x1, t_x2, t_x3)
            min_y = min(t_y1, t_y2, t_y3)
            max_x = max(t_x1, t_x2, t_x3)
            max_y = max(t_y1, t_y2, t_y3)
            
            # 计算包围盒对角线长度
            dx = max_x - min_x
            dy = max_y - min_y
            diagonal_length = math.sqrt(dx*dx + dy*dy)
            
            # 将结果写入输出图像
            output_image[y, x] = diagonal_length
            return

def draw_triangles_with_cuda(triangle_list, grid_size=32):
    # 1. 使用你提供的函数进行德劳内三角剖分
    triangle_list_cpu = triangle_list

    num_triangles = len(triangle_list_cpu)
    
    grid_cpu, grid_w, grid_h = create_grid(triangle_list_cpu, edges.shape, grid_size)
    
    grid_flat_offsets = np.zeros((grid_h, grid_w), dtype=np.int32)
    grid_flat_indices = []
    
    current_offset = 0
    for y in range(grid_h):
        for x in range(grid_w):
            grid_flat_offsets[y, x] = current_offset
            for index in grid_cpu[y][x]:
                grid_flat_indices.append(index)
            current_offset += len(grid_cpu[y][x])
            
    grid_flat_indices = np.array(grid_flat_indices, dtype=np.int32)
    
    triangle_coords_cpu = np.array(triangle_list_cpu, dtype=np.float32)
    d_triangle_coords = cuda.to_device(triangle_coords_cpu)
    
    d_grid_flat_offsets = cuda.to_device(grid_flat_offsets)
    d_grid_flat_indices = cuda.to_device(grid_flat_indices)

    h, w = edges.shape[:2]
    # 创建一个单通道的 float32 图像作为输出
    output_image_cpu = np.zeros((h, w), dtype=np.float32)
    d_output_image = cuda.to_device(output_image_cpu)
    
    threadsperblock = (16, 16)
    blockspergrid_x = (w + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (h + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    print("Launching optimized CUDA kernel...")
    triangles_kernel[blockspergrid, threadsperblock](
        d_triangle_coords, d_output_image, 
        d_grid_flat_offsets, d_grid_flat_indices, 
        np.int32(grid_w), np.int32(grid_h), np.int32(grid_size)
    )
    
    final_image_cpu = d_output_image.copy_to_host()

    return final_image_cpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    # parser.add_argument("--invdepth_dir", "-i", type=str, default=None, required=True, help="directory of invdepthmap")
    parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')
    parser.add_argument("--downsample_factor", "-d", type=float, default=1)
    parser.add_argument("--eval_ratio", "-r", type=float, default=0.1)
    parser.add_argument("--eval_image_select_mode", "-m", type=str, default="ratio")
    parser.add_argument("--multi_view_num", type=int, default=8)
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
    
    train_set = dataparser_outputs.train_set
    cameras = train_set.cameras

    name2idx = {value: idx for (idx, value) in enumerate(train_set.image_names)}

    sparse_dir = os.path.join(args.dataset_path, "sparse/0")
    rec = pycolmap.Reconstruction(sparse_dir)

    # 存储图像对的共视点数量
    co_visibility = defaultdict(int)

    # 遍历所有 3D 点，构建共视图
    for point3D_id, point3D in rec.points3D.items():
        track_image_ids = sorted([el.image_id for el in point3D.track.elements])
        
        # 使用 itertools.combinations 高效地生成所有图像对
        for imid1, imid2 in combinations(track_image_ids, 2):
            co_visibility[(imid1, imid2)] += 1

    top_k_neighbors = {}
    image_ids = list(rec.images.keys())
    
    # 遍历每张图像，找到其前 N 个邻居
    for imid in image_ids:
        ref_name = rec.images[imid].name
        if ref_name not in name2idx:
            continue
        ref_idx = name2idx[ref_name]

        neighbors = []
        for (imid1, imid2), count in co_visibility.items():
            if imid1 == imid:
                neighbors.append((imid2, count))
            elif imid2 == imid:
                neighbors.append((imid1, count))
        
        # 按共享点数降序排序
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        center_rays = []
        camera_centers = []
        neighbors_list = []
        for src_imid, count in neighbors:
            src_name = rec.images[src_imid].name
            if src_name not in name2idx or count < 100:
                continue
            src_idx = name2idx[src_name]
            neighbors_list.append(src_idx)

            center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda() @ cameras[src_idx].R.cuda()
            center = cameras[src_idx].camera_center
            center_rays.append(center_ray)
            camera_centers.append(center)

            if len(neighbors_list) >= args.multi_view_num * 2:
                break
        
        if len(neighbors_list) < args.multi_view_num:
            top_k_neighbors[ref_idx] = neighbors_list
            continue

        ref_center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda() @ cameras[ref_idx].R.cuda()
        ref_center = cameras[ref_idx].camera_center

        center_rays = torch.stack(center_rays)
        camera_centers = torch.stack(camera_centers)

        diss = torch.norm(ref_center - camera_centers, dim=-1).cpu().numpy()
        tmp = torch.sum(ref_center_ray * center_rays, dim=-1)
        angles = torch.arccos(tmp) * 180 / 3.14159
        angles = angles.cpu().numpy()
        sorted_indices = np.lexsort((angles, diss))
        mask = (angles[sorted_indices] < args.multi_view_max_angle) & \
                    (diss[sorted_indices] > args.multi_view_min_dis) & \
                    (diss[sorted_indices] < args.multi_view_max_dis)
        sorted_indices = sorted_indices[mask]
        multi_view_num = min(args.multi_view_num, len(sorted_indices))

        filter_neighbors_list = []
        for index in sorted_indices[:multi_view_num]:
            src_idx = neighbors_list[index]
            filter_neighbors_list.append(src_idx)

        top_k_neighbors[ref_idx] = filter_neighbors_list
    
    #### write multi-view-info
    # json_d = {}
    # for idx in top_k_neighbors:
    #     neighbors_list = top_k_neighbors[idx]
    #     ref_name = train_set.image_names[idx]
        
    #     src_names = []
    #     for index in neighbors_list:
    #         src_names.append(train_set.image_names[index])
    #     json_d[ref_name] = src_names
    
    # with open(os.path.join(args.dataset_path, "multi_view.json"), 'w') as file:
    #     json_str = json.dumps(json_d, separators=(',', ':'))
    #     file.write(json_str)
    #     file.write('\n')
    #### write multi-view-info


    #### depth filter
    os.makedirs("./filter/", exist_ok=True)
    for idx in top_k_neighbors:
        neighbors_list = top_k_neighbors[idx]
        camera = train_set.cameras[idx]

        ref_data = train_set.extra_data_processor(train_set.extra_data[idx])
        if ref_data is None:
            continue

        if 'DJI_202312191150' not in train_set.image_names[idx]:
            continue

        depth = ref_data[0].cpu().numpy()
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_ref_ori.png"), depth_color)
        
        src_data_list = []
        src_extrinsic_list = []
        for index in neighbors_list:
            src_data =  train_set.extra_data_processor(train_set.extra_data[index])
            if src_data is None:
                continue
            src_extrinsic = train_set.cameras[index].world_to_camera
            src_data_list.append(src_data)
            src_extrinsic_list.append(src_extrinsic)

            depth = src_data[0].cpu().numpy()
            depth[depth < 0] = 0
            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join("./filter/", train_set.image_names[index] + ".png"), depth_color)

            src_path = train_set.image_paths[index]
            src_img = cv2.imread(src_path)
            cv2.imwrite(os.path.join("./filter/", train_set.image_names[index] + "_raw.png"), src_img)
            break

        ref_data = train_set.extra_data_processor(train_set.extra_data[idx])
        ref_extrinsic = train_set.cameras[idx].world_to_camera.cuda()
        if ref_data is None:
            continue
        ref_depth = 1. / ref_data[0].clone().cuda()

        ref_path = train_set.image_paths[idx]
        ref_img = cv2.imread(ref_path)
        edges1 = canny_edge_detection(ref_path)
        edges2 = floyd_steinberg_dithering_pil(ref_img)
        edges = edges1 & edges2
        triangles, triangle_list = delaunay_triangulation(edges, ref_img)

        import time
        st = time.time()
        print("start compute")
        area_map = draw_triangles_with_cuda(triangle_list)
        print(area_map[100:-100, 100:-100], area_map.mean())
        normalized_image = cv2.normalize(area_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colored_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_cuda.png"), colored_image)
        ed = time.time()
        print("end compute", ed - st)

        depth_edges = depth_edge(ref_depth, rtol=0.01)
        # cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_dedge.png"), depth_edges)
        # cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_edge.png"), edges)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_tri.png"), triangles)
        ref_img = cv2.imread(ref_path)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_raw.png"), ref_img)

        ref_mask = ref_data[1]
        src_depth = torch.stack([1. / src_data[0].cuda() for src_data in src_data_list])
        src_extrinsic = torch.stack(src_extrinsic_list).cuda()

        print(train_set.image_names[idx], len(src_extrinsic_list))
        
        filter_mask = depth_filter(camera, ref_depth, ref_extrinsic, src_depth, src_extrinsic)
        filter_mask = filter_mask.cpu().numpy()

        depth = ref_data[0].cpu().numpy()
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + ".png"), depth_color)
        depth_color[~filter_mask] = 0 
        cv2.imwrite(os.path.join("./filter/", train_set.image_names[idx] + "_ref.png"), depth_color)
        break
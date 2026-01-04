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
from internal.dataparsers.colmap_dataparser import Colmap
from internal.utils.propagate_utils import depth_propagation, check_geometric_consistency

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')
parser.add_argument("--downsample_factor", "-d", type=float, default=1)
parser.add_argument('--split_mode', "-s", type=str, default="experiment", help='experiment or reconstruction')
parser.add_argument("--eval_ratio", "-r", type=float, default=0.1)
parser.add_argument("--eval_image_select_mode", "-m", type=str, default="ratio")
parser.add_argument("--multi_view_num", type=int, default=8)
parser.add_argument("--multi_view_max_angle", type=float, default=30)
parser.add_argument("--multi_view_min_dis", type=float, default=0.01)
parser.add_argument("--multi_view_max_dis", type=float, default=1.5)
args = parser.parse_args()

# dataparser_config = EstimatedDepthColmap(
#     down_sample_factor=args.downsample_factor,
#     eval_image_select_mode=args.eval_image_select_mode,
#     eval_ratio=args.eval_ratio,
#     split_mode=args.split_mode,
# )
dataparser_config = Colmap(
    down_sample_factor=args.downsample_factor,
    eval_image_select_mode=args.eval_image_select_mode,
    eval_ratio=args.eval_ratio,
    split_mode=args.split_mode,
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
        
    # print("number(1):", ref_idx, len(neighbors_list))
    
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
    if mask.sum() > 0:
        sorted_indices = sorted_indices[mask]
    multi_view_num = min(args.multi_view_num, len(sorted_indices))

    filter_neighbors_list = []
    for index in sorted_indices[:multi_view_num]:
        src_idx = neighbors_list[index]
        filter_neighbors_list.append(src_idx)
    
    # print("number(2):", ref_idx, len(filter_neighbors_list), mask.sum(), angles[sorted_indices], diss[sorted_indices])

    top_k_neighbors[ref_idx] = filter_neighbors_list

#### write multi-view-info
print("multi-view number:", len(top_k_neighbors))
json_d = {}
for idx in top_k_neighbors:
    neighbors_list = top_k_neighbors[idx]
    ref_name = train_set.image_names[idx]
    
    src_names = []
    for index in neighbors_list:
        src_names.append(train_set.image_names[index])
    json_d[ref_name] = src_names

with open(os.path.join(args.dataset_path, "multi_view.json"), 'w') as file:
    # json_str = json.dumps(json_d, separators=(',', ':'))
    # file.write(json_str)
    # file.write('\n')

    json_str = json.dumps(json_d, indent=4) 
    file.write(json_str)
    file.write('\n')
#### write multi-view-info


# #### depth filter
# from common import AsyncNDArraySaver
# ndarray_saver = AsyncNDArraySaver()
# try:
#     with torch.no_grad():
#         for idx in top_k_neighbors:
#             neighbors_list = top_k_neighbors[idx]
#             camera = train_set.cameras[idx]

#             ref_data = train_set.extra_data_processor(train_set.extra_data[idx])
#             if ref_data is None:
#                 continue

#             ref_mono_path = train_set.extra_data[idx][0]

#             ref_depth = 1. / ref_data[0].clone().cuda()

#             ref_K = camera.get_K()[:3, :3].cuda()
#             ref_pose = camera.world_to_camera.transpose(0, 1).inverse().cuda()

#             src_camera_list = []
#             src_gt_image_list = []

#             geometric_counts = None
#             for index in neighbors_list:
#                 src_data =  train_set.extra_data_processor(train_set.extra_data[index])
#                 if src_data is None:
#                     continue
#                 src_depth = 1. / src_data[0].cuda()
#                 src_camera = train_set.cameras[index]
#                 src_K = src_camera.get_K()[:3, :3].cuda()
#                 src_pose = src_camera.world_to_camera.transpose(0, 1).inverse().cuda()

#                 mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = \
#                     check_geometric_consistency(ref_depth.unsqueeze(0), ref_K.unsqueeze(0), 
#                                                 ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
#                                                 src_K.unsqueeze(0), src_pose.unsqueeze(0), 
#                                                 thre1=8, thre2=0.01)
                
#                 if geometric_counts is None:
#                     geometric_counts = mask.to(torch.uint8)
#                 else:
#                     geometric_counts += mask.to(torch.uint8)
            
#             if geometric_counts is not None:
#                 count = geometric_counts.squeeze()
#                 valid_mask = count >= 1
#             else:
#                 valid_mask = torch.zeros(ref_depth.shape, dtype=torch.bool).cuda()

#             ref_data_ori = np.load(ref_mono_path)

#             ref_mask = torch.tensor(ref_data_ori[..., 1], dtype=torch.bool).cuda()
#             ref_mask = ref_mask & valid_mask
#             ref_mask = ref_mask.cpu().numpy()

#             ref_inv_depth = ref_data_ori[..., 0]
#             combined_array = np.stack([ref_inv_depth, ref_mask], axis=-1)
#             file_name, file_extension = os.path.splitext(ref_mono_path)
#             ref_mono_mv_path = file_name + '.mv' + file_extension
#             # ndarray_saver.save(combined_array, ref_mono_mv_path)

#             # os.makedirs("./filter1/", exist_ok=True)
#             # depth = ref_inv_depth
#             # if ref_mask.sum() > 0:
#             #     depth[ref_mask] = (depth[ref_mask] - depth[ref_mask].min()) / (depth[ref_mask].max() - depth[ref_mask].min())
#             # depth[~ref_mask] = 0.0
#             # depth_i = depth
#             # depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
#             # depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
#             # cv2.imwrite(os.path.join("./filter1/", train_set.image_names[idx] + ".png"), depth_color)
# finally:
#     ndarray_saver.stop()
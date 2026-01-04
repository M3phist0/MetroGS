"""
For colmap dataset
"""

import numpy as np
import add_pypath
import os
import json
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from internal.utils.colmap import read_images_binary

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--train", type=str, default="train")
parser.add_argument("--test", type=str, default="val")
parser.add_argument("--k_num", type=int, default=4)
parser.add_argument("--use_name_sim", action="store_true", default=False)
args = parser.parse_args()

train_images_bin_path = os.path.join(args.dir, args.train, "sparse", "images.bin")
if os.path.exists(train_images_bin_path) is False:
    train_images_bin_path = os.path.join(args.dir, args.train, "sparse", "0", "images.bin")

test_images_bin_path = os.path.join(args.dir, args.test, "sparse", "images.bin")
if os.path.exists(test_images_bin_path) is False:
    test_images_bin_path = os.path.join(args.dir, args.test, "sparse", "0", "images.bin")

train_images = read_images_binary(train_images_bin_path)
test_images = read_images_binary(test_images_bin_path)

# print("train number:", len(train_images))
# print("test number:", len(test_images))

K = args.k_num
candi_count = 7 * K

train_tvecs = np.array([img.tvec for img in train_images.values()]) # (N, 3)
train_qvecs = np.array([img.qvec for img in train_images.values()]) # (N, 4)
train_ids = list(train_images.keys())

similar_images = {}

for test_id, test_image in test_images.items():
    test_name = test_image.name
    test_tvec = test_image.tvec
    test_qvec = test_image.qvec

    pos_dists = np.linalg.norm(test_tvec - train_tvecs, axis=1)
    coarse_indices = np.argsort(pos_dists)[:candi_count]

    candidate_ids = [train_ids[idx] for idx in coarse_indices]
    candidate_tvecs = train_tvecs[coarse_indices]
    candidate_qvecs = train_qvecs[coarse_indices]

    pos_dists_refined = np.linalg.norm(test_tvec - candidate_tvecs, axis=1)
    
    candidate_rotations = R.from_quat(candidate_qvecs)
    test_rotation = R.from_quat(test_qvec)
    rot_dists_refined = np.array([
        (test_rotation.inv() * candi_rot).magnitude() for candi_rot in candidate_rotations
    ])
    
    total_scores = pos_dists_refined + rot_dists_refined * 100
    
    final_indices = np.argsort(total_scores)[:K]
    
    top_k_neighbors = [candidate_ids[idx] for idx in final_indices]
    
    # 根据索引获取 top K 的ID和对应的分数
    top_k_neighbors = [candidate_ids[idx] for idx in final_indices]
    top_k_scores = total_scores[final_indices]
    
    # ---- 将分数转换为归一化权重 ----
    # 权重与分数成反比，分数越低（距离越近）权重越高。
    # 为了避免除以零，我们可以用一个很小的数（如 1e-6）来处理分数
    # 转换为相似度（分数越低，相似度越高）
    similarities = 1.0 / (top_k_scores + 1e-6)
    
    # 归一化权重，使总和为1
    normalized_weights = similarities / np.sum(similarities)
    
    # 过滤操作和重新归一化
    filtered_ids = []
    filtered_weights = []
    
    threshold = 1.0 / K
    
    for i in range(K):
        if normalized_weights[i] >= threshold:
            filtered_ids.append(top_k_neighbors[i])
            filtered_weights.append(normalized_weights[i])
    
    # 对过滤后的权重重新归一化
    if filtered_weights:
        filtered_weights = np.array(filtered_weights)
        filtered_weights /= np.sum(filtered_weights)
    # 如果过滤后列表为空，则保留权重最高的1个
    else:
        filtered_ids.append(top_k_neighbors[0])
        filtered_weights.append(1.0) # 只有一个项时权重为1

    filtered_names = [train_images[idx].name for idx in filtered_ids]
        
    similar_images[test_name] = {
        "ids": filtered_ids,
        "names": filtered_names,
        "weights": filtered_weights.tolist()
    }

output_path = os.path.join(args.dir, args.test, "similar_images.json")

# 使用 with 语句打开文件并写入数据
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(similar_images, f, indent=4, ensure_ascii=False)

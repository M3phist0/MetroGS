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
from numba import cuda
from plyfile import PlyData, PlyElement

import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.sparse import coo_matrix
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.dataparsers.estimated_mask_depth_colmap_dataparser import EstimatedDepthColmap
from internal.dataparsers.colmap_dataparser import Colmap

import internal.utils.colmap as colmap_utils

def build_scene_graph_from_sparse(train_set, rec):
    """
    Builds a scene graph from a pycolmap.Reconstruction object.
    
    The graph's nodes are image names, and the edge weights are the number
    of shared 3D points between images.
    
    Args:
        train_set: A data structure containing image names of the training set.
        rec (pycolmap.Reconstruction): The COLMAP reconstruction object.
        
    Returns:
        nx.Graph: The constructed scene graph.
    """
    print("Building a graph based on shared 3D points...")
    
    # 1. 确定图的节点（图像），并创建 ID 到名称的映射
    colmap_images_by_name = {image.name: image for image in rec.images.values()}
    node_names = [name for name in train_set.image_names if name in colmap_images_by_name]

    # 2. 映射 3D 点到图像
    # point_to_images = {point3D_id: [image_id1, image_id2, ...]}
    point_to_images = {}
    for point3D_id, point3D in rec.points3D.items():
        track_image_ids = [elem.image_id for elem in point3D.track.elements]
        
        if len(track_image_ids) >= 2:
            point_to_images[point3D_id] = track_image_ids

    # 3. 计算共享点数并存储
    # shared_points_counts = {(image_id1, image_id2): count, ...}
    shared_points_counts = {}
    
    for image_ids in point_to_images.values():
        # itertools.combinations 产生所有图像对，例如 (A,B), (A,C), (B,C)
        for img_id1, img_id2 in combinations(sorted(image_ids), 2):
            if (img_id1, img_id2) in shared_points_counts:
                shared_points_counts[(img_id1, img_id2)] += 1
            else:
                shared_points_counts[(img_id1, img_id2)] = 1
    
    # 4. 构建 NetworkX 图
    G = nx.Graph()
    G.add_nodes_from(node_names)
    
    for (img_id1, img_id2), count in shared_points_counts.items():
        image_name1 = rec.images[img_id1].name
        image_name2 = rec.images[img_id2].name
        
        # 仅添加在节点列表中的边
        if image_name1 in G.nodes and image_name2 in G.nodes:
            G.add_edge(image_name1, image_name2, weight=count)
    
    print(f"图构建完成。节点数：{G.number_of_nodes()}，边数：{G.number_of_edges()}")
    return G

def partition_scene_clusters(graph, num_clusters, max_image_overlap=0):
    """
    Partitions the scene graph into N clusters and expands them with related images.
    
    Args:
        graph (nx.Graph): The scene graph with image names as nodes and inlier counts as edge weights.
        num_clusters (int): The number of clusters (N) to partition the scene into.
        max_image_overlap (int): The max number of related images to add to each cluster for overlap.
    
    Returns:
        dict: A dictionary where keys are cluster IDs (0 to N-1) and values are
              lists of image names in each cluster.
    """
    if graph.number_of_nodes() < num_clusters:
        print("Warning: Not enough images to form the desired number of clusters.")
        # Return a single cluster with all images
        return {0: list(graph.nodes)}

    # 1. Prepare the graph for spectral clustering
    # Get the adjacency array from the NetworkX graph
    adj_array = nx.to_scipy_sparse_array(graph, weight='weight')

    adj_array_coo = adj_array.tocoo()

    if adj_array_coo.row.dtype != np.int32:
        adj_array = coo_matrix(
            (adj_array_coo.data, 
             (adj_array_coo.row.astype(np.int32), 
              adj_array_coo.col.astype(np.int32))), 
            shape=adj_array_coo.shape,
            dtype=adj_array_coo.dtype
        ).tocsr()
    else:
        adj_array = adj_array_coo.tocsr()

    # Get the ordered list of nodes (image names)
    node_names = list(graph.nodes)

    # 2. Perform normalized graph cut using Spectral Clustering
    print(f"Partitioning the graph into {num_clusters} clusters...")
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=0
    )
    labels = clustering.fit_predict(adj_array)

    # 3. Assign images to initial clusters
    clusters = {i: [] for i in range(num_clusters)}
    for i, image_name in enumerate(node_names):
        clusters[labels[i]].append(image_name)

    # 4. Expand clusters with overlapping images (matching the C++ logic)
    print("Expanding clusters with related images...")
    # Find all related images and sort by weight
    related_images = {name: [] for name in node_names}
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        related_images[u].append((v, weight))
        related_images[v].append((u, weight))
    
    for name in related_images:
        related_images[name].sort(key=lambda x: x[1], reverse=True)

    # Sequentially add related images to each cluster
    for i in range(num_clusters):
        original_images = clusters[i].copy()
        current_cluster_set = set(original_images)
        max_size = len(original_images) + max_image_overlap

        # Iterate through the best matches for each original image in the cluster
        for j in range(len(original_images)):
            if len(current_cluster_set) >= max_size:
                break
            
            image_name = original_images[j]
            if image_name in related_images:
                # Add up to num_image_matches related images (or until max_size is reached)
                for related_name, weight in related_images[image_name]:
                    if related_name not in current_cluster_set:
                        current_cluster_set.add(related_name)
                    if len(current_cluster_set) >= max_size:
                        break
        clusters[i] = list(current_cluster_set)

    return clusters

def generate_cluster_renaming_map(clusters, graph):
    """
    对每个聚类中的图像，根据其在场景图中的相关性进行排序并生成新的名称。

    Args:
        clusters (dict): 键为聚类ID，值为图像名列表的字典。
        graph (nx.Graph): 场景图，包含图像间的连接权重。

    Returns:
        dict: 一个字典，键是原始图像名，值是新的、经过重命名的图像名。
    """
    renaming_map = {}
    
    for cluster_id, image_list in clusters.items():
        if not image_list:
            continue
        
        # 1. 找到该聚类内连接最强的图像作为起点
        start_node = None
        max_total_weight = -1
        
        # 将图像列表转换为集合，以便进行快速查找
        images_set = set(image_list)
        
        for node in images_set:
            # 只考虑该节点在当前聚类内的连接
            total_weight = sum(data['weight'] for _, _, data in graph.edges(node, data=True) 
                               if _ in images_set and _ != node)
            if total_weight > max_total_weight:
                max_total_weight = total_weight
                start_node = node
        
        # 如果没有找到合适的起点，就用第一个图像作为起点
        if not start_node:
            start_node = list(images_set)[0]

        # 2. 执行贪心排序
        sorted_images = [start_node]
        visited = {start_node}
        current_node = start_node
        
        while len(sorted_images) < len(images_set):
            best_next_node = None
            max_edge_weight = -1
            
            # 寻找与当前节点连接最强的未访问节点
            if current_node in graph:
                neighbors = graph.adj[current_node]
                for neighbor, data in neighbors.items():
                    if neighbor in images_set and neighbor not in visited:
                        if data['weight'] > max_edge_weight:
                            max_edge_weight = data['weight']
                            best_next_node = neighbor
            
            # 如果没有找到相邻的未访问节点，则从剩余图像中任意选择一个
            if best_next_node:
                sorted_images.append(best_next_node)
                visited.add(best_next_node)
                current_node = best_next_node
            else:
                remaining_nodes = list(images_set - visited)
                if remaining_nodes:
                    next_node = remaining_nodes[0]
                    sorted_images.append(next_node)
                    visited.add(next_node)
                    current_node = next_node
                else:
                    break
        
        # 3. 根据排序结果为图像生成新名称
        for i, original_name in enumerate(sorted_images):
            # 新名称格式：cluster_ID_新索引.jpg
            new_name = f"cluster_{cluster_id:02d}_{i:06d}.jpg"
            renaming_map[original_name] = new_name
            
    return renaming_map

def balance_clusters(clusters, graph, max_iterations=5000, tolerance_k=100):
    """
    通过将图像从最大类移动到最小类来平衡类的大小。

    Args:
        clusters (dict): 键是类ID，值是图像名列表的字典。
        graph (nx.Graph): 用于分块的场景图。
        max_iterations (int): 移动图像的最大迭代次数。
        tolerance_k (int): 允许的最大和最小类之间的图像数量差。

    Returns:
        dict: 平衡后的类。
    """
    print("正在平衡类的大小...")

    for i in range(max_iterations):
        # 1. 找到最大和最小的类
        cluster_sizes = {cid: len(images) for cid, images in clusters.items()}
        largest_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
        smallest_cluster_id = min(cluster_sizes, key=cluster_sizes.get)
        
        if cluster_sizes[largest_cluster_id] - cluster_sizes[smallest_cluster_id] <= tolerance_k:
            print(f"在 {i+1} 次迭代后达到均衡，最大和最小类差为 {cluster_sizes[largest_cluster_id] - cluster_sizes[smallest_cluster_id]}")
            break

        # 2. 找到最适合移动的图像 (优化后)
        best_image_to_move = None
        max_connection_weight = -1
        
        largest_cluster_images = clusters[largest_cluster_id]
        smallest_cluster_images = clusters[smallest_cluster_id]
        
        # 关键优化：将最小类的图像列表转换为一个集合，提高查找速度
        smallest_cluster_set = set(smallest_cluster_images)

        for image_name in largest_cluster_images:
            total_weight_to_smallest = 0
            # 遍历该图像在图中的所有邻居
            for neighbor_name, data in graph[image_name].items():
                # 使用集合进行 O(1) 查找
                if neighbor_name in smallest_cluster_set:
                    total_weight_to_smallest += data['weight']
            
            if total_weight_to_smallest > max_connection_weight:
                max_connection_weight = total_weight_to_smallest
                best_image_to_move = image_name

        # 3. 移动图像
        if best_image_to_move:
            clusters[largest_cluster_id].remove(best_image_to_move)
            clusters[smallest_cluster_id].append(best_image_to_move)
            # print(f"第 {i+1} 次迭代: 将 '{best_image_to_move}' 从类 {largest_cluster_id} 移动到 {smallest_cluster_id}")
        else:
            # 如果没有找到可以移动的图像（两个类之间没有边），则跳出循环
            break
    
    return clusters

def save_clusters_as_sparse(clusters, remap, sparse_dir, images_dir, output_dir, downsample_factor=1.0):
    """
    将聚类结果保存为独立的 COLMAP sparse 文件夹。
    
    Args:
        rec (pycolmap.Reconstruction): 完整的 COLMAP 重建对象。
        clusters (dict): 键为聚类ID，值为图像名列表的字典。
        images_dir (str): 原始图像目录的路径（该函数未使用，但保留参数）。
        output_dir (str): 输出 sparse 文件夹的父目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cameras, images, points3D = colmap_utils.read_model(sparse_dir)

    name2idx = {images[idx].name : idx for idx in images}

    for cluster_id, image_names in clusters.items():
        print(f"正在保存聚类 {cluster_id} 的 sparse 文件...")
        
        cluster_images = {}
        for name in sorted(image_names):
            idx = name2idx[name]
            image = images[idx]
            remap_image = colmap_utils.Image(
                id=image.id, qvec=image.qvec, tvec=image.tvec,
                camera_id=image.camera_id, name=remap[image.name],
                xys=image.xys, point3D_ids=image.point3D_ids)
            cluster_images[idx] = remap_image

        # 保存到 sparse 目录
        cluster_output_dir = os.path.join(output_dir, f"block_{cluster_id}/sparse/0")
        if not os.path.exists(cluster_output_dir):
            os.makedirs(cluster_output_dir)

        cluster_images_dir = os.path.join(output_dir, f"block_{cluster_id}/images")
        if not os.path.exists(cluster_images_dir):
            os.makedirs(cluster_images_dir)
        for name in image_names:
            img_path = os.path.join(images_dir, name)
            if os.path.exists(img_path):
                # 将图像从原始位置拷贝到聚类目录
                shutil.copy(img_path, os.path.join(cluster_images_dir, remap[name]))
            else:
                print(f"Warning: Image {name} not found at {img_path}. Skipping.")

        colmap_utils.write_model(cameras, cluster_images, points3D, cluster_output_dir)
        print(f"聚类 {cluster_id} 保存完成，文件在 {cluster_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--split_num", "-k", type=int, default=4)
    parser.add_argument("--use_balance", action="store_true",  help="Enable balanced training strategy (default: False)")
    parser.add_argument("--downsample_factor", "-d", type=float, default=1)
    parser.add_argument('--split_mode', "-s", type=str, default="experiment", help='experiment or reconstruction')
    parser.add_argument("--eval_ratio", "-r", type=float, default=0.1)
    parser.add_argument("--eval_image_select_mode", "-m", type=str, default="ratio")

    args = parser.parse_args()
  
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
    images_dir = os.path.join(args.dataset_path, "images")
    rec = pycolmap.Reconstruction(sparse_dir)

    G = build_scene_graph_from_sparse(train_set, rec)

    clusters = partition_scene_clusters(G, args.split_num)

    if args.use_balance:
        clusters = balance_clusters(clusters, G)

    image_name_to_id = {image.name: image_id for image_id, image in rec.images.items()}
    clusters_with_ids = {}
    for cluster_id, image_names in clusters.items():
        # 使用列表推导式将图像名列表转换为图像 ID 列表
        image_ids = [image_name_to_id[name] for name in image_names]
        clusters_with_ids[cluster_id] = image_ids
        
        # 打印每个块的 ID 和包含的图像数，以供验证
        print(f"聚类 ID {cluster_id}: 包含 {len(image_ids)} 张图像")

    remap = generate_cluster_renaming_map(clusters, G)

    output_dir = os.path.join(args.dataset_path, "segments")
    save_clusters_as_sparse(clusters, remap, sparse_dir, images_dir, output_dir=output_dir, downsample_factor=args.downsample_factor)
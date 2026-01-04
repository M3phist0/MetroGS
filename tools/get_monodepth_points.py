import os
import sys
import torch
import argparse
import numpy as np
import cv2
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist

from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.dataparsers.estimated_depth_colmap_dataparser import EstimatedDepthColmap

def select_cameras_kmeans(cameras, K):
    """
    Selects K cameras from a set using K-means clustering.

    Args:
        cameras: NumPy array of shape (N, 16), representing N cameras with their 4x4 homogeneous matrices flattened.
        K: Number of clusters (cameras to select).

    Returns:
        selected_indices: List of indices of the cameras closest to the cluster centers.
    """
    # Ensure input is a NumPy array
    if not isinstance(cameras, np.ndarray):
        cameras = np.asarray(cameras)

    if cameras.shape[1] != 16:
        raise ValueError("Each camera must have 16 values corresponding to a flattened 4x4 matrix.")

    # Perform K-means clustering
    cluster_centers, _ = kmeans(cameras, K)

    # Assign each camera to a cluster and find distances to cluster centers
    cluster_assignments, _ = vq(cameras, cluster_centers)

    # Find the camera nearest to each cluster center
    selected_indices = []
    for k in range(K):
        cluster_members = cameras[cluster_assignments == k]
        distances = cdist([cluster_centers[k]], cluster_members)[0]
        nearest_camera_idx = np.where(cluster_assignments == k)[0][np.argmin(distances)]
        selected_indices.append(nearest_camera_idx)

    return selected_indices

def pairwise_distances(matrix):
    """
    Computes the pairwise Euclidean distances between all vectors in the input matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape [N, N].
    """
    # Compute squared pairwise distances
    squared_diff = torch.cdist(matrix, matrix, p=2)
    return squared_diff

def k_closest_vectors(matrix, k):
    """
    Finds the k-closest vectors for each vector in the input matrix based on Euclidean distance.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.
        k (int): Number of closest vectors to return for each vector.

    Returns:
        torch.Tensor: Indices of the k-closest vectors for each vector, excluding the vector itself.
    """
    # Compute pairwise distances
    distances = pairwise_distances(matrix)

    # For each vector, sort distances and get the indices of the k-closest vectors (excluding itself)
    # Set diagonal distances to infinity to exclude the vector itself from the nearest neighbors
    distances.fill_diagonal_(float('inf'))

    # Get the indices of the k smallest distances (k-closest vectors)
    _, indices = torch.topk(distances, k, largest=False, dim=1)

    return indices

def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src
                                ):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks=[]
    for i in range(2,11):
        mask = np.logical_and(dist < i/4, relative_depth_diff < i/1300)
        masks.append(mask)
    # vis_mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')
    parser.add_argument("--downsample_factor", "-d", type=float, default=1)
    parser.add_argument("--eval_ratio", "-r", type=float, default=0.1)
    parser.add_argument("--eval_image_select_mode", "-m", type=str, default="ratio")
    args = parser.parse_args()

    dataparser_config = EstimatedDepthColmap(
        down_sample_factor=args.downsample_factor,
        eval_image_select_mode=args.eval_image_select_mode,
        eval_ratio=args.eval_ratio,
    )

    # load dataset
    dataparser_outputs = dataparser_config.instantiate(
        path=args.dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()

    NUM_NNS_PER_REFERENCE = (len(dataparser_outputs.train_set) + 7) // 8
    viewpoint_cam_all = dataparser_outputs.train_set.cameras.world_to_camera.reshape(-1, 16)
    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_NNS_PER_REFERENCE)
    selected_indices = sorted(selected_indices)

    closest_indices = k_closest_vectors(viewpoint_cam_all, 7)
    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    os.makedirs("./tmp/", exist_ok=True)

    vertexs = []
    vertex_colors = []

    for ref_idx in sorted(selected_indices):
        img_path = dataparser_outputs.train_set.image_paths[ref_idx]
        ref_img = np.array(Image.open(img_path)) / 255.

        print("path:", img_path)

        chunk_idx_list = [ref_idx] + list(closest_indices_selected[ref_idx])
        invdepth_list = []
        for view_idx in chunk_idx_list:
            invdepth = dataparser_outputs.train_set.extra_data_processor(
                dataparser_outputs.train_set.extra_data[ref_idx]
            )
            invdepth_list.append(invdepth)
        ref_depth = 1. / invdepth_list[0].cpu().numpy()
        ref_camera = dataparser_outputs.train_set.cameras[ref_idx]
        ref_extri = ref_camera.world_to_camera.transpose(0, 1).cpu().numpy()
        ref_intri = np.array([
            [ref_camera.fx, 0, ref_camera.cx],
            [0, ref_camera.fy, ref_camera.cy],
            [0, 0, 1]
        ])

        print("H/W:", ref_camera.height, ref_camera.width)

        all_srcview_depth_ests = []

        ct = 0
        geo_mask_sum = 0
        geo_mask_sums=[]
        n = len(chunk_idx_list)
        for i, src_idx in enumerate(chunk_idx_list[1:]):
            ct = ct + 1

            src_depth = 1. / invdepth_list[i].cpu().numpy()
            src_camera = dataparser_outputs.train_set.cameras[src_idx]
            src_extri = src_camera.world_to_camera.transpose(0, 1).cpu().numpy()
            src_intri = np.array([
                [src_camera.fx, 0, src_camera.cx],
                [0, src_camera.fy, src_camera.cy],
                [0, 0, 1]
            ])

            masks, geo_mask, depth_reprojected, x2d_src, y2d_src = \
                check_geometric_consistency(ref_depth, ref_intri, ref_extri, src_depth, src_intri, src_extri)

            if (ct==1):
                for i in range(2,n):
                    geo_mask_sums.append(masks[i-2].astype(np.int32))
            else :
                for i in range(2,n):
                    geo_mask_sums[i-2]+=masks[i-2].astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)
        
        geo_mask=geo_mask_sum>=args.thres_view

        for i in range (2,n):
            geo_mask=np.logical_or(geo_mask,geo_mask_sums[i-2]>=i)
            # print(geo_mask.mean())

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth) / (geo_mask_sum + 1)

        height, width = ref_camera.height, ref_camera.width
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        downsample_mask = np.ones_like(geo_mask, dtype=bool)
        # downsample_mask[::4, ::4] = True
        valid_points = np.logical_and(geo_mask, downsample_mask)
        # print("valid_points", valid_points.mean())
        # x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        valid_points = downsample_mask
        x, y, depth = x[downsample_mask], y[downsample_mask], ref_depth[downsample_mask]
        # print("shape:", x.shape, y.shape)
        color = ref_img[:, :, :][valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intri),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extri),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))
        
        depth = depth_est_averaged
        depth[~valid_points] = 0.0
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join("./tmp/",  f"{ref_idx}_depth.png"), depth_color)

        img_array_uint8 = (ref_img * 255.0).astype(np.uint8)
        img_to_save = Image.fromarray(img_array_uint8)
        output_path = f"./tmp/image_{ref_idx}.png"  # 指定一个输出路径
        img_to_save.save(output_path)

        print(ref_intri)
        print(ref_extri)


    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write("./tmp/test.ply")
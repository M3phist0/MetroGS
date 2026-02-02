import numpy as np
import argparse

import open3d as o3d
import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2
from pathlib import Path

import gc

# try:
#     import onnxruntime
# except ImportError:
#     print("onnxruntime not found. Sky segmentation may not work.")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
# from loop_utils.visual_util import segment_sky, download_file_from_url

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, load_images_as_tensor_pi_long
from pi3.utils.geometry import depth_edge, homogenize_points

import numpy as np

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime

import loop_utils.colmap as colmap_utils

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from loop_utils.config_utils import load_config

import mkl
mkl.get_max_threads()

def sor_with_mask(points: np.ndarray, mask: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """
    使用 Open3D 的 Statistical Outlier Removal, 仅对 mask==True 的点执行，
    返回与输入等长的新掩码 (True 表示保留)
    - 优先尝试 Open3D Tensor/GPU 路径以加速，失败则自动回退到 legacy 路径.
    - 不改变函数签名与返回值。
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points 必须是 (N,3)"
    assert mask.ndim == 1 and mask.shape[0] == points.shape[0], "mask 尺寸不匹配"

    valid_idx = np.nonzero(mask)[0]
    n_valid = int(valid_idx.size)
    if n_valid == 0:
        return mask.copy()
    if n_valid <= k:
        return mask.copy()

    # 准备有效子集
    valid_points = points[valid_idx]
    # Open3D legacy Vector3dVector 需要 float64；Tensor 可用 float32/64
    valid_points64 = valid_points.astype(np.float64, copy=False)

    # 夹紧 k，避免越界与无意义计算
    k = min(max(1, k), n_valid - 1)

    # ---------- 优先：Tensor/GPU 路径（若可用则更快） ----------
    def _try_tensor_sor(vpts: np.ndarray) -> np.ndarray | None:
        try:
            has_t = hasattr(o3d, "t") and hasattr(o3d.t, "geometry")
            if not has_t:
                return None

            # 设备选择：有 CUDA 则用 GPU，否则 CPU
            try:
                dev = o3d.core.Device("CUDA:0") if hasattr(o3d.core, "cuda") and o3d.core.cuda.is_available() \
                      else o3d.core.Device("CPU:0")
            except Exception:
                dev = o3d.core.Device("CPU:0")

            tpcd = o3d.t.geometry.PointCloud(device=dev)
            # 用 float32 节省显存/内存（数值上对 SOR 足够）
            tpcd.point["positions"] = o3d.core.Tensor(vpts.astype(np.float32, copy=False),
                                                      dtype=o3d.core.Dtype.Float32, device=dev)

            # 兼容不同版本的 API 名称
            if hasattr(tpcd, "remove_statistical_outliers"):
                _, inlier_mask = tpcd.remove_statistical_outliers(nb_neighbors=k, std_ratio=std_ratio)
            elif hasattr(tpcd, "remove_statistical_outlier"):
                _, inlier_mask = tpcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
            else:
                return None

            inlier_mask = inlier_mask.cpu().numpy().astype(bool, copy=False)
            return inlier_mask
        except Exception:
            return None

    keep_local = _try_tensor_sor(valid_points)
    if keep_local is None:
        # ---------- 回退：legacy 路径（你原来的实现） ----------
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points64)
        # 返回 (filtered_pcd, inlier_indices)
        _, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
        keep_local = np.zeros(n_valid, dtype=bool)
        if len(inlier_indices) > 0:
            keep_local[np.asarray(inlier_indices, dtype=int)] = True

    # 映射回全局 mask
    new_mask = mask.copy()
    new_mask[valid_idx] = keep_local
    return new_mask

def ensure_tensor(x, device='cpu'):
    """Convert numpy array to torch tensor if needed."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    return x.to(device)

def depth_to_world_points(predictions, extrinsic_mode="c2w"):
    """
    Convert depth map + intrinsics + extrinsic into world 3D points.
    Supports numpy or torch input.
    
    predictions:
        depth:      (N,H,W) or (N,1,H,W)
        intrinsics: (N,3,3)
        extrinsic:  (N,3,4)

    Returns:
        points_world: (N, H, W, 3)
    """

    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- convert depth to tensor ----
    depth = ensure_tensor(predictions['depth'], device)

    # If input is (N,H,W), reshape to (N,1,H,W)
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)   # --> [N,1,H,W]
    elif depth.ndim == 4:
        pass
    else:
        raise ValueError("depth must be (N,H,W) or (N,1,H,W)")

    K = ensure_tensor(predictions['intrinsics'], device)            # [N,3,3]
    try:
        E = ensure_tensor(predictions['camera_poses'], device)      # [N,3,4]
    except:
        E = ensure_tensor(predictions['extrinsic'], device)         # [N,3,4]
        
    N, _, H, W = depth.shape

    # ---------- 1. pixel grid ----------
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    xs = xs.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W)
    ys = ys.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W)

    # ---------- 2. intrinsics ----------
    fx = K[:, 0, 0].view(N, 1, 1, 1)
    fy = K[:, 1, 1].view(N, 1, 1, 1)
    cx = K[:, 0, 2].view(N, 1, 1, 1)
    cy = K[:, 1, 2].view(N, 1, 1, 1)

    z = depth
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z

    points_cam = torch.cat([x_cam, y_cam, z], dim=1)  # [N,3,H,W]
    points_cam = points_cam.view(N, 3, -1)            # [N,3,HW]

    # ---------- 3. extrinsic transform ----------
    R = E[:, :, :3]        # [N,3,3]
    t = E[:, :, 3:].view(N, 3, 1)

    if extrinsic_mode == "c2w":
        points_world = torch.matmul(R, points_cam) + t
    elif extrinsic_mode == "w2c":
        points_world = torch.matmul(
            R.transpose(1, 2),
            points_cam - t
        )
    else:
        raise ValueError("extrinsic_mode must be 'c2w' or 'w2c'")

    points_world = points_world.view(N, 3, H, W)
    points_world = points_world.permute(0, 2, 3, 1)  # --> [N,H,W,3]

    return points_world.cpu().numpy()

def depth_to_local_points(depth, intrinsics):
    """
    depth:      (N, H, W)
    intrinsics: (N, 3, 3)  each with fx, fy, cx, cy
    return:     (N, H, W, 3) float32
    """
    print("shape:", depth.shape, intrinsics.shape)
    N, H, W = depth.shape

    # ----- pixel grid -----
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v, indexing="xy")  # (H, W)

    # broadcast to (N, H, W)
    u = np.broadcast_to(u, (N, H, W))
    v = np.broadcast_to(v, (N, H, W))

    # ----- intrinsics -----
    fx = intrinsics[:, 0, 0].reshape(N, 1, 1)
    fy = intrinsics[:, 1, 1].reshape(N, 1, 1)
    cx = intrinsics[:, 0, 2].reshape(N, 1, 1)
    cy = intrinsics[:, 1, 2].reshape(N, 1, 1)

    Z = depth
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z

    # stack into (N, H, W, 3)
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)

def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []

class X_Long:
    def __init__(self, image_dir, sparse_dir, save_dir, config, Xname: str = 'Pi3'):
        self.config = config

        self.Xname = Xname
        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']

        self.img_dir = image_dir
        self.img_list = None
        self.sparse_dir = sparse_dir
        self.output_dir = save_dir
        self.point_max_error = config['Model']['point_max_error']

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        
        self.all_camera_poses = []
        
        self.delete_temp_files = self.config['Model']['delete_temp_files']
        self.temp_files_location = self.config['Model']['temp_files_location'] # 'disk' or 'cpu_memory'
        
        # 初始化用于内存存储的字典
        if self.temp_files_location == 'cpu_memory':
            self.temp_storage = {}
        else:
            self.temp_storage = None

        print('Loading model...')

        if Xname == 'Pi3':
            self.model = Pi3().to(self.device).eval()
        elif Xname == 'VGGT':
            self.model = VGGT().to(self.device).eval()
        _URL = self.config['Weights'][Xname]

        if _URL.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(_URL)
        else:
            weight = torch.load(_URL, map_location='cuda', weights_only=False)
        self.model.load_state_dict(weight, strict=False)

        self.skyseg_session = None

        # if self.sky_mask:
        #     print('Loading skyseg.onnx...')
        #     # Download skyseg.onnx if it doesn't exist
        #     if not os.path.exists("skyseg.onnx"):
        #         print("Downloading skyseg.onnx...")
        #         download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

        #     self.skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        
        self.chunk_indices = None # [(begin_idx, end_idx), ...]

        self.loop_list = [] # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = [] # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = [] # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        print('init done.')

    def get_loop_pairs(self):

        if self.useDBoW: # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # gray to rgb
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori # (height, width, 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'], 
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands # e.g. cands = (812, 67)
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(frame_id)

        else: # DNIO v2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def align_with_colmap(self, image_paths, images_tensor, predictions):
        print("Align with colmap...")
        img_names = [os.path.basename(img_path) for img_path in image_paths]
        local_points = predictions["local_points"]
        camera_poses = predictions["camera_poses"]
        conf = predictions["conf"]

        if self.Xname in ['VGGT']:
            c2w_34 = camera_poses          # (N,3,4)
            R = c2w_34[..., :3]            # (N,3,3)
            t = c2w_34[..., 3]             # (N,3)

            Rt = np.transpose(R, (0, 2, 1))                    # (N,3,3)  = R^T
            t_inv = -np.einsum('nij,nj->ni', Rt, t)            # (N,3)    = -R^T t

            w2c_44 = np.tile(np.eye(4)[None, ...], (c2w_34.shape[0], 1, 1))  # (N,4,4)
            w2c_44[:, :3, :3] = Rt
            w2c_44[:, :3, 3] = t_inv
            camera_poses = w2c_44.astype(np.float32)
        
        cameras = self.colmap_cameras
        images = self.colmap_images
        name2key = self.colmap_name2key

        points3d_ordered = self.colmap_points3d_ordered
        points3d_error_ordered = self.colmap_points3d_error_ordered
        points3d_rgb_ordered = self.colmap_rgb_ordered

        aabb = None
        if 'aerial' in image_paths[0]: # chunk_len = 90
            aabb = [-12, -8, -1, 11, 9, 6]
        elif 'street' in image_paths[0]: # chunk_len = 15
            aabb = [-950, -650, -5, 200, 300, 100]

        if aabb is not None and not hasattr(self, 'colmap_mask'):
            valid_x = (points3d_ordered[:, 0] > aabb[0]) & (points3d_ordered[:, 0] < aabb[3])
            valid_y = (points3d_ordered[:, 1] > aabb[1]) & (points3d_ordered[:, 1] < aabb[4])
            valid_z = (points3d_ordered[:, 2] > aabb[2]) & (points3d_ordered[:, 2] < aabb[5])
            self.colmap_mask = valid_x & valid_y & valid_z

            if 'street' in image_paths[0]:
                sor_mask = self.colmap_mask
                for i in range(1):
                    sor_mask = sor_with_mask(points3d_ordered, sor_mask, k=6, std_ratio=1.0)
                self.colmap_mask = sor_mask

        N, _, H, W = images_tensor.shape

        point_max_error = self.point_max_error

        pts_list = []
        pred_pts_list = []
        pred_conf_list = []
        pred_colors_list = []

        for idx, name in enumerate(img_names):
            if name not in name2key:
                print("No image:", name)
                continue
            key = name2key[name]
            image_meta = images[key]
            cam_intrinsic = cameras[image_meta.camera_id]
            pts_idx = images[key].point3D_ids

            # filter out invalid 3D points
            mask = pts_idx >= 0
            mask *= pts_idx < len(points3d_ordered)

            # get valid 3D point indices and 2D point xy
            pts_idx = pts_idx[mask]
            valid_xys = image_meta.xys[mask]

            # reduce outliers
            if len(pts_idx) > 0:
                pts_errors = points3d_error_ordered[pts_idx]
                valid_errors = pts_errors < point_max_error
                if aabb is not None:
                    valid_mask = self.colmap_mask
                    valid_mask = valid_mask[pts_idx]
                    valid_errors = valid_errors & valid_mask

                pts_idx = pts_idx[valid_errors]
                valid_xys = valid_xys[valid_errors]

            else:
                print("no outliers")

            if len(pts_idx) > 0:
                # get 3D point xyz
                pts = points3d_ordered[pts_idx]
            else:
                pts = np.array([0, 0, 0])
            pts_ori = pts.copy()

            # transform from world to camera
            R = colmap_utils.qvec2rotmat(image_meta.qvec)
            pts = np.dot(pts, R.T) + image_meta.tvec

            invcolmapdepth = 1. / pts[..., 2]
            invmonodepthmap = 1. / local_points[idx][..., 2]
            pred_conf = conf[idx]
            pred_pose = camera_poses[idx]
            pred_xy = local_points[idx][..., :2] / local_points[idx][..., 2:]

            resized_invmonodepthmap = cv2.resize(
                invmonodepthmap, 
                dsize=(cam_intrinsic.width, cam_intrinsic.height), 
                interpolation=cv2.INTER_LINEAR
            )

            invdepth_norm = (resized_invmonodepthmap - resized_invmonodepthmap.min()) / (resized_invmonodepthmap.max() - resized_invmonodepthmap.min())
            depth = (invdepth_norm * 255).clip(0, 255).astype(np.uint8)
            colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            # os.makedirs("depths", exist_ok=True)
            # depth_save_path = os.path.join("depths", name)
            # cv2.imwrite(depth_save_path , colored_depth)

            s = W / cam_intrinsic.width, H / cam_intrinsic.height
            maps = (valid_xys * s).astype(np.float32)
            valid = (
                (maps[..., 0] >= 0) *
                (maps[..., 1] >= 0) *
                (maps[..., 0] < cam_intrinsic.width * s[0]) *
                (maps[..., 1] < cam_intrinsic.height * s[1]) * (invcolmapdepth > 0))
            
            if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
                pts_ori = pts_ori[valid, :]
                maps = maps[valid, :]
                invcolmapdepth = invcolmapdepth[valid]
                invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
                remap_x = cv2.remap(pred_xy[..., 0], maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
                remap_y = cv2.remap(pred_xy[..., 1], maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
                remap_conf = cv2.remap(pred_conf, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)[..., 0]
                monodepth = 1. / invmonodepth
                pred_xyz = np.column_stack([remap_x * monodepth, remap_y * monodepth, monodepth])
                pred_pts = torch.einsum('ij, nj -> ni', torch.tensor(pred_pose), homogenize_points(torch.tensor(pred_xyz)))[..., :3].numpy()
                pts_list.append(pts_ori)
                pred_pts_list.append(pred_pts)
                pred_conf_list.append(remap_conf)

                img_rgb = images_tensor[idx].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                img_rgb_255 = (img_rgb * 255).astype(np.uint8)
                remap_rgb = cv2.remap(img_rgb_255, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                
                pred_colors_list.append(remap_rgb)

        colmap_pts = np.concatenate(pts_list)[np.newaxis, ...]
        colmap_conf = np.ones(colmap_pts.shape[1])[np.newaxis, ...]
        pred_pts = np.concatenate(pred_pts_list)[np.newaxis, ...]
        pred_conf = np.concatenate(pred_conf_list)[np.newaxis, ...]

        conf_threshold = np.median(pred_conf) * 0.01
        s, R, t = weighted_align_point_maps(colmap_pts, colmap_conf, pred_pts, pred_conf, conf_threshold=conf_threshold, config=self.config)

        S = np.eye(4)
        S[:3, :3] = s * R
        S[:3, 3] = t
        w2c = camera_poses
        c2w = np.linalg.inv(w2c)
        transformed_c2w = S @ c2w  # Be aware of the left multiplication!
        # if self.Xname == 'Pi3':
        #     assert predictions['camera_poses'].shape == transformed_c2w.shape

        predictions['points'] = apply_sim3_direct(predictions['points'], s, R, t)
        if self.Xname == 'Pi3':
            predictions['camera_poses'] = np.linalg.inv(transformed_c2w)
        elif self.Xname in ['VGGT']:
            predictions['camera_poses'] = transformed_c2w[:, :3, :]

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        if self.Xname == 'Pi3':
            images = load_images_as_tensor_pi_long(chunk_image_paths).to(self.device)
        elif self.Xname == 'VGGT':
            images = load_and_preprocess_images(chunk_image_paths).to(self.device)

        print(f"Loaded {len(images)} images")
        
        # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        if self.Xname in ['Pi3', 'VGGT']:
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(images[None])

        predictions['images'] = images[None]

        # see issue https://github.com/yyfz/Pi3/issues/55
        if self.Xname == 'Pi3':
            conf = predictions['conf']
            conf = torch.sigmoid(conf)
            predictions['conf'] = conf
        torch.cuda.empty_cache()

        print("Processing model outputs...")

        if self.Xname == "VGGT":
            print("Converting pose encoding to extrinsic and intrinsic matrices...")
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["intrinsics"] = intrinsic.squeeze(0).cpu().numpy()
            predictions["camera_poses"] = extrinsic
            predictions["points"] = predictions.pop("world_points")
            predictions["conf"] = predictions.pop("world_points_conf")
            predictions["depth"] = predictions["depth"].squeeze(0).squeeze(-1).cpu().numpy()
        
        if self.Xname in ['VGGT']:
            local_points = depth_to_local_points(
                predictions['depth'],
                predictions['intrinsics']
            )  # (N, H, W, 3)
            predictions['local_points'] = local_points
        
        if self.temp_files_location == 'cpu_memory':
            if is_loop:
                key = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}"
            else:
                if chunk_idx is None:
                    raise ValueError("chunk_idx must be provided when is_loop is False")
                key = f"chunk_{chunk_idx}"
            
            predictions_cpu = {}
            for k, v in predictions.items():
                if isinstance(v, torch.Tensor):
                    predictions_cpu[k] = v.cpu().numpy().squeeze(0)
            
            if not is_loop and range_2 is None:
                extrinsics = predictions['camera_poses']
                chunk_range = self.chunk_indices[chunk_idx]
                self.all_camera_poses.append((chunk_range, extrinsics))
            
            self.temp_storage[key] = predictions_cpu
            return predictions_cpu if is_loop or range_2 is not None else None
        else:
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            
            if self.sparse_dir:
                self.align_with_colmap(chunk_image_paths, images, predictions)

            # Save predictions to disk instead of keeping in memory
            if is_loop:
                save_dir = self.result_loop_dir
                filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
            else:
                if chunk_idx is None:
                    raise ValueError("chunk_idx must be provided when is_loop is False")
                save_dir = self.result_unaligned_dir
                filename = f"chunk_{chunk_idx}.npy"
            
            save_path = os.path.join(save_dir, filename)
                        
            if not is_loop and range_2 is None:
                extrinsics = predictions['camera_poses']
                chunk_range = self.chunk_indices[chunk_idx]
                self.all_camera_poses.append((chunk_range, extrinsics))
            
            if self.Xname == "VGGT":
                predictions['depth'] = np.squeeze(predictions['depth'])

            # print("predictions analyze...")
            # for key in predictions.keys():
            #     print("[key|shape|type]:", key, predictions[key].shape, type(predictions[key]))

            np.save(save_path, predictions)
            
            return predictions if is_loop or range_2 is not None else None
    
    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                if start_idx + self.chunk_size > len(self.img_list): # my add
                    start_idx = max(0, len(self.img_list) - self.chunk_size)
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))
        
        if self.sparse_dir:
            cameras, images, points3d = colmap_utils.read_model(self.sparse_dir)
            name2key = {images[key].name: key for key in images}

            pts_indices = np.array([points3d[key].id for key in points3d])
            pts_xyzs = np.array([points3d[key].xyz for key in points3d])
            pts_errors = np.array([points3d[key].error for key in points3d])
            points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
            points3d_error_ordered = np.zeros([pts_indices.max() + 1, ])
            points3d_ordered[pts_indices] = pts_xyzs
            points3d_error_ordered[pts_indices] = pts_errors
            rgb_ordered = np.zeros([pts_indices.max() + 1, 3])

            self.colmap_cameras = cameras
            self.colmap_images = images
            self.colmap_name2key = name2key
            self.colmap_points3d_ordered = points3d_ordered
            self.colmap_points3d_error_ordered = points3d_error_ordered
            self.colmap_rgb_ordered = rgb_ordered

        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(self.chunk_indices, 
                                             self.loop_list, 
                                             half_window = int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

        print(f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        del self.model # Save GPU Memory
        torch.cuda.empty_cache()

        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices)-1):

            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            if self.temp_files_location == 'cpu_memory':
                chunk_data1 = self.temp_storage[f"chunk_{chunk_idx}"]
                chunk_data2 = self.temp_storage[f"chunk_{chunk_idx+1}"]
            else:
                chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
                chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            point_map1 = chunk_data1['points'][-self.overlap:]
            point_map2 = chunk_data2['points'][:self.overlap]
            conf1 = np.squeeze(chunk_data1['conf'][-self.overlap:])
            conf2 = np.squeeze(chunk_data2['conf'][:self.overlap])

            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            s, R, t = weighted_align_point_maps(point_map1, 
                                                conf1, 
                                                point_map2, 
                                                conf2, 
                                                conf_threshold=conf_threshold,
                                                config=self.config)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            self.sim3_list.append((s, R, t))


        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                print('chunk_a align')
                point_map_loop = item[1]['points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = np.squeeze(item[1]['conf'][:chunk_a_range[1] - chunk_a_range[0]])
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                print(self.chunk_indices[chunk_idx_a])
                print(chunk_a_range)
                print(chunk_a_rela_begin, chunk_a_rela_end)

                if self.temp_files_location == 'cpu_memory':
                    chunk_data_a = self.temp_storage[f"chunk_{chunk_idx_a}"]
                else:
                    chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
                
                point_map_a = chunk_data_a['points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = np.squeeze(chunk_data_a['conf'][chunk_a_rela_begin:chunk_a_rela_end])
            
                conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                s_a, R_a, t_a = weighted_align_point_maps(point_map_a, 
                                                          conf_a, 
                                                          point_map_loop, 
                                                          conf_loop, 
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                print('chunk_a align')
                point_map_loop = item[1]['points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = np.squeeze(item[1]['conf'][-chunk_b_range[1] + chunk_b_range[0]:])
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)

                if self.temp_files_location == 'cpu_memory':
                    chunk_data_b = self.temp_storage[f"chunk_{chunk_idx_b}"]
                else:
                    chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
                
                point_map_b = chunk_data_b['points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = np.squeeze(chunk_data_b['conf'][chunk_b_rela_begin:chunk_b_rela_end])
            
                conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                s_b, R_b, t_b = weighted_align_point_maps(point_map_b, 
                                                          conf_b, 
                                                          point_map_loop, 
                                                          conf_loop, 
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print('a -> b SIM 3')
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))


        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]
            
            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # Visual in png format
            plt.figure(figsize=(8, 8))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print('Apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices)-1):
            print(f'Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})')
            s, R, t = self.sim3_list[chunk_idx]
            
            if self.temp_files_location == 'cpu_memory':
                chunk_data = self.temp_storage[f"chunk_{chunk_idx+1}"]
            else:
                chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            chunk_data['points'] = apply_sim3_direct(chunk_data['points'], s, R, t)
            
            if self.temp_files_location == 'cpu_memory':
                self.temp_storage[f"chunk_{chunk_idx+1}"] = chunk_data
            else:
                aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
                np.save(aligned_path, chunk_data)
            
            if chunk_idx == 0:
                if self.temp_files_location == 'cpu_memory':
                    chunk_data_first = self.temp_storage[f"chunk_0"]
                else:
                    chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"), allow_pickle=True).item()
                points = chunk_data_first['points'].reshape(-1, 3)
                confs = chunk_data_first['conf'].reshape(-1)
                colors = (chunk_data_first['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
                ply_path = os.path.join(self.pcd_dir, f'{chunk_idx}_pcd.ply')
                save_confident_pointcloud_batch(
                    points=points,              # shape: (H, W, 3)
                    colors=colors,              # shape: (H, W, 3)
                    confs=confs,          # shape: (H, W)
                    output_path=ply_path,
                    conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                    sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
                )

            if self.temp_files_location == 'cpu_memory':
                aligned_chunk_data = self.temp_storage[f"chunk_{chunk_idx}"] if chunk_idx > 0 else chunk_data_first
            else:
                aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first
            
            points = aligned_chunk_data['points'].reshape(-1, 3)
            confs = aligned_chunk_data['conf'].reshape(-1)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx+1}_pcd.ply')
            save_confident_pointcloud_batch(
                points=points,              # shape: (H, W, 3)
                colors=colors,              # shape: (H, W, 3)
                confs=confs,          # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        if len(self.chunk_indices) == 1:
            chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"), allow_pickle=True).item()
            np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            
            aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first
            
            points = aligned_chunk_data['points'].reshape(-1, 3)
            confs = aligned_chunk_data['conf'].reshape(-1)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx}_pcd.ply')
            save_confident_pointcloud_batch(
                points=points,              # shape: (H, W, 3)
                colors=colors,              # shape: (H, W, 3)
                confs=confs,          # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )


        self.save_camera_poses()
        
        print('Done.')

    
    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) + 
                                glob.glob(os.path.join(self.img_dir, "*.png")) + 
                                glob.glob(os.path.join(self.img_dir, "*.JPG")))
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close() # Save CPU Memory
                gc.collect()
            else:
                del self.loop_detector # Save GPU Memory
        torch.cuda.empty_cache()

        self.process_long_sequence()

    def save_camera_poses(self):
        '''
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        '''
        chunk_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],    # Dark Red
            [0, 128, 0],    # Dark Green
            [0, 0, 128],    # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")
        
        all_poses = [None] * len(self.img_list)
        
        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):

            c2w = first_chunk_extrinsics[i] # camera pose of Pi3 is C2W while it is W2C in VGGT!
            all_poses[idx] = c2w

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            s, R, t = self.sim3_list[chunk_idx-1]   # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.
            
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                if self.Xname in ['Pi3']:
                    c2w = chunk_extrinsics[i] # camera pose of Pi3 is C2W while it is W2C in VGGT!
                elif self.Xname in ['VGGT']:
                    w2c = np.eye(4)
                    w2c[:3, :] = chunk_extrinsics[i]
                    c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')
        
        print(f"Camera poses saved to {poses_path}")
        
        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            
            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')
        
        print(f"Camera poses visualization saved to {ply_path}")
    
    def close(self):
        '''
            Clean up temporary files and calculate reclaimed disk space.
            
            This method deletes all temporary files generated during processing from three directories:
            - Unaligned results
            - Aligned results
            - Loop results
            
            ~50 GiB for 4500-frame KITTI 00, 
            ~35 GiB for 2700-frame KITTI 05, 
            or ~5 GiB for 300-frame short seq.
        '''
        if not self.delete_temp_files:
            return
        
        total_space = 0

        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print('Deleting temp files done.')

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil
def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
        
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='X-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='Image path')
    parser.add_argument('--sparse_dir', type=str, required=False, default=None,
                        help='sparse path')
    parser.add_argument('--save_dir', type=str, required=False, default=None,
                        help='sabe path')
    parser.add_argument('--Xname', '-x', type=str, required=False, default='Pi3',
                        help='sparse path')
    args = parser.parse_args()

    args.config = Path(args.config)
    base_dir = Path(__file__).resolve().parents[0]
    if not args.config.is_absolute():
        args.config = base_dir / args.config

    config = load_config(args.config)

    for key in config['Weights']:
        weight_path = Path(config['Weights'][key])
        if not weight_path.is_absolute():
            config['Weights'][key] = str(base_dir / weight_path)

    image_dir = args.image_dir
    path = image_dir.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'

    sparse_dir = args.sparse_dir
    if sparse_dir and os.path.isdir(os.path.join(sparse_dir, "0")):
        sparse_dir = os.path.join(sparse_dir, "0")

    if args.save_dir is None:
        save_dir = os.path.join(
                exp_dir, image_dir.replace("/", "_"), current_datetime
            )
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    import time
    start_time = time.time()
    x_long = X_Long(image_dir, sparse_dir, save_dir, config, args.Xname)
    x_long.run()
    x_long.close()
    end_time = time.time()
    print(f"Time cosume:{end_time - start_time:.2f}s")

    del x_long
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('X-Long done.')
    sys.exit()
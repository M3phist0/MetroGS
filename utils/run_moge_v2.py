import add_pypath
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
# from internal.utils.visualizers import Visualizers
from common import find_files, AsyncNDArraySaver, AsyncImageSaver, AsyncImageReader
from distibuted_tasks import configure_arg_parser, get_task_list_with_args
from utils.distibuted_tasks import get_task_list_with_args
from internal.utils.colmap import read_model
from internal.utils.graphics_utils import focal2fov

import json
from joblib import delayed, Parallel

parser = argparse.ArgumentParser()
parser.add_argument("image_dir")
parser.add_argument("--output", "-o", default=None)
parser.add_argument("--dataset_dir", "-i", default=None)
parser.add_argument("--downsample_factor", "-d", type=float, default=1)
parser.add_argument("--extensions", "-e", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
parser.add_argument("--preview", "-p", action="store_true", default=False)
parser.add_argument("--colormap", type=str, default="default")
parser.add_argument("--moge_path", type=str, default=os.path.join(os.path.dirname(__file__), "MoGe"))
parser.add_argument("--point-max-error", type=float, default=1.5)
configure_arg_parser(parser)
args = parser.parse_args()

sys.path.insert(0, args.moge_path)
from moge.model.v2 import MoGeModel

if args.output is None:
    args.output = os.path.join(os.path.dirname(args.image_dir), "estimated_mask_depths")
if args.dataset_dir is None:
    args.dataset_dir = os.path.dirname(args.image_dir)
sparse_model_dir = os.path.join(args.dataset_dir, "sparse")
if os.path.exists(os.path.join(sparse_model_dir, "images.bin")) is False:
    sparse_model_dir = os.path.join(sparse_model_dir, "0")

colmap_cameras, colmap_images, _ = read_model(sparse_model_dir)
name2key = {colmap_images[key].name: key for key in colmap_images}

images = get_task_list_with_args(args, find_files(args.image_dir, args.extensions, as_relative_path=False))
assert len(images) > 0, "not an image with extension name '{}' can be found in '{}'".format(args.extensions, args.image_dir)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

MoGe = MoGeModel.from_pretrained(f"{args.moge_path}/checkpoints/moge-2-vitl-normal.pt")                    
MoGe = MoGe.to(DEVICE).eval()

def apply_color_map(normalized_depth):
    depth = normalized_depth
    depth = (depth * 255).clip(0, 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return colored_depth

ndarray_saver = AsyncNDArraySaver()
image_reader = AsyncImageReader(image_list=images)
image_saver = AsyncImageSaver(is_rgb=True)
try:
    with torch.no_grad(), tqdm(range(len(images))) as t:
        for _ in t:
            image_path, raw_image = image_reader.get()
            image_name = image_path[len(args.image_dir):].lstrip(os.path.sep)
            if not image_name in name2key:
                continue
            camera_id = colmap_images[name2key[image_name]].camera_id
            intrinsics = colmap_cameras[camera_id]
            focal_length_x = intrinsics.params[0]

            input_image = torch.tensor(raw_image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)   

            height, width = input_image.shape[1:]
            if args.downsample_factor != 1:
                resized_height, resized_width = round(height / args.downsample_factor), round(width / args.downsample_factor)

                focal_length_x *= resized_width / width

                input_image_batch = input_image.unsqueeze(0)
                input_image_resized = torch.nn.functional.interpolate(
                    input_image_batch, 
                    size=(resized_height, resized_width), 
                    mode='bicubic', # 或者 'area'
                    align_corners=False
                )
                
                height, width = resized_height, resized_width
                input_image = input_image_resized.squeeze(0)

            fov_x = focal2fov(focal_length_x, width)

            output = MoGe.infer(input_image, fov_x=fov_x, use_fp16=False)
            depth = 1./ output["depth"].detach().cpu().numpy()
            mask = output["mask"].detach().cpu().numpy()
            depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min())
            depth[~mask] = 0
            normalized_depth = depth

            # if args.downsample_factor != 1:
            #     height, width = normalized_depth.shape
            #     resized_height, resized_width = round(height / args.downsample_factor), round(width / args.downsample_factor)
            #     normalized_depth = np.array(Image.fromarray(normalized_depth).resize((resized_width, resized_height)))
                
            #     mask_uint8 = mask.astype(np.uint8)
            #     mask = np.array(Image.fromarray(mask_uint8).resize((resized_width, resized_height)))
            # else:
            #     mask = mask.astype(np.uint8)

            combined_array = np.stack([normalized_depth, mask], axis=-1)
            
            output_filename = os.path.join(args.output, "{}.npy".format(image_name))
            ndarray_saver.save(combined_array, output_filename)
            # ndarray_saver.save(normalized_depth, output_filename)

            if args.preview is True:
                os.makedirs("./mogetmp/", exist_ok=True)
                image_saver.save(normalized_depth, os.path.join("./mogetmp/", "{}.png".format(image_name)), processor=apply_color_map)
                # image_saver.save(normalized_depth, os.path.join(args.output, "{}.png".format(image_name)), processor=apply_color_map)

finally:
    ndarray_saver.stop()
    image_reader.stop()
    image_saver.stop()
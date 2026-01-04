import os
import json
import numpy as np
import torch
from dataclasses import dataclass
from .colmap_block_dataparser import ColmapBlock, ColmapBlockDataParser
from internal.dataparsers import DataParserOutputs
from internal.dataparsers.dataparser import PointCloud


@dataclass
class EstimatedDepthBlockColmap(ColmapBlock):
    depth_dir: str = "estimated_mask_depths"

    use_mv_mask: bool = False

    depth_rescaling: bool = True

    depth_scale_name: str = "estimated_mask_depth_scales"

    depth_scale_lower_bound: float = 0.2

    depth_scale_upper_bound: float = 5.

    additional_ply_path: str = None

    overwrite_val_path: str = None

    def instantiate(self, path: str, output_path: str, global_rank: int) -> "EstimatedDepthBlockColmapDataParser":
        return EstimatedDepthBlockColmapDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


class EstimatedDepthBlockColmapDataParser(ColmapBlockDataParser):
    def overwrite_val_set(self, val_path):
        ori_path = self.path
        ori_mode = self.params.split_mode

        self.path = val_path
        self.params.split_mode = "reconstruction"
        new_val_set = super().get_outputs().train_set

        self.path = ori_path
        self.params.split_mode = ori_mode

        return new_val_set


    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        if self.params.additional_ply_path:
            from internal.utils.graphics_utils import fetch_ply_without_rgb_normalization
            basic_pcd = fetch_ply_without_rgb_normalization(self.params.additional_ply_path)
            xyz = basic_pcd.points
            rgb = basic_pcd.colors
            point_cloud=dataparser_outputs.point_cloud
            dataparser_outputs.point_cloud.xyz = np.vstack([point_cloud.xyz, xyz])
            dataparser_outputs.point_cloud.rgb = np.vstack([point_cloud.rgb, rgb])
            
            print("additional load {} points from {}".format(xyz.shape[0], self.params.additional_ply_path))
        
        if self.params.overwrite_val_path:
            new_val_set = self.overwrite_val_set(self.params.overwrite_val_path)
            dataparser_outputs.val_set = new_val_set
        
        print("final train set number:", len(dataparser_outputs.train_set), "val set number:", len(dataparser_outputs.val_set))

        if self.params.depth_rescaling is True:
            with open(os.path.join(self.path, self.params.depth_scale_name + ".json"), "r") as f:
                depth_scales = json.load(f)

            median_scale = np.median(np.asarray([i["scale"] for i in depth_scales.values()]))

        loaded_depth_count = 0
        skip_count = 0
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                depth_file_path = os.path.join(self.path, self.params.depth_dir, f"{image_name}.npy")
                if os.path.exists(depth_file_path) is False:
                    print("[WARNING] {} does not have a depth file".format(image_name))
                    continue

                depth_scale = {
                    "scale": 1.,
                    "offset": 0.,
                }
                if self.params.depth_rescaling is True:
                    depth_scale = depth_scales.get(image_name, None)
                    if depth_scale is None:
                        # print("[WARNING {} does not have a depth scale]".format(image_name))
                        skip_count += 1
                        continue
                    if depth_scale["scale"] < self.params.depth_scale_lower_bound * median_scale or depth_scale["scale"] > self.params.depth_scale_upper_bound * median_scale:
                        # print("[WARNING depth scale of {} out of bound]".format(image_name))
                        skip_count += 1
                        continue
                
                image_set.extra_data[idx] = (depth_file_path, depth_scale)
                loaded_depth_count += 1
            image_set.extra_data_processor = self.load_depth

        # assert loaded_depth_count > 0
        print("found {} depth maps".format(loaded_depth_count))
        print("skip {} depth maps".format(skip_count))

        return dataparser_outputs

    def load_depth(self, depth_info):
        if depth_info is None:
            return None

        depth_file_path, depth_scale = depth_info
        file_name, file_extension = os.path.splitext(depth_file_path)
        depth_file_mv_path = file_name + '.mv' + file_extension
        if self.params.use_mv_mask and os.path.exists(depth_file_mv_path):
            depth_file_path = depth_file_mv_path
        depth_data = np.load(depth_file_path)
        depth, mask = depth_data[..., 0], depth_data[..., 1]
        depth = depth.astype(np.float32)
        mask = mask.astype(np.uint8)
        depth = depth * depth_scale["scale"] + depth_scale["offset"]

        return torch.tensor(depth), torch.tensor(mask)

from typing import Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .citygsv2_metrics import CityGSV2Metrics, CityGSV2MetricsModule
import torch.nn.functional as F
import numpy as np
from internal.utils.general_utils import build_rotation, build_scaling_rotation
from internal.utils.propagate_utils import depth_propagation, check_geometric_consistency
from internal.utils.rank import get_random_depth_ranking_loss, get_random_depth_boundary_loss, get_depth_completion


@dataclass
class WeightScheduler:
    init: float = 1.0

    final_factor: float = 0.01

    max_steps: int = 30_000


@dataclass
class DistributedMetrics(CityGSV2Metrics):
    lambda_normal: float = 0.05
    lambda_dist: float = 0.0

    lambda_scale: float = 0.025

    pixel_noise_th: float = 2.0
    multi_view_from: int = 30000
    mv_epoch_interval: int = 5

    normal_regularization_from_iter: int = 7000

    depth_loss_type: Literal["l1", "l1+ssim", "l2", "kl"] = "l1"

    depth_loss_ssim_weight: float = 0.2

    depth_loss_weight: WeightScheduler = field(default_factory=lambda: WeightScheduler())

    depth_normalized: bool = False

    depth_output_key: str = "inverse_depth"

    def instantiate(self, *args, **kwargs) -> "DistributedMetricsModule":
        return DistributedMetricsModule(self)


class DistributedMetricsModule(CityGSV2MetricsModule):
    config: DistributedMetrics

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        assert isinstance(batch, list)
        bsz = len(batch)

        use_multi_view = pl_module.trainer.datamodule.use_multi_view
        current_epoch = pl_module.current_epoch
        need_propagate = False
        src_outputs = {}
        if use_multi_view and step >= self.config.multi_view_from:
            if not hasattr(self, 'mv_cache'):
                self.mv_cache = {}
            # if mv_cache not initialized for some image | epoch % interval == 0 -> need propagate
            max_src_num = 0
            for i, item in enumerate(batch):
                camera, image_info, extra_data = item
                image_name, _, _ = image_info
                if image_name not in self.mv_cache:
                    need_propagate = True
                mutli_view_data = extra_data[1]
                if mutli_view_data[-1] is not None:
                    max_src_num = max(max_src_num, len(mutli_view_data[0]))

            if not need_propagate:
                if current_epoch % self.config.mv_epoch_interval == 0:
                    need_propagate = True
            
            need_propagate_tmp = need_propagate

            need_propagate_tensor = torch.tensor([int(need_propagate)], dtype=torch.int, device="cuda")
            torch.distributed.all_reduce(need_propagate_tensor, op=torch.distributed.ReduceOp.SUM)
            need_propagate = need_propagate_tensor.item() > 0

            if need_propagate and max_src_num > 0:
                src_camera_list2 = [[] for _ in range(max_src_num)]
                src_name_list2 = [[] for _ in range(max_src_num)]
                for i, item in enumerate(batch):
                    _, image_info, extra_data = item
                    image_name, _, _ = image_info
                    mutli_view_data = extra_data[1]
                    src_camera_list, src_name_list, src_gt_image_list = mutli_view_data
                    if mutli_view_data[-1] is not None:
                        for j, (src_camera, src_name) in enumerate(zip(src_camera_list, src_name_list)):
                            src_camera_list2[j].append(src_camera)
                            src_name_list2[j].append(src_name)
                        for j in range(len(src_camera_list), max_src_num):
                            src_camera_list2[j].append(camera)
                            src_name_list2[j].append(image_name)
                    else:
                        for j in range(max_src_num):
                            src_camera_list2[j].append(camera)
                            src_name_list2[j].append(image_name)
                
                with torch.no_grad():
                    for i, (src_camera_list, src_name_list) in enumerate(zip(src_camera_list2, src_name_list2)):
                        src_output_list = pl_module(src_camera_list)
                        for j, (src_output, src_name) in enumerate(zip(src_output_list, src_name_list)):
                            if src_output['render'] is not None:
                                src_outputs[src_name] = src_output

            need_propagate = need_propagate_tmp

        metrics_list = []
        for i, (item, output) in enumerate(zip(batch, outputs)):
            if output["render"] is None:
                continue

            coverage_y = output["coverage_y"]

            camera, image_info, extra_data = item
            if use_multi_view:
                gt_inverse_depth = extra_data[0]
            else:
                gt_inverse_depth = extra_data
            image_name, gt_image, masked_pixels = image_info
            image = output["render"]

            # calculate loss
            if masked_pixels is not None:
                masked_pixels = masked_pixels.to(torch.uint8)  # False represents masked pixels
                # TODO: avoid repeatedly masking G.T. image
                gt_image = gt_image * masked_pixels
                image = image * masked_pixels

            image, gt_image = image.clone()[:, coverage_y[0]: coverage_y[1], :], gt_image.clone()[:, coverage_y[0]: coverage_y[1], :]

            rgb_diff_loss = self.rgb_diff_loss_fn(image, gt_image)
            ssim_metric = self.ssim(image, gt_image)
            loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1. - ssim_metric)

            metrics_i = {
                "loss": loss,
                "rgb_diff": rgb_diff_loss,
                "ssim": ssim_metric,
            }

            # regularization
            lambda_normal = self.config.lambda_normal if step > 7000 else 0.0
            lambda_dist = self.config.lambda_dist if step > 3000 else 0.0

            rend_dist = output["rend_dist"][:, coverage_y[0]: coverage_y[1], :]
            rend_normal = output['rend_normal'][:, coverage_y[0]: coverage_y[1], :]
            surf_normal = output['surf_normal'][:, coverage_y[0]: coverage_y[1], :]
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            # update metrics
            metrics_i["loss"] = metrics_i["loss"] + dist_loss + normal_loss
            metrics_i["normal_loss"] = normal_loss
            metrics_i["dist_loss"] = dist_loss

            d_reg_weight = self.get_weight(step)
            if gt_inverse_depth is None or (use_multi_view and step >= self.config.multi_view_from):
                d_reg = torch.tensor(0., device=camera.device)
            else:
                predicted_inverse_depth = 1. / (output["surf_depth"].clamp_min(0.).squeeze() + 1e-8)
                if self.config.depth_normalized:
                    # with torch.no_grad():
                    clamp_val = (predicted_inverse_depth.mean() + 2 * predicted_inverse_depth.std()).item()
                    predicted_inverse_depth = predicted_inverse_depth.clamp(max=clamp_val) / clamp_val
                    gt_inverse_depth = gt_inverse_depth.clamp(max=clamp_val) / clamp_val

                tot_mask_pixel = 0
                if isinstance(gt_inverse_depth, tuple):
                    gt_inverse_depth, gt_inverse_depth_mask = gt_inverse_depth

                    gt_inverse_depth = gt_inverse_depth * gt_inverse_depth_mask
                    predicted_inverse_depth = predicted_inverse_depth * gt_inverse_depth_mask

                    tot_mask_pixel = gt_inverse_depth_mask.sum()

                gt_inverse_depth, predicted_inverse_depth = \
                    gt_inverse_depth[coverage_y[0]: coverage_y[1], :], predicted_inverse_depth[coverage_y[0]: coverage_y[1], :]

                d_reg = self._get_inverse_depth_loss(gt_inverse_depth, predicted_inverse_depth) * d_reg_weight

                if tot_mask_pixel > 0:
                    tot_pixel = gt_inverse_depth.shape[0] * gt_inverse_depth.shape[1]
                    d_reg = d_reg * tot_pixel / tot_mask_pixel

                metrics_i["d_reg"] = d_reg
                metrics_i["d_w"] = d_reg_weight

                # n_reg
                # if use_multi_view:
                #     gt_inverse_depth = extra_data[0]
                # else:
                #     gt_inverse_depth = extra_data
                # if gt_inverse_depth is not None and not (use_multi_view and step >= self.config.multi_view_from):
                #     if isinstance(gt_inverse_depth, tuple):
                #         gt_inverse_depth, gt_inverse_depth_mask = gt_inverse_depth

                #     valid_mask = gt_inverse_depth > 0.0
                #     gt_depth = 1. / gt_inverse_depth.unsqueeze(0)
                #     with torch.no_grad():
                #         gt_normal = pl_module.renderer.depth_to_normal(camera, gt_depth).permute(2, 0, 1)
                #     n_reg = (1 - (rend_normal * gt_normal).sum(dim=0))[valid_mask].mean()
                #     n_reg_weight = d_reg_weight * 0.025
                #     n_reg = n_reg_weight * n_reg
                #     if torch.isnan(n_reg):
                #         n_reg = 0.0

                #     metrics_i["n_reg"] = n_reg
                #     metrics_i["loss"] = metrics_i["loss"] + n_reg

            if step < pl_module.hparams["density"].densify_until_iter:
                metrics_i["loss"] = pl_module.hparams["metric"].lambda_dssim * (1. - metrics_i["ssim"]) + metrics_i["dist_loss"] + metrics_i["normal_loss"] + d_reg
                metrics_i["extra_loss"] = (1.0 - pl_module.hparams["metric"].lambda_dssim) * metrics_i["rgb_diff"]
            else:
                metrics_i["loss"] = metrics_i["loss"] + d_reg

            # scale regularization
            visible_mask = output["visibility_filter"]
            scales = output["scales"][visible_mask]
            sorted_scale, _ = torch.sort(scales, dim=-1)
            s_reg = self.config.lambda_scale * (sorted_scale[:, -1] / sorted_scale[:, 0]).mean()
            metrics_i["loss"] = metrics_i["loss"] + s_reg

            # multi-view loss
            if use_multi_view and step >= self.config.multi_view_from:
                if step >= self.config.multi_view_from + 10000:
                    if self.config.pixel_noise_th != 1.0:
                        self.config.pixel_noise_th = 1.0
                elif step >= self.config.multi_view_from + 5000:
                    if self.config.pixel_noise_th != 1.5:
                        self.config.pixel_noise_th = 1.5
                else:
                    self.config.pixel_noise_th = 2.0

                ref_depth = output['surf_depth']
                mutli_view_data = extra_data[1]
                src_camera_list, src_name_list, src_gt_image_list = mutli_view_data
                if need_propagate and src_gt_image_list is not None:
                    with torch.no_grad():
                        propagated_depth, propagated_normal = \
                            depth_propagation(camera, gt_image, ref_depth, src_camera_list, src_gt_image_list, rend_normal, max_scale=1)
                        
                        ref_K = camera.get_K()[:3, :3]
                        ref_pose = camera.world_to_camera.transpose(0, 1).inverse()
                        geometric_counts = None
                        for src_camera, src_name in zip(src_camera_list, src_name_list):
                            src_K = src_camera.get_K()[:3, :3]
                            src_pose = src_camera.world_to_camera.transpose(0, 1).inverse()
                            src_depth = src_outputs[src_name]['surf_depth']

                            mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = \
                                check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                            ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                            src_K.unsqueeze(0), src_pose.unsqueeze(0), 
                                                            thre1=self.config.pixel_noise_th, thre2=0.01)

                            if geometric_counts is None:
                                geometric_counts = mask.to(torch.uint8)
                            else:
                                geometric_counts += mask.to(torch.uint8)
                        
                        count = geometric_counts.squeeze()
                        valid_mask = count >= 1

                        # use mono completion
                        gt_inverse_depth = extra_data[0]
                        if gt_inverse_depth is not None:
                            valid_mask = valid_mask.unsqueeze(0)
                            input_depth = 1. / propagated_depth.unsqueeze(0)
                            
                            if isinstance(gt_inverse_depth, tuple):
                                gt_inverse_depth, gt_inverse_depth_mask = gt_inverse_depth
                            for psz in [16, 32, 64, 128, 256, 512]:
                                transformed_depth = get_depth_completion(input_depth, gt_inverse_depth.unsqueeze(0), valid_mask, patch_size=psz)
                                if transformed_depth is not None:
                                    new_valid_mask = torch.abs(transformed_depth - input_depth) < 0.0025
                                    valid_mask = torch.logical_or(valid_mask, new_valid_mask)
                            valid_mask = valid_mask.squeeze(0)

                        inv_propagated_depth = 1. / propagated_depth
                        inv_propagated_depth[~valid_mask] = 0.0

                        # if image_name in self.mv_cache:
                        #     inv_propagated_depth_prev = self.mv_cache[image_name]
                        #     valid_mask_prev = inv_propagated_depth_prev > 0.0
                        #     valid_mask_prev[valid_mask] = False
                        #     inv_propagated_depth[valid_mask_prev] = inv_propagated_depth_prev[valid_mask_prev]
                        
                        self.mv_cache[image_name] = inv_propagated_depth

                        if 'DJI_202312191150' in image_name and need_propagate:
                        # if '00000' in image_name or '00002' in image_name:
                            import os
                            import cv2
                            import numpy as np
                            import torchvision
                            torch.cuda.synchronize()
                            os.makedirs('./tmp_split', exist_ok=True)
                            vis_inv_propagated_depth = inv_propagated_depth.clone()
                            vis_inv_propagated_depth[vis_inv_propagated_depth < 0] = 0.0
                            depth = vis_inv_propagated_depth.detach().cpu().numpy()
                            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                            cv2.imwrite(os.path.join("./tmp_split/",  f"{image_name}_propagate.png"), depth_color)

                            # torchvision.utils.save_image(gt_image, os.path.join("./tmp_split/",  f"{image_name}_gt.png"))
                            torchvision.utils.save_image(image, os.path.join("./tmp_split/",  f"{image_name}_ref.png"))
                            depth = 1. / output["surf_depth"].detach().cpu().numpy().squeeze(0)
                            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                            cv2.imwrite(os.path.join("./tmp_split/",  f"{image_name}_ori.png"), depth_color)

                            normal = output["rend_normal"].permute(1,2,0)
                            normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
                            normal = normal.detach().cpu().numpy()
                            normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
                            cv2.imwrite(os.path.join("./tmp_split/",  f"{image_name}_normal.png"), normal)

                mv_loss = 0.0
                if image_name in self.mv_cache:
                    mv_depth = self.mv_cache[image_name]
                    valid_mask = mv_depth > 0.0

                    if valid_mask.sum() > 0:
                        predicted_inverse_depth = 1. / (output["surf_depth"].clamp_min(0.).squeeze() + 1e-8)
                        mv_loss = 2.5 * torch.abs(predicted_inverse_depth - mv_depth)[valid_mask].mean()

                metrics_i["mv_loss"] = mv_loss
                metrics_i["loss"] = metrics_i["loss"] + mv_loss
            else:

                if 'DJI_202312191150' in image_name:
                # if '00000' in image_name or '00002' in image_name:
                    import os
                    import cv2
                    import numpy as np
                    import torchvision
                    torch.cuda.synchronize()
                    os.makedirs('./tmp_split', exist_ok=True)
                    # torchvision.utils.save_image(gt_image, os.path.join("./tmp_split/",  f"{image_name}_gt.png"))
                    torchvision.utils.save_image(image, os.path.join("./tmp_split/",  f"{image_name}_ref.png"))
                    depth = 1. / output["surf_depth"].detach().cpu().numpy().squeeze(0)
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join("./tmp_split/",  f"{image_name}_ori.png"), depth_color)

                    normal = output["rend_normal"].permute(1,2,0)
                    normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
                    normal = normal.detach().cpu().numpy()
                    normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
                    cv2.imwrite(os.path.join("./tmp_split/",  f"{image_name}_normal.png"), normal)

            metrics_list.append(metrics_i)

        # metrics["gs"] = gaussian_model.get_xyz.shape[0]
        gathered_size = torch.empty(torch.distributed.get_world_size(), dtype=torch.int, device=gaussian_model.get_xyz.device)
        local_size = torch.tensor(gaussian_model.get_xyz.shape[0], dtype=torch.int, device=gaussian_model.get_xyz.device)
        torch.distributed.all_gather_into_tensor(gathered_size, local_size)
        
        metrics = {'gs': sum(gathered_size), 'loss': 0.0}
        if len(metrics_list) > 0:
            metrics.update({key: sum(data.get(key, 0) for data in metrics_list) for key in metrics_list[0]})
        
        pbar = {
            "gs": True,
            "loss": True,
            "rgb_diff": False,
            "ssim": True,
            "normal_loss": False,
            "dist_loss": False,
            "d_reg": True,
            "n_reg": True,
            "d_w": False,
            "mv_loss": True,
        }

        if step < pl_module.hparams["density"].densify_until_iter:
            pbar["extra_loss"] = False

        return metrics, pbar
    
    def get_data_weights(self, pl_module, batch, outputs, data_weights):
        image_names = pl_module.trainer.datamodule.dataparser_outputs.train_set.image_names

        with torch.no_grad():
            for item, output in zip(batch, outputs):
                image = output["render"]
                
                if image is None:
                    continue

                camera, image_info, extra_data = item
                image_name, gt_image, masked_pixels = image_info

                index = image_names.index(image_name)
                weight = torch.exp(25.0 * (1.0 - self.ssim(image, gt_image)))

                data_weights[index] = weight

            data_weights = data_weights.cuda()

            torch.distributed.all_reduce(data_weights, op=torch.distributed.ReduceOp.MAX)

        return data_weights.cpu()

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        if not isinstance(batch, list):
            return super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)
        
        outputs_list = outputs

        bsz = len(batch)
        metrics_list = []
        for i, (item, outputs) in enumerate(zip(batch, outputs_list)):
            if outputs["render"] is None:
                continue

            camera, image_info, _ = item
            image_name, gt_image, masked_pixels = image_info
            image = outputs["render"]

            # calculate loss
            if masked_pixels is not None:
                masked_pixels = masked_pixels.to(torch.uint8)  # False represents masked pixels
                # TODO: avoid repeatedly masking G.T. image
                gt_image = gt_image * masked_pixels
                image = image * masked_pixels
            rgb_diff_loss = self.rgb_diff_loss_fn(image, gt_image)
            ssim_metric = self.ssim(image, gt_image)
            loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1. - ssim_metric)

            metrics_i = {
                "loss": loss,
                "rgb_diff": rgb_diff_loss,
                "ssim": ssim_metric,
            }

            metrics_i["psnr"] = self.psnr(outputs["render"], gt_image)
            metrics_i["lpips"] = self.no_state_dict_models["lpips"](outputs["render"].clamp(0., 1.).unsqueeze(0), gt_image.unsqueeze(0))

            # regularization
            step=1 << 30
            lambda_normal = self.config.lambda_normal if step > 7000 else 0.0
            lambda_dist = self.config.lambda_dist if step > 3000 else 0.0

            rend_dist = outputs["rend_dist"]
            rend_normal = outputs['rend_normal']
            surf_normal = outputs['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            # update metrics
            metrics_i["loss"] = metrics_i["loss"] + dist_loss + normal_loss
            metrics_i["normal_loss"] = normal_loss
            metrics_i["dist_loss"] = dist_loss

            d_reg = self.get_inverse_depth_metric(item, outputs)

            metrics_i["loss"] = metrics_i["loss"] + d_reg
            metrics_i["d_reg"] = d_reg
    
            metrics_list.append(metrics_i)

        metrics = {key: sum(data.get(key, 0) for data in metrics_list) / len(metrics_list) for key in metrics_list[0]}
        pbar = {
            "loss": True,
            "rgb_diff": True,
            "ssim": True,
            "psnr": True,
            "lpips": True,
            "ssim": True,
            "normal_loss": False,
            "dist_loss": False,
            "d_reg": True,
        }   

        return metrics, pbar

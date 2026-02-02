from typing import Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .citygsv2_metrics import CityGSV2Metrics, CityGSV2MetricsModule
import torch.nn.functional as F
import numpy as np
from internal.utils.psnr import color_correct
from internal.utils.general_utils import build_rotation, build_scaling_rotation
from internal.utils.propagate_utils import depth_propagation, check_geometric_consistency
from internal.utils.rank import get_random_depth_ranking_loss, get_random_depth_boundary_loss, get_depth_completion
import torch.distributed as dist

def my_color_correct(img, ref, num_iters=5, eps=0.5 / 255):
    device = img.device
    img = img.squeeze().permute(1, 2, 0).to(device)
    ref = ref.squeeze().permute(1, 2, 0).to(device)
    corrected = color_correct(img, ref, num_iters, eps)
    corrected = corrected.permute(2, 0, 1).contiguous().to(device)
    return corrected

@dataclass
class WeightScheduler:
    init: float = 1.0

    final_factor: float = 0.01

    max_steps: int = 30_000

@dataclass
class DistributedMetrics(CityGSV2Metrics):
    lambda_normal: float = 0.05
    lambda_dist: float = 0.0

    lambda_scale: float = 0.1
    scale_thresh: float = 0.0
    scale_reg_from: int = 0

    normal_factor: float = 0.0

    lambda_mapping: float = 0.1

    lambda_multi_view: float = 2.5
    pixel_noise_th: float = 2.0
    multi_view_from: int = 15000
    mv_epoch_interval: int = 5
    radius_increment: int = 2
    patch_size: int = 11
    max_scale: int = 1
    max_src_num: int = 1
    mv_shuffle: bool = True

    use_correct: bool = False

    single_view_from: int = 0

    normal_regularization_from_iter: int = 7000

    depth_loss_type: Literal["l1", "l1+ssim", "l2", "kl"] = "l1"

    depth_loss_ssim_weight: float = 0.2

    depth_loss_weight: WeightScheduler = field(default_factory=lambda: WeightScheduler())

    depth_normalized: bool = False

    depth_output_key: str = "inverse_depth"

    mono_thresh: float = 0.0025

    def instantiate(self, *args, **kwargs) -> "DistributedMetricsModule":
        return DistributedMetricsModule(self)


class DistributedMetricsModule(CityGSV2MetricsModule):
    config: DistributedMetrics

    def get_weight(self, step: int):
        if self.config.single_view_from > self.config.depth_loss_weight.max_steps:
            return 0.0
        return self.config.depth_loss_weight.init * (self.config.depth_loss_weight.final_factor ** min((step - self.config.single_view_from) / self.config.depth_loss_weight.max_steps, 1))

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        assert isinstance(batch, list)
        image_names = [item[1][1] for item in batch]
        assert len(image_names) == len(set(image_names))
        # assert each image only one
    
        bsz = len(batch)

        use_multi_view = pl_module.trainer.datamodule.use_multi_view
        current_epoch = pl_module.current_epoch
        need_propagate = False

        import time
        if use_multi_view and step >= self.config.multi_view_from:
            with torch.no_grad():
                src_outputs = {}
                H, W = batch[0][0].height.item(), batch[0][0].width.item()
                mv_tensor = torch.zeros((bsz, H, W), device=pl_module.device)

                if pl_module.trainer.global_rank == 0:
                    if not hasattr(self, 'mv_cache'):
                        self.mv_cache = dict()
                
                    for i, item in enumerate(batch):
                        camera, image_info, extra_data = item
                        image_name, _, _ = image_info
                        if image_name not in self.mv_cache:
                            need_propagate = True
                        else:
                            mv_tensor[i] = self.mv_cache[image_name].detach().cuda()

                    if not need_propagate:
                        if current_epoch % self.config.mv_epoch_interval == 0:
                            need_propagate = True
                dist.broadcast(mv_tensor, src=0)
                for i, output in enumerate(outputs):
                    if output['render'] is None:
                        mv_tensor[i] = torch.zeros((H, W), device=pl_module.device)
                
                need_propagate_tensor = torch.tensor([need_propagate], dtype=torch.int, device=pl_module.device)
                dist.broadcast(need_propagate_tensor, src=0)
                need_propagate = bool(need_propagate_tensor.item())

                if need_propagate:
                    # st = time.time()    
                    max_src_num = self.config.max_src_num
                    train_cached = pl_module.trainer.train_dataloader.cached
                    train_dataset = pl_module.trainer.train_dataloader.dataset
                    name2index = train_dataset.name2index
                    gray_weights = torch.tensor([0.299, 0.587, 0.114], device=pl_module.device).view(1, 3, 1, 1)

                    if pl_module.trainer.global_rank == 0:
                        src_index_list = []
                        for _, image_info, _ in batch:
                            ref_index = name2index[image_info[0]]
                            src_index_list.extend(train_dataset.get_multi_view_data_index(ref_index, N=max_src_num, mv_shuffle=self.config.mv_shuffle))
                        src_index_tensor = torch.tensor(src_index_list, dtype=torch.long, device=pl_module.device)
                    else:
                        src_index_tensor = torch.empty(bsz * max_src_num, dtype=torch.long, device=pl_module.device)
                    dist.broadcast(src_index_tensor, src=0)
                    src_index_list_batch_tensors = torch.split(src_index_tensor, max_src_num)

                    if pl_module.trainer.global_rank == 0:
                        images_to_process = torch.stack([
                            train_cached[idx.item()][1][1] for idx in src_index_tensor
                        ]).to(pl_module.device)
                        src_gt_image_tensor = (images_to_process * gray_weights).sum(dim=1, keepdim=True)
                        del images_to_process
                    else:
                        src_gt_image_tensor = torch.empty((bsz * max_src_num, 1, H, W), device=pl_module.device)
                    dist.broadcast(src_gt_image_tensor, src=0)
                    src_gt_image_list_batch_tensors = torch.split(src_gt_image_tensor, max_src_num)

                    mutli_view_data_list = []
                    src_camera_list2 = [[] for _ in range(max_src_num)]
                    src_name_list2 = [[] for _ in range(max_src_num)]

                    for i, item in enumerate(batch):
                        src_index_tensor_item = src_index_list_batch_tensors[i]
                        
                        src_camera_list = [train_dataset.image_cameras[index.item()] for index in src_index_tensor_item]
                        src_name_list = [train_dataset.image_set.image_names[index.item()] for index in src_index_tensor_item]
                            
                        if outputs[i]["render"] is None:
                            mutli_view_data = (None, None, None)
                        else:
                            src_gt_image_list = src_gt_image_list_batch_tensors[i]
                            mutli_view_data = (src_camera_list, src_name_list, src_gt_image_list)
                        mutli_view_data_list.append(mutli_view_data)
                        
                        for j, (src_camera, src_name) in enumerate(zip(src_camera_list, src_name_list)):
                            src_camera_list2[j].append(src_camera)
                            src_name_list2[j].append(src_name)
                    
                        for i, (src_camera_list, src_name_list) in enumerate(zip(src_camera_list2, src_name_list2)):
                            src_output_list = pl_module(src_camera_list, render_src=True)
                            for j, (src_output, src_name) in enumerate(zip(src_output_list, src_name_list)):
                                if outputs[j]["render"] is not None:
                                    src_outputs[src_name] = src_output

        metrics_list = []
        for i, (item, output) in enumerate(zip(batch, outputs)):
            if output["render"] is None:
                continue

            coverage_y = output["coverage_y"]

            camera, image_info, extra_data = item
            gt_inverse_depth = extra_data

            image_name, gt_image, masked_pixels = image_info
            image = output["render"]
            appearance = output["appearance"]
            mapping = output["mapping"]

            # calculate loss
            if masked_pixels is not None:
                masked_pixels = masked_pixels.to(torch.uint8)  # False represents masked pixels
                # TODO: avoid repeatedly masking G.T. image
                gt_image = gt_image * masked_pixels
                image = image * masked_pixels

            image, gt_image = image[:, coverage_y[0]: coverage_y[1], :], gt_image[:, coverage_y[0]: coverage_y[1], :]
            appearance = appearance[:, coverage_y[0]: coverage_y[1], :]
            if mapping is not None:
                mapping = mapping[:, coverage_y[0]: coverage_y[1], :]

            rgb_diff_loss = self.rgb_diff_loss_fn(appearance, gt_image)
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
            if step < self.config.single_view_from or gt_inverse_depth is None or (use_multi_view and step >= self.config.multi_view_from):
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

            # d_reg += n_reg

            if step < pl_module.hparams["density"].densify_until_iter:
                metrics_i["loss"] = pl_module.hparams["metric"].lambda_dssim * (1. - metrics_i["ssim"]) + metrics_i["dist_loss"] + metrics_i["normal_loss"] + d_reg
                metrics_i["extra_loss"] = (1.0 - pl_module.hparams["metric"].lambda_dssim) * metrics_i["rgb_diff"]
            else:
                metrics_i["loss"] = metrics_i["loss"] + d_reg

            # scale regularization
            if step >= self.config.scale_reg_from and step < self.config.multi_view_from:
                visible_mask = output["visibility_filter"]
                scales = output["scales"][visible_mask]
                sorted_scale, _ = torch.sort(scales, dim=-1)
                s_over = torch.clamp(sorted_scale[:, -1] - self.config.scale_thresh, min=0.0)
                s_reg = self.config.lambda_scale * s_over.mean()
                metrics_i["loss"] = metrics_i["loss"] + s_reg

            # multi-view loss
            if use_multi_view and step >= self.config.multi_view_from:
                if step >= self.config.multi_view_from + 10000:
                    self.config.pixel_noise_th = 1.0
                elif step >= self.config.multi_view_from + 5000:
                    self.config.pixel_noise_th = 1.5
                else:
                    self.config.pixel_noise_th = 2.0

                ref_depth = output['surf_depth']
                gt_image_gray = (0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]).unsqueeze(0)
                if need_propagate:
                    with torch.no_grad():
                        mutli_view_data = mutli_view_data_list[i]
                        src_camera_list, src_name_list, src_gt_image_list = mutli_view_data
                        if src_gt_image_list is not None:
                            try:
                                propagated_depth, propagated_normal = \
                                    depth_propagation(camera, gt_image_gray, ref_depth, src_camera_list, src_gt_image_list, rend_normal, 
                                                    patch_size=self.config.patch_size, max_scale=self.config.max_scale)
                            except:
                                print('mv error:', propagated_depth.min(), propagated_depth.max())
                            
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
                            gt_inverse_depth = extra_data
                            if gt_inverse_depth is not None:
                                valid_mask = valid_mask.unsqueeze(0)
                                input_depth = 1. / propagated_depth.unsqueeze(0)
                                
                                if isinstance(gt_inverse_depth, tuple):
                                    gt_inverse_depth, gt_inverse_depth_mask = gt_inverse_depth
                                for psz in [16, 32, 64, 128, 256, 512]:
                                    transformed_depth = get_depth_completion(input_depth, gt_inverse_depth.unsqueeze(0), valid_mask, patch_size=psz)
                                    if transformed_depth is not None:
                                        new_valid_mask = torch.abs(transformed_depth - input_depth) < self.config.mono_thresh
                                        valid_mask = torch.logical_or(valid_mask, new_valid_mask)
                                valid_mask = valid_mask.squeeze(0)
                            # else:
                            #     print(f"[WARNING] {image_name} have no gt_inverse_depth!")

                            inv_propagated_depth = 1. / propagated_depth
                            inv_propagated_depth[~valid_mask] = 0.0

                            inv_propagated_depth_prev = mv_tensor[i]
                            valid_mask_prev = inv_propagated_depth_prev > 0.0
                            valid_mask_prev[valid_mask] = False
                            inv_propagated_depth[valid_mask_prev] = inv_propagated_depth_prev[valid_mask_prev]

                            mv_tensor[i] = inv_propagated_depth

                mv_loss = 0.0
                mv_depth = mv_tensor[i]
                valid_mask = mv_depth > 0.0

                if valid_mask.sum() > 0:
                    predicted_inverse_depth = 1. / (output["surf_depth"].clamp_min(0.).squeeze() + 1e-8)
                    mv_loss = self.config.lambda_multi_view * torch.abs(predicted_inverse_depth - mv_depth)[valid_mask].mean()

                metrics_i["mv_loss"] = mv_loss
                metrics_i["loss"] = metrics_i["loss"] + mv_loss

            metrics_list.append(metrics_i)
        
        if use_multi_view and step >= self.config.multi_view_from:
            with torch.no_grad():
                dist.all_reduce(mv_tensor, op=dist.ReduceOp.SUM)
                mv_tensor = mv_tensor.cpu()
                if pl_module.trainer.global_rank == 0:
                    for i, item in enumerate(batch):
                        _, image_info, _ = item
                        image_name = image_info[0]
                        self.mv_cache[image_name] = mv_tensor[i].detach().cpu()
                del mv_tensor
        
        torch.cuda.empty_cache()

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
            "rgb_diff": True,
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

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        if not isinstance(batch, list):
            if self.config.use_correct:
                image = outputs["render"]
                camera, image_info, _ = batch
                image_name, gt_image, masked_pixels = image_info
                image = my_color_correct(image, gt_image)
                outputs["render"] = image
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

            if self.config.use_correct:
                image = my_color_correct(image, gt_image)
                outputs["render"] = image

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

            metrics_list.append(metrics_i)

        metrics = {key: sum(data.get(key, 0) for data in metrics_list) / len(metrics_list) for key in metrics_list[0]}
        pbar = {
            "loss": True,
            "rgb_diff": True,
            "ssim": True,
            "psnr": True,
            "lpips": True,
            "ssim": True,
            "normal_loss": True,
            "dist_loss": False,
        }   

        return metrics, pbar

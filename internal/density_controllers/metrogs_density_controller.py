from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl, build_rotation, VanillaGaussianModel, List, LightningModule
from dataclasses import dataclass
from internal.utils.general_utils import check_update_at_this_iter
from .density_controller import DensityController, DensityControllerImpl, Utils
from typing import Union
import torch

@dataclass    
class DistributedController(VanillaDensityController):
    densify_grad_scaler: float = 0.
    axis_ratio_threshold: float = 0.01

    densify_from_iter: int = 0

    voxel_size: float = 0.1
    voxel_thresh: int = 5
    pixel_thresh: int = 20

    opacity_reset_until: int = 99999999
    
    def instantiate(self, *args, **kwargs) -> "DistributedControllerModule":
        return DistributedControllerModule(self)

class DistributedControllerModule(VanillaDensityControllerImpl):
    def _init_state(self, n_gaussians: int, device):
        max_radii2D = torch.zeros((n_gaussians), device=device)
        xyz_gradient_accum = torch.zeros((n_gaussians, 1), device=device)
        denom = torch.zeros((n_gaussians, 1), device=device)
        acc_pix = torch.zeros((n_gaussians), device=device)

        self.register_buffer("max_radii2D", max_radii2D, persistent=True)
        self.register_buffer("xyz_gradient_accum", xyz_gradient_accum, persistent=True)
        self.register_buffer("denom", denom, persistent=True)
        self.register_buffer("acc_pix", acc_pix, persistent=True)
        
    def before_backward(self, outputs: dict, batch, gaussian_model, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        self.world_size = pl_module.trainer.world_size
        # Here, global_step is iteration
        if not isinstance(outputs, list):
            super().before_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)
            return
        
        outputs_list = outputs

        if global_step >= self.config.densify_until_iter or global_step < self.config.densify_from_iter:
            return
        
        for item, outputs in zip(batch, outputs_list):
            outputs["viewspace_points"].retain_grad()
    
    def after_backward(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        # Here, global_step is iteration
        if global_step >= self.config.densify_until_iter or global_step < self.config.densify_from_iter:
            return
        bsz = len(batch)

        with torch.no_grad():
            self.update_states(outputs)

            # densify and pruning
            if global_step > self.config.densify_from_iter \
                and check_update_at_this_iter(global_step, bsz, self.config.densification_interval, 0):
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                    # use_prune=global_step > self.config.densify_from_iter // 2
                )

            if  check_update_at_this_iter(global_step, bsz, self.config.opacity_reset_interval, 0) or \
                    (torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter):
                if global_step <= self.config.opacity_reset_until:
                    self._reset_opacities(gaussian_model, optimizers)
    
    def _prune_points(self, mask, gaussian_model: VanillaGaussianModel, optimizers: List):
        """
        Args:
            mask: `True` indicating the Gaussians to be pruned
            gaussian_model
            optimizers
        """
        valid_points_mask = ~mask  # `True` to keep
        new_parameters = Utils.prune_properties(valid_points_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

        # prune states
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.acc_pix = self.acc_pix[valid_points_mask]
    
    def _split_means_and_scales(self, gaussian_model, selected_pts_mask, N):
        scales = gaussian_model.get_scales()
        device = scales.device

        stds = scales[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(gaussian_model.get_property("rotations")[selected_pts_mask]).repeat(N, 1, 1)
        # Split means and scales, they are a little bit different
        new_means = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussian_model.get_means()[selected_pts_mask].repeat(N, 1)
        new_scales = gaussian_model.scale_inverse_activation(scales[selected_pts_mask].repeat(N, 1) / (0.8 * N))

        new_properties = {
            "means": new_means,
            "scales": new_scales,
        }

        return new_properties

    def update_states(self, outputs):
        if not isinstance(outputs, list):
            super().update_states(outputs)
            return
        else:
            outputs_list = outputs
        
        for outputs in outputs_list:
            if outputs['render'] is None:
                continue
            viewspace_point_tensor, visibility_filter, radii = outputs["viewspace_points"], outputs["visibility_filter"], outputs["radii"]
            # retrieve viewspace_points_grad_scale if provided
            viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)
            acc_pix = outputs["acc_pix"]

            self.acc_pix[visibility_filter] =  self.acc_pix[visibility_filter] + acc_pix[visibility_filter]

            # update states
            self.max_radii2D[visibility_filter] = torch.max(
                self.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )
            xys_grad = viewspace_point_tensor.grad
            if self.config.absgrad is True:
                xys_grad = viewspace_point_tensor.absgrad
            self._add_densification_stats(xys_grad, visibility_filter, scale=viewspace_points_grad_scale)
    
    def _densify_and_prune(self, max_screen_size, gaussian_model: VanillaGaussianModel, optimizers: List, use_prune: bool = True):
        min_opacity = self.config.cull_opacity_threshold
        prune_extent = self.prune_extent

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.

        if self.config.voxel_size > 0:
            pixels = self.acc_pix / self.denom.squeeze(-1)
        else:
            pixels = None

        # densify
        self._densify_and_clone(grads, gaussian_model, optimizers)
        self._densify_and_split(grads, gaussian_model, optimizers, pixels)

        # prune
        if use_prune:
            prune_mask = (gaussian_model.get_opacities() < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()
    
    def get_sparse_flag(self, points, voxel_size, threshold, device):
        N = points.shape[0]
        
        voxel_indices = torch.floor(points / voxel_size).long()
        max_indices = voxel_indices.max(dim=0).values
        
        multiplier_x = 1
        multiplier_y = (max_indices[0].item() + 1)
        multiplier_z = (max_indices[1].item() + 1) * multiplier_y
        
        voxel_keys = (voxel_indices[:, 0].long() * multiplier_y + 
                    voxel_indices[:, 1].long()) * multiplier_z + voxel_indices[:, 2].long()

        unique_keys, inverse_indices, counts = torch.unique(
            voxel_keys, 
            return_inverse=True, 
            return_counts=True
        )
        
        point_densities = counts[inverse_indices]
        
        labels = torch.where(point_densities >= threshold, 
                            torch.tensor(0, dtype=torch.int32, device=device), 
                            torch.tensor(1, dtype=torch.int32, device=device))
        
        # print(f"实际占用的体素数量 (K): {unique_keys.shape[0]}")
        # print(f"稀疏点 (1) 数量: {(labels == 1).sum().item()}")

        return labels
    
    def _densify_and_split(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List, pixels=None, N: int = 2):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        device = gaussian_model.get_property("means").device
        n_init_points = gaussian_model.n_gaussians
        scales = gaussian_model.get_scales()

        # The number of Gaussians and `grads` is different after cloning, so padding is required
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # Exclude small Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(
                scales,
                dim=1,
            ).values > percent_dense * scene_extent,
        )

        if pixels is not None:
            sparse_flag = self.get_sparse_flag(gaussian_model.get_xyz, voxel_size=self.config.voxel_size, threshold=self.config.voxel_thresh, device=device)
            need_split = torch.zeros((n_init_points,), device=device)
            need_split[:grads.shape[0]] = pixels >= self.config.pixel_thresh
            need_split = torch.logical_and(need_split, sparse_flag)
            selected_pts_mask = torch.logical_or(selected_pts_mask, need_split)

        axis_ratio = scales.min(dim=1).values / scales.max(dim=1).values
        selected_pts_mask = torch.logical_and(selected_pts_mask, axis_ratio > self.config.axis_ratio_threshold)

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

        # Prune selected Gaussians, since they are already split
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(
                N * selected_pts_mask.sum(),
                device=device,
                dtype=torch.bool,
            ),
        ))
        self._prune_points(prune_filter, gaussian_model, optimizers)
    
from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl, build_rotation, VanillaGaussianModel, List, LightningModule
from dataclasses import dataclass
from internal.utils.general_utils import check_update_at_this_iter
import torch

@dataclass
class CityGSV2DensityController(VanillaDensityController):
    densify_grad_scaler: float = 0.
    axis_ratio_threshold: float = 0.01
    
    def instantiate(self, *args, **kwargs) -> "CityGSV2DensityControllerModule":
        return CityGSV2DensityControllerModule(self)


class CityGSV2DensityControllerModule(VanillaDensityControllerImpl):

    def _densify_and_split(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List, N: int = 2):
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

    def _densify_and_clone(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # Exclude big Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
        )

        axis_ratio = gaussian_model.get_scales().min(dim=1).values / gaussian_model.get_scales().max(dim=1).values
        selected_pts_mask = torch.logical_and(selected_pts_mask, axis_ratio > self.config.axis_ratio_threshold)  # hard coded threshold

        # Copy selected Gaussians
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            new_properties[key] = value[selected_pts_mask]

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

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
    
    def before_backward(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        return
        if global_step >= self.config.densify_until_iter:
            return

        outputs["viewspace_points"].retain_grad()

    def after_backward(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        return
        if global_step >= self.config.densify_until_iter:
            return

        with torch.no_grad():
            self.update_states(outputs)

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                )

            if global_step % self.config.opacity_reset_interval == 0 or \
                    (
                            torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter
                    ):
                self._reset_opacities(gaussian_model, optimizers)
    
class DistributedController(CityGSV2DensityController):
    densify_grad_scaler: float = 0.
    axis_ratio_threshold: float = 0.01
    
    def instantiate(self, *args, **kwargs) -> "DistributedControllerModule":
        return DistributedControllerModule(self)

class DistributedControllerModule(CityGSV2DensityControllerModule):

    def before_backward(self, outputs: dict, batch, gaussian_model, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        self.world_size = pl_module.trainer.world_size
        # Here, global_step is iteration
        if not isinstance(outputs, list):
            super().before_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)
            return
        
        outputs_list = outputs

        if global_step >= self.config.densify_until_iter:
            return
        
        for item, outputs in zip(batch, outputs_list):
            outputs["viewspace_points"].retain_grad()
    
    def after_backward(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        # Here, global_step is iteration
        if global_step >= self.config.densify_until_iter:
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
                )

            if check_update_at_this_iter(global_step, bsz, self.config.opacity_reset_interval, 0) or \
                    (torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter):
                self._reset_opacities(gaussian_model, optimizers)

    def update_states(self, outputs):
        if not isinstance(outputs, list):
            super().update_states(outputs)
            return
        else:
            outputs_list = outputs
        
        for outputs in outputs_list:
            viewspace_point_tensor, visibility_filter, radii = outputs["viewspace_points"], outputs["visibility_filter"], outputs["radii"]
            # retrieve viewspace_points_grad_scale if provided
            viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

            # update states
            self.max_radii2D[visibility_filter] = torch.max(
                self.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )
            xys_grad = viewspace_point_tensor.grad
            if self.config.absgrad is True:
                xys_grad = viewspace_point_tensor.absgrad
            self._add_densification_stats(xys_grad, visibility_filter, scale=viewspace_points_grad_scale)
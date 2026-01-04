from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl, List, LightningModule
from dataclasses import dataclass
import torch
from typing import Tuple, Optional, Union, List, Dict
from .density_controller import Utils
from internal.utils.general_utils import inverse_sigmoid
from functools import reduce
from torch_scatter import scatter_max
from internal.models.anchor_gaussian import AnchorGaussianModel
from torch import nn


@dataclass
class AnchorDensityController(VanillaDensityController):
    densify_grad_scaler: float = 0.

    axis_ratio_threshold: float = 0.01

    densification_interval: int = 100

    densify_from_iter: int = 500

    densify_until_iter: int = 15_000

    update_init_factor: int = 16

    update_hierachy_factor: int = 4

    num_overlap: int = 5
    
    def instantiate(self, *args, **kwargs) -> "AnchorDensityControllerModule":
        return AnchorDensityControllerModule(self)


class AnchorDensityControllerModule(VanillaDensityControllerImpl):
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        if stage == "fit":
            self.cameras_extent = pl_module.trainer.datamodule.dataparser_outputs.camera_extent * self.config.camera_extent_factor
            self.prune_extent = pl_module.trainer.datamodule.prune_extent * self.config.camera_extent_factor

            if self.config.scene_extent_override > 0:
                self.cameras_extent = self.config.scene_extent_override
                self.prune_extent = self.config.scene_extent_override
                print(f"Override scene extent with {self.config.scene_extent_override}")

            self._init_state(pl_module.gaussian_model.n_gaussians, pl_module.gaussian_model.config.n_offsets, pl_module.device)
    
    def _init_state(self, n_gaussians: int, n_offsets, device):
        opacity_accum = torch.zeros((n_gaussians, 1), device=device)
        
        offset_gradient_accum = torch.zeros((n_gaussians*n_offsets, 1), device=device)
        offset_denom = torch.zeros((n_gaussians*n_offsets, 1), device=device)
        anchor_denom = torch.zeros((n_gaussians, 1), device=device)

        self.register_buffer("opacity_accum", opacity_accum, persistent=True)
        self.register_buffer("offset_gradient_accum", offset_gradient_accum, persistent=True)
        self.register_buffer("offset_denom", offset_denom, persistent=True)
        self.register_buffer("anchor_denom", anchor_denom, persistent=True)
    
    def cat_tensors_to_optimizers_(self, new_properties: Dict[str, torch.Tensor], optimizers: List[torch.optim.Optimizer]) -> Dict[str, torch.Tensor]:
        new_parameters = {}
        for opt in optimizers:
            for group in opt.param_groups:
                if  'mlp' in group['name'] or \
                    'conv' in group['name'] or \
                    'feat_base' in group['name'] or \
                    'embedding' in group['name']:
                    continue
                assert len(group["params"]) == 1
                assert group["name"] not in new_parameters, "parameter `{}` appears in multiple optimizers".format(group["name"])

                extension_tensor = new_properties[group["name"]]

                # get current sates
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    # append states for new properties
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    # delete old state key by old params from optimizer
                    del opt.state[group['params'][0]]
                    # append new parameters to optimizer
                    group["params"][0] = nn.Parameter(torch.cat(
                        (group["params"][0], extension_tensor),
                        dim=0,
                    ).requires_grad_(True))
                    # update optimizer states
                    opt.state[group['params'][0]] = stored_state
                else:
                    # append new parameters to optimizer
                    group["params"][0] = nn.Parameter(torch.cat(
                        (group["params"][0], extension_tensor),
                        dim=0,
                    ).requires_grad_(True))

                # add new `nn.Parameter` from optimizers to the dict returned later
                new_parameters[group["name"]] = group["params"][0]

        return new_parameters

    def cat_tensors_to_properties(self, new_properties: Dict[str, torch.Tensor], model: "internal.models.gaussian.GaussianModel", optimizers: List[torch.optim.Optimizer]):
        new_parameters = self.cat_tensors_to_optimizers_(
            new_properties=new_properties,
            optimizers=optimizers,
        )

        if len(new_properties) != len(new_parameters):
            # has non-optimizable parameters
            for k, v in new_properties.items():
                if k in new_parameters:
                    continue
                new_parameters[k] = torch.nn.Parameter(torch.concat([model.get_property(k), v], dim=0), requires_grad=False)

        return new_parameters

    def _densification_postfix(self, new_properties: Dict, gaussian_model: AnchorGaussianModel, optimizers):
        new_parameters = self.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

    @staticmethod
    def prune_optimizers_(mask, optimizers):
        """

        :param mask: The `False` indicating the ones to be pruned
        :param optimizers:
        :return: a new dict
        """

        new_parameters = {}
        for opt in optimizers:
            for group in opt.param_groups:
                if  'mlp' in group['name'] or \
                    'conv' in group['name'] or \
                    'feat_base' in group['name'] or \
                    'embedding' in group['name']:
                    continue
                assert len(group["params"]) == 1
                assert group["name"] not in new_parameters, "parameter `{}` appears in multiple optimizers".format(group["name"])

                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scales":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                new_parameters[group["name"]] = group["params"][0]

        return new_parameters

    def prune_properties(self, mask: torch.Tensor, model: "internal.models.gaussian.GaussianModel", optimizers: List[torch.optim.Optimizer]):
        new_parameters = self.prune_optimizers_(mask=mask, optimizers=optimizers)

        if len(model.property_names) != len(new_parameters):
            for k in model.property_names:
                if k in new_parameters:
                    continue

                new_parameters[k] = torch.nn.Parameter(model.get_property(k)[mask], requires_grad=False)

        return new_parameters

    def prune_anchor(self, mask, gaussian_model: AnchorGaussianModel, optimizers: List):
        """
        Args:
            mask: `True` indicating the Gaussians to be pruned
            gaussian_model
            optimizers
        """
        valid_points_mask = ~mask  # `True` to keep
        new_parameters = self.prune_properties(valid_points_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, num_overlap=1, use_chunk=True):
        counts = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.int, device=selected_grid_coords_unique.device)

        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            for i in range(max_iters):
                chunk = grid_coords[i * chunk_size:(i + 1) * chunk_size]
                matches = (selected_grid_coords_unique.unsqueeze(1) == chunk.unsqueeze(0)).all(-1)
                counts += matches.sum(dim=1)
        else:
            matches = (selected_grid_coords_unique.unsqueeze(1) == grid_coords.unsqueeze(0)).all(-1)
            counts = matches.sum(dim=1)

        remove_duplicates = counts >= num_overlap

        return remove_duplicates
    
    def anchor_growing(self, pc: AnchorGaussianModel, optimizers: List, grads, threshold, offset_mask, overlap):
        init_length = pc.get_anchor.shape[0]*pc.config.n_offsets
        for i in range(pc.config.update_depth):
            # update threshold
            cur_threshold = threshold*((pc.config.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = pc.get_anchor.shape[0]*pc.config.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = pc.get_anchor.unsqueeze(dim=1) + pc.gaussians["offset"] * pc.get_scaling[:,:3].unsqueeze(dim=1)
            
            size_factor = pc.config.update_init_factor // (pc.config.update_hierachy_factor**i)
            cur_size = pc.config.voxel_size*size_factor
            
            grid_coords = torch.round(pc.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            if overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + pc.config.padding * cur_size
            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + pc.config.padding * cur_size
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')
            
            if candidate_anchor.shape[0] > 0:
                if pc.get_scaling.shape[1] == 5:
                    new_scaling = torch.ones_like(candidate_anchor).repeat([1,2])[:, :5].float().cuda()*cur_size # *0.05 2DGS
                else:
                    new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05 3DGS
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0
                new_feat = pc.gaussians["anchor_feat"].unsqueeze(dim=1).repeat([1, pc.config.n_offsets, 1]).view([-1, pc.config.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,pc.config.n_offsets,1]).float().cuda()                

                temp_anchor_denom = torch.cat([self.anchor_denom, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_denom
                self.anchor_denom = temp_anchor_denom

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                new_properties = {
                    "anchor": candidate_anchor,
                    "scales": new_scaling,
                    "rotations": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                }

                self._densification_postfix(new_properties, pc, optimizers)            

    def adjust_anchor(self, pc: AnchorGaussianModel, optimizers: List, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(pc, optimizers, grads_norm, grad_threshold, offset_mask, self.config.num_overlap)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_denom = torch.zeros([pc.get_anchor.shape[0]*pc.config.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_denom], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([pc.get_anchor.shape[0]*pc.config.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_denom).squeeze(dim=1)
        anchors_mask = (self.anchor_denom > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, pc.config.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, pc.config.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_denom[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_denom = self.anchor_denom[~prune_mask]
        del self.anchor_denom
        self.anchor_denom = temp_anchor_denom

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask, pc, optimizers)

    def after_backward(self, outputs: dict, batch, gaussian_model: AnchorGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        with torch.no_grad():
            self.update_states(gaussian_model, outputs)

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                self.adjust_anchor(
                    pc=gaussian_model,
                    optimizers=optimizers,
                )

    def update_states(self, pc: AnchorGaussianModel, outputs):
        viewspace_point_tensor, visibility_filter = outputs["viewspace_points"], outputs["visibility_filter"]
        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        # update states
        xys_grad = viewspace_point_tensor.grad
        if self.config.absgrad is True:
            xys_grad = viewspace_point_tensor.absgrad

        offset_selection_mask, anchor_visible_mask, opacity = outputs["selection_mask"], outputs["visible_mask"], outputs["opacity"]
        self.training_statis(pc, xys_grad, opacity, visibility_filter, offset_selection_mask, anchor_visible_mask, scale=viewspace_points_grad_scale)

    # statis grad information to guide liftting. 
    def training_statis(self, pc: AnchorGaussianModel, grad, opacity, update_filter, offset_selection_mask, anchor_visible_mask, scale: Union[float, int, None]):
        # update opacity stats
        temp_opacity = torch.zeros(offset_selection_mask.shape[0], dtype=torch.float32, device="cuda")
        temp_opacity[offset_selection_mask] = opacity.clone().view(-1).detach()
        
        temp_opacity = temp_opacity.view([-1, pc.config.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_denom[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, pc.config.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        scaled_grad = grad[update_filter, :2]
        if scale is not None:
            scaled_grad = scaled_grad * scale
        
        grad_norm = torch.norm(scaled_grad, dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional

import torch
import numpy as np
from torch import nn


from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel
from internal.utils.general_utils import inverse_sigmoid
from internal.optimizers import OptimizerConfig, Adam, SelectiveAdam, SparseGaussianAdam
from internal.schedulers import Scheduler, ExponentialDecayScheduler


@dataclass
class OptimizationConfig:
    anchor_lr_init: float = 0.0

    # only in below format can work with jsonargparse
    anchor_lr_scheduler: Scheduler = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 0.0,
            "max_steps": 30_000,
        },
    })

    offset_lr_init: float = 0.01

    offset_lr_scheduler: Scheduler = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 0.0001,
            "max_steps": 30_000,
        },
    })

    mlp_opacity_lr_init: float = 0.002

    mlp_opacity_lr_scheduler: Scheduler = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 0.00002,
            "max_steps": 30_000,
        },
    })

    mlp_cov_lr_init: float = 0.004

    mlp_cov_lr_scheduler: Scheduler = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 0.004,
            "max_steps": 30_000,
        },
    })

    mlp_color_lr_init: float = 0.008

    mlp_color_lr_scheduler: Scheduler = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 0.00005,
            "max_steps": 30_000,
        },
    })

    spatial_lr_scale: float = -1  # auto calculate from camera poses if <= 0

    feature_lr: float = 0.0025

    opacities_lr: float = 0.02

    scales_lr: float = 0.007

    rotations_lr: float = 0.002

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})

@dataclass
class AnchorGaussian(VanillaGaussian):
    feat_dim: int = 32
    padding: float = 0.0
    n_offsets: int = 10
    voxel_size: float = 0.001
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4
    ratio : int = 1
    add_opacity_dist : bool = False
    add_cov_dist : bool = False
    add_color_dist : bool = False

    sh_degree: int = 3

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "AnchorGaussianModel":
        return AnchorGaussianModel(self)


class AnchorGaussianModel(VanillaGaussianModel):
    def __init__(self, config: AnchorGaussian) -> None:
        super().__init__(config)
        self.config = config

        names = [
                    "anchor",
                    "offset",
                    "anchor_feat",
                    "scales",
                    "rotations",
                ] + self.get_extra_property_names()
        self._names = tuple(names)

        self.is_pre_activated = False

        self.create_net()

    def create_net(self):
        self.config.opacity_dist_dim = 1 if self.config.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.config.feat_dim+3+self.config.opacity_dist_dim, self.config.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.config.feat_dim, self.config.n_offsets),
            nn.Tanh()
        )

        self.config.add_cov_dist = self.config.add_cov_dist
        self.config.cov_dist_dim = 1 if self.config.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.config.feat_dim+3+self.config.cov_dist_dim, self.config.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.config.feat_dim, 7*self.config.n_offsets),
        )

        self.color_dist_dim = 1 if self.config.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(self.config.feat_dim+3+self.color_dist_dim, self.config.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.config.feat_dim, 3*self.config.n_offsets),
            nn.Sigmoid()
        )
    
    def setup_from_number(self, n: int, *args, **kwargs):
        anchor = torch.zeros((n, 3))
        offset = torch.zeros((n, self.config.n_offsets, 3))
        anchor_feat = torch.zeros((n, self.config.feat_dim))
        # scales = torch.zeros((n, 5))
        scales = torch.zeros((n, 6))
        rotations = torch.zeros((n, 4))

        anchor = nn.Parameter(anchor.requires_grad_(True))
        offset = nn.Parameter(offset.requires_grad_(True))
        anchor_feat = nn.Parameter(anchor_feat.requires_grad_(True))
        scales = nn.Parameter(scales.requires_grad_(True))
        rotations = nn.Parameter(rotations.requires_grad_(False))

        property_dict = {
            "anchor": anchor,
            "offset": offset,
            "anchor_feat": anchor_feat,
            "scales": scales,
            "rotations": rotations,
        }
        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        self.set_properties(property_dict)

        self.active_sh_degree = 0

    def pre_activate_all_properties(self):
        self.is_pre_activated = True

        self.scales = self.get_scales()
        self.rotations = self.get_rotations()

        # concat `shs_dc` and `shs_rest` and store it to dict, then remove `shs_dc` and `shs_rest`
        names = list(self._names)
        ## replace `names`
        self._names = tuple(names)

        self.scale_activation = self._return_as_is
        self.scale_inverse_activation = self._return_as_is
        self.rotation_activation = self._return_as_is
        self.rotation_inverse_activation = self._return_as_is

    @property
    def get_anchor(self):
        return self.gaussians["anchor"]
    
    @property
    def get_xyz(self):
        return self.get_anchor
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    def voxelize_sample(self, data: torch.Tensor, voxel_size: float = 0.01) -> torch.Tensor:
        perm = torch.randperm(data.shape[0], device=data.device)
        data = data[perm]

        rounded_data = torch.round(data / voxel_size) * voxel_size
        data = torch.unique(rounded_data, dim=0)
        
        return data

    def setup_from_pcd(self, xyz: Union[torch.Tensor, np.ndarray], rgb: Union[torch.Tensor, np.ndarray], *args, **kwargs):
        if isinstance(xyz, np.ndarray):
            xyz = torch.tensor(xyz)

        if self.config.voxel_size <= 0:
            init_xyz = xyz.float()
            init_dist = distCUDA2(init_xyz).float()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.config.voxel_size = median_dist.item()
            del init_xyz
            del init_dist

        print(f'Initial voxel_size: {self.config.voxel_size}')

        xyz = self.voxelize_sample(xyz, voxel_size=self.config.voxel_size)
        fused_point_cloud = xyz.float()

        n_gaussians = fused_point_cloud.shape[0]
        offsets = torch.zeros((n_gaussians, self.config.n_offsets, 3)).float()
        anchors_feat = torch.zeros((n_gaussians, self.config.feat_dim)).float()

        # scales
        # TODO: replace `simple_knn`
        from simple_knn._C import distCUDA2
        # the parameter device may be "cpu", so tensor must move to cuda before calling distCUDA2()
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device)
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 5)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        # rotations
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        offset = nn.Parameter(offsets.requires_grad_(True))
        anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        scales = nn.Parameter(scales.requires_grad_(True))
        rotations = nn.Parameter(rots.requires_grad_(False))

        property_dict = {
            "anchor": anchor,
            "offset": offset,
            "anchor_feat": anchor_feat,
            "scales": scales,
            "rotations": rotations,
        }
        self.before_setup_set_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        self.set_properties(property_dict)

    def training_setup(self, module: "lightning.LightningModule") -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        spatial_lr_scale = self.config.optimization.spatial_lr_scale
        if spatial_lr_scale <= 0:
            spatial_lr_scale = module.trainer.datamodule.dataparser_outputs.camera_extent
        assert spatial_lr_scale > 0

        optimization_config = self.config.optimization

        optimizer_factory = self.config.optimization.optimizer

        # anchor
        anchor_lr_init = optimization_config.anchor_lr_init * spatial_lr_scale
        anchor_optimizer = optimizer_factory.instantiate(
            [{'params': [self.gaussians["anchor"]], "name": "anchor"}],
            lr=anchor_lr_init,
            eps=1e-15,
        )
        self._add_optimizer_after_backward_hook_if_available(anchor_optimizer, module)
        # TODO: other scheduler may not contain `lr_final`, but does not need to change scheduler currently
        optimization_config.anchor_lr_scheduler.lr_final *= spatial_lr_scale
        anchor_scheduler = optimization_config.anchor_lr_scheduler.instantiate().get_scheduler(
            anchor_optimizer,
            anchor_lr_init,
        )

        # offset
        offset_lr_init = optimization_config.offset_lr_init * spatial_lr_scale
        offset_optimizer = optimizer_factory.instantiate(
            [{'params': [self.gaussians["offset"]], "name": "offset"}],
            lr=offset_lr_init,
            eps=1e-15,
        )
        self._add_optimizer_after_backward_hook_if_available(offset_optimizer, module)
        # TODO: other scheduler may not contain `lr_final`, but does not need to change scheduler currently
        optimization_config.offset_lr_scheduler.lr_final *= spatial_lr_scale
        offset_scheduler = optimization_config.offset_lr_scheduler.instantiate().get_scheduler(
            offset_optimizer,
            offset_lr_init,
        )

        # opacity
        mlp_opacity_lr_init = optimization_config.mlp_opacity_lr_init
        mlp_opacity_optimizer = optimizer_factory.instantiate(
            [{'params': self.mlp_opacity.parameters(), "name": "mlp_opacity"}],
            lr=mlp_opacity_lr_init,
            eps=1e-15,
        )
        self._add_optimizer_after_backward_hook_if_available(mlp_opacity_optimizer, module)
        mlp_opacity_scheduler = optimization_config.mlp_opacity_lr_scheduler.instantiate().get_scheduler(
            mlp_opacity_optimizer,
            mlp_opacity_lr_init,
        )

        # cov
        mlp_cov_lr_init = optimization_config.mlp_cov_lr_init
        mlp_cov_optimizer = optimizer_factory.instantiate(
            [{'params': self.mlp_cov.parameters(), "name": "mlp_cov"}],
            lr=mlp_opacity_lr_init,
            eps=1e-15,
        )
        self._add_optimizer_after_backward_hook_if_available(mlp_cov_optimizer, module)
        mlp_cov_scheduler = optimization_config.mlp_cov_lr_scheduler.instantiate().get_scheduler(
            mlp_cov_optimizer,
            mlp_cov_lr_init,
        )

        # color
        mlp_color_lr_init = optimization_config.mlp_color_lr_init
        mlp_color_optimizer = optimizer_factory.instantiate(
            [{'params': self.mlp_color.parameters(), "name": "mlp_color"}],
            lr=mlp_opacity_lr_init,
            eps=1e-15,
        )
        self._add_optimizer_after_backward_hook_if_available(mlp_color_optimizer, module)
        mlp_color_scheduler = optimization_config.mlp_color_lr_scheduler.instantiate().get_scheduler(
            mlp_color_optimizer,
            mlp_color_lr_init,
        )

        # the params with constant LR
        l = [
            {'params': [self.gaussians["anchor_feat"]], 'lr': optimization_config.feature_lr, "name": "anchor_feat"},
            {'params': [self.gaussians["scales"]], 'lr': optimization_config.scales_lr, "name": "scales"},
            {'params': [self.gaussians["rotations"]], 'lr': optimization_config.rotations_lr, "name": "rotations"},
        ]
        constant_lr_optimizer = optimizer_factory.instantiate(l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(constant_lr_optimizer, module)

        print("spatial_lr_scale={}, learning_rates=".format(spatial_lr_scale))
        print("  anchor={}->{}".format(anchor_lr_init, optimization_config.anchor_lr_scheduler.lr_final))
        print("  offset={}->{}".format(offset_lr_init, optimization_config.offset_lr_scheduler.lr_final))
        print("  mlp_opa={}->{}".format(mlp_opacity_lr_init, optimization_config.mlp_opacity_lr_scheduler.lr_final))
        print("  mlp_cov={}->{}".format(mlp_cov_lr_init, optimization_config.mlp_cov_lr_scheduler.lr_final))
        print("  mlp_rgb={}->{}".format(mlp_color_lr_init, optimization_config.mlp_color_lr_scheduler.lr_final))
        for i in l:
            print("  {}={}".format(i["name"], i["lr"]))

        return [
            anchor_optimizer, 
            offset_optimizer, 
            mlp_opacity_optimizer,
            mlp_cov_optimizer,
            mlp_color_optimizer,
            constant_lr_optimizer,
        ], [
            anchor_scheduler,
            offset_scheduler,
            mlp_opacity_scheduler,
            mlp_cov_scheduler,
            mlp_color_scheduler,
        ]
    
    def on_train_batch_end(self, step: int, module: "lightning.LightningModule"):
        pass
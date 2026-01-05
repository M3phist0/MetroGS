from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional

import torch
import numpy as np
import math
from torch import nn
from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel, OptimizationConfig
from internal.optimizers import OptimizerConfig
from internal.schedulers import Scheduler
from internal.utils.general_utils import inverse_sigmoid

@dataclass
class Gaussian2D(VanillaGaussian):
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        means_lr_scheduler={
            "class_path": "ExponentialDecayScheduler",
            "init_args": {
                "lr_final": 0.0000016,
                "max_steps": 50_000,
            },
        }
    ))

    def instantiate(self, *args, **kwargs) -> "Gaussian2DModel":
        return Gaussian2DModel(self)


class Gaussian2DModelMixin:    
    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_pcd(
            xyz=xyz,
            rgb=rgb,
            property_dict=property_dict,
            *args,
            **kwargs,
        )
        with torch.no_grad():
            property_dict["scales"] = property_dict["scales"][..., :2]
            # key to a quality comparable to hbb1/2d-gaussian-splatting
            property_dict["rotations"].copy_(torch.rand_like(property_dict["rotations"]))

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_number(
            n=n,
            property_dict=property_dict,
            *args,
            **kwargs,
        )
        property_dict["scales"] = property_dict["scales"][..., :2]


class Gaussian2DModel(Gaussian2DModelMixin, VanillaGaussianModel):
    def __init__(self, config):
        super().__init__(config)

    def setup_from_number(self, n: int, *args, **kwargs):
        means = torch.zeros((n, 3))
        shs = torch.zeros((n, 3, (self.max_sh_degree + 1) ** 2))
        shs_dc = shs[:, :, 0:1].transpose(1, 2).contiguous()
        shs_rest = shs[:, :, 1:].transpose(1, 2).contiguous()
        scales = torch.zeros((n, 3))
        rotations = torch.zeros((n, 4))
        opacities = torch.zeros((n, 1))

        means = nn.Parameter(means.requires_grad_(True))
        shs_dc = nn.Parameter(shs_dc.requires_grad_(True))
        shs_rest = nn.Parameter(shs_rest.requires_grad_(True))
        scales = nn.Parameter(scales.requires_grad_(True))
        rotations = nn.Parameter(rotations.requires_grad_(True))
        opacities = nn.Parameter(opacities.requires_grad_(True))

        property_dict = {
            "means": means,
            "shs_dc": shs_dc,
            "shs_rest": shs_rest,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }

        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        self.set_properties(property_dict)

        self.active_sh_degree = 0

    def setup_from_pcd(self, xyz: Union[torch.Tensor, np.ndarray], rgb: Union[torch.Tensor, np.ndarray], *args, **kwargs):
        from internal.utils.sh_utils import RGB2SH

        if isinstance(xyz, np.ndarray):
            xyz = torch.tensor(xyz)
        if isinstance(rgb, np.ndarray):
            rgb = torch.tensor(rgb)

        fused_point_cloud = xyz.float()
        fused_color = RGB2SH(rgb.float())

        n_gaussians = fused_point_cloud.shape[0]

        # SHs
        shs = torch.zeros((n_gaussians, 3, (self.config.sh_degree + 1) ** 2)).float()
        shs[:, :3, 0] = fused_color
        shs[:, 3:, 1:] = 0.0

        # scales
        # TODO: replace `simple_knn`
        from simple_knn._C import distCUDA2
        # the parameter device may be "cpu", so tensor must move to cuda before calling distCUDA2()
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # rotations
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        # opacities
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))

        means = nn.Parameter(fused_point_cloud.requires_grad_(True))
        shs_dc = nn.Parameter(shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        shs_rest = shs[:, :, 1:].transpose(1, 2).contiguous()
        shs_rest = nn.Parameter(shs_rest.requires_grad_(True))

        scales = nn.Parameter(scales.requires_grad_(True))
        rotations = nn.Parameter(rots.requires_grad_(True))
        opacities = nn.Parameter(opacities.requires_grad_(True))

        property_dict = {
            "means": means,
            "shs_dc": shs_dc,
            "shs_rest": shs_rest,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }

        self.before_setup_set_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        self.set_properties(property_dict)

        self.active_sh_degree = 0
    
    @property
    def get_offset(self):
        return self.gaussians["offset"]
    
    @property
    def get_anchor_feat(self):
        return self.gaussians["anchor_feat"]

    def set_requires_grad(self, param_names: list, requires_grad: bool):
        for name in self.property_names:
            if name in param_names:
                self.gaussians[name].requires_grad = requires_grad
                print(f"参数 '{name}' 已被设置为 requires_grad={requires_grad}")
    
    def sample_points_from_gaussians_uniform(self, num_points_per_gaussian, visible_mask=None):
        from internal.utils.general_utils import build_rotation
        """
        从每个 2D 高斯圆盘上均匀采样指定数量的点。

        该函数会创建一个规则网格，并将其变换到每个高斯圆盘的表面。
        采样点的总数应为 num_points_per_gaussian 的完全平方数，
        例如 9, 16, 25, 49 等，以便形成一个完整的网格。

        参数:
        gaussians (object): 包含高斯参数的对象，至少应有 _xyz (N, 3), 
                            _scaling (N, 2) 和 _rotation (N, 4) 参数。
        num_points_per_gaussian (int): 每个高斯需要采样的点数。

        返回:
        torch.Tensor: 形状为 (N * num_points, 3) 的张量，包含所有采样点的世界坐标。
        """
        if visible_mask is None:
            visible_mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device = self.get_xyz.device)

        # 1. 获取高斯参数
        num_gaussians = self.get_xyz[visible_mask].shape[0]
        scales = self.get_scaling[visible_mask]  # 形状: (N, 2)
        rotations = self.get_rotation[visible_mask]  # 形状: (N, 4) - 四元数
        device = self.get_xyz.device

        # 2. 将 2D 缩放扩展为 3D。2D 高斯圆盘在 z 轴方向上的缩放为0。
        scales_3d = torch.cat([scales, torch.zeros_like(scales[:, :1])], dim=-1)  # 形状: (N, 3)

        # 3. 将四元数旋转转换为 3x3 旋转矩阵
        gs_rotation = build_rotation(rotations)  # 形状: (N, 3, 3)

        # 4. 构建完整的变换矩阵 L = R * S
        # diag_embed 用于创建对角矩阵，然后进行矩阵乘法
        L = gs_rotation @ torch.diag_embed(scales_3d)  # 形状: (N, 3, 3)

        # 5. 生成均匀分布的标准圆盘采样点
        # 为了实现均匀采样，我们生成一个 patch_num x patch_num 的网格
        patch_num = int(num_points_per_gaussian**0.5)
        if patch_num * patch_num != num_points_per_gaussian:
            raise ValueError(f"num_points_per_gaussian ({num_points_per_gaussian}) must be a perfect square for uniform grid sampling.")
        
        # 在 [-1, 1] 范围内生成网格坐标
        step = 2.0 / (patch_num - 1) if patch_num > 1 else 0
        y_range = torch.arange(-1.0, 1.0 + 1e-6, step, device=device)
        x_range = torch.arange(-1.0, 1.0 + 1e-6, step, device=device)
        
        # meshgrid 生成网格，然后 reshape 成点列表
        sample_points2d = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij'), dim=-1).reshape(-1, 2)  # 形状: (N_points, 2)
        
        # 将 2D 点扩展到 3D，z 坐标为0
        sample_points3d = torch.cat([sample_points2d, torch.zeros_like(sample_points2d[:, :1])], dim=-1)  # 形状: (N_points, 3)

        # 6. 将标准采样点通过变换矩阵 L 变换到高斯圆盘的实际位置
        # L: (N, 3, 3), sample_points3d: (N_points, 3)
        # 扩展维度以进行批量矩阵乘法
        L_expanded = L.unsqueeze(1)  # 形状: (N, 1, 3, 3)
        sample_points_expanded = sample_points3d.unsqueeze(0).unsqueeze(-1)  # 形状: (1, N_points, 3, 1)

        # 执行变换: (N, 1, 3, 3) @ (1, N_points, 3, 1) -> (N, N_points, 3, 1)
        samples_relative = torch.matmul(L_expanded, sample_points_expanded).squeeze(-1)  # 形状: (N, N_points, 3)
        
        # 7. 加上高斯圆盘的中心位置
        mean = self.get_xyz[visible_mask].unsqueeze(1)  # 形状: (N, 1, 3)
        samples_absolute = samples_relative + mean  # 形状: (N, N_points, 3)

        # 8. 重新塑形，返回所有点的列表
        return samples_absolute.reshape(-1, 3)  # 形状: (N * N_points, 3)
    
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

        # the param name and property name must be identical

        # means
        means_lr_init = optimization_config.means_lr_init * spatial_lr_scale
        means_optimizer = optimizer_factory.instantiate(
            [{'params': [self.gaussians["means"]], "name": "means"}],
            lr=means_lr_init,
            eps=1e-15,
        )
        self._add_optimizer_after_backward_hook_if_available(means_optimizer, module)
        # TODO: other scheduler may not contain `lr_final`, but does not need to change scheduler currently
        optimization_config.means_lr_scheduler.lr_final *= spatial_lr_scale
        means_scheduler = optimization_config.means_lr_scheduler.instantiate().get_scheduler(
            means_optimizer,
            means_lr_init,
        )

        # the params with constant LR
        l = [
            {'params': [self.gaussians["shs_dc"]], 'lr': optimization_config.shs_dc_lr, "name": "shs_dc"},
            {'params': [self.gaussians["shs_rest"]], 'lr': optimization_config.shs_rest_lr, "name": "shs_rest"},
            {'params': [self.gaussians["opacities"]], 'lr': optimization_config.opacities_lr, "name": "opacities"},
            {'params': [self.gaussians["scales"]], 'lr': optimization_config.scales_lr, "name": "scales"},
            {'params': [self.gaussians["rotations"]], 'lr': optimization_config.rotations_lr, "name": "rotations"},
        ]
        constant_lr_optimizer = optimizer_factory.instantiate(l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(constant_lr_optimizer, module)

        print("spatial_lr_scale={}, learning_rates=".format(spatial_lr_scale))
        print("  means={}->{}".format(means_lr_init, optimization_config.means_lr_scheduler.lr_final))
        for i in l:
            print("  {}={}".format(i["name"], i["lr"]))

        return [means_optimizer, constant_lr_optimizer], [means_scheduler]
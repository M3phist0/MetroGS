from dataclasses import dataclass
from typing import Dict

import torch
from typing import Dict, Union
import numpy as np
from torch import nn
from internal.utils.general_utils import inverse_sigmoid

from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel



@dataclass
class Gaussian2D(VanillaGaussian):
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
    def setup_from_pcd(self, xyz: Union[torch.Tensor, np.ndarray], rgb: Union[torch.Tensor, np.ndarray], *args, **kwargs):
        from internal.utils.sh_utils import RGB2SH

        if isinstance(xyz, np.ndarray):
            xyz = torch.tensor(xyz)
        if isinstance(rgb, np.ndarray):
            rgb = torch.tensor(rgb)

        fused_point_cloud = xyz.float()
        fused_color = RGB2SH(rgb.float())

        n_gaussians = fused_point_cloud.shape[0] * 3

        # SHs
        shs = torch.zeros((n_gaussians, 3, (self.config.sh_degree + 1) ** 2)).float()
        shs[:, :3, 0] = fused_color.repeat(3, 1)
        shs[:, 3:, 1:] = 0.0

        # scales
        # TODO: replace `simple_knn`
        from simple_knn._C import distCUDA2
        # the parameter device may be "cpu", so tensor must move to cuda before calling distCUDA2()
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device).repeat(3)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        perturb_factor = torch.rand(n_gaussians, 3) - 0.5
        fused_point_cloud = fused_point_cloud.repeat(3, 1) + perturb_factor * torch.exp(scales)

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

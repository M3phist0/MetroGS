import torch
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F

@dataclass
class RayBundle:
    origins: Optional[torch.Tensor] = None
    """Ray origins (XYZ)"""

    directions: Optional[torch.Tensor] = None
    """Unit ray direction vector"""

    radiis: Optional[torch.Tensor] = None
    """Ray image plane intersection circle radii"""

    ray_cos: Optional[torch.Tensor] = None
    """Ray cos"""

    def __len__(self):
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    @property
    def shape(self):
        return list(super().shape)


@dataclass
class RayBundleExt(RayBundle):

    ray_depth: Optional[torch.Tensor] = None


@dataclass
class RayBundleRast(RayBundleExt):

    ray_uv: Optional[torch.Tensor] = None
    ray_mip_level: Optional[torch.Tensor] = None

def build_ray_bundle(camera, device='cuda', coord_type='opencv', normalize_ray=True):
    if coord_type == 'opencv':
        sign_z = 1.0
    elif coord_type == 'opengl':
        sign_z = -1.0
    else:
        raise ValueError
    
    x, y = torch.meshgrid(
        torch.arange(camera.width, device=device),
        torch.arange(camera.height, device=device),
        indexing="xy",
    )
    K = camera.get_K()
    directions = F.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.0) / K[0, 0],
                (y - K[1, 2] + 0.0) / K[1, 1] * sign_z,
            ],
            dim=-1,
        ),
        (0, 1),
        value=sign_z,
    )  # [H,W,3]
    # Distance from each unit-norm direction vector to its x-axis neighbor
    dx = torch.linalg.norm(
        (directions[:, :-1, :] - directions[:, 1:, :]),
        dim=-1,
        keepdims=True,
    )  # [H,W-1,1]
    dx = torch.cat([dx, dx[:, -2:-1, :]], 1)  # [H,W,1]
    dy = torch.linalg.norm(
        (directions[:-1, :, :] - directions[1:, :, :]),
        dim=-1,
        keepdims=True,
    )  # [H-1,W,1]
    dy = torch.cat([dy, dy[-2:-1, :, :]], 0)  # [H,W,1]
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    area = dx * dy
    radii = torch.sqrt(area / torch.pi)
    if normalize_ray:
        directions = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
    ray_bundle = RayBundle(
        origins=torch.zeros_like(directions),
        directions=directions,
        radiis=radii,
        ray_cos=torch.matmul(
            directions,
            torch.tensor([[0.0, 0.0, sign_z]], device=device).T,
        ),
    )

    return ray_bundle

from torch.cuda.amp import autocast
def build_world_ray_bundle(camera, device='cuda', normalize_ray=False):
    device = torch.device(device)

    # 在这个函数内部禁用 AMP，全部用 float32 计算，避免 inverse 报错
    with autocast(enabled=False):
        c2w = camera.world_to_camera.T.to(device=device, dtype=torch.float32).inverse()
        W, H = camera.width, camera.height

        ndc2pix = torch.tensor(
            [
                [W / 2, 0, 0, W / 2],
                [0, H / 2, 0, H / 2],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        ).T

        projection_matrix = c2w.T @ camera.full_projection.to(device=device, dtype=torch.float32)
        intrins = (projection_matrix @ ndc2pix)[:3, :3].T  # float32

        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            indexing='xy',
        )
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)

        # 这里用 float32 做矩阵求逆 + 乘法，就不会再报 Half 的错
        directions_cam = points @ torch.linalg.inv(intrins).T
        directions = directions_cam @ c2w[:3, :3].T
        directions = directions.view(H, W, 3)

        rays_o = c2w[:3, 3]
        cam_z_axis = c2w[:3, 2]

        dx = torch.linalg.norm(
            (directions[:, :-1, :] - directions[:, 1:, :]),
            dim=-1,
            keepdims=True,
        )  # [H, W-1, 1]
        dx = torch.cat([dx, dx[:, -2:-1, :]], 1)  # [H, W, 1]

        dy = torch.linalg.norm(
            (directions[:-1, :, :] - directions[1:, :, :]),
            dim=-1,
            keepdims=True,
        )  # [H-1, W, 1]
        dy = torch.cat([dy, dy[-2:-1, :, :]], 0)  # [H, W, 1]

        area = dx * dy
        radii = torch.sqrt(area / torch.pi)

        directions_norm = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if normalize_ray:
            directions = directions_norm

        ray_cos = torch.matmul(
            directions_norm,
            cam_z_axis.T,
        ).unsqueeze(-1)

        ray_bundle = RayBundle(
            origins=torch.ones_like(directions) * rays_o.unsqueeze(0).unsqueeze(1),
            directions=directions,
            radiis=radii,
            ray_cos=ray_cos,
        )

    # 返回的 ray_bundle 是 float32，后面的网络在 AMP 里会自动按需转换
    return ray_bundle

# def build_world_ray_bundle(camera, device='cuda', normalize_ray=False):
#     c2w = (camera.world_to_camera.T).inverse()
#     W, H = camera.width, camera.height
#     ndc2pix = torch.tensor([
#         [W / 2, 0, 0, W / 2],
#         [0, H / 2, 0, H / 2],
#         [0, 0, 0, 1]]).float().cuda().T
#     projection_matrix = c2w.T @ camera.full_projection
#     intrins = (projection_matrix @ ndc2pix)[:3, :3].T

#     grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
#     points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
#     directions_cam = points @ intrins.inverse().T
#     directions = directions_cam @ c2w[:3, :3].T
#     directions = directions.view(H, W, 3)
#     rays_o = c2w[:3, 3]

#     cam_z_axis = c2w[:3, 2]
    
#     # Distance from each unit-norm direction vector to its x-axis neighbor
#     dx = torch.linalg.norm(
#         (directions[:, :-1, :] - directions[:, 1:, :]),
#         dim=-1,
#         keepdims=True,
#     )  # [H,W-1,1]
#     dx = torch.cat([dx, dx[:, -2:-1, :]], 1)  # [H,W,1]
#     dy = torch.linalg.norm(
#         (directions[:-1, :, :] - directions[1:, :, :]),
#         dim=-1,
#         keepdims=True,
#     )  # [H-1,W,1]
#     dy = torch.cat([dy, dy[-2:-1, :, :]], 0)  # [H,W,1]
#     # Cut the distance in half, and then round it out so that it's
#     # halfway between inscribed by / circumscribed about the pixel.
#     area = dx * dy
#     radii = torch.sqrt(area / torch.pi)

#     directions_norm = directions / torch.linalg.norm(
#         directions, dim=-1, keepdims=True
#     )
    
#     if normalize_ray:
#         directions = directions_norm

#     ray_cos = torch.matmul(
#         directions_norm,
#         cam_z_axis.T,
#     ).unsqueeze(-1)

#     ray_bundle = RayBundle(
#         origins=torch.ones_like(directions) * rays_o.unsqueeze(0).unsqueeze(1),
#         directions=directions,
#         radiis=radii,
#         ray_cos=ray_cos,
#     )
#     return ray_bundle
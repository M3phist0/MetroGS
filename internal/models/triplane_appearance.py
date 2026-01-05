import math
import torch
from torch import nn

import nvdiffrast.torch
import tinycudann as tcnn
import torch.distributed as dist

from utils.ray import build_ray_bundle, build_world_ray_bundle

from typing import Union, List, Tuple

class TriMipEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int,
        plane_size: int,
        feature_dim: int,
        include_xyz: bool = False,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.feature_dim = feature_dim
        self.include_xyz = include_xyz

        self.register_parameter(
            "fm",
            nn.Parameter(torch.zeros(3, plane_size, plane_size, feature_dim)),
        )
        self.init_parameters()
        self.dim_out = (
            self.feature_dim * 3 + 3 if include_xyz else self.feature_dim * 3
        )

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, level):
        # x in [0,1], level in [0,max_level]
        # x is Nx3, level is Nx1
        if 0 == x.shape[0]:
            return torch.zeros([x.shape[0], self.feature_dim * 3]).to(x)
        decomposed_x = torch.stack(
            [
                x[:, None, [1, 2]],
                x[:, None, [0, 2]],
                x[:, None, [0, 1]],
            ],
            dim=0,
        )  # 3xNx1x2
        if 0 == self.n_levels:
            level = None
        else:
            # assert level.shape[0] > 0, [level.shape, x.shape]
            torch.stack([level, level, level], dim=0)
            level = torch.broadcast_to(
                level, decomposed_x.shape[:3]
            ).contiguous()
        enc = nvdiffrast.torch.texture(
            self.fm,
            decomposed_x,
            mip_level_bias=level,
            boundary_mode="clamp",
            max_mip_level=self.n_levels - 1,
        )  # 3xNx1xC
        enc = (
            enc.permute(1, 2, 0, 3)
            .contiguous()
            .view(
                x.shape[0],
                self.feature_dim * 3,
            )
        )  # Nx(3C)
        if self.include_xyz:
            enc = torch.cat([x, enc], dim=-1)
        return enc

class TriMipModel(nn.Module):
    def __init__(
        self,
        optimization,
        n_appearance_count: int=6000,
        geo_feat_dim: int = 15,
        n_rgb_dims: int = 3,
        app_emb_dim: int = 64,
        std: float = 1e-4,
        n_levels: int = 8,
        plane_size: int = 512,
        feature_dim: int = 16,
        net_depth_base: int = 2,
        net_depth_color: int = 3,
        net_width: int = 128,
        occ_grid_resolution: int = 128,
        aabb: Union[torch.Tensor, List[float]] = torch.tensor([-3, -3, -3, 3, 3, 3]),
        version=1,
        final_activation="None",
    ) -> None:
        super().__init__()
        self.optimization = optimization

        self._appearance_embeddings = nn.Parameter(torch.empty(n_appearance_count, app_emb_dim).cuda())
        self._appearance_embeddings.data.normal_(0, std)

        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)
        self.app_emb_dim = app_emb_dim

        self.occ_grid_resolution = occ_grid_resolution
        self.aabb = aabb.cuda()
        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        self.aabb_size = aabb_max - aabb_min

        self.feature_vol_radii = self.aabb_size[0] / 2.0
        self.register_buffer(
            "occ_level_vol",
            torch.log2(
                self.aabb_size[0]
                / occ_grid_resolution
                / 2.0
                / self.feature_vol_radii
            ),
        )

        self.encoding = TriMipEncoding(n_levels, plane_size, feature_dim)

        self.version = version
        if version == 1:
            self.mlp = tcnn.Network(
                n_input_dims=self.encoding.dim_out + self.app_emb_dim,
                n_output_dims=n_rgb_dims,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": final_activation,
                    "n_neurons": net_width,
                    "n_hidden_layers": net_depth_base,
                },
            )
        else:
            self.mlp = tcnn.Network(
                n_input_dims=self.encoding.dim_out,
                n_output_dims=geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": net_width,
                    "n_hidden_layers": net_depth_base,
                },
            )

            self.mlp_head = tcnn.Network(
                n_input_dims=self.app_emb_dim + geo_feat_dim,
                n_output_dims=n_rgb_dims,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": final_activation,
                    "n_neurons": net_width,
                    "n_hidden_layers": net_depth_color,
                },
            )

    def forward(self, image, depth, camera, emb_precomp=None):

        detached_depth = depth.detach()

        view_idx = camera.appearance_id

        H, W = image.size(1), image.size(2)

        rays = build_world_ray_bundle(camera)
        
        t_origins = rays.origins
        t_dirs = rays.directions
        radiis = rays.radiis
        cos = rays.ray_cos
        distance = detached_depth.permute(1, 2, 0)

        positions = t_origins + t_dirs * distance
        positions = self.contraction(positions)

        sample_ball_radii = self.compute_ball_radii(distance, radiis, cos)
        level_vol = torch.log2(
            sample_ball_radii / self.feature_vol_radii
        )  # real level should + log2(feature_resolution)

        if emb_precomp is None:
            embedding = self.get_appearance(view_idx)
        else:
            embedding = emb_precomp

        level = (
            level_vol if level_vol is None else level_vol + self.log2_plane_size
        )
        encoded_x = self.encoding(positions.view(-1, 3), level.view(-1, 1))

        if self.version == 1:
            embedding_expand = embedding.expand(encoded_x.shape[0], -1)

            combined_features = torch.cat([encoded_x, embedding_expand], dim=-1)

            mapping = self.mlp(combined_features).permute(1, 0).view(3, H, W)
        else:
            geo_feature = self.mlp(encoded_x)

            embedding_expand = embedding.expand(geo_feature.shape[0], -1)

            combined_features = torch.cat([geo_feature, embedding_expand], dim=-1)

            mapping = self.mlp_head(combined_features).permute(1, 0).view(3, H, W)

        transformed_image = image * mapping

        return transformed_image, mapping
    
    def get_appearance(self, view_idx: Union[float, torch.Tensor]):
        return self._appearance_embeddings[view_idx]
    
    def sync_model(self):
        with torch.no_grad():
            for param in self.mlp.parameters():
                dist.broadcast(param.data, 0)
            for param in self.encoding.parameters():
                dist.broadcast(param.data, 0)
            if hasattr(self, 'mlp_head'):
                for param in self.mlp_head.parameters():
                    dist.broadcast(param.data, 0)
            dist.broadcast(self._appearance_embeddings.data, 0)

    def all_reduce(self, bsz=1):
        with torch.no_grad():
            # all reduce mlp grad
            for param in self.mlp.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    torch.cuda.synchronize()
                    dist.barrier()
                    param.grad = param.grad / bsz

            for param in self.encoding.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    torch.cuda.synchronize()
                    dist.barrier()
                    param.grad = param.grad / bsz

            if hasattr(self, 'mlp_head'):
                for param in self.mlp_head.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad)
                        torch.cuda.synchronize()
                        dist.barrier()
                        param.grad = param.grad / bsz
        
            if self._appearance_embeddings.grad is not None:
                dist.all_reduce(self._appearance_embeddings.grad)
                torch.cuda.synchronize()
                dist.barrier()
    
    def update_aabb(self, aabb):
        self.aabb = aabb.cuda()
        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        self.aabb_size = aabb_max - aabb_min

        self.feature_vol_radii = self.aabb_size[0] / 2.0
        self.register_buffer(
            "occ_level_vol",
            torch.log2(
                self.aabb_size[0]
                / self.occ_grid_resolution
                / 2.0
                / self.feature_vol_radii
            ),
        )

    @staticmethod
    def compute_ball_radii(distance, radiis, cos):
        inverse_cos = 1.0 / cos
        tmp = (inverse_cos * inverse_cos - 1).sqrt() - radiis
        sample_ball_radii = distance * radiis * cos / (tmp * tmp + 1.0).sqrt()
        return sample_ball_radii

    def contraction(self, x):
        aabb_min, aabb_max = self.aabb[:3].unsqueeze(0), self.aabb[
            3:
        ].unsqueeze(0)
        x = (x - aabb_min) / (aabb_max - aabb_min)

        # print("x_val:", x)
        return x
    
    def create_optimizer_and_scheduler(self):
        embedding_optimizer, embedding_scheduler = self._create_optimizer_and_scheduler(
            [self._appearance_embeddings],
            "appearance_embeddings",
            lr_init=self.optimization.embedding_lr_init,
            lr_final_factor=self.optimization.lr_final_factor,
            max_steps=self.optimization.max_steps,
            eps=self.optimization.eps,
            warm_up=self.optimization.warm_up,
        )

        enc_optimizer, enc_scheduler = self._create_optimizer_and_scheduler(
            self.encoding.parameters(),
            "appearance_encoding",
            lr_init=self.optimization.lr_init,
            lr_final_factor=self.optimization.lr_final_factor,
            max_steps=self.optimization.max_steps,
            eps=self.optimization.eps,
            warm_up=self.optimization.warm_up,
        )

        mlp_optimizer, mlp_scheduler = self._create_optimizer_and_scheduler(
            self.mlp.parameters(),
            "appearance_mlp",
            lr_init=self.optimization.lr_init,
            lr_final_factor=self.optimization.lr_final_factor,
            max_steps=self.optimization.max_steps,
            eps=self.optimization.eps,
            warm_up=self.optimization.warm_up,
        )

        if self.version == 1:
            return [embedding_optimizer, enc_optimizer, mlp_optimizer], [embedding_scheduler, enc_scheduler, mlp_scheduler]
        else:
            head_optimizer, head_scheduler = self._create_optimizer_and_scheduler(
                self.mlp_head.parameters(),
                "appearance_mlp_head",
                lr_init=self.optimization.lr_init,
                lr_final_factor=self.optimization.lr_final_factor,
                max_steps=self.optimization.max_steps,
                eps=self.optimization.eps,
                warm_up=self.optimization.warm_up,
            )
            return [embedding_optimizer, enc_optimizer, mlp_optimizer, head_optimizer], [embedding_scheduler, enc_scheduler, mlp_scheduler, head_scheduler]
    
    @staticmethod
    def _create_optimizer_and_scheduler(
            params,
            name,
            lr_init,
            lr_final_factor,
            max_steps,
            eps,
            warm_up,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            params=[
                {"params": list(params), "name": name}
            ],
            lr=lr_init,
            eps=eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: lr_final_factor ** min(max(iter - warm_up, 0) / max_steps, 1),
            verbose=False,
        )

        return optimizer, scheduler


import math
import torch
from torch import nn
from typing import Union, List, Tuple, Optional
import torch.distributed as dist

import nvdiffrast.torch
import tinycudann as tcnn

from utils.ray import build_world_ray_bundle


class TriMipEncodingCUDA(nn.Module):
    def __init__(self, n_levels: int, plane_size: int, feature_dim: int, include_xyz: bool = False):
        super().__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.feature_dim = feature_dim
        self.include_xyz = include_xyz

        self.fm = nn.Parameter(torch.zeros(3, plane_size, plane_size, feature_dim, device="cuda"))
        self.dim_out = (feature_dim * 3 + 3) if include_xyz else (feature_dim * 3)

    def forward(self, x, level):
        if x.shape[0] == 0:
            return torch.zeros((0, self.feature_dim * 3), device=x.device, dtype=x.dtype)

        decomposed_x = torch.stack(
            [x[:, None, [1, 2]], x[:, None, [0, 2]], x[:, None, [0, 1]]],
            dim=0,
        )

        if self.n_levels == 0:
            level = None
        else:
            # torch.stack([level, level, level], dim=0)  # 原来无效，这里不需要
            level = torch.broadcast_to(level, decomposed_x.shape[:3]).contiguous()

        enc = nvdiffrast.torch.texture(
            self.fm, decomposed_x,
            mip_level_bias=level,
            boundary_mode="clamp",
            max_mip_level=self.n_levels - 1,
        )
        enc = enc.permute(1, 2, 0, 3).contiguous().view(x.shape[0], self.feature_dim * 3)
        if self.include_xyz:
            enc = torch.cat([x, enc], dim=-1)
        return enc


class _CPUMaster:
    """
    非 nn.Module 容器：Lightning 不会 .to() / DDP 包它，因此 CPU master 永远留在 CPU。
    里面用 nn.Parameter 只是为了让 torch.optim.Adam 能更新。
    """
    def __init__(self, pin_cpu: bool = True):
        self.pin_cpu = pin_cpu

        self.emb: Optional[nn.Parameter] = None
        self.fm: Optional[nn.Parameter] = None
        self.tcnn_params: List[nn.Parameter] = []  # CPU master tcnn weights

        # embedding 稀疏 grad buffer（CPU Tensor）
        self.emb_grad: Optional[torch.Tensor] = None

    def _pin(self, t: torch.Tensor) -> torch.Tensor:
        if self.pin_cpu:
            return t.contiguous().pin_memory()
        return t.contiguous()

    def set_embeddings(self, t: torch.Tensor):
        t = self._pin(t.to("cpu", dtype=torch.float32))
        self.emb = nn.Parameter(t)  # 不属于 module tree，所以不会被 Lightning move
        self.emb.grad = None

    def set_fm(self, t: torch.Tensor):
        t = self._pin(t.to("cpu", dtype=torch.float32))
        self.fm = nn.Parameter(t)
        self.fm.grad = None

    def add_tcnn_param(self, t: torch.Tensor):
        t = self._pin(t.to("cpu", dtype=torch.float32))
        p = nn.Parameter(t)
        p.grad = None
        self.tcnn_params.append(p)


class TriMipModel(nn.Module):
    """
    最稳妥：
    - CPU master 全部放在 self.cpu_master（非 Module 容器），Lightning 不会把它搬到 CUDA
    - CUDA shadow 用于 forward/backward
    - backward 后：NCCL allreduce shadow grads -> 拷回 CPU master grads
    - optimizer 仅更新 CPU master（外部 optimizer.step()）
    - step 后：调用 sync_cuda_from_cpu() 推回 shadow
    """
    def __init__(
        self,
        optimization,
        n_appearance_count: int = 6000,
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
        final_activation: str = "None",
        pin_cpu: bool = True,
        cuda_device: str = "cuda",
    ) -> None:
        super().__init__()
        self.optimization = optimization
        self.cuda_device = torch.device(cuda_device)

        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)
        self.app_emb_dim = app_emb_dim
        self.occ_grid_resolution = occ_grid_resolution

        # buffers（Lightning 会把 buffer 搬，但它们小且不影响 offload；也可保持 CPU）
        self.register_buffer("aabb", torch.zeros(6, dtype=torch.float32))
        self.register_buffer("aabb_size", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("feature_vol_radii", torch.zeros((), dtype=torch.float32))
        self.register_buffer("occ_level_vol", torch.zeros((), dtype=torch.float32))
        self.update_aabb(aabb)

        # ========= CPU master（不注册成 module params）=========
        self.cpu_master = _CPUMaster(pin_cpu=pin_cpu)

        emb = torch.empty(n_appearance_count, app_emb_dim, device="cpu", dtype=torch.float32)
        emb.normal_(0, std)
        self.cpu_master.set_embeddings(emb)

        fm = torch.zeros(3, plane_size, plane_size, feature_dim, device="cpu", dtype=torch.float32)
        nn.init.uniform_(fm, -1e-2, 1e-2)
        self.cpu_master.set_fm(fm)

        # ========= CUDA shadow =========
        self.encoding_cuda = TriMipEncodingCUDA(n_levels, plane_size, feature_dim, include_xyz=False).to(self.cuda_device)

        self.mlp_cuda = tcnn.Network(
            n_input_dims=self.encoding_cuda.dim_out + self.app_emb_dim,
            n_output_dims=n_rgb_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": final_activation,
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_base,
            },
        ).to(self.cuda_device)
        self.mlp_head_cuda = None

        # shadow params 列表（CUDA）+ CPU master 对应拷贝（放 cpu_master.tcnn_params）
        self._shadow_params: List[nn.Parameter] = []

        def _register_shadow_params(module: nn.Module):
            for p in module.parameters():
                assert p.is_cuda
                self._shadow_params.append(p)
                self.cpu_master.add_tcnn_param(p.detach())  # 初始值从 shadow 拷贝到 CPU master

        _register_shadow_params(self.mlp_cuda)
        if self.mlp_head_cuda is not None:
            _register_shadow_params(self.mlp_head_cuda)

        assert len(self._shadow_params) == len(self.cpu_master.tcnn_params)

        # 本 step 的 view_idx 与 embedding row grad（CUDA）
        self._last_view_idx: Optional[int] = None
        self._last_emb_grad_cuda: Optional[torch.Tensor] = None

        # 初次同步：CPU->CUDA
        self.sync_cuda_from_cpu()

    # ----------------------------
    # 同步：CPU master -> CUDA shadow
    # ----------------------------
    @torch.no_grad()
    def sync_cuda_from_cpu(self):
        # fm
        self.encoding_cuda.fm.data.copy_(self.cpu_master.fm.to(self.cuda_device, non_blocking=True))

        # tcnn weights
        for shadow_p, cpu_p in zip(self._shadow_params, self.cpu_master.tcnn_params):
            shadow_p.data.copy_(cpu_p.to(self.cuda_device, non_blocking=True))

    # ----------------------------
    # NCCL allreduce：CUDA grads
    # ----------------------------
    @torch.no_grad()
    def all_reduce(self, bsz: int = 1):
        if not (dist.is_available() and dist.is_initialized()):
            return

        for p in self._shadow_params:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            if bsz != 1:
                p.grad.div_(bsz)

        if self.encoding_cuda.fm.grad is not None:
            dist.all_reduce(self.encoding_cuda.fm.grad, op=dist.ReduceOp.SUM)
            if bsz != 1:
                self.encoding_cuda.fm.grad.div_(bsz)

        if self._last_emb_grad_cuda is not None:
            dist.all_reduce(self._last_emb_grad_cuda, op=dist.ReduceOp.SUM)
            if bsz != 1:
                self._last_emb_grad_cuda.div_(bsz)

    # ----------------------------
    # CUDA grads -> CPU master grads
    # ----------------------------
    @torch.no_grad()
    def sync_cpu_grads_from_cuda(self):
        # tcnn grads
        for shadow_p, cpu_p in zip(self._shadow_params, self.cpu_master.tcnn_params):
            if shadow_p.grad is None:
                cpu_p.grad = None
                continue
            g_cpu = shadow_p.grad.detach().float().cpu().contiguous()  # 强制 CPU，最稳
            if cpu_p.grad is None:
                cpu_p.grad = torch.empty_like(cpu_p.data)
            cpu_p.grad.copy_(g_cpu)

        # fm grad
        if self.encoding_cuda.fm.grad is None:
            self.cpu_master.fm.grad = None
        else:
            g_cpu = self.encoding_cuda.fm.grad.detach().float().cpu().contiguous()
            if self.cpu_master.fm.grad is None:
                self.cpu_master.fm.grad = torch.empty_like(self.cpu_master.fm.data)
            self.cpu_master.fm.grad.copy_(g_cpu)

        # embedding 稀疏写回（CPU）
        if self._last_view_idx is not None and self._last_emb_grad_cuda is not None:
            if self.cpu_master.emb_grad is None:
                self.cpu_master.emb_grad = torch.zeros_like(self.cpu_master.emb.data)  # CPU

            g_row_cpu = self._last_emb_grad_cuda.detach().float().cpu().contiguous()
            self.cpu_master.emb_grad[self._last_view_idx].add_(g_row_cpu)

            # 把 emb_grad 接到 emb.param.grad 上（让 Adam 看到）
            self.cpu_master.emb.grad = self.cpu_master.emb_grad

    # ----------------------------
    # forward：CUDA shadow 计算
    # ----------------------------
    def forward(self, image, depth, camera, emb_precomp=None):
        self.sync_cuda_from_cpu()

        view_idx = camera.appearance_id
        self._last_view_idx = int(view_idx) if not torch.is_tensor(view_idx) else int(view_idx.item())
        self._last_emb_grad_cuda = None

        detached_depth = depth.detach()
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
        level_vol = torch.log2(sample_ball_radii / self.feature_vol_radii)
        level = level_vol + self.log2_plane_size

        # embedding：CPU master row -> CUDA tensor (requires_grad)，用 hook 拿 grad
        if emb_precomp is None:
            emb_row_cpu = self.cpu_master.emb[self._last_view_idx]  # CPU row
            emb_row_cuda = emb_row_cpu.to(self.cuda_device, non_blocking=True).requires_grad_(True)

            def _save_grad(g):
                self._last_emb_grad_cuda = g
                return g

            emb_row_cuda.register_hook(_save_grad)
            embedding = emb_row_cuda
        else:
            embedding = emb_precomp

        encoded_x = self.encoding_cuda(positions.view(-1, 3), level.view(-1, 1))

        embedding_expand = embedding.expand(encoded_x.shape[0], -1)
        combined_features = torch.cat([encoded_x, embedding_expand], dim=-1)
        mapping = self.mlp_cuda(combined_features).permute(1, 0).view(3, H, W)

        transformed_image = image * mapping
        return transformed_image, mapping

    def update_aabb(self, aabb):
        if not torch.is_tensor(aabb):
            aabb = torch.tensor(aabb, dtype=torch.float32)

        if aabb.shape[-2:] == (2, 3):
            aabb_flat = aabb.reshape(*aabb.shape[:-2], 6)
        else:
            aabb_flat = aabb
        if aabb_flat.shape[-1] != 6:
            raise ValueError(f"aabb last dim must be 6, got {tuple(aabb.shape)}")
        if aabb_flat.dim() > 1:
            aabb_flat = aabb_flat.reshape(-1, 6)[0]
        aabb_flat = aabb_flat.detach().float()

        self.aabb.copy_(aabb_flat.to(device=self.aabb.device, dtype=self.aabb.dtype))
        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        aabb_size = aabb_max - aabb_min
        self.aabb_size.copy_(aabb_size.to(self.aabb_size.device, dtype=self.aabb_size.dtype))

        feature_vol_radii = self.aabb_size[0] / 2.0
        self.feature_vol_radii.copy_(feature_vol_radii.to(self.feature_vol_radii.device, dtype=self.feature_vol_radii.dtype))

        occ_level_vol = torch.log2(self.aabb_size[0] / self.occ_grid_resolution / 2.0 / self.feature_vol_radii)
        self.occ_level_vol.copy_(occ_level_vol.to(self.occ_level_vol.device, dtype=self.occ_level_vol.dtype))

    @staticmethod
    def compute_ball_radii(distance, radiis, cos):
        inverse_cos = 1.0 / cos
        tmp = (inverse_cos * inverse_cos - 1).sqrt() - radiis
        sample_ball_radii = distance * radiis * cos / (tmp * tmp + 1.0).sqrt()
        return sample_ball_radii

    def contraction(self, x):
        aabb_min, aabb_max = self.aabb[:3].unsqueeze(0), self.aabb[3:].unsqueeze(0)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        return x

    def create_optimizer_and_scheduler(self):
        emb_opt, emb_sch = self._create_optimizer_and_scheduler(
            [self.cpu_master.emb],
            "appearance_embeddings",
            lr_init=self.optimization.embedding_lr_init,
            lr_final_factor=self.optimization.lr_final_factor,
            max_steps=self.optimization.max_steps,
            eps=self.optimization.eps,
            warm_up=self.optimization.warm_up,
        )

        fm_opt, fm_sch = self._create_optimizer_and_scheduler(
            [self.cpu_master.fm],
            "appearance_encoding",
            lr_init=self.optimization.lr_init,
            lr_final_factor=self.optimization.lr_final_factor,
            max_steps=self.optimization.max_steps,
            eps=self.optimization.eps,
            warm_up=self.optimization.warm_up,
        )

        tcnn_opt, tcnn_sch = self._create_optimizer_and_scheduler(
            list(self.cpu_master.tcnn_params),
            "appearance_mlp",
            lr_init=self.optimization.lr_init,
            lr_final_factor=self.optimization.lr_final_factor,
            max_steps=self.optimization.max_steps,
            eps=self.optimization.eps,
            warm_up=self.optimization.warm_up,
        )

        return [emb_opt, fm_opt, tcnn_opt], [emb_sch, fm_sch, tcnn_sch]

    @staticmethod
    def _create_optimizer_and_scheduler(params, name, lr_init, lr_final_factor, max_steps, eps, warm_up):
        p_list = list(params)
        optimizer = torch.optim.Adam(
            params=[{"params": p_list, "name": name}],
            lr=lr_init,
            eps=eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda it: lr_final_factor ** min(max(it - warm_up, 0) / max_steps, 1),
        )
        return optimizer, scheduler

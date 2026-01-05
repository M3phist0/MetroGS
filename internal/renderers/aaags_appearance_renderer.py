from typing import Dict, Tuple, Union, Callable, Optional, List

import traceback
import lightning
import torch
import math
from einops import repeat
from dataclasses import dataclass
from .renderer import RendererOutputTypes, RendererOutputInfo, Renderer, RendererConfig
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from internal.density_controllers.density_controller import Utils as DensityControllerUtils
from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer

import torch.distributed.nn.functional as dist_func

from gsplat.sh_decomposed import spherical_harmonics_decomposed
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as AnchorSettings, GaussianRasterizer as AnchorRasterizer
from diff_surfel_rasterization._C import get_local2j_ids_bool

from internal.models.vast_appearance import VastModel, VastModelV2
from .gsplat_appearance_embedding_renderer import OptimizationConfig as AppearanceModelOptimizationConfig
import torch.distributed as dist

DEFAULT_BLOCK_SIZE: int =  16

@dataclass
class DistributedRenderer(RendererConfig):
    block_size: int = DEFAULT_BLOCK_SIZE

    filter_2d_kernel_size: float = 0.3

    tile_based_culling: bool = False

    # Since the density controllers are replaceable, below parameters should be updated manually when the parameters of density controller changed

    redistribute_interval: int = 1000
    """This value should be the result of `n` times the densify interval, where `n` is an integer"""

    redistribute_until: int = 30_000
    """Should be the same as the densify until iteration"""

    redistribute_threshold: float = 1.1
    """Redistribute if min*threshold < max"""

    depth_ratio: float = 1.0
    K: int = 5
    v_pow: float = 0.1
    prune_ratio: float = 0.1
    contribution_prune_from_iter : int = 1000
    contribution_prune_interval: int = 500
    start_prune_ratio: float = 0.0
    diable_start_trimming: bool = False
    diable_trimming: bool = True
    """SepDepthTrim2DGS Params"""

    def instantiate(self, *args, **kwargs) -> Renderer:
        return DistributedRendererImpl(self)

class DistributedRendererImpl(SepDepthTrim2DGSRenderer):
    @staticmethod
    def depths_to_points(view, depthmap):
        device = view.world_to_camera.device
        c2w = (view.world_to_camera.T).inverse()
        W, H = view.width, view.height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, W / 2],
            [0, H / 2, 0, H / 2],
            [0, 0, 0, 1]]).float().cuda().T
        projection_matrix = c2w.T @ view.full_projection
        intrins = (projection_matrix @ ndc2pix)[:3, :3].T

        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
        rays_o = c2w[:3, 3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
        return points

    @classmethod
    def depth_to_normal(cls, view, depth):
        """
            view: view camera
            depth: depthmap
        """
        points = cls.depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
        output = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output[1:-1, 1:-1, :] = normal_map
        return output
    
    def __init__(self, config: DistributedRenderer):
        super().__init__(config.depth_ratio, config.K, config.v_pow, config.prune_ratio, config.contribution_prune_from_iter, 
                         config.contribution_prune_interval, config.start_prune_ratio, config.diable_start_trimming, config.diable_trimming)

        self.block_size = 16
        self.world_size = 1
        self.global_rank = 0

        self.redistribute_threshold = config.redistribute_threshold
        self.redistribute_interval = config.redistribute_interval
        self.redistribute_until = config.redistribute_until

        self.n_appearances = -1

    def setup(self, stage: str, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        super().setup(stage, *args, **kwargs)

        lightning_module = kwargs.get("lightning_module", None)

        if lightning_module is not None:
            if self.n_appearances <= 0:
                max_input_id = 0
                appearance_group_ids = lightning_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
                if appearance_group_ids is not None:
                    for i in appearance_group_ids.values():
                        if i[0] > max_input_id:
                            max_input_id = i[0]
                n_appearances = max_input_id + 1
                self.n_appearances = n_appearances
            self.n_appearances = 1657
            self._setup_model()
            print(self.model)
    
    def load_state_dict(self, state_dict, strict: bool = False):
        self.n_appearances = state_dict["model.embedding.weight"].shape[0]
        self._setup_model(device=state_dict["model.embedding.weight"].device)
        return super().load_state_dict(state_dict, strict)
    
    def _setup_model(self, device=None):
        self.model = VastModel(n_appearance_count=self.n_appearances)
        # self.model = VastModelV2(n_appearance_count=self.n_appearances)

        if device is not None:
            self.model.to(device=device)

    def training_setup(self, module: lightning.LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        self.world_size = module.trainer.world_size
        self.global_rank = module.trainer.global_rank

        # divide gaussians evenly
        n_gaussians = module.gaussian_model.n_gaussians
        n_gaussians_per_member = round(n_gaussians / self.world_size)

        l = n_gaussians_per_member * self.global_rank
        r = l + n_gaussians_per_member
        if self.global_rank + 1 == self.world_size:
            r = n_gaussians

        new_param_tensors = {}
        for attr_name, attr_value in module.gaussian_model.properties.items():
            new_param_tensors[attr_name] = attr_value[l:r]

        self.replace_tensors_to_optimizer(new_param_tensors, module.gaussian_model, module.gaussian_optimizers)

        print("Start from gaussian number:", module.gaussian_model.get_xyz.shape[0])

        # notify module
        self.on_density_changed = module.density_updated_by_renderer
        self.on_density_changed()

        try:
            self.profiler = module.trainer.profiler
        except:
            traceback.print_exc()
            pass

        module.density_updated_by_renderer()

        print(f"rank={self.global_rank}, l={l}, r={r}")

        def get_trainer():
            return module.trainer

        self.get_trainer = get_trainer

        train_set_cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras

        self.img_height = train_set_cameras[0].height.item()
        self.img_width = train_set_cameras[0].width.item()
        self.tile_width = math.ceil(self.img_width / float(self.block_size))
        self.tile_height = math.ceil(self.img_height / float(self.block_size))

        self.strategy_history = DivisionStrategyHistory(
            train_set_cameras, self.world_size, self.global_rank, self.tile_height
        )

        self.optimization: AppearanceModelOptimizationConfig = AppearanceModelOptimizationConfig()

        embedding_optimizer, embedding_scheduler = VastModel._create_optimizer_and_scheduler(
            [self.model._appearance_embeddings],
            "appearance_embeddings",
            lr_init=self.optimization.embedding_lr_init,
            lr_final_factor=self.optimization.lr_final_factor,
            max_steps=30000,
            eps=self.optimization.eps,
            warm_up=1000,
        )
        network_optimizer, network_scheduler = VastModel._create_optimizer_and_scheduler(
            self.model.appearance_network.parameters(),
            "appearance_network",
            lr_init=self.optimization.lr_init,
            lr_final_factor=self.optimization.lr_final_factor,
            max_steps=30000,
            eps=self.optimization.eps,
            warm_up=1000,
        )

        # return None, None
        return [embedding_optimizer, network_optimizer], [embedding_scheduler, network_scheduler]

    @staticmethod
    def replace_tensors_to_optimizer(tensors_dict, gaussian_model, optimizers):
        gaussian_model.properties = DensityControllerUtils.replace_tensors_to_properties(
            tensors_dict,
            optimizers,
        )

    def training_forward(
            self,
            step: int,
            module: lightning.LightningModule,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            render_types: list = None,
            **kwargs,
    ):
        return self(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            render_types=render_types,
            iteration=step,
            **kwargs,
        )

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            record_transmittance=False,
            **kwargs,
    ):
        
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        if self.world_size > 1:
            with torch.no_grad():
                for param in self.model.appearance_network.parameters():
                    dist.broadcast(param.data, 0)

        if not isinstance(viewpoint_camera, list):
            return self.forward_eval(viewpoint_camera, pc, bg_color, scaling_modifier, record_transmittance, **kwargs)

        viewpoint_camera_list = viewpoint_camera
        bsz = len(viewpoint_camera_list)

        iteration = kwargs.get('iteration', -1)
        render_src = kwargs.get('render_src', False)
        
        strategy_list, gpuid2tasks =  self.strategy_history.start_strategy(
            viewpoint_camera, self.tile_height, self.tile_width, self.global_rank
        )

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        gs_count = pc.get_xyz.shape[0]
        global_idx = torch.arange(gs_count, device=pc.get_xyz.device)
        if self.world_size > 1:
            local_gs_count_tensor = torch.tensor([gs_count], dtype=torch.int64, device=pc.get_xyz.device)
            all_gs_counts = [torch.zeros(1, dtype=torch.int64, device=pc.get_xyz.device) for _ in range(self.world_size)]
            torch.distributed.all_gather(all_gs_counts, local_gs_count_tensor)
            all_gs_counts = torch.cat(all_gs_counts)
            global_offset = torch.sum(all_gs_counts[:self.global_rank])
            global_idx = global_idx + global_offset.item()
            gs_count = torch.sum(all_gs_counts)
        gloabal_acc_pix = torch.ones(gs_count, device=pc.get_xyz.device)
        
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        cov3D_precomp = None
        scales = pc.get_scaling[..., :2]
        rotations = pc.get_rotation

        shs = pc.get_features

        rasterizer_list = []  # One rasterizer for each picture in a batch
        cuda_args_list = []  # Per picture in a batch
        screenspace_params_list = []  # Per picture in a batch
        means2D_list = []
        radii_list = []

        for i, (viewpoint_camera, strategy) in enumerate(zip(viewpoint_camera_list, strategy_list)):
            cuda_args = self.strategy_history.get_cuda_args(iteration, strategy, bsz=bsz)
            cuda_args_list.append(cuda_args)

            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
            tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.height),
                image_width=int(viewpoint_camera.width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_to_camera,
                projmatrix=viewpoint_camera.full_projection,
                sh_degree=pc.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                record_transmittance=record_transmittance,
                debug=False
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means2d, rgb, normal_opacity, radii, transMat, depths = rasterizer.preprocess_gaussians(
                means3D=means3D,
                means2D=means2D,
                scales=scales,
                rotations=rotations,
                shs=shs,
                opacities=opacity,
                cuda_args=cuda_args,
            )

            means2D_list.append(means2D)
            screenspace_params_list.append([
                means2d, 
                rgb, 
                normal_opacity, 
                radii, 
                transMat, 
                depths,
                global_idx
            ])
            rasterizer_list.append(rasterizer)
            radii_list.append(radii)

        # print("rank:", self.global_rank, [(radii > 0).sum().int() for radii in radii_list])

        if self.world_size > 1:
            screenspace_params_redistributed_list, _ = \
                self.rasterizer_required_data_all2all(rasterizer_list, screenspace_params_list, cuda_args_list, strategy_list)
        else:
            screenspace_params_redistributed_list = screenspace_params_list

        # print(self.global_rank, [self.global_rank in strategy_list[cam_id].gpu_ids for cam_id in range(len(viewpoint_camera_list))])

        rets_list = []
        for cam_id in range(len(viewpoint_camera_list)):
            radii = radii_list[cam_id]
            strategy = strategy_list[cam_id]
            screenspace_points = means2D_list[cam_id]
            if self.global_rank not in strategy.gpu_ids:
                rets = {
                    "render": None,
                    "coverage_y": None,
                    "viewspace_points": screenspace_points,
                    "visibility_filter": radii > 0,
                    "radii": radii,
                    'rend_alpha': None,
                    'rend_normal': None,
                    'view_normal': None,
                    'rend_dist': None,
                    'surf_depth': None,
                    'surf_normal': None,
                    'scales': None,
                    'pixels': None,
                    'acc_pix': None,
                }
                rets_list.append(rets)
                continue

            viewpoint_camera = viewpoint_camera_list[cam_id]

            compute_locally = strategy.get_compute_locally()
            cuda_args = cuda_args_list[cam_id]
            
            rank = strategy.gpu_ids.index(self.global_rank)
            tile_ids_l, tile_ids_r = (
                strategy.division_pos[rank],
                strategy.division_pos[rank + 1],
            )
            coverage_min_y, coverage_max_y = self.get_coverage_y_min_max(tile_ids_l, tile_ids_r)

            means2d_redist, rgb_redist, normal_opacity_redist, radii_redist, transMat_redist, depths_redist, global_idx_redist \
                = screenspace_params_redistributed_list[cam_id]

            rasterizer = rasterizer_list[cam_id]

            output = rasterizer.render_gaussians(
                means2D=means2d_redist,
                normal_opacity=normal_opacity_redist,
                rgb=rgb_redist,
                transMat=transMat_redist,
                depths=depths_redist,
                radii=radii_redist,
                compute_locally=compute_locally,
                cuda_args=cuda_args,
            )

            if record_transmittance:
                transmittance_sum, num_covered_pixels, radii_redist = output
                transmittance = transmittance_sum / (num_covered_pixels + 1e-6)
                return transmittance
            else:
                rendered_image, radii_redist, allmap, accum_count_redist = output

            appearance_image, mapping = self.model(rendered_image, viewpoint_camera.appearance_id)
            rendered_image = appearance_image

            rets = {
                "appearance": appearance_image,
                "render": rendered_image,
                "coverage_y": (coverage_min_y, coverage_max_y),
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
            }

            render_alpha = allmap[1:2]

            render_depth_median = allmap[5:6]
            render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

            render_depth_expected = allmap[0:1]
            render_depth_expected = (render_depth_expected / render_alpha)
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

            surf_depth = render_depth_expected * (1 - self.depth_ratio) + (self.depth_ratio) * render_depth_median
            
            if render_src:
                rets.update({
                    'rend_alpha': None,
                    'rend_normal': None,
                    'view_normal': None,
                    'rend_dist': None,
                    'surf_depth': surf_depth,
                    'surf_normal': None,
                    'scales': None,
                    'acc_pix': None,
                })
            else:
                gloabal_acc_pix[global_idx_redist] = accum_count_redist
                acc_pix = gloabal_acc_pix[global_idx]

                render_normal = allmap[2:5]
                render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_to_camera[:3, :3].T)).permute(2, 0, 1)

                render_dist = allmap[6:7]
                surf_normal = self.depth_to_normal(viewpoint_camera, surf_depth)
                surf_normal = surf_normal.permute(2, 0, 1)
                surf_normal = surf_normal * (render_alpha).detach()

                rets.update({
                    'rend_alpha': render_alpha,
                    'rend_normal': render_normal,
                    'view_normal': -allmap[2:5],
                    'rend_dist': render_dist,
                    'surf_depth': surf_depth,
                    'surf_normal': surf_normal,
                    'scales': scales,
                    'acc_pix': acc_pix,
                })
            rets_list.append(rets)

        return rets_list
    
    def forward_eval(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            record_transmittance=False,
            **kwargs,
    ):

        iteration = kwargs.get('iteration', -1)
        render_src = kwargs.get('render_src', False)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=bg_color.device) + 0
        
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        cov3D_precomp = None
        scales = pc.get_scaling[..., :2]
        rotations = pc.get_rotation

        shs = pc.get_features

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            record_transmittance=record_transmittance,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2d, rgb, normal_opacity, radii, transMat, depths = rasterizer.preprocess_gaussians(
            means3D=means3D,
            means2D=means2D,
            scales=scales,
            rotations=rotations,
            shs=shs,
            opacities=opacity,
        )

        output = rasterizer.render_gaussians(
            means2D=means2d,
            normal_opacity=normal_opacity,
            rgb=rgb,
            transMat=transMat,
            depths=depths,
            radii=radii,
            compute_locally=None,
        )

        if record_transmittance:
            transmittance_sum, num_covered_pixels, radii = output
            transmittance = transmittance_sum / (num_covered_pixels + 1e-6)
            return transmittance
        else:
            rendered_image, radii, allmap, accum_count = output

        # rendered_image, mapping = self.model(rendered_image, viewpoint_camera.appearance_id)

        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

        render_alpha = allmap[1:2]

        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        surf_depth = render_depth_expected * (1 - self.depth_ratio) + (self.depth_ratio) * render_depth_median
        
        if render_src:
            rets.update({
                'rend_alpha': None,
                'rend_normal': None,
                'view_normal': None,
                'rend_dist': None,
                'surf_depth': surf_depth,
                'surf_normal': None,
                'scales': None,
                'acc_pix': None,
            })
        else:
            render_normal = allmap[2:5]
            render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_to_camera[:3, :3].T)).permute(2, 0, 1)

            render_dist = allmap[6:7]
            surf_normal = self.depth_to_normal(viewpoint_camera, surf_depth)
            surf_normal = surf_normal.permute(2, 0, 1)
            surf_normal = surf_normal * (render_alpha).detach()

            rets.update({
                'rend_alpha': render_alpha,
                'rend_normal': render_normal,
                'view_normal': -allmap[2:5],
                'rend_dist': render_dist,
                'surf_depth': surf_depth,
                'surf_normal': surf_normal,
                'scales': scales,
            })

        return rets
    
    def rasterizer_required_data_all2all(
            self, 
            rasterizer_list, 
            screenspace_params_list, 
            cuda_args_list, 
            strategy_list
    ):
        num_cameras = len(rasterizer_list)
        local_to_gpuj_camk_size = [[] for j in range(self.world_size)]
        local_to_gpuj_camk_send_ids = [[] for j in range(self.world_size)]

        for k in range(num_cameras):
            strategy = strategy_list[k]
            means2d, rgb, normal_opacity, radii, transMat, depths, global_idx = screenspace_params_list[k]
            local2j_ids, local2j_ids_bool = strategy_list[k].get_local2j_ids(
                means2d, radii, rasterizer_list[k].raster_settings, cuda_args_list[k]
            )

            # local_to_gpuj_camk_send_ids 记录了与GPUi相交的高斯列表 Ngpu * [Ncam * [(Ngs, 1)]]
            for local_id, global_id in enumerate(strategy.gpu_ids):
                local_to_gpuj_camk_size[global_id].append(len(local2j_ids[local_id]))
                local_to_gpuj_camk_send_ids[global_id].append(local2j_ids[local_id])

            for j in range(self.world_size):
                if len(local_to_gpuj_camk_size[j]) == k:
                    local_to_gpuj_camk_size[j].append(0)
                    local_to_gpuj_camk_send_ids[j].append(
                        torch.empty((0, 1), dtype=torch.int64)
                    )

        gpui_to_gpuj_imgk_size = torch.zeros(
            (self.world_size, self.world_size, num_cameras),
            dtype=torch.int,
            device="cuda",
        )
        local_to_gpuj_camk_size_tensor = torch.tensor(
            local_to_gpuj_camk_size, dtype=torch.int, device="cuda"
        )
        
        # 每个进程发送local_to_gpuj_camk_size_tensor，在local_to_gpuj_camk_size_tensor中聚合，顺序根据GPU序号
        torch.distributed.all_gather_into_tensor(
            gpui_to_gpuj_imgk_size,
            local_to_gpuj_camk_size_tensor,
        )
        gpui_to_gpuj_imgk_size = gpui_to_gpuj_imgk_size.cpu().numpy().tolist()
        # print("local:", self.global_rank, gpui_to_gpuj_imgk_size)

        def one_all_to_all(batched_tensors, use_function_version=False):
            tensor_to_rki = [] # 加载到ranki
            tensor_from_rki = [] # 从ranki加载
            for i in range(self.world_size):
                tensor_to_rki_list = []
                tensor_from_rki_size = 0
                for k in range(num_cameras):
                    tensor_to_rki_list.append(
                        batched_tensors[k][local_to_gpuj_camk_send_ids[i][k]]
                    ) # 相机k分配给GPUi的数据列表
                    tensor_from_rki_size += gpui_to_gpuj_imgk_size[i][
                        self.global_rank
                    ][k]
                tensor_to_rki.append(torch.cat(tensor_to_rki_list, dim=0).contiguous() ) # 分配给GPUi的数据汇总
                tensor_from_rki.append(
                    torch.empty(
                        (tensor_from_rki_size,) + batched_tensors[0].shape[1:], 
                        dtype=batched_tensors[0].dtype, 
                        device="cuda",
                    )
                )
            
            if use_function_version:
                dist_func.all_to_all(
                    output_tensor_list=tensor_from_rki,
                    input_tensor_list=tensor_to_rki,
                )  # The function version could naturally enable communication during backward.
            else:
                torch.distributed.all_to_all(
                    output_tensor_list=tensor_from_rki,
                    input_tensor_list=tensor_to_rki,
                )

            # tensor_from_rki: (world_size, (all data received from all other GPUs))
            for i in range(self.world_size):
                # -> (world_size, num_cameras, *)
                tensor_from_rki[i] = tensor_from_rki[i].split(gpui_to_gpuj_imgk_size[i][self.global_rank], dim=0)

            tensors_per_camera = []
            for k in range(num_cameras):
                tensors_per_camera.append(torch.cat([tensor_from_rki[i][k] for i in range(self.world_size)], dim=0,).contiguous())

            return tensors_per_camera

        catted_screenspace_states_list = []
        catted_screenspace_auxiliary_states_list = []

        for k in range(num_cameras):
            means2d, rgb, normal_opacity, radii, transMat, depths, global_idx = screenspace_params_list[k]
            if k == 0:
                mean2d_dim1 = means2d.shape[1]
                rgb_dim1 = rgb.shape[1]
                normal_opacity_dim1 = normal_opacity.shape[1]
                transMat_dim1 = transMat.shape[1]
            catted_screenspace_states_list.append(
                torch.cat([means2d, rgb, normal_opacity, transMat], dim=1).contiguous()
            )
            catted_screenspace_auxiliary_states_list.append(
                torch.cat(
                    [radii.float().unsqueeze(1), depths.unsqueeze(1), global_idx.float().unsqueeze(1)], dim=1
                ).contiguous()
            )

        params_redistributed_list = one_all_to_all(catted_screenspace_states_list, use_function_version=True)
        radii_depth_redistributed_list = one_all_to_all(catted_screenspace_auxiliary_states_list, use_function_version=False)

        screenspace_params_redistributed_list = []
        for k in range(num_cameras):
            means2d, rgb, normal_opacity, transMat = torch.split(
                params_redistributed_list[k],
                [mean2d_dim1, rgb_dim1, normal_opacity_dim1, transMat_dim1],
                dim=-1,
            )
            radii, depths, global_idx = torch.split(
                radii_depth_redistributed_list[k], [1, 1, 1], dim=1
            )

            screenspace_params_redistributed_list.append([
                means2d, rgb, normal_opacity, radii.squeeze(1).int(), transMat, depths.squeeze(1), global_idx.squeeze(1).int()
            ])

        return screenspace_params_redistributed_list, gpui_to_gpuj_imgk_size

    def get_coverage_y_min_max(self, tile_ids_l, tile_ids_r):
        return tile_ids_l * self.block_size, min(tile_ids_r * self.block_size, self.img_height)
    def get_coverage_y_min(self, tile_ids_l):
        return tile_ids_l * self.block_size
    def get_coverage_y_max(self, tile_ids_r):
        return min(tile_ids_r * self.block_size, self.img_height)
    
    def after_training_step(self, step: int, module):
        super().after_training_step(step, module)

        bsz = module.batch_size
        if self.redistribute_interval < 0:
            return
        if step >= self.redistribute_until:
            return
        if all([i % self.redistribute_interval != 0 for i in range(step, step + bsz)]): 
            return
        if self.world_size > 1:
            self.redistribute(module)

    def redistribute(self, module):
        with torch.no_grad():
            # gather number of Gaussians
            member_n_gaussians = [0 for _ in range(self.world_size)]
            torch.distributed.all_gather_object(member_n_gaussians, module.gaussian_model.get_xyz.shape[0])
            if self.global_rank == 0:
                # print(f"[rank={self.global_rank}] member_n_gaussians={member_n_gaussians}")
                pass

            if min(member_n_gaussians) * self.redistribute_threshold >= max(member_n_gaussians):
                # print(f"[rank={self.global_rank}] skip redistribution: under threshold")
                return

            # print(f"[rank={self.global_rank}] begin redistribution")
            self.random_redistribute(module)

    def random_redistribute(self, module):
        destination = torch.randint(0, self.world_size, (module.gaussian_model.get_xyz.shape[0],), device=module.device)
        count_by_destination = list(torch.bincount(destination, minlength=self.world_size).chunk(self.world_size))

        # print(f"[rank={self.global_rank}] destination_count={[i.item() for i in count_by_destination]}")

        # number of gaussians to receive all-to-all
        number_of_gaussians_to_receive = list(torch.zeros((self.world_size,), dtype=count_by_destination[0].dtype, device=module.device).chunk(self.world_size))
        dist_func.all_to_all(number_of_gaussians_to_receive, count_by_destination)

        self.optimizer_all2all(destination, number_of_gaussians_to_receive, module.gaussian_model, module.gaussian_optimizers)

        new_number_of_gaussians = module.gaussian_model.get_xyz.shape[0]
        print(f"[rank={self.global_rank}] redistributed: n_gaussians={new_number_of_gaussians}")

        self.on_density_changed()

    def all2all_gaussian_state(self, local_tensor, destination, number_of_gaussians_to_receive):
        output_tensor_list = []
        input_tensor_list = []

        for i in range(self.world_size):
            output_tensor_list.append(torch.empty(
                [number_of_gaussians_to_receive[i]] + list(local_tensor.shape[1:]),
                dtype=local_tensor.dtype,
                device=local_tensor.device,
            ))
            input_tensor_list.append(local_tensor[destination == i])

        dist_func.all_to_all(output_tensor_list, input_tensor_list)

        return torch.concat(output_tensor_list, dim=0).contiguous()
    
    def optimizer_all2all(self, destination, number_of_gaussians_to_receive, gaussian_model, optimizers):
        def invoke_all2all(local_tensor):
            return self.all2all_gaussian_state(local_tensor, destination=destination, number_of_gaussians_to_receive=number_of_gaussians_to_receive)

        new_tensors = {}
        # optimizable
        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = invoke_all2all(stored_state["exp_avg"])
                    stored_state["exp_avg_sq"] = invoke_all2all(stored_state["exp_avg_sq"])

                    # replace with new tensor and state
                    del opt.state[group['params'][0]]
                    group["params"][0] = torch.nn.Parameter(invoke_all2all(group["params"][0]).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = torch.nn.Parameter(invoke_all2all(group["params"][0]).requires_grad_(True))

                new_tensors[group["name"]] = group["params"][0]

        # tensors
        for name in gaussian_model.get_property_names():
            if name in new_tensors:
                continue
            new_tensors[name] = invoke_all2all(gaussian_model.get_property(name))

        # update
        gaussian_model.properties = new_tensors

class DivisionStrategy:
    def __init__(
        self, camera, tile_height, tile_width, global_rank, world_size, gpu_ids, 
        division_pos, gpu_for_this_camera_tilelr, 
    ):
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.global_rank = global_rank
        self.camera = camera
        self.world_size = world_size
        self.gpu_ids = gpu_ids
        assert world_size > 0, "The world_size must be greater than 0."
        assert (
            len(gpu_ids) == world_size
        ), "The number of gpu_ids must be equal to the world_size."
        assert (
            len(division_pos) == world_size + 1
        ), "The number of division_pos must be equal to the world_size+1."
        assert division_pos[0] == 0, "The first element of division_pos must be 0."
        assert (
            division_pos[-1] == self.tile_height
        ), "The last element of division_pos must be equal to the total number of tiles."
        for i in range(1, len(division_pos)):
            assert (
                division_pos[i] > division_pos[i - 1]
            ), "The division_pos must be in ascending order."

        for idx in range(len(gpu_for_this_camera_tilelr)):
            assert (
                gpu_for_this_camera_tilelr[idx][0] == division_pos[idx]
                and gpu_for_this_camera_tilelr[idx][1] == division_pos[idx + 1]
            ), "The division_pos must be consistent with gpu_for_this_camera_tilelr."

        if self.global_rank in gpu_ids:
            self.rank = gpu_ids.index(self.global_rank)
        else:
            self.rank = -1

        self.division_pos = division_pos

    def get_local2j_ids(self, means2D, radii, raster_settings, avoid_pixel_all2all=False):
        dist_global_strategy_tensor = (
            torch.tensor(self.division_pos, dtype=torch.int, device=means2D.device)
            * self.tile_width
        )

        args = (
            raster_settings.image_height,
            raster_settings.image_width,
            self.rank,
            self.world_size,
            means2D,
            radii,
            dist_global_strategy_tensor,
            avoid_pixel_all2all,
        )

        local2j_ids_bool = get_local2j_ids_bool(*args)

        local2j_ids = []
        for rk in range(self.world_size):
            local2j_ids.append(local2j_ids_bool[:, rk].nonzero())

        return local2j_ids, local2j_ids_bool

    def get_compute_locally(self):
        if self.global_rank not in self.gpu_ids:
            return None
        rank = self.gpu_ids.index(self.global_rank)

        tile_ids_l, tile_ids_r = (
            self.division_pos[rank] * self.tile_width,
            self.division_pos[rank + 1] * self.tile_width,
        )
        compute_locally = torch.zeros(
            self.tile_height * self.tile_width, dtype=torch.bool, device="cuda"
        )
        compute_locally[tile_ids_l:tile_ids_r] = True
        compute_locally = compute_locally.view(self.tile_height, self.tile_width)
        return compute_locally

    def get_compute_locally_all(self):
        if self.global_rank not in self.gpu_ids:
            return None
        rank = self.gpu_ids.index(self.global_rank)

        # tile_ids_l, tile_ids_r = self.division_pos[rank]*utils.TILE_X, self.division_pos[rank+1]*utils.TILE_X
        compute_locally = torch.zeros(
            self.tile_height, self.tile_width, dtype=torch.bool, device="cuda"
        )
        compute_locally[:] = True
        compute_locally = compute_locally.view(self.tile_height, self.tile_width)
        return compute_locally
    
class DivisionStrategyHistory:
    def __init__(self, cameras, world_size, rank, tile_height):
        self.world_size = world_size
        self.rank = rank
        self.tile_height = tile_height
        self.accum_heuristic = {}
        for camera in cameras:
            self.accum_heuristic[camera.idx.cpu().item()] = torch.ones(
                (tile_height,), dtype=torch.float32, device="cuda", requires_grad=False
            )

        self.history = []

        self.heuristic_decay = 0.0
        self.no_heuristics_update = False
        self.adjust_strategy_warmp_iterations = -1
        self.save_strategy_history = False

        self.log_interval = 250
        self.zhx_debug = False
        self.zhx_time = False
        self.log_folder = "./tmp/gaussian-splatting"
    
    def division_pos_heuristic(self, heuristic, tile_num, world_size, right=False):
        assert (
            heuristic.shape[0] == tile_num
        ), "the length of heuristics should be the same as the number of tiles."
        heuristic_prefix_sum = torch.cumsum(heuristic, dim=0)
        heuristic_sum = heuristic_prefix_sum[-1]
        heuristic_per_worker = heuristic_sum / world_size
        thresholds = torch.arange(1, world_size, device="cuda") * heuristic_per_worker
        division_pos = [0]

        # Use searchsorted to find the positions
        division_indices = torch.searchsorted(heuristic_prefix_sum, thresholds, right=right)

        # check_division_indices_globally_same(division_indices)

        # Convert to a Python list and prepend the initial division at 0.
        division_pos = [0] + division_indices.cpu().tolist() + [tile_num]

        return division_pos
    
    def start_strategy(self, camera_list, tile_height, tile_width, global_rank):
        n_tiles_per_image = self.tile_height
        border_divpos_coeff = 1.0
        total_tiles = n_tiles_per_image * len(camera_list)

        accum_heuristic_list = [
            self.accum_heuristic[camera.idx.cpu().item()] for camera in camera_list
        ]  # batch_size * tile_y
        catted_accum_heuristic = torch.cat(
            accum_heuristic_list, dim=0
        )  # batch_size * tile_y

        division_pos = self.division_pos_heuristic(
            catted_accum_heuristic, total_tiles, self.world_size, right=True
        )
        # slightly adjust the division_pos to avoid redundant kernel launch overheads.
        for i in range(1, len(division_pos) - 1):
            if (
                division_pos[i] % n_tiles_per_image + border_divpos_coeff
                >= n_tiles_per_image
            ):
                division_pos[i] = (
                    division_pos[i] // n_tiles_per_image * n_tiles_per_image
                    + n_tiles_per_image
                )
            elif division_pos[i] % n_tiles_per_image - border_divpos_coeff <= 0:
                division_pos[i] = (
                    division_pos[i] // n_tiles_per_image * n_tiles_per_image
                )
        for i in range(0, len(division_pos) - 1):
            assert (
                division_pos[i] + border_divpos_coeff < division_pos[i + 1]
            ), f"Each part between division_pos must be large enough."

        strategy_list = []
        gpuid2tasks = [
            [] for _ in range(self.world_size)
        ]  # map from gpuid to a list of tasks (camera_id, tile_l, tile_r) it should do.
        for idx, camera in enumerate(camera_list):
            offset = idx * n_tiles_per_image

            gpu_for_this_camera = []
            gpu_for_this_camera_tilelr = []
            for gpu_id in range(self.world_size):
                gpu_tile_l, gpu_tile_r = division_pos[gpu_id], division_pos[gpu_id + 1]
                if gpu_tile_r <= offset or offset + n_tiles_per_image <= gpu_tile_l:
                    continue
                gpu_for_this_camera.append(gpu_id)
                local_tile_l, local_tile_r = (
                    max(gpu_tile_l, offset) - offset,
                    min(gpu_tile_r, offset + n_tiles_per_image) - offset,
                )
                gpu_for_this_camera_tilelr.append((local_tile_l, local_tile_r))
                gpuid2tasks[gpu_id].append((idx, local_tile_l, local_tile_r))

            ws_for_this_camera = len(gpu_for_this_camera)
            division_pos_for_this_viewpoint = [0] + [
                tilelr[1] for tilelr in gpu_for_this_camera_tilelr
            ]
            strategy = DivisionStrategy(
                camera,
                tile_height, tile_width, global_rank,
                ws_for_this_camera,
                gpu_for_this_camera,
                division_pos_for_this_viewpoint,
                gpu_for_this_camera_tilelr,
            )
            strategy_list.append(strategy)
        
        return strategy_list, gpuid2tasks
    
    def finish_strategy(self, iteration, camera_list, strategiey_list, statistic_collector_list, bsz=1):
        running_time_list = []
        for idx, strategy in enumerate(strategiey_list):
            if self.rank not in strategy.gpu_ids:
                running_time_list.append(-1.0)
                continue

            running_time_list.append(
                statistic_collector_list[idx]["forward_render_time"]
                + statistic_collector_list[idx]["backward_render_time"]
                + statistic_collector_list[idx]["forward_loss_time"] * 2
            )

        gpu_camera_running_time = self.our_allgather_among_cpu_processes_float_list(running_time_list)
        self.store_stats(
            camera_list, gpu_camera_running_time, strategiey_list
        )

        img_height, img_width = camera_list[0].height.item(), camera_list[0].width.item()

        if (
            iteration <= self.adjust_strategy_warmp_iterations or self.world_size == 1 or self.no_heuristics_update
            or (bsz >= self.world_size and (img_height <= 1080 or img_width <= 1920))
            or (img_height <= 600 or img_width <= 1000)
        ):
            return

        for camera_id, (camera, strategy) in enumerate(
            zip(camera_list, strategiey_list)
        ):
            new_heuristic = torch.zeros((self.tile_height,), dtype=torch.float32, device="cuda")
            for local_id, gpu_id in enumerate(strategy.gpu_ids):
                tile_ids_l, tile_ids_r = strategy.division_pos[local_id], strategy.division_pos[local_id + 1]
                new_heuristic[tile_ids_l:tile_ids_r] = gpu_camera_running_time[gpu_id][camera_id] / (tile_ids_r - tile_ids_l)
            if self.heuristic_decay == 0:
                self.accum_heuristic[camera.idx.cpu().item()] = new_heuristic
            else:
                self.accum_heuristic[camera.idx.cpu().item()] = (
                    self.accum_heuristic[camera.idx.cpu().item()] * self.heuristic_decay + new_heuristic * (1 - self.heuristic_decay)
                )

    def get_cuda_args(self, iteration, strategy, mode="train", bsz=1):

        if mode == "train":
            for x in range(bsz):
                if (iteration + x) % self.log_interval == 1:
                    iteration += x
                    break
        elif mode == "test":
            iteration = -1
        else:
            raise ValueError("mode should be train or test.")

        cuda_args = {
            "mode": mode,
            "world_size": str(self.world_size),
            "global_rank": str(self.rank),
            "mp_world_size": str(strategy.world_size),
            "mp_rank": str(strategy.rank),
            "log_folder": self.log_folder,
            "log_interval": str(self.log_interval),
            "iteration": str(iteration),
            "zhx_debug": str(self.zhx_debug),
            "zhx_time": str(self.zhx_time),
            "avoid_pixel_all2all": False,
            "stats_collector": {},
        }
        return cuda_args

    def our_allgather_among_cpu_processes_float_list(self, data):
        assert isinstance(data, list) and isinstance(data[0], float), "data should be a list of float"
        data_gpu = torch.tensor(data, dtype=torch.float32, device="cuda")
        all_data_gpu = torch.empty((self.world_size, len(data_gpu)), dtype=torch.float32, device="cuda")
        if self.world_size > 1:
            torch.distributed.all_gather_into_tensor(all_data_gpu, data_gpu)
        else:
            all_data_gpu = data_gpu.unsqueeze(0)

        all_data = all_data_gpu.cpu().tolist()
        return all_data

    def store_stats(self, step, camera_list, gpu_camera_running_time, strategy_list):
        camera_info_list = []
        all_camera_running_time = [0 for _ in range(len(camera_list))]
        all_gpu_running_time = [0 for _ in range(self.world_size)]
        for camera_id, camera in enumerate(camera_list):
            each_gpu_running_time = []
            for gpu_i in strategy_list[camera_id].gpu_ids:
                all_camera_running_time[camera_id] += gpu_camera_running_time[gpu_i][camera_id]
                all_gpu_running_time[gpu_i] += gpu_camera_running_time[gpu_i][camera_id]
                each_gpu_running_time.append(gpu_camera_running_time[gpu_i][camera_id])

            camera_info_list.append({
                    "camera_id": camera.idx.cpu().item(),
                    "gpu_ids": strategy_list[camera_id].gpu_ids,
                    "division_pos": strategy_list[camera_id].division_pos,
                    "each_gpu_running_time": each_gpu_running_time,
                })
        self.history.append({
                "step": step,
                "all_gpu_running_time": all_gpu_running_time,
                "all_camera_running_time": all_camera_running_time,
                "camera_info_list": camera_info_list,
            })

    def to_json(self):
        return self.history
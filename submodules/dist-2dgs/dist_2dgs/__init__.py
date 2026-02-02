#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
import time

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


########################### Preprocess ###########################



def preprocess_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    cuda_args,
):
    return _PreprocessGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        cuda_args,
    )

class _PreprocessGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        cuda_args,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
             means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            cuda_args
        )

        # TODO: update this. 
        num_rendered, means2d, depths, radii, transMat, normal_opacity, rgb, clamped = _C.preprocess_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.cuda_args = cuda_args
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, sh, means2d, depths, radii, transMat, normal_opacity, rgb, clamped)
        ctx.mark_non_differentiable(radii, depths)

        # # TODO: double check. means2D is padded to (P, 3) in python. It is (P, 2) in cuda code.
        # means2D_pad = torch.zeros((means2D.shape[0], 1), dtype = means2D.dtype, device = means2D.device)
        # means2D = torch.cat((means2D, means2D_pad), dim = 1).contiguous()

        return means2d, rgb, normal_opacity, radii, transMat, depths

    @staticmethod # TODO: gradient for conic_opacity is tricky. because cuda render backward generate dL_dconic and dL_dopacity sperately. 
    def backward(ctx, grad_means2D, grad_rgb, grad_normal_opacity, grad_radii, grad_transMat, grad_depths):
        # grad_radii, grad_depths should be all None. 

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        cuda_args = ctx.cuda_args
        means3D, scales, rotations, sh, means2D, depths, radii, transMat, normal_opacity, rgb, clamped = ctx.saved_tensors

        # change dL_dmeans2D from (P, 2) to (P, 3)
        # grad_means2D is (P, 2) now. Need to pad it to (P, 3) because preprocess_gaussians_backward's cuda implementation.
        grad_means2D_pad = torch.zeros((grad_means2D.shape[0], 1), dtype = grad_means2D.dtype, device = grad_means2D.device)
        grad_means2D = torch.cat((grad_means2D, grad_means2D_pad), dim = 1).contiguous()

        # Restructure args as C++ method expects them
        args = (radii,
                transMat,
                clamped,#the above are all per-Gaussian intemediate results.
                means3D,
                scales,
                rotations, 
                sh, #input of this operator
                raster_settings.scale_modifier, 
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.sh_degree,
                raster_settings.campos,#rasterization setting.
                grad_transMat,
                grad_means2D,
                grad_normal_opacity,
                grad_rgb,#gradients of output of this operator
                raster_settings.debug,
                cuda_args)

        dL_dmeans3D, dL_dmeans2D, dL_dscales, dL_drotations, dL_dsh, dL_dopacity = _C.preprocess_gaussians_backward(*args)

        grads = (
            dL_dmeans3D.contiguous(),
            dL_dmeans2D.contiguous(),
            dL_dsh.contiguous(),
            grad_rgb.contiguous(),#colors_precomp
            dL_dopacity.contiguous(),
            dL_dscales.contiguous(),
            dL_drotations.contiguous(),
            grad_transMat.contiguous(),#cov3Ds_precomp
            None,#raster_settings
            None,#cuda_args
        )

        return grads




########################### Render ###########################



def render_gaussians(
    means2D,
    normal_opacity,
    rgb,
    transMat,
    depths,
    radii,
    compute_locally,
    raster_settings,
    cuda_args,
):
    return _RenderGaussians.apply(
        means2D,
        normal_opacity,
        rgb,
        transMat,
        depths,
        radii,
        compute_locally,
        raster_settings,
        cuda_args,
    )

class _RenderGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2D,
        normal_opacity,
        rgb,
        transMat,
        depths,
        radii,
        compute_locally,
        raster_settings,
        cuda_args,
    ):
        # means2D = means2D[:,:2].contiguous()
        # NOTE: double check.
        # means2D is padded to (P, 3) before being output from preprocess_gaussians.
        # because _RenderGaussians.backward will give dL_dmeans2D with shape (P, 3).
        # Here, because the means2D in cuda code is (P, 2), we need to remove the padding.
        # Basically, means2D is (P, 3) in python. But it is (P, 2) in cuda code.
        # dL_dmeans2D is alwayds (P, 3) in both python and cuda code.

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,# image setting
            means2D,
            depths,
            radii,
            normal_opacity,
            rgb,# 3dgs intermediate results
            transMat,
            compute_locally,
            raster_settings.record_transmittance,
            raster_settings.debug,
            cuda_args
        )

        # This time is to measure: render forward+loss forward+loss backward+render backward . And then used to do load balancing.
        # Used when cuda_args["avoid_pixel_all2all"] is True.
        # It is not useful for now.
        # torch.cuda.synchronize()
        # render_forward_start_time = time.time()

        num_rendered, color, others, geomBuffer, binningBuffer, imgBuffer, accum_count, transmittance, num_covered_pixels = _C.render_gaussians(*args)

        if raster_settings.record_transmittance:
            return transmittance, num_covered_pixels, radii

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.cuda_args = cuda_args
        ctx.num_rendered = num_rendered
        # ctx.render_forward_start_time = render_forward_start_time
        ctx.save_for_backward(means2D, transMat, normal_opacity, depths, rgb, geomBuffer, binningBuffer, imgBuffer, compute_locally)
        ctx.mark_non_differentiable(radii, accum_count)

        return color, radii, others, accum_count

    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_others, grad_accum_count):
        # grad_n_render, grad_n_consider, grad_n_contrib should be all None. 

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        cuda_args = ctx.cuda_args
        # render_forward_start_time = ctx.render_forward_start_time
        means2D, transMat, normal_opacity, depths, rgb, geomBuffer, binningBuffer, imgBuffer, compute_locally = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                num_rendered,
                transMat,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                compute_locally,# buffer
                grad_color,# gradient of output of this operator
                grad_others,
                means2D,
                normal_opacity,
                depths,
                rgb,# 3dgs intermediate results
                raster_settings.debug,
                cuda_args)
        

        dL_dmeans2D, dL_dnormal_opacity, dL_dcolors, dL_dtransMat = _C.render_gaussians_backward(*args)

        # Used when cuda_args["avoid_pixel_all2all"] is True.
        # torch.cuda.synchronize()
        # render_backward_end_time = time.time()
        # cuda_args["stats_collector"]["pixelwise_workloads_time"] = (render_backward_end_time - render_forward_start_time)*1000

        # change dL_dmeans2D from (P, 3) to (P, 2)
        # dL_dmeans2D is now (P, 3) because of render backwards' cuda implementation.
        dL_dmeans2D = dL_dmeans2D[:,:2]

        grads = (
            dL_dmeans2D.contiguous(),
            dL_dnormal_opacity.contiguous(),
            dL_dcolors.contiguous(),
            dL_dtransMat.contiguous(),
            None, #depths
            None, #radii
            None, #compute_locally
            None, #raster_settings
            None #cuda_args
        )

        # print("dL_dmeans2D:", dL_dmeans2D)
        # print("dL_dnormal_opacity:", dL_dnormal_opacity)
        # print("dL_dcolors:", dL_dcolors)
        # print("dL_dtransMat:", dL_dtransMat)

        return grads





########################### Settings ###########################


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    record_transmittance: bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def preprocess_gaussians(self, means3D, means2D, scales, rotations, shs, opacities, colors_precomp=None, cov3D_precomp=None, cuda_args=None):
        
        raster_settings = self.raster_settings

        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()
        
        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()
        if cuda_args is None:
            cuda_args = dict()

        # Invoke C++/CUDA rasterization routine
        return preprocess_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            cuda_args)
    
    def visible_filter(self, means3D, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.rasterize_gaussians_filter(means3D,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.prefiltered,
            raster_settings.debug)
        return radii

    def render_gaussians(self, means2D, normal_opacity, rgb, transMat, depths, radii, compute_locally, cuda_args = None):

        raster_settings = self.raster_settings

        if compute_locally is None:
            compute_locally = torch.empty(0, dtype=torch.bool).cuda()
        if cuda_args is None:
            cuda_args = dict()

        # Invoke C++/CUDA rasterization routine
        return render_gaussians(
            means2D,
            normal_opacity,
            rgb,
            transMat,
            depths,
            radii,
            compute_locally,
            raster_settings,
            cuda_args
        )

class _LoadImageTilesByPos(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        local_image_rect,
        all_tiles_pos,
        image_height, image_width,
        touched_pixels_rect,
        touched_tiles_rect
    ):
        ctx.save_for_backward(local_image_rect, all_tiles_pos)
        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.touched_pixels_rect = touched_pixels_rect
        ctx.touched_tiles_rect = touched_tiles_rect

        min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x = touched_pixels_rect

        return _C.load_image_tiles_by_pos(local_image_rect,
                                          all_tiles_pos,
                                          image_height,
                                          image_width,
                                          min_pixel_y,
                                          min_pixel_x,
                                          max_pixel_y-min_pixel_y,
                                          max_pixel_x-min_pixel_x)
        # return shape: (N, 3, BLOCK_Y, BLOCK_X)

    @staticmethod
    def backward(ctx, grad_image_tiles):
        # grad_image_tiles: (N, 3, BLOCK_Y, BLOCK_X)

        local_image_rect, all_tiles_pos = ctx.saved_tensors
        image_height = ctx.image_height
        image_width = ctx.image_width
        touched_pixels_rect = ctx.touched_pixels_rect

        min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x = touched_pixels_rect

        grad_local_image_rect = _C.set_image_tiles_by_pos(all_tiles_pos,
                                                          grad_image_tiles,
                                                          image_height,
                                                          image_width,
                                                          min_pixel_y,
                                                          min_pixel_x,
                                                          max_pixel_y-min_pixel_y,
                                                          max_pixel_x-min_pixel_x)

        # return tensor in which the grad_image_tiles are set to the right position, and the rest are zeros.
        return grad_local_image_rect, None, None, None, None, None

class _MergeImageTilesByPos(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        all_tiles_pos,
        image_tiles,
        image_height, image_width,
        touched_pixels_rect,
        touched_tiles_rect
    ):
        ctx.save_for_backward(all_tiles_pos, image_tiles)
        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.touched_pixels_rect = touched_pixels_rect

        min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x = touched_pixels_rect

        merged_local_image_rect = _C.set_image_tiles_by_pos(all_tiles_pos,
                                                            image_tiles,
                                                            image_height,
                                                            image_width,
                                                            min_pixel_y,
                                                            min_pixel_x,
                                                            max_pixel_y-min_pixel_y,
                                                            max_pixel_x-min_pixel_x)
        return merged_local_image_rect # (3, H, W)

    @staticmethod
    def backward(ctx, grad_merged_local_image_rect):
        # grad_image_tiles: (N, 3, BLOCK_Y, BLOCK_X)

        all_tiles_pos, image_tiles = ctx.saved_tensors
        image_height = ctx.image_height
        image_width = ctx.image_width
        touched_pixels_rect = ctx.touched_pixels_rect

        min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x = touched_pixels_rect

        grad_image_tiles = _C.load_image_tiles_by_pos(grad_merged_local_image_rect,
                                                      all_tiles_pos,
                                                      image_height,
                                                      image_width,
                                                      min_pixel_y,
                                                      min_pixel_x,
                                                      max_pixel_y-min_pixel_y,
                                                      max_pixel_x-min_pixel_x)

        return None, grad_image_tiles, None, None, None, None

def load_image_tiles_by_pos(
    local_image_rect,# in local coordinate
    all_tiles_pos,# in global coordinate
    image_height, image_width,
    touched_pixels_rect,
    touched_tiles_rect
):
    return _LoadImageTilesByPos.apply(
        local_image_rect,
        all_tiles_pos,
        image_height, image_width,
        touched_pixels_rect,
        touched_tiles_rect
    )

def merge_image_tiles_by_pos(
    all_tiles_pos,# in global coordinate
    image_tiles,
    image_height, image_width,
    touched_pixels_rect,
    touched_tiles_rect
):
    return _MergeImageTilesByPos.apply(
        all_tiles_pos,
        image_tiles,
        image_height, image_width,
        touched_pixels_rect,
        touched_tiles_rect
    )# return image should be in local coordinate.
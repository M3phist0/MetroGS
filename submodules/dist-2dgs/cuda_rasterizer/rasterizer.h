/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <torch/extension.h>

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// Either of the above two headers could make the compilation successful; 
// however, <pybind11/pybind11.h> will make the compilation very slow.

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static void visible_filter(
			const int P, int M,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			int* radii,
			float* cov3D,
			bool debug);

		/////////////////////////////// Preprocess ///////////////////////////////

		// Forward rendering procedure for differentiable rasterization
		// of Gaussians.
		static int preprocessForward(
			const int P, int D, int M,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			int* radii,
			float2* means2D,
			float* depths,
			float* transMat,
			float* rgb,
			float4* normal_opacity,
			bool* clamped,
			bool debug,
			const pybind11::dict &args);

		static void preprocessBackward(
			const int* radii,
			const float* transMats,
			const bool* clamped,//the above are all per-Gaussian intemediate results.
			const int P, int D, int M,
			const int width, int height,//rasterization setting.
			const float* means3D,
			const float* scales,
			const float* rotations,
			const float* shs,//input of this operator
			const float scale_modifier,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,//rasterization setting.
			float* dL_dmean2D,
			float* dL_dnormal3D,
			float* dL_dcolor,//gradients of output of this operator
			float* dL_dmean3D,
			float* dL_dtransMats,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dsh,//gradients of input of this operator
			bool debug,
			const pybind11::dict &args);
		

		////////////////////// GetDistributionStrategy ////////////////////////

		static void getDistributionStrategy(
			std::function<char* (size_t)> distBuffer,
			const int P,
			const int width, int height,
			float2* means2D,
			int* radii,
			bool* compute_locally,
			bool debug,
			const pybind11::dict &args);



		/////////////////////////////// Render ///////////////////////////////

		static int renderForward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const float* background,
			const int width, int height,
			const float tan_fovx, float tan_fovy,
			float2* means2D,//TODO: do I have to add const for it? However, internal means2D is not const type. 
			float* depths,
			int* radii,
			float4* normal_opacity,
			float* rgb,
			float* transMat,
			bool* compute_locally,
			float* out_color,
			float* out_others,
			float* accum_max_count,
			float* transmittance,
			int* num_covered_pixels,
			bool record_transmittance,
			bool debug,
			const pybind11::dict &args);

		static void renderBackward(
			const int P, int R,
			const float* background,
			const int width, int height,//rasterization settings.
			const float* transMats,
			const float2* means2D,
			const float4* normal_opacity,
			const float* depths,
			const float tan_fovx, float tan_fovy,
			const float* rgb,//inputs
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,//buffer that contains intermedia results
			bool* compute_locally,
			const float* dL_dpix,//gradient of output
			const float* dL_depths,
			float* dL_dmean2D,
			float* dL_dnormal,
			float* dL_dopacity,
			float* dL_dcolor,//gradient of inputs
			float* dL_dtransMat,
			bool debug,
			const pybind11::dict &args);


	};
};

#endif
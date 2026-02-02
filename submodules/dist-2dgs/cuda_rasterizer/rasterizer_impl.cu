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

#include "rasterizer_impl.h"
#include "timers.cu"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	bool* compute_locally,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		if (compute_locally == nullptr) {
			for (int y = rect_min.y; y < rect_max.y; y++)
			{
				for (int x = rect_min.x; x < rect_max.x; x++)
				{
					uint64_t key = y * grid.x + x;
					key <<= 32;
					key |= *((uint32_t*)&depths[idx]);
					gaussian_keys_unsorted[off] = key;
					gaussian_values_unsorted[off] = idx;
					off++;
				}
			}
		} else {
			for (int y = rect_min.y; y < rect_max.y; y++)
			{
				for (int x = rect_min.x; x < rect_max.x; x++)
				{ 
					if (compute_locally[y * grid.x + x])
					{
						uint64_t key = y * grid.x + x;
						key <<= 32;
						key |= *((uint32_t*)&depths[idx]);
						gaussian_keys_unsorted[off] = key;
						gaussian_values_unsorted[off] = idx;
						off++;
					}
				}
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P, bool sep_rendering=false)
{
	GeometryState geom;
	if (!sep_rendering)
	{
		obtain(chunk, geom.depths, P, 128);
		obtain(chunk, geom.clamped, P * 3, 128);
		obtain(chunk, geom.internal_radii, P, 128);
		obtain(chunk, geom.means2D, P, 128);
		obtain(chunk, geom.transMat, P * 6, 128);
		obtain(chunk, geom.normal_opacity, P, 128);
		obtain(chunk, geom.rgb, P * 3, 128);
	}
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	// no work is done and the required allocation size is returned in geom.scan_size.
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 3, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.n_contrib2loss, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::DistributedState CudaRasterizer::DistributedState::fromChunk(char*& chunk, size_t tile_num, bool sep_rendering=false)
{
	DistributedState dist;
	obtain(chunk, dist.gs_on_tiles, tile_num, 128);
	obtain(chunk, dist.gs_on_tiles_offsets, tile_num, 128);
	cub::DeviceScan::InclusiveSum(nullptr, dist.scan_size, dist.gs_on_tiles, dist.gs_on_tiles_offsets, tile_num);
	obtain(chunk, dist.scanning_space, dist.scan_size, 128);
	if (!sep_rendering)
		obtain(chunk, dist.compute_locally, tile_num, 128);
	return dist;
}

__global__ void get_n_render(
	const int tile_num,
	const uint2* ranges,
	int* n_render
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= tile_num)
		return;
	n_render[idx] = ranges[idx].y - ranges[idx].x;
}

__global__ void reduce_data_per_block(
	const int width,
	const int height,
	const uint32_t* n_data_per_pixel,
	int* n_data_per_block,
	bool* compute_locally
) {
	auto block = cg::this_thread_block();
	if (!compute_locally[block.group_index().y * gridDim.x + block.group_index().x])
		return;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	int pix_id = width * pix.y + pix.x;

	int blocksz = block.size(), beta;
	__shared__ int reduction_s[BLOCK_X * BLOCK_Y];
	int tid = block.thread_rank();

	bool inside = pix.x < width && pix.y < height;
	int data = inside ? (int)n_data_per_pixel[pix_id] : 0;// TODO: make sure the explicit cast is correct, i.e. n_data_per_pixel < 2^31-1;

	cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
	reduction_s[tid] = cg::reduce(tile, data, cg::plus<int>());
    cg::sync(block);

    if (tid == 0) {
        beta = 0;
        for (int i = 0; i < blocksz; i += tile.num_threads()) {
            beta += reduction_s[i];
        }
        n_data_per_block[block.group_index().y * gridDim.x + block.group_index().x] = beta;
    }
}

__global__ void updateTileTouched(
	const int P,
	const dim3 tile_grid,
	int* radii,
	float2* means2D,
	uint32_t* tiles_touched,
	bool* compute_locally
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	uint32_t cnt = 0;
	if (radii[idx] > 0)
	{
		uint2 rect_min, rect_max;
		getRect(means2D[idx], radii[idx], rect_min, rect_max, tile_grid);
		if (compute_locally == nullptr) {
			cnt = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
		} else {
			for (int y = rect_min.y; y < rect_max.y; y++)
				for (int x = rect_min.x; x < rect_max.x; x++)
					if (compute_locally[y * tile_grid.x + x])
						cnt++;
		}
	}

	tiles_touched[idx] = cnt;
}

void save_log_in_file(int iteration, int global_rank, int world_size, std::string log_folder, const char* filename_prefix, const char* log_content) {
	char* filename = new char[256];
	sprintf(filename, "%s/%s_ws=%d_rk=%d.log", log_folder.c_str(), filename_prefix, world_size, global_rank);
	std::ofstream outfile;
	outfile.open(filename, std::ios_base::app);
	outfile << "iteration: " << iteration << ", " << log_content << "\n";
	outfile.close();
	delete[] filename;
}


/////////////////////////////// Preprocess ///////////////////////////////


// Define the function to process the dict and return a tuple with all variables
std::tuple<int, int, int, int, int, bool, bool, std::string, std::string, std::string>
 prepareArgs(const pybind11::dict &args) {
    std::string mode = args["mode"].cast<std::string>();
    std::string global_rank_str = args["global_rank"].cast<std::string>();
    std::string world_size_str = args["world_size"].cast<std::string>();
    std::string iteration_str = args["iteration"].cast<std::string>();
    std::string log_interval_str = args["log_interval"].cast<std::string>();
    std::string log_folder_str = args["log_folder"].cast<std::string>();
    std::string zhx_debug_str = args["zhx_debug"].cast<std::string>();
    std::string zhx_time_str = args["zhx_time"].cast<std::string>();
    // std::string dist_division_mode_str = args["dist_division_mode"].cast<std::string>();
	std::string dist_division_mode_str = "";

    int global_rank = std::stoi(global_rank_str);
    int world_size = std::stoi(world_size_str);
    int iteration = std::stoi(iteration_str);
    int log_interval = std::stoi(log_interval_str);
    bool zhx_debug = zhx_debug_str == "True";
    bool zhx_time = zhx_time_str == "True";

	int device;
	cudaError_t status = cudaGetDevice(&device);

	// Pack and return the variables in a tuple
    return std::make_tuple(global_rank, world_size, iteration, log_interval, device,
			zhx_debug, zhx_time,
			mode, dist_division_mode_str, log_folder_str);
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::preprocessForward(
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
	const pybind11::dict &args)
{
	// auto [global_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);
	// char* log_tmp = new char[500];

	// // print out the environment variables
	// if (mode == "train" && zhx_debug && iteration % log_interval == 1) {
	// 	sprintf(log_tmp, "world_size: %d, global_rank: %d, iteration: %d, log_folder: %s, zhx_debug: %d, zhx_time: %d, device: %d, log_interval: %d, dist_division_mode: %s", 
	// 			world_size, global_rank, iteration, log_folder.c_str(), zhx_debug, zhx_time, device, log_interval, dist_division_mode.c_str());
	// 	save_log_in_file(iteration, global_rank, world_size, log_folder, "cuda", log_tmp);
	// }

	// MyTimerOnGPU timer;
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	int tile_num = tile_grid.x * tile_grid.y;

	// allocate temporary buffer for tiles_touched.
	// In sep_rendering==True case, we will compute tiles_touched in the renderForward. 
	// TODO: remove it later by modifying FORWARD::preprocess when we deprecate sep_rendering==False case
	uint32_t* tiles_touched = nullptr;
	// CHECK_CUDA(cudaMalloc(&tiles_touched_temp_buffer, P * sizeof(uint32_t)), debug);
	// CHECK_CUDA(cudaMemset(tiles_touched_temp_buffer, 0, P * sizeof(uint32_t)), debug);

	// timer.start("10 preprocess");
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec2*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		means2D,
		depths,
		transMat,
		rgb,
		normal_opacity,
		tile_grid,
		tiles_touched,
		prefiltered
	), debug)
	// timer.stop("10 preprocess");

	int num_rendered = 0;//TODO: should I calculate this here?

	// Print out timing information
	// if (zhx_time && iteration % log_interval == 1) {
	// 	timer.printAllTimes(iteration, world_size, global_rank, log_folder, true);
	// }
	// delete log_tmp;
	// free temporary buffer for tiles_touched. TODO: remove it. 
	// CHECK_CUDA(cudaFree(tiles_touched_temp_buffer), debug);
	return num_rendered;
}

void CudaRasterizer::Rasterizer::preprocessBackward(
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
	float* dL_dnormal,
	float* dL_dcolor,//gradients of output of this operator. TODO: dL_dcolor is not const here because low-level implementation does not use const. Even though, we never modify it. 
	float* dL_dmean3D,
	float* dL_dtransMats,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dsh,
	bool debug,
	const pybind11::dict &args)
{
	// auto [global_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);

	// MyTimerOnGPU timer;
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const float* transMats_ptr = transMats;
	// timer.start("b20 preprocess");
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		transMats_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dnormal,
		dL_dtransMats,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dmean3D,
		(glm::vec2*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
	// timer.stop("b20 preprocess");

	// Print out timing information
	// if (zhx_time && iteration % log_interval == 1) {
	// 	timer.printAllTimes(iteration, world_size, global_rank, log_folder, false);
	// }
}





/////////////////////////////// GetDistributionStrategy ///////////////////////////////

void CudaRasterizer::Rasterizer::getDistributionStrategy(
	std::function<char* (size_t)> distBuffer,
	const int P,
	const int width, int height,
	float2* means2D,
	int* radii,
	bool* compute_locally,
	bool debug,
	const pybind11::dict &args)
{
	// This function is deprecated for now. But I keed the structure of code here potentially for future use.
	throw std::runtime_error("getDistributionStrategy is deprecated.");
}

/////////////////////////////// Render ///////////////////////////////




int CudaRasterizer::Rasterizer::renderForward(
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
	const pybind11::dict &args)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	// auto [global_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);	
	// char* log_tmp = new char[500];

	// MyTimerOnGPU timer;

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, true); // do not allocate extra memory here if sep_rendering==True.

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	int tile_num = tile_grid.x * tile_grid.y;

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// timer.start("24 updateDistributedStatLocally.updateTileTouched");
	// For sep_rendering==True case (here), we only compute tiles_touched in the renderForward.
	updateTileTouched <<<(P + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >>> (
		P,
		tile_grid,
		radii,
		means2D,
		geomState.tiles_touched,
		compute_locally
	);
	// timer.stop("24 updateDistributedStatLocally.updateTileTouched");

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// timer.start("30 InclusiveSum");
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	// timer.stop("30 InclusiveSum");

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// timer.start("40 duplicateWithKeys");
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >> > (
		P,
		means2D,
		depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		compute_locally,
		tile_grid)
	CHECK_CUDA(, debug)
	// timer.stop("40 duplicateWithKeys");

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// timer.start("50 SortPairs");
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)
	// timer.stop("50 SortPairs");

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// timer.start("60 identifyTileRanges");
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)
	// timer.stop("60 identifyTileRanges");

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = rgb;
	const float* transMat_ptr = transMat;
	// timer.start("70 render");
	CHECK_CUDA(FORWARD::render(//TODO: only deal with local tiles. do not even load other tiles.
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		means2D,
		feature_ptr,
		transMat_ptr,
		depths,
		normal_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.n_contrib2loss,
		compute_locally,
		background,
		out_color, 
		out_others,
		accum_max_count,
		transmittance,
		num_covered_pixels,
		record_transmittance), debug)
	// timer.stop("70 render");

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::renderBackward(
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
	char* img_buffer,
	bool* compute_locally,//buffer that contains intermedia results
	const float* dL_dpix,//gradient of output
	const float* dL_depths,
	float* dL_dmean2D,//(P, 3)
	float* dL_dnormal,
	float* dL_dopacity,
	float* dL_dcolor,//gradient of inputs
	float* dL_dtransMat,
	bool debug,
	const pybind11::dict &args)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	// auto [global_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);

	// MyTimerOnGPU timer;

	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = rgb;
	const float* depth_ptr = depths;
	const float* transMat_ptr = transMats;
	// timer.start("b10 render");
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		background,
		means2D,
		normal_opacity,
		color_ptr,
		transMat_ptr,
		depth_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		compute_locally,
		dL_dpix,
		dL_depths,
		dL_dtransMat,
		(float3*)dL_dmean2D,
		dL_dnormal,
		dL_dopacity,
		dL_dcolor), debug)
	// timer.stop("b10 render");

	// float backward_render_time = timer.elapsedMilliseconds("b10 render", "sum");
	// // save backward_render_time in args["stats_collector"] which is a python::dict. Then it could be sent back to python.
	// args["stats_collector"]["backward_render_time"] = backward_render_time;
	
	// // Print out timing information
	// if (zhx_time && iteration % log_interval == 1) {
	// 	timer.printAllTimes(iteration, world_size, global_rank, log_folder, false);
	// }
}

void CudaRasterizer::Rasterizer::visible_filter(
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
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::filter_preprocess(
		P, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		cov3D_precomp,
		viewmatrix, projmatrix,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		cov3D,
		tile_grid,
		prefiltered
	), debug)

}

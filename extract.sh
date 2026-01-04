# CUDA_VISIBLE_DEVICES=4 python utils/gs2d_mesh_extraction.py outputs/CUHK_UPPER --voxel_size 0.005 --sdf_trunc 0.02 --depth_trunc 999.0
CUDA_VISIBLE_DEVICES=1 python utils/gs2d_mesh_extraction.py outputs/LFLS_new --voxel_size 0.01 --sdf_trunc 0.04 --depth_trunc 2.0
# CUDA_VISIBLE_DEVICES=3 python utils/gs2d_mesh_extraction.py outputs/UPPER_30K --voxel_size 0.01 --sdf_trunc 0.04 --depth_trunc 2.0
# CUDA_VISIBLE_DEVICES=2 python utils/gs2d_mesh_extraction.py outputs/MatrixCity-Aerial --voxel_size 0.01 --sdf_trunc 0.04 --depth_trunc 5.0
# CUDA_VISIBLE_DEVICES=1 python utils/gs2d_mesh_extraction.py outputs/aerial2_new --voxel_size 0.01 --sdf_trunc 0.04 --depth_trunc 5.0
# CUDA_VISIBLE_DEVICES=3 python utils/gs2d_mesh_extraction.py outputs/MatrixCity-Street --voxel_size 1 --sdf_trunc 4 --depth_trunc 500
# CUDA_VISIBLE_DEVICES=2 python utils/gs2d_mesh_extraction.py /mnt/sharedisk/chenkehua/GS+Nerf/others/CityGaussian_V2/checkpoints/MatrixCity_Street --voxel_size 1 --sdf_trunc 4 --depth_trunc 500
# CUDA_VISIBLE_DEVICES=3 python utils/gs2d_mesh_extraction.py /mnt/sharedisk/chenkehua/GS+Nerf/others/CityGaussian_V2/checkpoints/MatrixCity_Street --voxel_size 1 --sdf_trunc 4 --depth_trunc 500
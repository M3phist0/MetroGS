SCENE=aerial
ADD=mc_aerial
NAME=metrogs_mc_aerial
YAML=MatrixCity-Aerial

# ============================================= downsample images ============================================
# python utils/image_downsample.py data/matrix_city/${SCENE}/train/block_all/images --factor 1.2
# python utils/image_downsample.py data/matrix_city/${SCENE}/test/block_all_test/images --factor 1.2

# ===================================== generate depth with MoGe-2 ===========================================
# python utils/estimate_dataset_mask_depths.py data/matrix_city/${SCENE}/train/block_all -d 1.2 --preview

# ====================================== generate multi-view info ============================================
# python utils/multi_view_filter.py data/matrix_city/${SCENE}/train/block_all --split_mode reconstruction

# ===================================== pointmap assisted initalization ======================================
# python pointmap/scene_images_segment.py data/matrix_city/${SCENE}/train/block_all -k 4 --split_mode reconstruction # '-k' indicates number of segments
# bash pointmap/run_para.sh -g "0 1 2 3" -k 4 -b data/matrix_city/${SCENE}/train/block_all -c ./configs/mc_aerial.yaml # '-g' indicates gpu ids
# python pointmap/merge_all.py --base_dir data/matrix_city/${SCENE}/train/block_all --output add_ply/${ADD}.ply
# rm -rf data/matrix_city/${SCENE}/train/block_all/segments

# ============================================= train&eval model =============================================
export NCCL_SHM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_bsz.py fit --config configs/metrogs/train/${YAML}.yaml -n ${NAME}

python utils/merge_distributed_ckpts.py outputs/${NAME}

rm outputs/${NAME}/checkpoints/*150000-rank*

python main.py test --config configs/metrogs/val/${YAML}.yaml --weights_only false -n ${NAME}

CUDA_VISIBLE_DEVICES=0 python utils/gs2d_mesh_extraction.py outputs/${NAME} --post --voxel_size 0.01 --sdf_trunc 0.04 --depth_trunc 5.0

python tools/eval_tnt/run.py --scene Block_all_ds --dataset-dir data/geometry_gt/MC_Aerial --ply-path outputs/${NAME}/fuse.ply
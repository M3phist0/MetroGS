SCENE=residence-pixsfm
ADD=residence
NAME=metrogs_residence

# ============================================= downsample images ============================================
# python utils/image_downsample.py data/urban_scene_3d/${SCENE}/train/images --factor 4
# python utils/image_downsample.py data/urban_scene_3d/${SCENE}/val/images --factor 4

# ===================================== generate depth with MoGe-2 ===========================================
# python utils/estimate_dataset_mask_depths.py data/urban_scene_3d/${SCENE}/train -d 4 --preview

# ====================================== generate multi-view info ============================================
# python utils/multi_view_filter.py data/urban_scene_3d/${SCENE}/train --split_mode reconstruction

# ===================================== pointmap assisted initalization ======================================
# python pointmap/scene_images_segment.py data/urban_scene_3d/${SCENE}/train -k 4 --split_mode reconstruction # '-k' indicates number of segments
# bash pointmap/run_para.sh -g "0 1 2 3" -k 4 -b data/urban_scene_3d/${SCENE}/train -c ./configs/mill19.yaml # '-g' indicates gpu ids
# python pointmap/merge_all.py --base_dir data/urban_scene_3d/${SCENE}/train --output add_ply/${ADD}.ply
# rm -rf data/urban_scene_3d/${SCENE}/train/segments

# ============================================= train&eval model =============================================
export NCCL_SHM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_bsz.py fit --config configs/metrogs/train/${SCENE}.yaml -n ${NAME}

python utils/merge_distributed_ckpts.py outputs/${NAME}

rm outputs/${NAME}/checkpoints/*100000-rank*

python main.py test --config configs/metrogs/val/${SCENE}.yaml --weights_only false -n ${NAME}

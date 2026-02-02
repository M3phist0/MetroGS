SCENE=SMBU
PLYGT=SMBU
ADD=smbu
NAME=metrogs_smbu

# ============================================= downsample images ============================================
# python utils/image_downsample.py data/GauU_Scene/${SCENE}/images --factor 3.4175

# ===================================== generate depth with MoGe-2 ===========================================
# python utils/estimate_dataset_mask_depths.py data/GauU_Scene/${SCENE} -d 3.4175 --preview

# ====================================== generate multi-view info ============================================
# python utils/multi_view_filter.py data/GauU_Scene/${SCENE}

# ===================================== pointmap assisted initalization ======================================
# python pointmap/scene_images_segment.py data/GauU_Scene/${SCENE} -k 4 # '-k' indicates number of segments
# bash pointmap/run_para.sh -g "0 1 2 3" -k 4 -b data/GauU_Scene/${SCENE} -c ./configs/gauuscene.yaml # '-g' indicates gpu ids
# python pointmap/merge_all.py --base_dir data/GauU_Scene/${SCENE} --output add_ply/${ADD}.ply
# rm -rf data/GauU_Scene/${SCENE}/segments

# ============================================= train&eval model =============================================
export NCCL_SHM_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_bsz.py fit --config configs/metrogs/train/${SCENE}.yaml -n ${NAME}

python utils/merge_distributed_ckpts.py outputs/${NAME}

rm outputs/${NAME}/checkpoints/*60000-rank*

python main.py test --config configs/metrogs/val/${SCENE}.yaml -n ${NAME}

python utils/gs2d_mesh_extraction.py outputs/${NAME} --voxel_size 0.01 --sdf_trunc 0.04 --depth_trunc 2.0

python tools/eval_tnt/run.py --scene ${PLYGT}_ds --dataset-dir data/geometry_gt/${PLYGT} \
    --transform-path data/geometry_gt/${PLYGT}/transform.txt --ply-path outputs/${NAME}/fuse.ply
    
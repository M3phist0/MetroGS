SCENE=SMBU

# 获取单目深度图(MoGeV2)
python utils/estimate_dataset_mask_depths.py data/GauU_Scene/${SCENE} -d 3.4175

# 获取相邻视角信息
python utils/multi_view_filter.py data/GauU_Scene/${SCENE} # --split_mode reconstruction

# 划分图像方便Pi3处理
python utils/scene_images_segment.py data/GauU_Scene/${SCENE}


# cd utils/VGGT-Long # 改后的VGGT-Long

# bash run_blocks_parallel.sh

# python merge_all.py --base_dir ../../data/GauU_Scene/${SCENE}


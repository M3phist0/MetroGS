PROJECT_PATH="/mnt/sharedisk/chenkehua/GS+Nerf/AAAGS/data/surgarcane"
IMAGE_PATH="sugarcane_1600px"

# colmap feature_extractor \
#     --database_path $PROJECT_PATH/database.db \
#     --image_path $PROJECT_PATH/$IMAGE_PATH \
#     --ImageReader.camera_model RADIAL \
#     --SiftExtraction.use_gpu 1 \
#     --SiftExtraction.gpu_index 0,1,2,3,4,5,6

# colmap spatial_matcher \
#     --database_path $PROJECT_PATH/database.db \
#     --SpatialMatching.is_gps 1 \
#     --SiftMatching.gpu_index 0,1,2,3,4,5,6

# colmap vocab_tree_matcher \
#     --database_path $PROJECT_PATH/database.db \
#     --VocabTreeMatching.vocab_tree_path $PROJECT_PATH/vocab_tree_flickr100K_words256K.bin \
#     --VocabTreeMatching.num_images 50 \
#     --VocabTreeMatching.num_checks 256 \
#     --SiftMatching.gpu_index 0,1,2,3,4,5,6

# mkdir $PROJECT_PATH/sparse

# colmap hierarchical_mapper \
#     --database_path $PROJECT_PATH/database.db \
#     --image_path $PROJECT_PATH/$IMAGE_PATH \
#     --output_path $PROJECT_PATH/sparse

# colmap mapper \
#     --database_path $PROJECT_PATH/database.db \
#     --image_path $PROJECT_PATH/$IMAGE_PATH \
#     --output_path $PROJECT_PATH/sparse \
#     --Mapper.num_threads $(nproc) # 使用所有 CPU 核心

# colmap image_undistorter \
#     --image_path $PROJECT_PATH/$IMAGE_PATH  \
#     --input_path $PROJECT_PATH/sparse/0 \
#     --output_path $PROJECT_PATH/dense3 \
#     --output_width 1600 \
#     --output_height 1000 \
#     --output_type COLMAP \
#     --blank_pixels 1
    # COLMAP 或者 PLY, TEXT, 等，但 COLMAP 格式用于后续MVS最方便


colmap model_converter \
    --input_path $PROJECT_PATH/dense2/sparse \
    --output_path $PROJECT_PATH/dense2/sparse \
    --output_type TXT

### delete
# colmap database_cleaner \
#     --database_path $PROJECT_PATH/database.db \
#     --type all
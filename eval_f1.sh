# CUDA_VISIBLE_DEVICES=2 \
# python tools/eval_tnt/run.py --scene SMBU_ds --dataset-dir data/geometry_gt/SMBU \
#     --ply-path /mnt/sharedisk/chenkehua/GS+Nerf/others/CityGaussian_V2/outputs/citygsv2_smbu_sh2_trim/fuse_post.ply \
#     --transform-path data/geometry_gt/SMBU/transform.txt \
    # --ply-path outputs/SMBU/fuse.ply \

# CUDA_VISIBLE_DEVICES=1 \
# python tools/eval_tnt/run.py --scene CUHK_UPPER_ds --dataset-dir data/geometry_gt/CUHK_UPPER \
#     --ply-path outputs/CUHK_UPPER_acc/fuse.ply \
#     --transform-path data/geometry_gt/CUHK_UPPER/transform.txt \
    # --ply-path /mnt/sharedisk/chenkehua/GS+Nerf/others/CityGaussian_V2/checkpoints/UPPER_CAMPUS/fuse_post.ply \

CUDA_VISIBLE_DEVICES=1 \
python tools/eval_tnt/run.py --scene LFLS_ds --dataset-dir data/geometry_gt/LFLS \
    --ply-path outputs/LFLS_new/fuse.ply \
    --transform-path data/geometry_gt/LFLS/transform.txt \
    # --ply-path /mnt/sharedisk/chenkehua/GS+Nerf/others/CityGaussian_V2/outputs/citygsv2_lfls_sh2_trim/fuse.ply \

# CUDA_VISIBLE_DEVICES=2 \
# python tools/eval_tnt/run.py --scene Block_all_ds --dataset-dir data/geometry_gt/MC_Aerial --ply-path "/mnt/sharedisk/chenkehua/GS+Nerf/others/CityGS-X/output/MatrixCity_aerial/possion_mesh/tsdf_fusion.ply"
# python tools/eval_tnt/run.py --scene Block_all_ds --dataset-dir data/geometry_gt/MC_Aerial --ply-path "outputs/aerial2_new/fuse_post.ply"

# CUDA_VISIBLE_DEVICES=0 \
# python tools/eval_tnt/run.py --scene Block_A_ds --dataset-dir data/geometry_gt/MC_Street --ply-path "outputs/MatrixCity-Street/fuse_post.ply"
# python tools/eval_tnt/run.py --scene Block_A_ds --dataset-dir data/geometry_gt/MC_Street --ply-path "/mnt/sharedisk/chenkehua/GS+Nerf/others/CityGaussian_V2/checkpoints/MatrixCity_Street/fuse_post.ply"

# python eval_tnt/pgsr/run.py --dataset-dir data/TNT/Truck \
#     --traj-path data/TNT/Truck/Truck_COLMAP_SfM.log \
#     --ply-path outputs/Truck_geo/train/ours_29997/fuse_post.ply \
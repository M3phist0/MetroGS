# UrbanScene3D, Residence, Sci-Art
python tools/copy_images.py --image_path data/urban_scene_3d/Residence/photos --dataset_path data/urban_scene_3d/residence-pixsfm
python tools/copy_images.py --image_path data/urban_scene_3d/Sci-Art/photos --dataset_path data/urban_scene_3d/sci-art-pixsfm

cp -r data/colmap_results/residence/train/sparse data/mill19/residence-pixsfm/train
cp -r data/colmap_results/residence/val/sparse data/mill19/residence-pixsfm/val

cp -r data/colmap_results/sciart/train/sparse data/mill19/sci-art-pixsfm/train
cp -r data/colmap_results/sciart/val/sparse data/mill19/sci-art-pixsfm/val

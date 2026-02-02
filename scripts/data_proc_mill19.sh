# Mill19, Building, Rubble
ln -s rgbs data/mill19/building-pixsfm/train/images
ln -s rgbs data/mill19/building-pixsfm/val/images

ln -s rgbs data/mill19/rubble-pixsfm/train/images
ln -s rgbs data/mill19/rubble-pixsfm/val/images

cp -r data/colmap_results/building/train/sparse data/mill19/building-pixsfm/train
cp -r data/colmap_results/building/val/sparse data/mill19/building-pixsfm/val

cp -r data/colmap_results/rubble/train/sparse data/mill19/rubble-pixsfm/train
cp -r data/colmap_results/rubble/val/sparse data/mill19/rubble-pixsfm/val


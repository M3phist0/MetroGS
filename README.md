# MetroGS
MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes

The current implementation is preliminary and subject to ongoing development. The repository will be continuously updated and improved in future.

The data preparation refers to: [CityGS-data](https://github.com/Linketic/CityGaussian/blob/main/doc/data_preparation.md).

Run example:
```
# Obtain monocular depth maps (MoGeV2)
python utils/estimate_dataset_mask_depths.py data/GauU_Scene/SMBU -d 3.4175

# Obtain adjacent view information
python utils/multi_view_filter.py data/GauU_Scene/SMBU

# Train
# Note: Preprocessed PLY files are currently not provided.
# Please comment out `additional_ply_path` in the configuration.
python main_bsz.py fit --config config/merogs/SMBU.yaml -n SMBU

# Test
# Please set `aabb` in the YAML file to the computed array.
# The array values will be printed during the initial stage of training.
python utils/merge_distributed_ckpts.py outputs/SMBU
python main.py test --config config/merogs/SMBU_TEST.yaml -n SMBU
```

We sincerely thank the contributors of [CityGaussianV2](https://github.com/Linketic/CityGaussian.git) for their valuable contributions.

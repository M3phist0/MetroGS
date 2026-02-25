<h2 align="center">
  MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes
</h2>


<div align="center">
    <a href="https://arxiv.org/html/2511.19172"">
      <img src="https://img.shields.io/badge/arXiv-2511.19172-b31b1b?style=flat-square" height="20">
    </a>
    <a href="https://m3phist0.github.io/MetroGS/">
      <img src="https://img.shields.io/badge/Project-Website-green?style=flat-square" height="20">
    </a>
</div>


<p align="center">
  <img src="asset/demo.gif" width="80%" />
</p>




## Getting Started
- [Installation](doc/installation.md)
- [Data Preparation](doc/data_preparation.md)
- [Video Rendering](doc/render_video.md)
<!-- - [Run and Eval](doc/run&eval.md) -->

## Run & Evaluate
The detailed setting of each step on GauU-Scene and MatrixCity Dataset can be found in `./scripts/run_metrogs_NAME.sh`. 


### Train model and merge
```bash
# ${SCENE} is the name of training config file.
# ${NAME} is the name of model output file.
[Optional] export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SHM_DISABLE=1 # Shared memory (SHM) is currently not supported.
python main_bsz.py fit --config configs/metrogs/train/${SCENE}.yaml -n ${NAME}
python utils/merge_distributed_ckpts.py outputs/${NAME}
```

### Evaluate Rendering Performance
```bash
python main.py test --config configs/metrogs/val/${SCENE}.yaml -n ${NAME}
```

### Mesh extraction and evaluation
```bash
python utils/gs2d_mesh_extraction.py outputs/${NAME} \
        --post ${POST_FLAG} \
        --voxel_size ${VOXEL_SIZE} \
        --sdf_trunc ${SDF_TRUNC} \
        --depth_trunc ${DEPTH_TRUNC} \

python tools/eval_tnt/run.py \
        --scene your_gt_pcd \
        --dataset-dir data/geometry_gt/your_scene \
        --transform-path data/geometry_gt/your_scene/transform.txt \
        --ply-path "outputs/${NAME}/fuse.ply"
```


### Checkpoints
Preprocessed pointmaps and checkpoints are available [here](https://pan.baidu.com/s/1XsEjppkuRi-6NivTdth4ww?pwd=ek36).

## TODO List

- \[ \] Incorporate CLM-GS.
- \[x\] Release the checkpoints.


## Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```latex
@article{chen2025metrogs,
  title={MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes},
  author={Chen, Kehua and Mao, Tianlu and Ma, Zhuxin and Jiang, Hao and Li, Zehao and Liu, Zihan and Gao, Shuqi and Zhao, Honglong and Dai, Feng and Zhang, Yucheng and others},
  journal={arXiv preprint arXiv:2511.19172},
  year={2025}
}
```

## Acknowledgements

This repo benefits from [CityGSV2](https://github.com/Linketic/CityGaussian), [CityGS-X](https://github.com/gyy456/CityGS-X), [Grendel-GS](https://github.com/nyu-systems/Grendel-GS), [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning), [TriMip-RF](https://github.com/wbhu/Tri-MipRF). Thanks for their great work!

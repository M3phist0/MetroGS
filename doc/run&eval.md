## Run & Evaluate
The detailed setting of each step on GauU-Scene and MatrixCity Dataset can be found in `./scripts/run_metrogs_NAME.sh`. 


### A. Train model and merge
```bash
# ${SCENE} is the name of training config file.
# ${NAME} is the name of model output file.
[Optional] export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SHM_DISABLE=1 # Shared memory (SHM) is currently not supported.
python main_bsz.py fit --config configs/metrogs/train/${SCENE}.yaml -n ${NAME}
python utils/merge_distributed_ckpts.py outputs/${NAME}
```

### B. Evaluate Rendering Performance
```bash
# For finetuned model, since the split mode and eval ratios are changed for per-block tuning, the parameters have to be reappointed. Please see the script for details.
python main.py test --config configs/metrogs/val/${SCENE}.yaml -n ${NAME}
```

### C. Mesh extraction and evaluation
```bash
python utils/gs2d_mesh_extraction.py outputs/${NAME} \
        --voxel_size ${VOXEL_SIZE} \
        --sdf_trunc ${SDF_TRUNC} \
        --depth_trunc ${DEPTH_TRUNC} \

python tools/eval_tnt/run.py \
        --scene your_gt_pcd \
        --dataset-dir data/geometry_gt/your_scene \
        --transform-path data/geometry_gt/your_scene/transform.txt \
        --ply-path "outputs/${NAME}/fuse.ply"
```

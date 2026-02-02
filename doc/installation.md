## Installation
### A. Clone repository

```bash
# clone repository
git clone https://github.com/M3phist0/MetroGS.git
cd MetroGS
```

### B. Create virtual environment

```bash
# create virtual environment
conda create -n metrogs python=3.10
conda activate metrogs
```

### C. Install PyTorch
* We reimplemented the method on RTX 5090 due to an equipment upgrade in our group.
* Tested on `PyTorch==2.10`
* You must install the one match to the version of your nvcc (nvcc --version)
* For CUDA 12.8

  ```bash
  pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple -f https://mirrors.aliyun.com/pytorch-wheels/cu128
  ```

### D. Install submodules
Use `--no-build-isolation` to resolve the error “ModuleNotFoundError: No module named 'torch'”.
```bash
# basic
pip install submodules/dist-2dgs --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
# patchmatch
pip install submodules/propagation --no-build-isolation
# apperance
pip install submodules/nvdiffrast --no-build-isolation
pip install submodules/tiny-cuda-nn/bindings/torch --no-build-isolation
```
Note: The `-gencode=arch=compute_xx,code=sm_xx` setting in submodules/propagation/setup.py must be adjusted to match your hardware environment.

`diff-gaussian-rasterization` is required by the base classes, even though we do not use it directly.
```bash
pip install submodules/diff-gaussian-rasterization --no-build-isolation
```

### E. Install requirements

```bash
pip install -r requirements.txt
```

### F. Download Prior Models

#### MoGe-2
# NOTE: do not run `pip install -r utils/MoGe/requirements.txt`
```bash
# Mode: moge-2-vitl-normal
Download link: https://huggingface.co/Ruicheng/moge-2-vitl-normal/blob/main/model.pt
# Please download `model.pt` and move it to `utils/MoGe/checkpoints`
```

#### Pi3-Align

```bash
# Please follow the official script to download the weights:
cd pointmap/Pi3-Align 
bash scripts/download_weights.sh
```
#!/usr/bin/env bash
set -e

########## 1. 配置区域（你只需要改这里） ##########
PROJ="/mnt/sharedisk/chenkehua/GS+Nerf/AAAGS/data/surgarcane/dense2"          # 工程根目录
TARGET_W=1600        # 目标宽度
TARGET_H=1000        # 目标高度
###################################################

SPARSE_BIN="$PROJ/sparse"
IMAGES_IN="$PROJ/images"
WORK="$PROJ/resize_${TARGET_W}x${TARGET_H}"

mkdir -p "$WORK"
mkdir -p "$WORK/sparse"
mkdir -p "$WORK/sparse_txt"

echo "==> 1. 将 sparse/bin 转为 txt ..."
colmap model_converter \
    --input_path "$SPARSE_BIN" \
    --output_path "$WORK/sparse" \
    --output_type TXT

echo "==> 2. 拷贝并 resize 图像到统一分辨率 ${TARGET_W}x${TARGET_H} ..."
mkdir -p "$WORK/images"

# 这里用 Python + Pillow，你需要: pip install pillow
python3 - "$IMAGES_IN" "$WORK/images" $TARGET_W $TARGET_H << 'PYCODE'
import sys
from pathlib import Path
from PIL import Image

src_dir = Path(sys.argv[1])
dst_dir = Path(sys.argv[2])
target_w = int(sys.argv[3])
target_h = int(sys.argv[4])

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}

files = [p for p in src_dir.iterdir() if p.suffix in image_exts]
if not files:
    print(f"No images found in {src_dir}")
    sys.exit(1)

for src in files:
    img = Image.open(src)
    # 强制拉伸到指定分辨率，如果你不想拉伸，可以自己改成按长边缩放等策略
    resized = img.resize((target_w, target_h), Image.BILINEAR)
    dst = dst_dir / src.name
    resized.save(dst)
    print(f"Resized {src.name} -> {target_w}x{target_h}")
PYCODE

echo "==> 3. 按新分辨率缩放 cameras.txt 中的内参 ..."

python3 - "$WORK/sparse_txt/cameras.txt" $TARGET_W $TARGET_H << 'PYCODE'
import sys
from pathlib import Path

cam_path = Path(sys.argv[1])
new_w = int(sys.argv[2])
new_h = int(sys.argv[3])

lines = cam_path.read_text().splitlines()
out_lines = []

for line in lines:
    if line.startswith("#") or not line.strip():
        out_lines.append(line)
        continue

    parts = line.split()
    if len(parts) < 5:
        out_lines.append(line)
        continue

    cam_id = parts[0]
    model = parts[1]
    old_w = int(parts[2])
    old_h = int(parts[3])
    params = list(map(float, parts[4:]))

    sx = new_w / old_w
    sy = new_h / old_h

    # 如果宽高比例不一致，给个提示（但仍然继续运行）
    if abs(sx - sy) > 1e-3:
        print(f"[WARN] camera {cam_id}: sx != sy (sx={sx:.4f}, sy={sy:.4f}), "
              f"说明你指定的分辨率和原始图像长宽比不一致，会产生拉伸失真。")

    # 常见模型的参数布局（参考 COLMAP 官方文档）
    # PINHOLE:         fx fy cx cy
    # OPENCV:          fx fy cx cy k1 k2 p1 p2 k3
    # OPENCV_FISHEYE:  fx fy cx cy k1 k2 k3 k4
    # SIMPLE_PINHOLE:  f cx cy
    # SIMPLE_RADIAL:   f cx cy k
    # RADIAL:          f cx cy k1 k2
    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
        # fx, fy, cx, cy
        params[0] *= sx   # fx
        params[1] *= sy   # fy
        params[2] *= sx   # cx
        params[3] *= sy   # cy
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        # 这些模型只有一个 f，理论上要求 sx ≈ sy
        scale = sx  # 默认用 sx
        params[0] *= scale   # f
        params[1] *= sx      # cx
        params[2] *= sy      # cy
        # 后面的畸变参数照常保留
    else:
        print(f"[INFO] 未特别处理的相机模型 {model}，只修改 width/height，不缩放参数。")

    new_params_str = " ".join(f"{p:.12f}" for p in params)
    new_line = f"{cam_id} {model} {new_w} {new_h} {new_params_str}"
    out_lines.append(new_line)

cam_path.write_text("\n".join(out_lines) + "\n")
PYCODE

echo "==> 4. （可选）将新的 txt 模型转回 BIN 方便 COLMAP 使用 ..."

colmap model_converter \
    --input_path "$WORK/sparse_txt" \
    --output_path "$WORK/sparse" \
    --output_type BIN

echo "完成！"
echo "统一分辨率的图像在:   $WORK/images"
echo "对应的模型 txt 在:     $WORK/sparse_txt"
echo "对应的模型 bin 在:     $WORK/sparse"


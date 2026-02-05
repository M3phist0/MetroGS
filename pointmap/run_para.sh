#!/bin/bash

set -euo pipefail

# --- 1. 配置区域（默认值，可被参数覆盖） ---
AVAILABLE_GPUS_DEFAULT="0 1 2 3"
BLOCK_COUNT_DEFAULT=4
BASE_DIR_DEFAULT="data/GauU_Scene/CUHK_UPPER_COLMAP"
IMAGE_SUBDIR_DEFAULT="images"
CONFIG_DEFAULT="./configs/base_config.yaml"
SPARSE_SUBDIR_DEFAULT="segments/block_"

AVAILABLE_GPUS="$AVAILABLE_GPUS_DEFAULT"
BLOCK_COUNT="$BLOCK_COUNT_DEFAULT"
BASE_DIR="$BASE_DIR_DEFAULT"
IMAGE_SUBDIR="$IMAGE_SUBDIR_DEFAULT"
CONFIG="$CONFIG_DEFAULT"
SPARSE_SUBDIR="$SPARSE_SUBDIR_DEFAULT"

usage() {
  cat <<'EOF'
用法:
  run_blocks.sh [-g "0 1 2 3"] [-k 4] [-b BASE_DIR] [-i IMAGE_SUBDIR] [-c CONFIG] [-s SPARSE_SUBDIR]

参数:
  -g  可用GPU列表（空格分隔），例如: "0 1 2 3"
  -k  block数量（整数）
  -b  BASE_DIR
  -i  IMAGE_SUBDIR
  -c  CONFIG
  -s  SPARSE_SUBDIR（默认: segments/block_）
EOF
}

while getopts ":g:k:b:i:c:s:h" opt; do
  case "$opt" in
    g) AVAILABLE_GPUS="$OPTARG" ;;
    k) BLOCK_COUNT="$OPTARG" ;;
    b) BASE_DIR="$OPTARG" ;;
    i) IMAGE_SUBDIR="$OPTARG" ;;
    c) CONFIG="$OPTARG" ;;
    s) SPARSE_SUBDIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "未知参数: -$OPTARG"; usage; exit 1 ;;
    :)  echo "参数 -$OPTARG 需要值"; usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

# 简单校验
if ! [[ "$BLOCK_COUNT" =~ ^[0-9]+$ ]] || [ "$BLOCK_COUNT" -le 0 ]; then
  echo "BLOCK_COUNT 必须是正整数，但你传的是: $BLOCK_COUNT"
  exit 1
fi

# ----------------------------------------------------


# 任务数组：存储所有待处理的 Block ID (0, 1, 2, 3...)
TASKS=()
for i in $(seq 0 $((BLOCK_COUNT - 1))); do
    TASKS+=($i)
done

# ----------------------------------------------------

# 将可用 GPU ID 转换为数组
GPU_ARRAY=($AVAILABLE_GPUS)
NUM_GPUS=${#GPU_ARRAY[@]}

# 空闲 GPU 队列（初始化：全部 GPU 都空闲）
FREE_GPUS=("${GPU_ARRAY[@]}")

# PID -> GPU 映射（任务结束后归还 GPU 用）
declare -A PID_TO_GPU


echo "--- 开始并行调度任务 ---"
echo "可用 GPU 数量: $NUM_GPUS"
echo "任务总数 (Blocks): $BLOCK_COUNT"
echo "--------------------------------"

# 核心调度循环
for TASK_ID in "${TASKS[@]}"; do
    # --- 1&2. 等待并拿到一个空闲 GPU（不会重复分配） ---
    while [ ${#FREE_GPUS[@]} -eq 0 ]; do
        echo "当前没有空闲 GPU，等待其中一个任务完成..."
        wait -n  # 等任意一个后台任务结束

        # 回收：把已经结束的任务占用的 GPU 放回 FREE_GPUS
        running_set=" $(jobs -pr) "
        for pid in "${!PID_TO_GPU[@]}"; do
            if [[ "$running_set" != *" $pid "* ]]; then
                echo "GPU 进程 $pid 已完成，归还 GPU ${PID_TO_GPU[$pid]}."
                FREE_GPUS+=("${PID_TO_GPU[$pid]}")
                unset "PID_TO_GPU[$pid]"
            fi
        done
    done

    # 从空闲队列取一个 GPU（FIFO）
    CURRENT_GPU="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    # --- 3. 构造并运行命令 ---
    
    # 构造当前 Block 的目录
    CURRENT_IMAGE_DIR="${BASE_DIR}/${SPARSE_SUBDIR}${TASK_ID}/${IMAGE_SUBDIR}"
    CURRENT_SPARSE_DIR="${BASE_DIR}/${SPARSE_SUBDIR}${TASK_ID}/sparse"
    CURRENT_SAVE_DIR="${BASE_DIR}/${SPARSE_SUBDIR}${TASK_ID}/output"

    # 如果保存目录已存在，先删除（加安全防呆）
    if [ -d "$CURRENT_SAVE_DIR" ]; then
        # 安全检查：必须在 BASE_DIR 下，且不是空/根目录
        case "$CURRENT_SAVE_DIR" in
            ""|"/") 
                echo "错误：CURRENT_SAVE_DIR 异常（为空或根目录），拒绝删除"
                exit 1
                ;;
        esac
        if [[ "$CURRENT_SAVE_DIR" != "$BASE_DIR"* ]]; then
            echo "错误：CURRENT_SAVE_DIR 不在 BASE_DIR 下，拒绝删除: $CURRENT_SAVE_DIR"
            exit 1
        fi

        echo "保存目录已存在，删除: $CURRENT_SAVE_DIR"
        rm -rf -- "$CURRENT_SAVE_DIR"
    fi
    
    echo "--- 任务 $TASK_ID 分配给 GPU $CURRENT_GPU ---"
    echo "image_dir:  $CURRENT_IMAGE_DIR"
    echo "sparse_dir: $CURRENT_SPARSE_DIR"
    echo "save_dir:   $CURRENT_SAVE_DIR"

    CUDA_VISIBLE_DEVICES="$CURRENT_GPU" \
    python pointmap/Pi3-Align/X_long.py \
    --image_dir "$CURRENT_IMAGE_DIR" \
    --sparse_dir "$CURRENT_SPARSE_DIR" \
    --save_dir "$CURRENT_SAVE_DIR" \
    --config "$CONFIG" -x Pi3 &

    PID=$!
    PID_TO_GPU["$PID"]="$CURRENT_GPU"
    echo "后台进程 ID: $PID (占用 GPU $CURRENT_GPU)"

done

# --- 4. 结束所有任务 ---
echo "所有任务已分配，等待所有后台进程完成..."
wait

echo "--- 所有 Blocks 处理完成！ ---"

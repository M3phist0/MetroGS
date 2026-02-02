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

# 存储当前正在运行的任务的进程ID (PID)
RUNNING_PIDS=()

echo "--- 开始并行调度任务 ---"
echo "可用 GPU 数量: $NUM_GPUS"
echo "任务总数 (Blocks): $BLOCK_COUNT"
echo "--------------------------------"

# 核心调度循环
for TASK_ID in "${TASKS[@]}"; do
    # --- 1. 等待空闲 GPU (如果运行中的任务数达到 GPU 限制) ---
    while [ ${#RUNNING_PIDS[@]} -ge $NUM_GPUS ]; do
        
        # 使用 wait -n 等待**任一**后台子进程结束 (等待第一个空闲的 GPU)
        echo "当前有 $NUM_GPUS 个任务在运行中，等待其中一个完成..."
        wait -n  # Bash 4.3+ 支持

        # 清理已完成的 PID，更新 RUNNING_PIDS 列表
        NEW_PIDS=()
        for PID in "${RUNNING_PIDS[@]}"; do
            # 检查进程是否存在
            if kill -0 "$PID" 2>/dev/null; then
                NEW_PIDS+=($PID)
            else
                echo "GPU 进程 $PID 已完成。"
            fi
        done
        RUNNING_PIDS=("${NEW_PIDS[@]}")
    done


    # --- 2. 查找下一个可用 GPU ---
    
    # 找到一个未被占用的 GPU ID
    CURRENT_GPU=""
    for gpu_id in "${GPU_ARRAY[@]}"; do
        IS_USED=false
        # 简化处理：由于我们使用 wait -n 确保了 RUNNING_PIDS 数量 <= NUM_GPUS，
        # 这里的 GPU 分配可以简化为：将当前任务索引映射到 GPU_ARRAY 索引。
        
        # 简单分配逻辑：使用 RUNNING_PIDS 的数量作为 GPU 数组的索引
        # 这种逻辑在 wait -n 后可能导致 GPU ID 重复，但在后台进程数受限时是有效的负载均衡。
        # 最简单的方式是：直接使用当前任务ID对GPU总数取模来决定使用哪个 GPU，
        # 但我们这里采用更严谨的方式：从 GPU 数组中，根据当前 RUNNING_PIDS 的数量来分配 GPU
        if [ ${#RUNNING_PIDS[@]} -lt $NUM_GPUS ]; then
             CURRENT_GPU=${GPU_ARRAY[${#RUNNING_PIDS[@]}]}
             break
        fi
    done
    
    if [ -z "$CURRENT_GPU" ]; then
        # 理论上不会发生，因为上面的 while 循环确保了有空间
        echo "错误：无法找到空闲 GPU，但 RUNNING_PIDS 数量不足！"
        exit 1
    fi

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
    
    COMMAND="CUDA_VISIBLE_DEVICES=${CURRENT_GPU} python pointmap/Pi3-Align/X_long.py --image_dir ${CURRENT_IMAGE_DIR} --sparse_dir ${CURRENT_SPARSE_DIR} --save_dir ${CURRENT_SAVE_DIR} --config ${CONFIG} -x Pi3"

    
    echo "--- 任务 $TASK_ID 分配给 GPU $CURRENT_GPU ---"
    echo "执行命令: $COMMAND"

    # 运行命令到后台，并捕获 PID
    eval "$COMMAND" &
    PID=$!
    RUNNING_PIDS+=($PID) # 将新的 PID 加入到运行中列表
    echo "后台进程 ID: $PID"
done

# --- 4. 结束所有任务 ---
echo "所有任务已分配，等待所有后台进程完成..."
wait

echo "--- 所有 Blocks 处理完成！ ---"

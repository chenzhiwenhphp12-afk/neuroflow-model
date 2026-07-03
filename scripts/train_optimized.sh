#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# NeuroFlow v2 训练启动脚本 (双路 Xeon E5-2666 v3 优化版)
#
# 硬件: 2× Intel Xeon E5-2666 v3, 40 threads, AVX2, 64GB DDR4
# 语料: D:\语料\ (~420GB, 10个目录, 含大学215万文件)
#
# 用法:
#   chmod +x scripts/train_optimized.sh
#   bash scripts/train_optimized.sh
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ═══ 配置 ═══════════════════════════════════════════════════
CORPUS_DIR="/mnt/d/语料"
TOKENIZED_DIR="${HOME}/neuroflow_data"
OUTPUT_DIR="${PROJECT_ROOT}/output"
CONFIG_PATH="${PROJECT_ROOT}/configs/config.json"
TOKENIZER_PATH="${PROJECT_ROOT}/configs/tokenizer_128k.json"

# 训练超参
EPOCHS=20
BATCH_SIZE=64
GRAD_ACCUM=4              # 梯度累积步数 (等效 batch=256)
LEARNING_RATE=3e-5
GRAD_CLIP=4.0
LOG_INTERVAL=50
SAVE_INTERVAL=5000
SEED=42

# 经验回放
REPLAY_BUFFER=10000
REPLAY_RATIO=0.25

# ═══ Step 1: 编译 ═══════════════════════════════════════════
echo "═════════════════════════════════════════════════════"
echo "Step 1: 编译 NeuroFlow"
echo "═════════════════════════════════════════════════════"

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEUROFLOW_USE_BLAS=ON \
    -DNEUROFLOW_USE_AVX2=ON \
    -DNEUROFLOW_BUILD_TESTS=OFF
make -j$(nproc)
cd "$PROJECT_ROOT"
echo "编译完成."

# ═══ Step 2: 数据预处理 (首次运行) ══════════════════════════
echo ""
echo "═════════════════════════════════════════════════════"
echo "Step 2: 数据预处理"
echo "═════════════════════════════════════════════════════"

if [ ! -f "${TOKENIZED_DIR}/train.tok1" ]; then
    echo "首次运行, 预处理语料到 ${TOKENIZED_DIR} ..."
    echo "语料: ${CORPUS_DIR}"
    echo "输出: ${TOKENIZED_DIR}"
    python3 scripts/prepare_training_data.py \
        --corpus "${CORPUS_DIR}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --output "${TOKENIZED_DIR}" \
        --max-seq-len 128 \
        --max-samples 5000000
    echo "预处理完成."
else
    echo "已存在 tokenized 数据: ${TOKENIZED_DIR}/train.tok1"
    ls -lh "${TOKENIZED_DIR}/train.tok1"
fi

# ═══ Step 3: 训练 ═══════════════════════════════════════════
echo ""
echo "═════════════════════════════════════════════════════"
echo "Step 3: 启动训练"
echo "═════════════════════════════════════════════════════"

# OpenBLAS 线程优化 (双路Xeon, 每socket 10物理核)
# 使用一半物理核心给BLAS, 另一半给OpenMP
NCPU=$(nproc)
BLAS_THREADS=$((NCPU / 4))  # 10 threads for BLAS (one socket)
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export GOTO_NUM_THREADS=${BLAS_THREADS}
export OMP_NUM_THREADS=${BLAS_THREADS}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "硬件配置:"
echo "  CPU 核心: ${NCPU}"
echo "  BLAS 线程: ${BLAS_THREADS}"
echo "  批大小:    ${BATCH_SIZE} × ${GRAD_ACCUM} grad_accum = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  学习率:    ${LEARNING_RATE}"
echo "  数据:      ${TOKENIZED_DIR}/train.tok1"

mkdir -p "${OUTPUT_DIR}"

# 训练命令
./build/neuroflow_train_v2 \
    --config "${CONFIG_PATH}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --data "${TOKENIZED_DIR}" \
    --output "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --grad-clip ${GRAD_CLIP} \
    --grad-accum ${GRAD_ACCUM} \
    --seed ${SEED} \
    --log-interval ${LOG_INTERVAL} \
    --save-interval ${SAVE_INTERVAL} \
    --replay-buffer ${REPLAY_BUFFER} \
    --replay-ratio ${REPLAY_RATIO} \
    --init-weights xavier

echo ""
echo "═════════════════════════════════════════════════════"
echo "训练完成! 模型保存于: ${OUTPUT_DIR}/model_final.nfv1"
echo "═════════════════════════════════════════════════════"

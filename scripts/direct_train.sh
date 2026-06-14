#!/bin/bash
# 直接C++训练: 小学 → 高中 → 大学
set -e
B=/mnt/d/neuroflow-C++
O=$B/output
CKPT=
export LD_LIBRARY_PATH=$HOME/local/openblas/usr/lib/x86_64-linux-gnu/openblas-openmp:$LD_LIBRARY_PATH
N=/tmp/numactl-extracted/usr/bin/numactl

train() {
  local DATA=$1 EPOCHS=$2 MSG=$3
  local FLAGS=""
  [ -n "$CKPT" ] && FLAGS="--resume $CKPT"
  echo; echo "===== $MSG ====="
  OPENBLAS_NUM_THREADS=10 OMP_NUM_THREADS=20 \
  $N --cpunodebind=0 --membind=0 \
  $B/neuroflow_train_v2 \
    --config $B/configs/config.json \
    --tokenizer $B/configs/tokenizer_cn_013.json \
    --output $O \
    --data "$DATA" \
    $FLAGS \
    --epochs $EPOCHS --lr 1e-4 --batch-size 64 \
    --log-interval 10 --save-interval 1000 --grad-clip 1.0
  # 找最新checkpoint (嵌套目录)
  CKPT=$(ls -td $O/checkpoint_step*/checkpoint_step* 2>/dev/null | head -1)
  echo "  checkpoint: $CKPT"
}

train "/mnt/d/语料/小学" 2 "小学 2 epochs"
train "/mnt/d/语料/高中" 5 "高中 3 epochs (续小学)"
train "/mnt/d/语料/大学" 10 "大学 5 epochs (续高中)"
echo; echo "✅ 课程训练完成!"

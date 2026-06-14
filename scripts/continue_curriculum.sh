#!/bin/bash
B=/mnt/d/neuroflow-C++
O=$B/output
CKPT=$(ls -td $O/checkpoint_step*/checkpoint_step* 2>/dev/null | head -1)
export LD_LIBRARY_PATH=$HOME/local/openblas/usr/lib/x86_64-linux-gnu/openblas-openmp:$LD_LIBRARY_PATH
N=/tmp/numactl-extracted/usr/bin/numactl

train() {
  local DATA=$1 EPOCHS=$2 MSG=$3
  local FLAGS=""
  [ -n "$CKPT" ] && FLAGS="--resume $CKPT"
  echo; echo "===== $MSG ====="
  OPENBLAS_NUM_THREADS=10 OMP_NUM_THREADS=20 \
  $N --cpunodebind=0 --membind=0 $B/neuroflow_train_v2 \
    --config $B/configs/config.json --tokenizer $B/configs/tokenizer_cn_013.json \
    --output $O --data "$DATA" $FLAGS \
    --epochs $EPOCHS --lr 1e-4 --batch-size 64 \
    --log-interval 10 --save-interval 1000 --grad-clip 1.0
  CKPT=$(ls -td $O/checkpoint_step*/checkpoint_step* 2>/dev/null | head -1)
  echo "  → $CKPT"
}

train "/mnt/d/语料/初中" 5 "初中 3 epochs (续小学)"
train "/mnt/d/语料/高中" 8 "高中 3 epochs (续初中)"
train "/mnt/d/语料/大学" 13 "大学 5 epochs (续高中)"
echo; echo "✅ 全部完成!"
ls -la $O/model_final.nfv1 2>/dev/null

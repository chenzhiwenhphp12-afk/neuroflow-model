#!/bin/bash
# NeuroFlow 课程训练: 小学 → 高中 → 大学
set -e

BASE=/mnt/d/neuroflow-C++
OBLAS=$HOME/local/openblas/usr/lib/x86_64-linux-gnu/openblas-openmp
NUMACTL=/tmp/numactl-extracted/usr/bin/numactl
BINARY=$BASE/neuroflow_train_v2
OUTPUT=$BASE/output
TOK1_DIR=$OUTPUT/tok1
CKPT=

mkdir -p $OUTPUT $TOK1_DIR
rm -f $OUTPUT/model_final.nfv1

export LD_LIBRARY_PATH=$OBLAS:$LD_LIBRARY_PATH

echo "===== 预处理 大学 (后台) ====="
python3 $BASE/scripts/preprocess_corpus.py process \
  --input /mnt/d/语料/大学 \
  --tokenizer $BASE/configs/tokenizer_cn_013.json \
  --output $TOK1_DIR/university.tok1 \
  --max-seq-len 512 --max-samples 500000 --min-tokens 4 \
  > /tmp/tok1_university.log 2>&1 &
UNI_PID=$!

echo "===== 预处理 小学 ====="
python3 $BASE/scripts/preprocess_corpus.py process \
  --input /mnt/d/语料/小学 \
  --tokenizer $BASE/configs/tokenizer_cn_013.json \
  --output $TOK1_DIR/primary.tok1 \
  --max-seq-len 512 --max-samples 500000 --min-tokens 4 \
  2>&1 | tail -3

echo "===== 预处理 高中 ====="
python3 $BASE/scripts/preprocess_corpus.py process \
  --input /mnt/d/语料/高中 \
  --tokenizer $BASE/configs/tokenizer_cn_013.json \
  --output $TOK1_DIR/highschool.tok1 \
  --max-seq-len 512 --max-samples 500000 --min-tokens 4 \
  2>&1 | tail -3

train() {
  local DATA=$1 EPOCHS=$2 MSG=$3
  local RESUME=""
  [ -n "$CKPT" ] && RESUME="--resume $CKPT"
  echo -e "\n===== $MSG ====="
  OPENBLAS_NUM_THREADS=10 OMP_NUM_THREADS=20 \
  $NUMACTL --cpunodebind=0 --membind=0 \
  $BINARY --config $BASE/configs/config.json \
    --tokenizer $BASE/configs/tokenizer_cn_013.json \
    --data $DATA --output $OUTPUT $RESUME \
    --epochs $EPOCHS --lr 1e-4 --batch-size 64 \
    --log-interval 10 --save-interval 1000 --grad-clip 1.0
  CKPT=$(ls -td $OUTPUT/checkpoint_step*/checkpoint_step* 2>/dev/null | head -1)
  echo "  checkpoint: $CKPT"
}

train $TOK1_DIR/primary.tok1 2 "小学 2 epochs"
train $TOK1_DIR/highschool.tok1 5 "高中 3 epochs"

echo -e "\n===== 等待大学预处理 ====="
wait $UNI_PID
ls -la $TOK1_DIR/university.tok1

train $TOK1_DIR/university.tok1 10 "大学 5 epochs"

echo -e "\n✅ 全部完成!"
ls -la $OUTPUT/model_final.nfv1
#!/bin/bash
# NeuroFlow 课程训练: 预处理→小学→初中→高中→大学
set -e
B=/mnt/d/neuroflow-C++
O=$B/output
T=/tmp/neuroflow_tok1
CKPT=
export LD_LIBRARY_PATH=$HOME/local/openblas/usr/lib/x86_64-linux-gnu/openblas-openmp:$LD_LIBRARY_PATH
N=/tmp/numactl-extracted/usr/bin/numactl

mkdir -p $O $T

echo "========================================"
echo "STEP 0: 预处理好 (TOK1)"
echo "  大学数据量最大，后台预处理"
echo "========================================"
python3 $B/scripts/preprocess_corpus.py process \
  --input /mnt/d/语料/大学 --tokenizer $B/configs/tokenizer_cn_013.json \
  --output $T/university.tok1 --max-seq-len 512 --max-samples 500000 --min-tokens 4 \
  > /tmp/tok1_university.log 2>&1 &
UNI_PID=$!

for DS in 小学 初中 高中; do
  echo "预处理 $DS..."
  python3 $B/scripts/preprocess_corpus.py process \
    --input "/mnt/d/语料/$DS" --tokenizer $B/configs/tokenizer_cn_013.json \
    --output "$T/$DS.tok1" --max-seq-len 512 --max-samples 500000 --min-tokens 4 \
    2>&1 | tail -1
done

train() {
  local DATA=$1 EPOCHS=$2 MSG=$3
  local FLAGS=""
  [ -n "$CKPT" ] && FLAGS="--resume $CKPT"
  echo; echo "========================================"
  echo ">>> $MSG"
  echo "========================================"
  OPENBLAS_NUM_THREADS=10 OMP_NUM_THREADS=20 \
  $N --cpunodebind=0 --membind=0 \
  $B/neuroflow_train_v2 --config $B/configs/config.json \
    --tokenizer $B/configs/tokenizer_cn_013.json \
    --output $O --data "$DATA" $FLAGS \
    --epochs $EPOCHS --lr 1e-4 --batch-size 64 \
    --log-interval 10 --save-interval 1000 --grad-clip 1.0
  CKPT=$(ls -td $O/checkpoint_step*/checkpoint_step* 2>/dev/null | head -1)
  echo "  → checkpoint: $CKPT"
}

train "$T/小学.tok1" 2 "小学 2 epochs (从头)"
train "$T/初中.tok1" 3 "初中 3 epochs (续小学)"

echo "等待大学 TOK1预处理完成..."
wait $UNI_PID

train "$T/高中.tok1" 3 "高中 3 epochs (续初中)"
train "$T/university.tok1" 5 "大学 5 epochs (续高中)"

echo; echo "========================================"
echo "✅ 全部课程训练完成!"
ls -la $O/model_final.nfv1 2>/dev/null

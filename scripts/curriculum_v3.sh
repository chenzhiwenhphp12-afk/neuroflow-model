#!/bin/bash
set -e
B=/mnt/d/neuroflow-C++
O=$B/output
CKPT=
export LD_LIBRARY_PATH=$HOME/local/openblas/usr/lib/x86_64-linux-gnu/openblas-openmp:$LD_LIBRARY_PATH
N=/tmp/numactl-extracted/usr/bin/numactl
RUN() { OPENBLAS_NUM_THREADS=10 OMP_NUM_THREADS=20 $N --cpunodebind=0 --membind=0 $B/neuroflow_train_v2 --config $B/configs/config.json --tokenizer $B/configs/tokenizer_cn_013.json --output $O --batch-size 64 --lr 1e-4 --log-interval 10 --save-interval 1000 --grad-clip 1.0 "$@"; }

mkdir -p $O
echo "===== 小学 2 epochs =====" && RUN --data /mnt/d/语料/小学 --epochs 2
CKPT=$(ls -td $O/checkpoint_step*/checkpoint_step* 2>/dev/null | head -1)
echo "===== 初中 3 epochs =====" && RUN --data /mnt/d/语料/初中 --epochs 5 --resume $CKPT
CKPT=$(ls -td $O/checkpoint_step*/checkpoint_step* 2>/dev/null | head -1)
echo "===== 高中 3 epochs =====" && RUN --data /mnt/d/语料/高中 --epochs 8 --resume $CKPT
CKPT=$(ls -td $O/checkpoint_step*/checkpoint_step* 2>/dev/null | head -1)
echo "===== 大学 5 epochs =====" && RUN --data /mnt/d/语料/大学 --epochs 13 --resume $CKPT
echo; echo "✅ 全部完成!"
ls -la $O/model_final.nfv1 2>/dev/null

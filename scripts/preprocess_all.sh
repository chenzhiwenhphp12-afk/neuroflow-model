#!/bin/bash
T=/tmp/neuroflow_tok1
S=/mnt/d/neuroflow-C++/scripts/preprocess_corpus.py
K=/mnt/d/neuroflow-C++/configs/tokenizer_cn_013.json
M=/mnt/d/neuroflow-C++/scripts/mix_training_data.py

mkdir -p "$T"

echo ">>> 大学 (后台)"
python3 "$S" process --input "/mnt/d/语料/大学" --tokenizer "$K" --output "$T/university.tok1" --max-seq-len 512 --max-samples 500000 --min-tokens 4 > /tmp/uni_prep.log 2>&1 &
UPID=$!

echo ">>> 初中"
python3 "$S" process --input "/mnt/d/语料/初中" --tokenizer "$K" --output "$T/junior.tok1" --max-seq-len 512 --max-samples 500000 --min-tokens 4
echo "初中 done"

echo ">>> 高中"
python3 "$S" process --input "/mnt/d/语料/高中" --tokenizer "$K" --output "$T/senior.tok1" --max-seq-len 512 --max-samples 500000 --min-tokens 4
echo "高中 done"

echo ">>> 等待大学..."
wait $UPID
echo "大学 done"

echo ">>> 混合"
python3 "$M" --inputs "$T/primary.tok1" "$T/junior.tok1" "$T/senior.tok1" "$T/university.tok1" --weights 1 1.5 2 2.5 --output "$T/mixed.tok1" --max-samples 500000

echo ">>> 全部就绪"
ls -lh "$T/"*.tok1

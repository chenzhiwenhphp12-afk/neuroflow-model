#!/bin/bash
TOK=/mnt/d/neuroflow-C++/configs/tokenizer_128k.json
S=/mnt/d/neuroflow-C++/scripts/preprocess_corpus.py
M=/mnt/d/neuroflow-C++/scripts/mix_training_data.py
T=/tmp/neuroflow_tok1_v2
mkdir -p "$T"

echo ">>> 大学 (后台)"
python3 "$S" process --input "/mnt/d/语料/大学" --tokenizer "$TOK" --output "$T/university.tok1" --max-seq-len 512 --max-samples 500000 --min-tokens 4 > /tmp/prep_uni.log 2>&1 &
UPID=$!

echo ">>> 初中"
python3 "$S" process --input "/mnt/d/语料/初中" --tokenizer "$TOK" --output "$T/junior.tok1" --max-seq-len 512 --max-samples 500000 --min-tokens 4
echo ">>> 高中"
python3 "$S" process --input "/mnt/d/语料/高中" --tokenizer "$TOK" --output "$T/senior.tok1" --max-seq-len 512 --max-samples 500000 --min-tokens 4
echo ">>> 小学"
python3 "$S" process --input "/mnt/d/语料/小学" --tokenizer "$TOK" --output "$T/primary.tok1" --max-seq-len 512 --max-samples 500000 --min-tokens 4

echo ">>> 等待大学..."
wait $UPID

echo ">>> 混合"
python3 "$M" --inputs "$T/primary.tok1" "$T/junior.tok1" "$T/senior.tok1" "$T/university.tok1" --weights 1 1.5 2 2.5 --output "$T/mixed_v2.tok1" --max-samples 500000

echo ">>> TOK1就绪"
ls -lh "$T/"*.tok1 | awk '{print $5, $9}'

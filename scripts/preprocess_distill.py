#!/usr/bin/env python3
import json, sys, os

input_path = sys.argv[1] if len(sys.argv) > 1 else "D:/语料/deepseek蒸馏/teacher_data.jsonl"
output_path = sys.argv[2] if len(sys.argv) > 2 else "D:/neuroflow-C++/data/distill_train.txt"
max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 0

count = 0
skipped = 0
total_chars = 0

with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            prompt = rec.get('prompt', '')
            completion = rec.get('completion', '')
            text = prompt + '\n' + completion
            if len(text) < 20:
                skipped += 1
                continue
            fout.write(text + '\n')
            count += 1
            total_chars += len(text)
            if count % 50000 == 0:
                print(f"  已处理 {count} 条, 跳过 {skipped} 条, {total_chars/1e6:.1f}M 字符", file=sys.stderr)
            if max_samples > 0 and count >= max_samples:
                break
        except json.JSONDecodeError:
            skipped += 1
            continue

print(f"完成: {count} 条训练样本, 跳过 {skipped} 条, 总计 {total_chars/1e6:.1f}M 字符", file=sys.stderr)
print(f"输出: {output_path}", file=sys.stderr)
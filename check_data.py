#!/usr/bin/env python3
"""Check if distill_train.txt was generated from the current teacher_data.jsonl"""
import json, os

jsonl_path = "D:/语料/deepseek蒸馏/teacher_data.jsonl"
txt_path = "D:/neuroflow-C++/data/distill_train.txt"

# Check first few lines of text file
print("=== distill_train.txt first 3 lines ===")
with open(txt_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        print(f"  [{i}] {line[:120]}...")

# Check first JSONL record
print("\n=== teacher_data.jsonl first record ===")
with open(jsonl_path, 'r', encoding='utf-8') as f:
    rec = json.loads(f.readline())
    print(f"  Keys: {list(rec.keys())}")
    print(f"  Subject: {rec.get('subject','')[:60]}")
    print(f"  Prompt: {rec.get('prompt','')[:100]}...")
    print(f"  Completion: {rec.get('completion','')[:100]}...")

# Check if text file starts with the same content as prompt+completion from JSONL
print("\n=== Cross-check ===")
f_jsonl = open(jsonl_path, 'r', encoding='utf-8')
f_txt = open(txt_path, 'r', encoding='utf-8')
first_json = json.loads(f_jsonl.readline())
first_txt_line = f_txt.readline().strip()
expected = (first_json.get('prompt','') + '\n' + first_json.get('completion','')).strip()
match = first_txt_line == expected
print(f"  First txt line matches JSONL prompt+completion: {match}")
if not match:
    print(f"  TXT first line: {first_txt_line[:100]}...")
    print(f"  JSONL expected: {expected[:100]}...")
f_jsonl.close()
f_txt.close()

# Count total samples
print(f"\n=== Statistics ===")
with open(jsonl_path, 'r', encoding='utf-8') as f:
    jsonl_count = sum(1 for _ in f)
print(f"  teacher_data.jsonl: {jsonl_count} lines")

with open(txt_path, 'r', encoding='utf-8') as f:
    txt_count = sum(1 for _ in f)
print(f"  distill_train.txt: {txt_count} lines")

# File sizes
print(f"\n  jsonl size: {os.path.getsize(jsonl_path)/1e6:.1f} MB")
print(f"  txt size: {os.path.getsize(txt_path)/1e6:.1f} MB")

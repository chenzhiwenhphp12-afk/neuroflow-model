#!/usr/bin/env python3
"""Build vocabulary from knowledge base - use os.listdir (faster than subprocess.find)"""
import os, re, json
from collections import Counter

kb_dir = "/mnt/d/neuroflow-model/knowledge_base"
files = [f for f in os.listdir(kb_dir) if f.endswith('.txt')]
print(f"Found {len(files)} files, scanning first 3000...")

word_counts = Counter()
for fname in files[:3000]:
    try:
        with open(os.path.join(kb_dir, fname)) as f:
            text = f.read(500)
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        word_counts.update(set(words))
    except: pass

# Build vocabulary: top 2000
vocab = ['<PAD>', '<UNK>'] + [w for w, _ in word_counts.most_common(1998)]
out_file = "/mnt/d/neuroflow-model/vocab.json"
with open(out_file, 'w') as f:
    json.dump(vocab, f)
print(f"Vocabulary: {len(vocab)} words, unique: {len(word_counts)}")
print(f"Top 20: {vocab[2:22]}")
print(f"Saved to {out_file}")

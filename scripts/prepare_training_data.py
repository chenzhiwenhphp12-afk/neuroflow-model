#!/usr/bin/env python3
"""
NeuroFlow 训练数据预处理工具

将 D:\语料\ 中的原始语料预处理为高效的 .tok1 二进制格式。
存放于 WSL 虚拟盘 (速度 ~200 MB/s vs HDD ~125 MB/s)。

用法:
  python3 scripts/prepare_training_data.py \
    --corpus D:/语料 \
    --tokenizer configs/tokenizer_128k.json \
    --output /home/user/neuroflow_data \
    --max-seq-len 128 \
    --max-samples 5000000

输出目录结构:
  output/
    train.tok1       # 训练数据
    stats.json        # 统计信息
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

# ─── BPE Tokenizer ───────────────────────────────────────────
class BPETokenizer:
    """与 C++ BPETokenizer 兼容的 Python 实现"""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.vocab = config.get('vocab', {})
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = config.get('merges', [])
        self.vocab_size = len(self.vocab)

        # Build merge ranks
        self.merge_ranks = {}
        for i, (a, b) in enumerate(self.merges):
            self.merge_ranks[(a, b)] = i

        # Special tokens
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def apply_bpe(self, token: str) -> str:
        """Apply BPE merges to a single token using priority queue algorithm"""
        if len(token) <= 1 or not self.merge_ranks:
            return token

        symbols = list(token)
        n = len(symbols)

        import heapq

        # Build linked list
        next_link = list(range(1, n)) + [None]
        prev_link = [None] + list(range(0, n - 1))

        pq = []

        def push_pair(i):
            j = next_link[i]
            if j is None:
                return
            pair = (symbols[i], symbols[j])
            rank = self.merge_ranks.get(pair)
            if rank is not None:
                heapq.heappush(pq, (rank, i))

        for i in range(n):
            push_pair(i)

        while pq:
            rank, i = heapq.heappop(pq)
            j = next_link[i]
            if j is None:
                continue
            pair = (symbols[i], symbols[j])
            if self.merge_ranks.get(pair) != rank:
                continue

            # Merge
            symbols[i] = symbols[i] + symbols[j]
            k = next_link[j]
            next_link[i] = k
            if k is not None:
                prev_link[k] = i

            if prev_link[i] is not None:
                push_pair(prev_link[i])
            push_pair(i)

        # Collect result
        result = []
        i = 0
        while i is not None:
            result.append(symbols[i])
            i = next_link[i]
        return ''.join(result)

    def encode(self, text: str, max_len: int = 128) -> list[int]:
        """Encode text to token IDs"""
        ids = [self.bos_id]

        i = 0
        while i < len(text) and len(ids) < max_len - 1:
            # Handle UTF-8 multi-byte
            char = text[i]
            byte_len = 1
            if ord(char) >= 0x80:
                if ord(char) < 0xE0:
                    byte_len = 2
                elif ord(char) < 0xF0:
                    byte_len = 3
                else:
                    byte_len = 4

            byte_seq = text[i:i + byte_len]
            bpe_result = self.apply_bpe(byte_seq)

            if bpe_result in self.vocab:
                ids.append(self.vocab[bpe_result])
            else:
                for c in bpe_result:
                    ids.append(self.vocab.get(c, self.unk_id))
            i += byte_len

        ids.append(self.eos_id)
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text"""
        result = []
        for id in ids:
            if id in (self.pad_id, self.unk_id, self.bos_id, self.eos_id):
                continue
            token = self.id_to_token.get(id, '')
            if token:
                result.append(token)
        return ''.join(result)


# ─── TOK1 Format Writer ─────────────────────────────────────
# TOK1 Binary Format:
#   Magic:     "TOK1" (4 bytes)
#   Version:   uint16 (2 bytes)
#   VocabSize: uint32 (4 bytes)
#   MaxSeqLen: uint32 (4 bytes)
#   NumSamples:uint32 (4 bytes)
#   For each sample:
#     SeqLen:   uint16 (2 bytes)
#     TokenIDs: uint32[SeqLen] (4 bytes each)


class TOK1Writer:
    def __init__(self, path: str, vocab_size: int, max_seq_len: int):
        self.f = open(path, 'wb')
        self.f.write(b'TOK1')
        self.f.write(struct.pack('<H', 1))  # version
        self.f.write(struct.pack('<I', vocab_size))
        self.f.write(struct.pack('<I', max_seq_len))
        self.f.write(struct.pack('<I', 0))  # num_samples placeholder
        self.count = 0
        self.max_seq_len = max_seq_len
        self._pos_count = self.f.tell()

    def add(self, token_ids: list[int]):
        # Truncate if needed
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        self.f.write(struct.pack('<H', len(token_ids)))
        self.f.write(struct.pack(f'<{len(token_ids)}I', *token_ids))
        self.count += 1

    def close(self):
        # Update num_samples
        self.f.seek(self._pos_count - 4)  # back to num_samples field
        self.f.write(struct.pack('<I', self.count))
        self.f.close()
        return self.count


# ─── Corpus Scanner ─────────────────────────────────────────
def scan_corpus(corpus_root: str) -> list[Path]:
    """Scan corpus directory and return list of text files"""
    root = Path(corpus_root)
    if not root.exists():
        print(f"Error: corpus root not found: {corpus_root}")
        sys.exit(1)

    supported_exts = {'.txt', '.json', '.jsonl', '.csv', '.tsv', '.md'}
    files = []
    for ext in supported_exts:
        found = list(root.rglob(f'*{ext}'))
        files.extend(found)
        if found:
            print(f"  {ext}: {len(found)} files")

    # Sort for reproducibility
    files.sort()
    return files


def extract_text_from_file(filepath: Path) -> list[str]:
    """Extract text content from various file formats"""
    texts = []
    ext = filepath.suffix.lower()

    try:
        if ext == '.txt' or ext == '.md':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Split into paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) >= 10]
            texts.extend(paragraphs)

        elif ext == '.json':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Try JSON array
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for key in ('text', 'content', 'title', 'question', 'answer'):
                                if key in item and isinstance(item[key], str):
                                    texts.append(item[key])
                elif isinstance(data, dict):
                    for key in ('text', 'content', 'title', 'question', 'answer'):
                        if key in data and isinstance(data[key], str):
                            texts.append(data[key])
            except json.JSONDecodeError:
                # Fallback: treat as plain text
                if len(content) >= 10:
                    texts.append(content)

        elif ext == '.jsonl':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        item = json.loads(line)
                        for key in ('text', 'content', 'title', 'question', 'answer'):
                            if key in item and isinstance(item[key], str):
                                texts.append(item[key])
                                break
                    except json.JSONDecodeError:
                        pass

        elif ext in ('.csv', '.tsv'):
            delim = '\t' if ext == '.tsv' else ','
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Take longest field as text
                    fields = line.split(delim)
                    if fields:
                        longest = max(fields, key=len)
                        if len(longest) >= 10:
                            texts.append(longest)
    except Exception as e:
        pass  # Skip problematic files

    return texts


# ─── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='NeuroFlow Training Data Preparer')
    parser.add_argument('--corpus', required=True, help='Corpus root directory')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer config path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--max-seq-len', type=int, default=128, help='Max sequence length')
    parser.add_argument('--max-samples', type=int, default=5000000, help='Max total samples')
    parser.add_argument('--min-text-len', type=int, default=10, help='Minimum text length')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load tokenizer
    print(f"\n{'='*60}")
    print(f"NeuroFlow 训练数据预处理")
    print(f"{'='*60}")
    print(f"语料根目录: {args.corpus}")
    print(f"输出目录:   {args.output}")
    print(f"最大序列:   {args.max_seq_len}")
    print(f"最大样本:   {args.max_samples:,}")
    print()

    print("加载分词器...")
    tok = BPETokenizer(args.tokenizer)
    print(f"  词表大小: {tok.vocab_size}")
    print(f"  Merges:   {len(tok.merges)}")

    # Scan corpus
    print("\n扫描语料...")
    files = scan_corpus(args.corpus)
    print(f"总计: {len(files):,} 文件")

    # Process files
    print(f"\n开始处理...")
    t0 = time.time()
    total_chars = 0
    total_samples = 0
    skipped = 0

    train_path = os.path.join(args.output, 'train.tok1')
    writer = TOK1Writer(train_path, tok.vocab_size, args.max_seq_len)

    for i, filepath in enumerate(files):
        texts = extract_text_from_file(filepath)

        for text in texts:
            if len(text) < args.min_text_len:
                skipped += 1
                continue

            try:
                ids = tok.encode(text, args.max_seq_len)
                if len(ids) >= 4:  # at least bos + 2 tokens + eos
                    writer.add(ids)
                    total_samples += 1
                    total_chars += len(text)
            except Exception:
                skipped += 1
                continue

            if total_samples >= args.max_samples:
                break

        if total_samples >= args.max_samples:
            break

        # Progress report
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = total_samples / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(files)}] "
                  f"samples={total_samples:,} "
                  f"chars={total_chars:,} "
                  f"rate={rate:.0f} samples/s "
                  f"elapsed={elapsed:.1f}s")

    final_count = writer.close()
    elapsed = time.time() - t0

    # Stats
    file_size = os.path.getsize(train_path)
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"{'='*60}")
    print(f"  样本数:    {final_count:,}")
    print(f"  字符数:    {total_chars:,}")
    print(f"  跳过:      {skipped:,}")
    print(f"  文件大小:  {file_size / (1024**3):.2f} GB")
    print(f"  耗时:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  速率:      {final_count/elapsed:.0f} samples/s")
    print(f"  输出:      {train_path}")

    # Save stats
    stats = {
        'num_samples': final_count,
        'total_chars': total_chars,
        'skipped': skipped,
        'file_size_bytes': file_size,
        'vocab_size': tok.vocab_size,
        'max_seq_len': args.max_seq_len,
        'corpus_root': args.corpus,
        'elapsed_seconds': elapsed,
    }
    stats_path = os.path.join(args.output, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计:      {stats_path}")


if __name__ == '__main__':
    main()

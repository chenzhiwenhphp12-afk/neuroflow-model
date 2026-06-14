#!/usr/bin/env python3
"""
mix_training_data.py — 多TOK1文件按比例混合

用法:
  # 等比例混合
  python mix_training_data.py --inputs a.tok1 b.tok1 c.tok1 --output mixed.tok1

  # 加权混合（小学1份, 初中1.5份, 高中2份）
  python mix_training_data.py --inputs a.tok1 b.tok1 c.tok1 --weights 1 1.5 2 --output mixed.tok1
"""
import struct, sys, os, random
from pathlib import Path

MAGIC = b'TOK1'
FOOTER = b'END\x00'

def read_tok1(path):
    """读取TOK1文件, 返回(version, vocab_size, max_seq_len, samples)"""
    samples = []
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        version = struct.unpack('<H', f.read(2))[0]
        vocab_size = struct.unpack('<I', f.read(4))[0]
        max_seq_len = struct.unpack('<I', f.read(4))[0]
        total = struct.unpack('<I', f.read(4))[0]
        for _ in range(total):
            seq_len = struct.unpack('<H', f.read(2))[0]
            ids = list(struct.unpack(f'<{seq_len}I', f.read(seq_len * 4)))
            samples.append(ids)
        footer = f.read(4)
        assert footer == FOOTER, f"Bad footer: {footer}"
    return version, vocab_size, max_seq_len, samples

def write_tok1(path, version, vocab_size, max_seq_len, samples):
    """写入混合后的TOK1文件"""
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<H', version))
        f.write(struct.pack('<I', vocab_size))
        f.write(struct.pack('<I', max_seq_len))
        f.write(struct.pack('<I', len(samples)))
        for ids in samples:
            f.write(struct.pack('<H', len(ids)))
            f.write(struct.pack(f'<{len(ids)}I', *ids))
        f.write(FOOTER)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='多TOK1文件按比例混合')
    parser.add_argument('--inputs', nargs='+', required=True, help='输入TOK1文件列表')
    parser.add_argument('--output', required=True, help='输出混合TOK1文件')
    parser.add_argument('--weights', nargs='+', type=float, default=None, help='各文件采样权重(默认等比例)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max-samples', type=int, default=500000, help='最大样本数')
    args = parser.parse_args()

    random.seed(args.seed)

    # 读取所有TOK1
    all_data = []
    total_samples = 0
    for path in args.inputs:
        version, vocab, max_seq, samples = read_tok1(path)
        all_data.append((path, samples))
        total_samples += len(samples)
        sz = os.path.getsize(path)
        print(f"  {Path(path).name}: {len(samples)} samples, {sz/1024/1024:.0f}MB")

    # 权重
    n = len(all_data)
    weights = args.weights or [1.0] * n
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    # 按加权比例计算每份采样数
    total_out = min(total_samples, args.max_samples)
    counts = [int(total_out * w) for w in weights]

    # 确保至少各1条
    for i in range(n):
        counts[i] = max(1, counts[i])
    # 调整到总数
    diff = total_out - sum(counts)
    counts[-1] += diff

    # 从每份中随机采样
    mixed = []
    for (path, samples), cnt in zip(all_data, counts):
        picked = random.sample(samples, min(cnt, len(samples)))
        mixed.extend(picked)
        print(f"  取 {Path(path).name}: {len(picked)} 条 (权{weights[i]:.2f})")

    random.shuffle(mixed)
    print(f"\n混合后: {len(mixed)} 条样本")

    # 使用第一个文件的版本参数
    _, v, ms, _ = read_tok1(args.inputs[0])
    write_tok1(args.output, 1, v, ms, mixed[:total_out])
    print(f"写出: {args.output} ({os.path.getsize(args.output)/1024/1024:.0f}MB)")

if __name__ == '__main__':
    main()

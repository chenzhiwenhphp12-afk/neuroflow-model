"""
Vocab Minification Tool for NeuroFlow NFv1.1

Scans training corpus to identify which tokens in the 128K vocabulary
actually appear. Generates:
1. active_vocab_mask.json  - binary mask (1=active, 0=cold) for gradient gating
2. vocab_coverage.json     - coverage statistics

Usage:
    python vocab_minification.py --corpus <corpus_file> --tokenizer <tokenizer.json> --output <output_dir>
    python vocab_minification.py --corpus data/train.tok1 --tokenizer configs/tokenizer_cn_013.json --output configs/
"""

import json
import argparse
import os
import sys
from collections import Counter


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def scan_tok1_corpus(corpus_path, vocab_size):
    token_counts = Counter()
    total_tokens = 0
    line_count = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tokens = [int(x) for x in line.split()]
                for t in tokens:
                    if 0 <= t < vocab_size:
                        token_counts[t] += 1
                        total_tokens += 1
            except ValueError:
                continue
            line_count += 1
            if line_count % 10000 == 0:
                print(f"  Scanned {line_count} lines, {total_tokens} tokens, {len(token_counts)} unique", flush=True)

    return token_counts, total_tokens, line_count


def scan_text_corpus(corpus_path, tokenizer_data):
    from preprocess_corpus import BPETokenizer
    tokenizer = BPETokenizer(tokenizer_data)
    vocab_size = tokenizer_data.get('vocab_size', 128000)

    token_counts = Counter()
    total_tokens = 0
    line_count = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = tokenizer.encode(line)
            for t in tokens:
                if 0 <= t < vocab_size:
                    token_counts[t] += 1
                    total_tokens += 1
            line_count += 1
            if line_count % 10000 == 0:
                print(f"  Scanned {line_count} lines, {total_tokens} tokens, {len(token_counts)} unique", flush=True)

    return token_counts, total_tokens, line_count


def generate_mask(token_counts, vocab_size, min_freq=1):
    mask = [0] * vocab_size
    active_count = 0
    for token_id in range(vocab_size):
        if token_counts.get(token_id, 0) >= min_freq:
            mask[token_id] = 1
            active_count += 1
    return mask, active_count


def main():
    parser = argparse.ArgumentParser(description='NeuroFlow Vocab Minification')
    parser.add_argument('--corpus', required=True, help='Path to training corpus')
    parser.add_argument('--tokenizer', default='configs/tokenizer_cn_013.json', help='Tokenizer config')
    parser.add_argument('--output', default='configs/', help='Output directory')
    parser.add_argument('--min-freq', type=int, default=1, help='Minimum token frequency to be active')
    parser.add_argument('--format', choices=['tok1', 'text'], default='tok1', help='Corpus format')
    args = parser.parse_args()

    tokenizer_data = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer_data.get('vocab_size', 128000)
    print(f"Vocab size: {vocab_size}")
    print(f"Scanning corpus: {args.corpus} (format: {args.format})")

    if args.format == 'tok1':
        token_counts, total_tokens, line_count = scan_tok1_corpus(args.corpus, vocab_size)
    else:
        token_counts, total_tokens, line_count = scan_text_corpus(args.corpus, tokenizer_data)

    print(f"\nScan complete:")
    print(f"  Lines: {line_count}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Unique tokens: {len(token_counts)}")
    print(f"  Coverage: {len(token_counts)/vocab_size*100:.1f}%")

    mask, active_count = generate_mask(token_counts, vocab_size, args.min_freq)
    cold_count = vocab_size - active_count

    print(f"\nMask generated (min_freq={args.min_freq}):")
    print(f"  Active tokens: {active_count} ({active_count/vocab_size*100:.1f}%)")
    print(f"  Cold tokens:   {cold_count} ({cold_count/vocab_size*100:.1f}%)")

    os.makedirs(args.output, exist_ok=True)

    mask_path = os.path.join(args.output, 'active_vocab_mask.json')
    with open(mask_path, 'w', encoding='utf-8') as f:
        json.dump({
            'vocab_size': vocab_size,
            'active_count': active_count,
            'cold_count': cold_count,
            'min_freq': args.min_freq,
            'mask': mask
        }, f)
    print(f"  Saved: {mask_path}")

    coverage_path = os.path.join(args.output, 'vocab_coverage.json')
    top_tokens = token_counts.most_common(100)
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump({
            'vocab_size': vocab_size,
            'total_tokens': total_tokens,
            'unique_tokens': len(token_counts),
            'coverage_pct': len(token_counts) / vocab_size * 100,
            'active_count': active_count,
            'cold_count': cold_count,
            'top_100': [[str(t), c] for t, c in top_tokens],
            'freq_distribution': {
                '1_occurrence': sum(1 for c in token_counts.values() if c == 1),
                '2_10_occurrences': sum(1 for c in token_counts.values() if 2 <= c <= 10),
                '11_100_occurrences': sum(1 for c in token_counts.values() if 11 <= c <= 100),
                '100_plus': sum(1 for c in token_counts.values() if c > 100),
            }
        }, f, indent=2)
    print(f"  Saved: {coverage_path}")

    print(f"\nMemory savings estimate:")
    embed_params = vocab_size * 512
    cold_params = cold_count * 512
    print(f"  Embedding params: {embed_params:,} ({embed_params*4/1024/1024:.1f} MB)")
    print(f"  Cold token params: {cold_params:,} ({cold_params*4/1024/1024:.1f} MB)")
    print(f"  Gradient zeroed: {cold_params:,} params (saves {cold_params*4/1024/1024:.1f} MB gradient memory)")


if __name__ == '__main__':
    main()
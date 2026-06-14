"""
NeuroFlow 128K BPE Tokenizer Trainer

从真实语料训练完整的128K BPE词表，覆盖：
- 中文汉字 + 常见子词
- 英文子词
- 代码token
- 数字、标点、符号
- 多语言字符片段

用法:
    # 第1步: 从语料提取词频
    python train_128k_tokenizer.py --corpus data/mixed_train.txt --output configs/tokenizer_128k.json

    # 第2步: 指定已有词表大小
    python train_128k_tokenizer.py --corpus data/mixed_train.txt --output configs/tokenizer_128k.json --vocab-size 128000

输入格式: 纯文本，每行一个句子/段落
输出格式: NeuroFlow BPE tokenizer JSON (兼容 tokenizer_cn_013.json 格式)
"""

import json
import argparse
import os
import sys
import re
from collections import Counter, defaultdict


def extract_initial_vocab(corpus_path, min_char_freq=2):
    """从语料中提取初始字符级词表"""
    char_counts = Counter()
    line_count = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for ch in line:
                char_counts[ch] += 1
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  字符统计: {line_count} 行, {len(char_counts)} 唯一字符", flush=True)

    vocab = {}
    idx = 4  # 0-3 reserved for <pad>, <s>, </s>, <unk>

    # 特殊token
    special_tokens = ['<pad>', '<s>', '</s>', '<unk>']
    for i, t in enumerate(special_tokens):
        vocab[t] = i

    # 高频字符
    for ch, count in char_counts.most_common():
        if count >= min_char_freq and ch not in vocab:
            vocab[ch] = idx
            idx += 1

    print(f"  初始字符词表: {len(vocab)} 个token (来自 {line_count} 行语料)")
    return vocab, char_counts


def pretokenize(text):
    """预分词：按空格和标点切分，保留数字和代码结构"""
    pattern = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d|"""
        r"""[^\W\d_]+|"""          # 字母单词
        r"""\d+|"""                # 数字
        r"""[^\s\w]+|"""           # 标点符号
        r"""\s+|"""                # 空白
        r"""[\u4e00-\u9fff]|"""    # 单个CJK汉字
        r"""[\u3400-\u4dbf]|"""    # CJK扩展A
        r"""[\uf900-\ufaff]|"""    # CJK兼容
        r"""[\u3040-\u309f]|"""    # 平假名
        r"""[\u30a0-\u30ff]|"""    # 片假名
        r"""[\uac00-\ud7af]""",    # 韩文
        re.UNICODE
    )
    return pattern.findall(text)


def train_bpe(corpus_path, vocab_size=128000, min_char_freq=2):
    """训练BPE词表"""
    print(f"=== 训练128K BPE词表 ===")
    print(f"语料: {corpus_path}")
    print(f"目标词表大小: {vocab_size}")

    # 第1步: 提取初始字符词表
    print("\n[1/3] 提取初始字符词表...")
    vocab, char_counts = extract_initial_vocab(corpus_path, min_char_freq)

    if len(vocab) >= vocab_size:
        print(f"  初始词表已超过目标大小，截断到 {vocab_size}")
        vocab = dict(list(vocab.items())[:vocab_size])
        return vocab, []

    # 第2步: 统计词对频率
    print("\n[2/3] 统计BPE词对...")
    pair_counts = Counter()
    word_freqs = Counter()
    line_count = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = pretokenize(line)
            for token in tokens:
                word_freqs[token] += 1
                chars = list(token)
                for i in range(len(chars) - 1):
                    pair_counts[(chars[i], chars[i+1])] += 1
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  词对统计: {line_count} 行, {len(pair_counts)} 唯一词对", flush=True)

    print(f"  唯一token: {len(word_freqs)}")
    print(f"  唯一词对: {len(pair_counts)}")

    # 第3步: 迭代合并最高频词对
    print(f"\n[3/3] BPE合并 (目标: {vocab_size} tokens)...")
    merges = []
    current_vocab_size = len(vocab)
    merge_round = 0

    while current_vocab_size < vocab_size and pair_counts:
        best_pair = pair_counts.most_common(1)[0][0]
        best_count = pair_counts[best_pair]

        if best_count < 2:
            print(f"  词对频率低于2，停止合并 (已合并 {len(merges)} 次)")
            break

        new_token = best_pair[0] + best_pair[1]
        if new_token not in vocab:
            vocab[new_token] = current_vocab_size
            merges.append(best_pair[0] + ' ' + best_pair[1])
            current_vocab_size += 1

        # 更新词对计数（简化：移除已合并的词对）
        del pair_counts[best_pair]

        merge_round += 1
        if merge_round % 10000 == 0:
            print(f"  合并进度: {current_vocab_size}/{vocab_size} ({current_vocab_size/vocab_size*100:.1f}%)", flush=True)

    # 如果还没到128K，用常见子词填充
    if current_vocab_size < vocab_size:
        print(f"\n  BPE合并后: {current_vocab_size} tokens，需填充 {vocab_size - current_vocab_size} 个")
        print("  从语料中提取高频子词填充...")

        subword_counts = Counter()
        for token, freq in word_freqs.most_common(500000):
            if len(token) >= 2 and token not in vocab:
                subword_counts[token] += freq
            if len(subword_counts) >= vocab_size * 2:
                break

        for subword, freq in subword_counts.most_common():
            if current_vocab_size >= vocab_size:
                break
            if subword not in vocab:
                vocab[subword] = current_vocab_size
                current_vocab_size += 1

    # 如果还不够，用数字和常见模式填充
    if current_vocab_size < vocab_size:
        print(f"  子词填充后: {current_vocab_size} tokens，用模式填充剩余...")
        patterns = []
        # 常见代码token
        code_tokens = [
            'def', 'class', 'return', 'import', 'from', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'with', 'as', 'not', 'and', 'or',
            'True', 'False', 'None', 'self', 'super', 'yield', 'lambda', 'pass',
            'break', 'continue', 'global', 'nonlocal', 'assert', 'del', 'raise',
            'async', 'await', 'print', 'input', 'range', 'len', 'str', 'int',
            'float', 'list', 'dict', 'set', 'tuple', 'bool', 'type', 'isinstance',
            'function', 'const', 'var', 'let', 'new', 'delete', 'this', 'that',
            'public', 'private', 'protected', 'static', 'virtual', 'override',
            'template', 'namespace', 'using', 'struct', 'enum', 'typedef',
            'void', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t',
            'size_t', 'float32', 'float64', 'auto', 'nullptr', 'constexpr',
            '#include', '#define', '#ifdef', '#ifndef', '#endif', '#pragma',
            '->', '=>', '::', '&&', '||', '!=', '==', '<=', '>=', '+=', '-=',
            '*=', '/=', '++', '--', '<<', '>>', '...', '??', '?.',
        ]
        patterns.extend(code_tokens)

        # 常见英文子词
        common_prefixes = [
            'un', 're', 'pre', 'dis', 'mis', 'over', 'out', 'sub', 'inter',
            'trans', 'anti', 'semi', 'multi', 'mini', 'macro', 'micro',
            'super', 'hyper', 'ultra', 'meta', 'proto', 'neo', 'pseudo',
        ]
        common_suffixes = [
            'tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful', 'less',
            'ous', 'ive', 'al', 'ial', 'ic', 'ical', 'ly', 'ally', 'ing',
            'ed', 'er', 'est', 'ize', 'ise', 'ify', 'ate', 'en', 'dom',
            'ship', 'hood', 'ward', 'wards', 'wise', 'like', 'worth',
        ]
        for p in common_prefixes:
            for s in common_suffixes:
                patterns.append(p + s)

        for p in patterns:
            if current_vocab_size >= vocab_size:
                break
            if p not in vocab:
                vocab[p] = current_vocab_size
                current_vocab_size += 1

        # 最后用编号填充
        while current_vocab_size < vocab_size:
            token = f'<extra_{current_vocab_size}>'
            vocab[token] = current_vocab_size
            current_vocab_size += 1

    print(f"\n=== 训练完成 ===")
    print(f"最终词表大小: {len(vocab)}")
    print(f"BPE合并数: {len(merges)}")

    # 统计词表组成
    categories = defaultdict(int)
    for token in vocab:
        if token.startswith('<') and token.endswith('>'):
            categories['special'] += 1
        elif re.match(r'^[\u4e00-\u9fff]$', token):
            categories['cjk_single'] += 1
        elif re.match(r'^[\u4e00-\u9fff]+', token):
            categories['cjk_multi'] += 1
        elif re.match(r'^[a-zA-Z]+$', token):
            categories['latin'] += 1
        elif re.match(r'^[0-9]+$', token):
            categories['number'] += 1
        else:
            categories['other'] += 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(vocab)*100:.1f}%)")

    return vocab, merges


def main():
    parser = argparse.ArgumentParser(description='NeuroFlow 128K BPE Tokenizer Trainer')
    parser.add_argument('--corpus', required=True, help='训练语料(纯文本)')
    parser.add_argument('--output', default='configs/tokenizer_128k.json', help='输出词表文件')
    parser.add_argument('--vocab-size', type=int, default=128000, help='目标词表大小')
    parser.add_argument('--min-char-freq', type=int, default=2, help='字符最低频率')
    args = parser.parse_args()

    vocab, merges = train_bpe(args.corpus, args.vocab_size, args.min_char_freq)

    output = {
        'model_type': 'bpe',
        'vocab_size': len(vocab),
        'special_tokens': {
            '<pad>': 0,
            '<s>': 1,
            '</s>': 2,
            '<unk>': 3,
        },
        'vocab': vocab,
        'merges': merges,
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n词表已保存: {args.output}")
    print(f"文件大小: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
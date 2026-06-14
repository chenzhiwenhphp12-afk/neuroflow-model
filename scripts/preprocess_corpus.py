"""
preprocess_corpus.py — 语料预处理：文本 → TOK1 二进制格式

TOK1 格式规范：
  Magic:    4 bytes "TOK1"
  Version:  2 bytes (uint16 LE) = 1
  Header:   4 bytes (uint32 LE) = tokenizer_vocab_size
            4 bytes (uint32 LE) = max_seq_len
            4 bytes (uint32 LE) = total_sample_count (占位，写完回填)
  Samples:  [2 bytes seq_len (uint16 LE)] [seq_len * 4 bytes token_ids (uint32 LE each)] ...
  Footer:   4 bytes "END\0"

支持输入格式：txt, json, jsonl, csv, 目录递归
"""

import os
import sys
import json
import struct
import argparse
import logging
from pathlib import Path
from collections.abc import Iterator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MAGIC = b'TOK1'
VERSION = 1
FOOTER = b'END\x00'

SPECIAL_TOKENS = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}


class SimpleBPETokenizer:
    """轻量级BPE分词器，兼容tokenizer_cn_013.json格式"""

    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'model' in data:
            self.vocab = data['model'].get('vocab', {})
            self.merges = data['model'].get('merges', [])
        elif isinstance(data, dict) and 'vocab' in data:
            self.vocab = data['vocab']
            self.merges = data.get('merges', [])
        else:
            self.vocab = data if isinstance(data, dict) else {}
            self.merges = []

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        self.merge_ranks = {}
        for i, merge in enumerate(self.merges):
            if isinstance(merge, list) and len(merge) == 2:
                pair = (merge[0], merge[1])
            elif isinstance(merge, str) and ' ' in merge:
                parts = merge.split(' ', 1)
                pair = (parts[0], parts[1])
            else:
                continue
            self.merge_ranks[pair] = i

        logger.info(f"词表加载: vocab_size={self.vocab_size}, merges={len(self.merge_ranks)}")

    def _tokenize_word(self, word: str) -> list:
        tokens = list(word) if len(word) > 0 else []
        if not tokens:
            return tokens

        while len(tokens) >= 2:
            best_pair = None
            best_rank = float('inf')
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_pos = i

            if best_pair is None or best_rank == float('inf'):
                break

            tokens[best_pos] = best_pair[0] + best_pair[1]
            tokens.pop(best_pos + 1)

        return tokens

    def encode(self, text: str, max_seq_len: int = 128) -> list:
        if not text or not text.strip():
            return []

        text = text.strip()
        words = []
        current = []
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                if current:
                    words.append(''.join(current))
                    current = []
                words.append(ch)
            elif ch.isspace():
                if current:
                    words.append(''.join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            words.append(''.join(current))

        token_ids = [SPECIAL_TOKENS.get("<s>", 1)]
        for word in words:
            sub_tokens = self._tokenize_word(word)
            for st in sub_tokens:
                tid = self.vocab.get(st, SPECIAL_TOKENS.get("<unk>", 3))
                token_ids.append(tid)
                if len(token_ids) >= max_seq_len - 1:
                    break
            if len(token_ids) >= max_seq_len - 1:
                break

        token_ids.append(SPECIAL_TOKENS.get("</s>", 2))
        return token_ids[:max_seq_len]


def read_texts_from_json(file_path: str) -> Iterator[str]:
    """流式读取JSON文件中的文本字段"""
    with open(file_path, 'r', encoding='utf-8') as f:
        brace_depth = 0
        record = ''
        for line in f:
            for ch in line:
                if ch == '{':
                    if brace_depth == 0:
                        record = ''
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and record:
                        record += '}'
                        for text in _extract_texts_from_record(record):
                            yield text
                        record = ''
                if brace_depth > 0:
                    record += ch


def _extract_texts_from_record(record: str) -> Iterator[str]:
    """从JSON record中提取文本字段"""
    for field in ['"text"', '"content"', '"title"', '"question"', '"answer"']:
        pos = record.find(field)
        if pos == -1:
            continue
        colon = record.find(':', pos + len(field))
        if colon == -1:
            continue
        colon += 1
        while colon < len(record) and record[colon] != '"':
            colon += 1
        if colon >= len(record):
            continue
        colon += 1
        end = colon
        while end < len(record) and record[end] != '"':
            if record[end] == '\\':
                end += 1
            end += 1
        text = record[colon:end]
        text = _unescape_json(text)
        if len(text) >= 10:
            yield text


def _unescape_json(s: str) -> str:
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')


def read_texts_from_jsonl(file_path: str) -> Iterator[str]:
    """逐行读取JSONL文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                for text in _extract_texts_from_record(line):
                    yield text
                continue
            for field in ['text', 'content', 'title', 'question', 'answer']:
                if field in obj and isinstance(obj[field], str) and len(obj[field]) >= 10:
                    yield obj[field]


def read_texts_from_txt(file_path: str) -> Iterator[str]:
    """按段落读取纯文本文件"""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        paragraph = ''
        for line in f:
            if line.strip() == '':
                if len(paragraph) >= 10:
                    yield paragraph.strip()
                paragraph = ''
            else:
                paragraph += line
            if len(paragraph) > 10000:
                yield paragraph.strip()
                paragraph = ''
        if len(paragraph) >= 10:
            yield paragraph.strip()


def read_texts_from_csv(file_path: str) -> Iterator[str]:
    """读取CSV，第一列作为文本"""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',', 1)
            text = parts[0].strip().strip('"')
            if len(text) >= 10:
                yield text


def read_texts(file_path: str) -> Iterator[str]:
    """根据扩展名自动分派读取器"""
    ext = Path(file_path).suffix.lower()
    if ext == '.json':
        return read_texts_from_json(file_path)
    elif ext == '.jsonl':
        return read_texts_from_jsonl(file_path)
    elif ext == '.txt':
        return read_texts_from_txt(file_path)
    elif ext in ('.csv', '.tsv'):
        return read_texts_from_csv(file_path)
    else:
        logger.warning(f"跳过不支持的格式: {file_path}")
        return iter([])


def read_texts_recursive(path: str) -> Iterator[tuple]:
    """递归遍历目录，返回 (文件路径, 文本) 对"""
    p = Path(path)
    if p.is_file():
        for text in read_texts(str(p)):
            yield (str(p), text)
    elif p.is_dir():
        files = sorted(p.rglob('*'))
        for f in files:
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in ('.json', '.jsonl', '.txt', '.csv', '.tsv'):
                logger.info(f"读取: {f}")
                try:
                    for text in read_texts(str(f)):
                        yield (str(f), text)
                except Exception as e:
                    logger.warning(f"跳过 {f}: {e}")
    else:
        raise ValueError(f"路径不存在: {path}")


def write_tok1(output_path: str, tokenizer: SimpleBPETokenizer,
                text_iter: Iterator[tuple], max_seq_len: int = 128,
                max_samples: int = 0, min_tokens: int = 4):
    """将文本流写入TOK1二进制格式"""
    total_written = 0
    total_skipped = 0

    with open(output_path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<H', VERSION))
        f.write(struct.pack('<I', tokenizer.vocab_size))
        f.write(struct.pack('<I', max_seq_len))
        count_offset = f.tell()
        f.write(struct.pack('<I', 0))

        for file_path, text in text_iter:
            token_ids = tokenizer.encode(text, max_seq_len)
            if len(token_ids) < min_tokens:
                total_skipped += 1
                continue

            seq_len = len(token_ids)
            f.write(struct.pack('<H', seq_len))
            for tid in token_ids:
                f.write(struct.pack('<I', tid))

            total_written += 1
            if total_written % 10000 == 0:
                logger.info(f"已写入 {total_written} 样本, 跳过 {total_skipped}")

            if max_samples > 0 and total_written >= max_samples:
                logger.info(f"达到采样上限 {max_samples}，停止")
                break

        f.write(FOOTER)

        f.seek(count_offset)
        f.write(struct.pack('<I', total_written))

    file_size = os.path.getsize(output_path)
    logger.info(f"写入完成: {total_written} 样本, 跳过 {total_skipped}")
    logger.info(f"输出文件: {output_path} ({file_size / 1024 / 1024:.1f} MB)")


def split_large_json(input_path: str, output_dir: str, chunk_size: int = 50000):
    """将大型JSON文件拆分为JSONL分片"""
    os.makedirs(output_dir, exist_ok=True)
    chunk_idx = 0
    count = 0
    out = None

    for text in read_texts_from_json(input_path):
        if out is None:
            chunk_path = os.path.join(output_dir, f'chunk_{chunk_idx:04d}.jsonl')
            out = open(chunk_path, 'w', encoding='utf-8')
            logger.info(f"开始分片: {chunk_path}")

        obj = json.dumps({"text": text}, ensure_ascii=False)
        out.write(obj + '\n')
        count += 1

        if count >= chunk_size:
            out.close()
            logger.info(f"分片 {chunk_idx} 完成: {count} 条")
            chunk_idx += 1
            count = 0
            out = None

    if out:
        out.close()
        logger.info(f"分片 {chunk_idx} 完成: {count} 条")

    logger.info(f"拆分完成: {chunk_idx + 1} 个分片")


def verify_tok1(file_path: str, max_show: int = 5):
    """验证TOK1文件格式"""
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Magic不匹配: {magic}"
        version = struct.unpack('<H', f.read(2))[0]
        vocab_size = struct.unpack('<I', f.read(4))[0]
        max_seq_len = struct.unpack('<I', f.read(4))[0]
        total = struct.unpack('<I', f.read(4))[0]

        logger.info(f"TOK1文件: version={version}, vocab_size={vocab_size}, "
                     f"max_seq_len={max_seq_len}, total_samples={total}")

        shown = 0
        for i in range(total):
            seq_len = struct.unpack('<H', f.read(2))[0]
            ids = []
            for _ in range(seq_len):
                ids.append(struct.unpack('<I', f.read(4))[0])
            if shown < max_show:
                logger.info(f"  样本{i}: len={seq_len}, ids[:10]={ids[:10]}")
                shown += 1

        footer = f.read(4)
        assert footer == FOOTER, f"Footer不匹配: {footer}"

    logger.info("验证通过")


def main():
    parser = argparse.ArgumentParser(description='语料预处理: 文本 → TOK1 二进制格式')
    sub = parser.add_subparsers(dest='command')

    p_process = sub.add_parser('process', help='预处理语料为TOK1格式')
    p_process.add_argument('--input', required=True, help='输入路径(文件或目录)')
    p_process.add_argument('--tokenizer', required=True, help='tokenizer JSON路径')
    p_process.add_argument('--output', required=True, help='输出TOK1文件路径')
    p_process.add_argument('--max-seq-len', type=int, default=128, help='最大序列长度')
    p_process.add_argument('--max-samples', type=int, default=0, help='最大样本数(0=不限)')
    p_process.add_argument('--min-tokens', type=int, default=4, help='最小token数')

    p_split = sub.add_parser('split', help='拆分大型JSON为JSONL分片')
    p_split.add_argument('--input', required=True, help='输入JSON文件')
    p_split.add_argument('--output-dir', required=True, help='输出目录')
    p_split.add_argument('--chunk-size', type=int, default=50000, help='每片条数')

    p_verify = sub.add_parser('verify', help='验证TOK1文件')
    p_verify.add_argument('--input', required=True, help='TOK1文件路径')

    args = parser.parse_args()

    if args.command == 'process':
        tokenizer = SimpleBPETokenizer(args.tokenizer)
        text_iter = read_texts_recursive(args.input)
        write_tok1(args.output, tokenizer, text_iter,
                    max_seq_len=args.max_seq_len,
                    max_samples=args.max_samples,
                    min_tokens=args.min_tokens)
    elif args.command == 'split':
        split_large_json(args.input, args.output_dir, args.chunk_size)
    elif args.command == 'verify':
        verify_tok1(args.input)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
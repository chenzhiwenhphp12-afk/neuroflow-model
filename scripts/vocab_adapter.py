import json
import os
import argparse
import logging

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}


def build_chinese_chars(start_id=4, count=3755):
    vocab = {}
    idx = start_id
    for code in range(0x4E00, 0x4E00 + count):
        if idx >= start_id + count:
            break
        char = chr(code)
        vocab[char] = idx
        idx += 1
    return vocab


def build_english_subwords(start_id, count=500):
    vocab = {}
    idx = start_id
    prefixes = ["un", "re", "pre", "dis", "mis", "over", "under", "out", "sub", "inter", "anti", "non", "semi", "multi", "bi", "co", "ex", "in", "im", "il", "ir"]
    suffixes = ["ing", "ed", "er", "est", "ly", "tion", "sion", "ment", "ness", "ity", "ous", "ive", "able", "ible", "ful", "less", "al", "ial", "ic", "ical"]
    roots = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"]
    for r in roots:
        if idx >= start_id + count:
            break
        vocab[r] = idx
        idx += 1
    for p in prefixes:
        for r in roots[:50]:
            if idx >= start_id + count:
                break
            token = p + r
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    for s in suffixes:
        for r in roots[:50]:
            if idx >= start_id + count:
                break
            token = r + s
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def build_punctuation(start_id):
    vocab = {}
    idx = start_id
    puncts = list("。，、；：！？…—""''（）【】《》·～、.,!?;:\"'()[]{}<>-_/\\|@#$%^&*+=~`")
    for p in puncts:
        vocab[p] = idx
        idx += 1
    return vocab


def build_digits(start_id):
    vocab = {}
    idx = start_id
    for d in "0123456789":
        vocab[d] = idx
        idx += 1
    for combo in ["10", "00", "01", "20", "30", "50", "100", "200", "500", "1000"]:
        vocab[combo] = idx
        idx += 1
    return vocab


def build_merges(vocab, max_merges=2000):
    merges = []
    tokens = sorted(vocab.keys(), key=lambda t: len(t), reverse=True)
    for i, t1 in enumerate(tokens):
        if len(merges) >= max_merges:
            break
        for t2 in tokens[i:i+10]:
            if len(merges) >= max_merges:
                break
            combined = t1 + t2
            if combined in vocab:
                merges.append(f"{t1} {t2}")
    return merges


def generate_tokenizer_json(vocab_size=5000):
    vocab = dict(SPECIAL_TOKENS)
    next_id = len(vocab)

    cn = build_chinese_chars(next_id, 3755)
    vocab.update(cn)
    next_id = max(vocab.values()) + 1

    en = build_english_subwords(next_id, 500)
    vocab.update(en)
    next_id = max(vocab.values()) + 1

    punct = build_punctuation(next_id)
    vocab.update(punct)
    next_id = max(vocab.values()) + 1

    digits = build_digits(next_id)
    vocab.update(digits)

    while len(vocab) > vocab_size:
        max_id = max(vocab.values())
        for k, v in list(vocab.items()):
            if v == max_id:
                del vocab[k]
                break

    merges = build_merges(vocab)

    return {
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "vocab": vocab,
        "merges": merges,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="NeuroFlow BPE词表适配器")
    parser.add_argument("--vocab-size", type=int, default=5000, help="词表大小")
    parser.add_argument("--output", type=str, default="configs/tokenizer_cn_013.json", help="输出路径")
    args = parser.parse_args()

    tokenizer = generate_tokenizer_json(args.vocab_size)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, indent=2, ensure_ascii=False)
    logger.info(f"词表已保存到 {args.output}, vocab_size={len(tokenizer['vocab'])}, merges={len(tokenizer['merges'])}")
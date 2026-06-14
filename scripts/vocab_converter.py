import json
import os
import argparse
import logging

logger = logging.getLogger(__name__)


def export_to_huggingface(nf_json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(nf_json_path, "r", encoding="utf-8") as f:
        nf_data = json.load(f)

    vocab = nf_data.get("vocab", {})
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    merges = nf_data.get("merges", [])
    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for merge in merges:
            f.write(merge + "\n")

    logger.info(f"HuggingFace格式已导出到 {output_dir}")


def import_from_huggingface(vocab_json_path, merges_txt_path, output_path):
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    merges = []
    with open(merges_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            merges.append(line)

    special_tokens = {}
    for name, tid in [("<pad>", 0), ("<s>", 1), ("</s>", 2), ("<unk>", 3)]:
        if name in vocab:
            special_tokens[name] = vocab[name]

    nf_data = {
        "model_type": "bpe",
        "vocab_size": len(vocab),
        "special_tokens": special_tokens,
        "vocab": vocab,
        "merges": merges,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nf_data, f, indent=2, ensure_ascii=False)
    logger.info(f"NeuroFlow格式已导入到 {output_path}")


def verify_roundtrip(nf_json_path):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        export_to_huggingface(nf_json_path, tmpdir)
        roundtrip_path = os.path.join(tmpdir, "roundtrip.json")
        import_from_huggingface(
            os.path.join(tmpdir, "vocab.json"),
            os.path.join(tmpdir, "merges.txt"),
            roundtrip_path,
        )
        with open(nf_json_path, "r", encoding="utf-8") as f:
            original = json.load(f)
        with open(roundtrip_path, "r", encoding="utf-8") as f:
            roundtrip = json.load(f)
        vocab_match = original["vocab"] == roundtrip["vocab"]
        merges_match = original["merges"] == roundtrip["merges"]
        passed = vocab_match and merges_match
        if not passed:
            logger.warning(f"往返验证失败: vocab_match={vocab_match}, merges_match={merges_match}")
        return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="NeuroFlow词表格式转换器")
    subparsers = parser.add_subparsers(dest="command")

    p_export = subparsers.add_parser("export", help="导出为HuggingFace格式")
    p_export.add_argument("--input", type=str, required=True, help="NeuroFlow词表JSON路径")
    p_export.add_argument("--output-dir", type=str, default="configs/huggingface", help="输出目录")

    p_import = subparsers.add_parser("import", help="从HuggingFace格式导入")
    p_import.add_argument("--vocab", type=str, required=True, help="vocab.json路径")
    p_import.add_argument("--merges", type=str, required=True, help="merges.txt路径")
    p_import.add_argument("--output", type=str, default="configs/tokenizer_cn_013.json", help="输出路径")

    p_verify = subparsers.add_parser("verify", help="验证往返无损性")
    p_verify.add_argument("--input", type=str, required=True, help="NeuroFlow词表JSON路径")

    args = parser.parse_args()
    if args.command == "export":
        export_to_huggingface(args.input, args.output_dir)
    elif args.command == "import":
        import_from_huggingface(args.vocab, args.merges, args.output)
    elif args.command == "verify":
        result = verify_roundtrip(args.input)
        print(f"往返验证: {'通过' if result else '失败'}")
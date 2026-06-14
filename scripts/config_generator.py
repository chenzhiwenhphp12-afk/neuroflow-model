import re
import json
import argparse
import logging
import os

logger = logging.getLogger(__name__)

MODEL_CONFIG_FIELDS = [
    ("input_dim", 512, "int", "输入维度(d_model)", "hidden_dim"),
    ("hidden_dim", 256, "int", "隐藏层维度", "hidden_dim"),
    ("output_dim", 10, "int", "输出维度(分类数)", "num_labels"),
    ("memory_dim", 128, "int", "记忆维度(d_mem)", None),
    ("memory_slots", 64, "int", "记忆槽数(MEM_SLOTS)", None),
    ("num_layers", 2, "int", "网络层数", "num_layers"),
    ("num_associations", 8, "int", "DMN关联头数", None),
    ("use_quantization", False, "bool", "是否启用量化", None),
    ("use_mla", False, "bool", "是否启用MLA注意力", "use_mla"),
    ("mla_latent_dim", 32, "int", "MLA潜在维度", None),
    ("use_causal_lm", False, "bool", "是否启用因果LM", None),
    ("vocab_size", 5000, "int", "词表大小(VOCAB_SIZE)", "vocab_size"),
    ("max_seq_len", 128, "int", "最大序列长度", "max_seq_len"),
    ("causal_window_size", 32, "int", "因果窗口大小", None),
    ("sae_k", 64, "int", "SAE稀疏度", None),
    ("ntm_memory_slots", 16, "int", "NTM记忆槽数", None),
]

CAUSAL_LM_FIELDS = [
    ("vocab_size", 5000, "int", "词表大小", "vocab_size"),
    ("d_model", 256, "int", "模型维度", "hidden_dim"),
    ("max_seq_len", 128, "int", "最大序列长度", "max_seq_len"),
    ("causal_window_size", 32, "int", "因果窗口大小", None),
    ("sae_k", 64, "int", "SAE稀疏度", None),
    ("ntm_memory_slots", 16, "int", "NTM记忆槽数", None),
    ("use_mla", True, "bool", "启用MLA", None),
    ("mla_latent_dim", 32, "int", "MLA潜在维度", None),
    ("mla_n_heads", 8, "int", "MLA头数", None),
    ("mla_max_cache_len", 4096, "int", "MLA最大缓存长度", None),
    ("use_quantization", False, "bool", "启用量化", None),
]

GENERATE_FIELDS = [
    ("max_new_tokens", 50, "int", "最大生成token数", None),
    ("temperature", 1.0, "float", "采样温度(0.0,2.0]", None),
    ("top_k", 40, "int", "Top-K采样K值", None),
    ("top_p", 0.9, "float", "Top-P采样P值[0.0,1.0]", None),
    ("repetition_penalty", 1.0, "float", "重复惩罚[1.0,2.0]", None),
    ("punct_penalty", 0.0, "float", "标点惩罚", None),
    ("random_seed", 0, "int", "随机种子(0=随机)", None),
    ("strategy", "top_k", "str", "采样策略(greedy/top_k/top_p/top_k_top_p)", None),
    ("eos_id", 3, "int", "EOS token ID", None),
]


def parse_config_from_hpp(hpp_path, struct_name):
    fields = []
    try:
        with open(hpp_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        pattern = rf'struct\s+{struct_name}\s*\{{([^}}]+)\}}'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            logger.warning(f"未找到结构体 {struct_name}，使用默认字段")
            return None
        body = match.group(1)
        for line in body.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            m = re.match(r'(size_t|bool|float|std::string|int)\s+(\w+)\s*=\s*([^;]+);', line)
            if m:
                ctype, name, default = m.groups()
                default = default.strip().rstrip('f')
                if ctype == "size_t" or ctype == "int":
                    default = int(default)
                elif ctype == "float":
                    default = float(default)
                elif ctype == "bool":
                    default = default == "true"
                elif "SamplingStrategyType" in default:
                    default = default.split("::")[-1].lower()
                fields.append((name, default, ctype))
    except Exception as e:
        logger.warning(f"解析hpp失败: {e}，使用默认字段")
    return fields


def generate_config_json():
    config = {}
    comments = {}
    python_alias = {}
    for name, default, _, comment, alias in MODEL_CONFIG_FIELDS + CAUSAL_LM_FIELDS:
        config[name] = default
        comments[name] = comment
        if alias:
            python_alias[name] = alias
    config["_comment"] = comments
    config["_python_alias"] = python_alias
    return config


def generate_special_tokens_map():
    return {
        "pad_token": "<pad>",
        "pad_token_id": 0,
        "bos_token": "<s>",
        "bos_token_id": 1,
        "eos_token": "</s>",
        "eos_token_id": 2,
        "unk_token": "<unk>",
        "unk_token_id": 3,
        "_comment": {
            "pad_token": "填充token，用于batch对齐",
            "bos_token": "序列起始token",
            "eos_token": "序列结束token，生成时遇到此token停止",
            "unk_token": "未知token，词表外字符映射到此",
        },
    }


def generate_generation_config_json():
    config = {}
    comments = {}
    for name, default, _, comment, _ in GENERATE_FIELDS:
        config[name] = default
        comments[name] = comment
    config["_comment"] = comments
    return config


def validate_config(config, schema_type):
    errors = []
    if schema_type == "config":
        if config.get("vocab_size", 0) <= 0:
            errors.append("vocab_size必须>0")
        if config.get("input_dim", 0) <= 0:
            errors.append("input_dim必须>0")
    elif schema_type == "generation_config":
        t = config.get("temperature", 0)
        if t <= 0 or t > 2.0:
            errors.append(f"temperature须在(0.0,2.0]，当前={t}")
        s = config.get("strategy", "")
        if s not in ("greedy", "top_k", "top_p", "top_k_top_p"):
            errors.append(f"strategy须为greedy/top_k/top_p/top_k_top_p，当前={s}")
    return errors


def save_all_configs(output_dir, model_hpp=None, generative_hpp=None):
    os.makedirs(output_dir, exist_ok=True)

    config = generate_config_json()
    if model_hpp:
        parsed = parse_config_from_hpp(model_hpp, "Config")
        if parsed:
            for name, default, _ in parsed:
                if name in config and name not in ("_comment", "_python_alias"):
                    config[name] = default
    errors = validate_config(config, "config")
    if errors:
        logger.warning(f"config.json校验问题: {errors}")
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    special = generate_special_tokens_map()
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special, f, indent=2, ensure_ascii=False)

    gen_config = generate_generation_config_json()
    if generative_hpp:
        parsed = parse_config_from_hpp(generative_hpp, "GenerateConfig")
        if parsed:
            for name, default, _ in parsed:
                if name in gen_config and name not in ("_comment",):
                    gen_config[name] = default
    errors = validate_config(gen_config, "generation_config")
    if errors:
        logger.warning(f"generation_config.json校验问题: {errors}")
    with open(os.path.join(output_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2, ensure_ascii=False)

    logger.info(f"配置文件已保存到 {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="NeuroFlow配置文件生成器")
    parser.add_argument("--model-hpp", type=str, default="", help="model.hpp路径")
    parser.add_argument("--generative-hpp", type=str, default="", help="generative.hpp路径")
    parser.add_argument("--output-dir", type=str, default="configs", help="输出目录")
    parser.add_argument("--reference-config", type=str, default="", help="Python训练系统参考配置")
    args = parser.parse_args()
    save_all_configs(args.output_dir, args.model_hpp or None, args.generative_hpp or None)
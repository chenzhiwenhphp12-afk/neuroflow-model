#!/usr/bin/env python3
"""
LMH2 → LMH1 转换器 / LMH1 生成器

模式1 (有lm_head.nfv1): 从 C++ LMH2 提取 w_embed/w_proj，新建 bridge
模式2 (无lm_head.nfv1): 从 model.nfv1 读取维度，全部 xavier 初始化

LMH1 张量: bridge.weight, bridge.bias, w_proj.weight, w_proj.bias, w_embed

用法:
  # 模式1: 从LMH2转换
  python3 scripts/lmh2_to_lmh1.py \
    --nf-model output/model.nfv1 \
    --lmh2 output/lm_head.nfv1 \
    --output lm_head_lmh1.nfv1

  # 模式2: 从model.nfv1直接生成 (无需lm_head)
  python3 scripts/lmh2_to_lmh1.py \
    --nf-model output/model.nfv1 \
    --vocab-size 128000 \
    --d-model 512 \
    --output lm_head_lmh1.nfv1
"""

import argparse, struct, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_full import load_nfv1, load_lmh1


def load_lmh2(path):
    weights = {}
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic not in (b'LMH2', b'LMH1'):
            raise ValueError(f"Bad magic: {magic} (expected LMH2 or LMH1)")
        while True:
            nl = struct.unpack('<I', f.read(4))[0]
            if nl == 0:
                break
            name = f.read(nl).decode('utf-8', errors='replace')
            ndim = struct.unpack('<I', f.read(4))[0]
            shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
            dsize = struct.unpack('<I', f.read(4))[0]
            arr = np.frombuffer(f.read(dsize), dtype=np.float32).reshape(shape).copy()
            weights[name] = arr
    return weights, magic.decode('utf-8')


def save_lmh1(path, weights):
    with open(path, 'wb') as f:
        f.write(b'LMH1')
        for name, arr in weights.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(arr.shape)))
            for d in arr.shape:
                f.write(struct.pack('<I', d))
            data = arr.astype(np.float32).tobytes()
            f.write(struct.pack('<I', len(data)))
            f.write(data)
        f.write(struct.pack('<I', 0))


def xavier_uniform(shape, rng):
    fan_in = shape[1] if len(shape) == 2 else shape[0]
    fan_out = shape[0] if len(shape) == 2 else 1
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='LMH2 → LMH1 converter / LMH1 generator')
    parser.add_argument('--nf-model', required=True, help='NF 模型文件 (用于获取 hidden_dim)')
    parser.add_argument('--lmh2', default='', help='C++ LMH2 格式 LM head 文件 (可选)')
    parser.add_argument('--output', required=True, help='输出 LMH1 文件路径')
    parser.add_argument('--vocab-size', type=int, default=128000, help='词表大小 (模式2需要)')
    parser.add_argument('--d-model', type=int, default=0, help='LM d_model (0=从NF模型推断)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    nf_w = load_nfv1(args.nf_model)
    hidden_dim = nf_w['input_proj.weight'].shape[0]
    print(f"NF hidden_dim: {hidden_dim}")

    rng = np.random.RandomState(args.seed)
    lmh1_w = {}

    if args.lmh2 and os.path.exists(args.lmh2):
        lmh2_w, fmt = load_lmh2(args.lmh2)
        print(f"模式1: 从 {args.lmh2} 转换 (格式: {fmt}, {len(lmh2_w)} 个张量)")

        if fmt == 'LMH1':
            print("已经是 LMH1 格式，直接复制")
            import shutil
            shutil.copy2(args.lmh2, args.output)
            print(f"输出: {args.output}")
            return

        vocab_size, d_model = lmh2_w['w_embed'].shape
        print(f"vocab_size: {vocab_size}, d_model: {d_model}")

        lmh1_w['w_embed'] = lmh2_w['w_embed']
        print(f"  w_embed: {lmh1_w['w_embed'].shape} (from LMH2)")

        if 'w_proj.weight' in lmh2_w:
            lmh1_w['w_proj.weight'] = lmh2_w['w_proj.weight']
            lmh1_w['w_proj.bias'] = lmh2_w['w_proj.bias']
            print(f"  w_proj.weight: {lmh1_w['w_proj.weight'].shape} (from LMH2)")
            print(f"  w_proj.bias: {lmh1_w['w_proj.bias'].shape} (from LMH2)")
        else:
            lmh1_w['w_proj.weight'] = xavier_uniform((d_model, d_model), rng)
            lmh1_w['w_proj.bias'] = np.zeros(d_model, dtype=np.float32)
            print(f"  w_proj.weight: {lmh1_w['w_proj.weight'].shape} (xavier init)")
            print(f"  w_proj.bias: {lmh1_w['w_proj.bias'].shape}")
    else:
        if args.lmh2:
            print(f"警告: --lmh2 文件不存在: {args.lmh2}")
        print(f"模式2: 从 model.nfv1 生成全新 LMH1")

        d_model = args.d_model
        if d_model == 0:
            d_model = int(nf_w['output_fusion.up.weight'].shape[0])
            print(f"  d_model 从NF模型推断: {d_model}")
        vocab_size = args.vocab_size
        print(f"vocab_size: {vocab_size}, d_model: {d_model}")

        lmh1_w['w_embed'] = xavier_uniform((vocab_size, d_model), rng)
        print(f"  w_embed: {lmh1_w['w_embed'].shape} (xavier init)")

        lmh1_w['w_proj.weight'] = xavier_uniform((d_model, d_model), rng)
        lmh1_w['w_proj.bias'] = np.zeros(d_model, dtype=np.float32)
        print(f"  w_proj.weight: {lmh1_w['w_proj.weight'].shape} (xavier init)")
        print(f"  w_proj.bias: {lmh1_w['w_proj.bias'].shape}")

    lmh1_w['bridge.weight'] = xavier_uniform((d_model, hidden_dim), rng)
    lmh1_w['bridge.bias'] = np.zeros(d_model, dtype=np.float32)
    print(f"  bridge.weight: {lmh1_w['bridge.weight'].shape} (xavier init)")
    print(f"  bridge.bias: {lmh1_w['bridge.bias'].shape}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_lmh1(args.output, lmh1_w)
    print(f"\n输出: {args.output} (LMH1, {len(lmh1_w)} 个张量)")

    verify_w = load_lmh1(args.output)
    print(f"验证: {len(verify_w)} 个张量加载成功")
    for name in lmh1_w:
        if name in verify_w:
            match = np.allclose(lmh1_w[name], verify_w[name])
            print(f"  {name}: {'OK' if match else 'MISMATCH'}")
        else:
            print(f"  {name}: MISSING")


if __name__ == '__main__':
    main()

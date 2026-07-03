#!/usr/bin/env python3
"""NeuroFlow 完整推理 — 加载 NF + LM Head + Tokenizer"""
import struct, json, numpy as np, time, sys, math

# ═══════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════

def load_nfv1(path):
    """加载 NFv1 格式权重 (NeuroFlowModel)"""
    weights = {}
    with open(path, 'rb') as f:
        assert f.read(4) == b'NFv1', f"Bad magic in {path}"
        while True:
            nl = struct.unpack('<I', f.read(4))[0]
            if nl == 0: break
            name = f.read(nl).decode('utf-8', errors='replace')
            ndim = struct.unpack('<I', f.read(4))[0]
            shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
            dsize = struct.unpack('<I', f.read(4))[0]
            arr = np.frombuffer(f.read(dsize), dtype=np.float32).reshape(shape).copy()
            weights[name] = arr
    return weights

def load_lmh1(path):
    """加载 LMH1 格式权重 (bridge + LM head)"""
    weights = {}
    with open(path, 'rb') as f:
        assert f.read(4) == b'LMH1', f"Bad magic in {path}"
        while True:
            nl = struct.unpack('<I', f.read(4))[0]
            if nl == 0: break
            name = f.read(nl).decode('utf-8', errors='replace')
            ndim = struct.unpack('<I', f.read(4))[0]
            shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
            dsize = struct.unpack('<I', f.read(4))[0]
            arr = np.frombuffer(f.read(dsize), dtype=np.float32).reshape(shape).copy()
            weights[name] = arr
    return weights

# ═══════════════════════════════════════════════════════════
# 分词器
# ═══════════════════════════════════════════════════════════

def load_tokenizer(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = data.get('vocab', {})
    merges = data.get('merges', [])
    merge_ranks = {}
    for i, s in enumerate(merges):
        parts = s.split(' ')
        if len(parts) == 2:
            merge_ranks[(parts[0], parts[1])] = i
    id2token = {v: k for k, v in vocab.items()}
    return vocab, id2token, merge_ranks

def apply_bpe(token, merge_ranks):
    if len(token) <= 1 or not merge_ranks:
        return token
    symbols = list(token)
    while True:
        best_rank = float('inf')
        best_i = -1
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair in merge_ranks and merge_ranks[pair] < best_rank:
                best_rank = merge_ranks[pair]
                best_i = i
        if best_i < 0:
            break
        symbols[best_i] = symbols[best_i] + symbols[best_i + 1]
        del symbols[best_i + 1]
    return ''.join(symbols)


_cache_vocab = None
_cache_prefix_set = None


def _get_prefix_set(vocab):
    global _cache_vocab, _cache_prefix_set
    if vocab is not _cache_vocab:
        _cache_vocab = vocab
        _cache_prefix_set = set()
        for token in vocab:
            for end in range(1, len(token)):
                _cache_prefix_set.add(token[:end])
    return _cache_prefix_set


def encode(text, vocab, merge_ranks, max_len=128):
    ids = [2]
    prefix_set = _get_prefix_set(vocab)
    i = 0
    n = len(text)
    while i < n and len(ids) < max_len - 1:
        best_end = i + 1
        best_token = None
        for end in range(min(i + 64, n), i, -1):
            candidate = text[i:end]
            if candidate in vocab:
                best_end = end
                best_token = candidate
                break
            if len(candidate) > 1 and candidate not in prefix_set:
                continue
        if best_token is not None:
            ids.append(vocab[best_token])
            i = best_end
        else:
            ch = text[i]
            byte_len = 1
            if ord(ch) >= 0x80:
                if ord(ch) < 0xE0:
                    byte_len = 2
                elif ord(ch) < 0xF0:
                    byte_len = 3
                else:
                    byte_len = 4
            byte_seq = text[i:i + byte_len]
            bpe_result = apply_bpe(byte_seq, merge_ranks)
            if bpe_result in vocab:
                ids.append(vocab[bpe_result])
            else:
                for c in bpe_result:
                    ids.append(vocab.get(c, 1))
            i += byte_len
    ids.append(3)
    return ids

def decode(ids, id2token):
    parts = []
    for tid in ids:
        if tid in (0, 1, 2, 3): continue
        if tid in id2token:
            t = id2token[tid]
            if not t.startswith('<extra_'): parts.append(t)
    return ''.join(parts)

# ═══════════════════════════════════════════════════════════
# 前向传播
# ═══════════════════════════════════════════════════════════

def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(); var = x.var()
    return w * (x - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def neuroflow_full_forward(token_ids, nf_w, lm_w, vocab_size=128000, d_model=512, hidden_dim=2048):
    """完整前向: NF → bridge → LM head → logits"""
    batch = 1
    vocab_scale = 1.0 / float(vocab_size)

    # ── NF Input ──
    x = np.zeros(d_model, dtype=np.float32)
    copy_len = min(len(token_ids), d_model)
    for j in range(copy_len):
        x[j] = float(token_ids[j]) * vocab_scale

    # ── Input Projection ──
    h = nf_w['input_proj.weight'] @ x + nf_w['input_proj.bias']          # [2048]
    h = layernorm(h, nf_w['input_proj_norm.weight'], nf_w['input_proj_norm.bias'])
    h = gelu(h)

    # ── SN: gates ──
    g1 = gelu(nf_w['sn.gate1.weight'] @ h + nf_w['sn.gate1.bias'])        # [1024]
    gates = softmax(nf_w['sn.gate2.weight'] @ g1 + nf_w['sn.gate2.bias']) # [2]
    ecn_gate = gates[0]

    # ── ECN: 12-layer DLPFC ──
    h_ecn = h.copy()
    for i in range(12):
        h_ecn = gelu(nf_w[f'ecn.dlpfc{i}.weight'] @ h_ecn + nf_w[f'ecn.dlpfc{i}.bias'])
    ecn_last = h_ecn

    # ── ECN: decision ──
    vmpfc = gelu(nf_w['ecn.vmpfc1.weight'] @ ecn_last + nf_w['ecn.vmpfc1.bias'])  # [1024]
    decision = nf_w['ecn.vmpfc2.weight'] @ vmpfc + nf_w['ecn.vmpfc2.bias']          # [2048]

    # ── Memory ──
    mem_encoded = nf_w['memory.encode.weight'] @ h + nf_w['memory.encode.bias']     # [512]

    # ── DMN ──
    dmn_enc = gelu(nf_w['dmn.mem_encoder1.weight'] @ mem_encoded + nf_w['dmn.mem_encoder1.bias'])
    dmn_latent = nf_w['dmn.mem_encoder2.weight'] @ dmn_enc + nf_w['dmn.mem_encoder2.bias']  # [1024]
    assoc_outs = []
    for i in range(8):
        a1 = gelu(nf_w[f'dmn.head{i}.1.weight'] @ dmn_latent + nf_w[f'dmn.head{i}.1.bias'])
        a2 = nf_w[f'dmn.head{i}.2.weight'] @ a1 + nf_w[f'dmn.head{i}.2.bias']
        assoc_outs.append(a2)
    dmn_vision = gelu(nf_w['dmn.future_proj1.weight'] @ np.concatenate(assoc_outs) + nf_w['dmn.future_proj1.bias'])

    # ── Memory bank retrieval ──
    mem_bank = nf_w['memory.bank']  # [64, 512]
    att = softmax(mem_encoded @ mem_bank.T)
    retrieved = att @ mem_bank
    mem_retrieved = nf_w['memory.retrieve.weight'] @ retrieved + nf_w['memory.retrieve.bias']

    # ── Output Fusion ──
    ecn_w = decision * ecn_gate
    dmn_w = dmn_vision * gates[1]
    dmn_w_pad = np.zeros(hidden_dim, dtype=np.float32)
    dmn_w_pad[:dmn_w.shape[0]] = dmn_w
    mem_w = np.zeros(hidden_dim, dtype=np.float32)
    mem_w[:mem_retrieved.shape[0]] = mem_retrieved
    combined = np.concatenate([ecn_w, dmn_w_pad, mem_w])  # [6144]

    fused = nf_w['output_fusion.down.weight'] @ combined + nf_w['output_fusion.down.bias']        # [256]
    fused = layernorm(fused, nf_w['output_fusion.bn_norm.weight'], nf_w['output_fusion.bn_norm.bias'])
    fused = np.maximum(0, fused)  # relu
    nf_output = nf_w['output_fusion.up.weight'] @ fused + nf_w['output_fusion.up.bias']            # [2048]
    nf_output = layernorm(nf_output, nf_w['output_fusion.norm.weight'], nf_w['output_fusion.norm.bias'])

    # ── Bridge projection (learned, from training) ──
    bridge_h = lm_w['bridge.weight'] @ nf_output + lm_w['bridge.bias']  # [2048] → [512]

    # ── LM Head ──
    projected = lm_w['w_proj.weight'] @ bridge_h + lm_w['w_proj.bias']  # [512] → [512]
    logits = lm_w['w_embed'] @ projected                                  # [128000]
    return logits


# ═══════════════════════════════════════════════════════════
# 生成
# ═══════════════════════════════════════════════════════════

def generate(prompt, nf_w, lm_w, vocab, id2token, merge_ranks,
             max_tokens=30, temp=0.8, top_k=40, seed=42):
    rng = np.random.RandomState(seed)
    ids = encode(prompt, vocab, merge_ranks)
    generated = []

    for step in range(max_tokens):
        ctx = ids[-d_model:]  # truncate to d_model
        logits = neuroflow_full_forward(ctx, nf_w, lm_w)

        if temp > 0.01:
            logits = logits / temp

        # Top-K filter
        if 0 < top_k < len(logits):
            topk_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full(len(logits), -np.inf, dtype=np.float32)
            mask[topk_indices] = logits[topk_indices]
            logits = mask

        probs = softmax(logits)
        next_id = int(rng.choice(len(probs), p=probs))

        if next_id == 3:
            break  # eos
        generated.append(next_id)
        ids.append(next_id)

        if step < 5:
            tok = id2token.get(next_id, f'<{next_id}>')
            print(f"  [{step}] id={next_id} '{tok}' p={probs[next_id]:.4f}")

    return decode(generated, id2token)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    CKPT_DIR = '/home/administrator/output_final2/checkpoint_step5000'
    NF_MODEL = f'{CKPT_DIR}/checkpoint_step5000/model.nfv1'
    LM_MODEL = f'{CKPT_DIR}/lm_head.nfv1'
    TOKENIZER = '/mnt/d/neuroflow-C++/configs/tokenizer_128k.json'

    print("⏳ 加载模型...")
    t0 = time.time()
    nf_w = load_nfv1(NF_MODEL)
    lm_w = load_lmh1(LM_MODEL)
    print(f"✅ NF: {len(nf_w)}层 | LM: {len(lm_w)}层 ({time.time()-t0:.1f}s)")

    vocab, id2token, merge_ranks = load_tokenizer(TOKENIZER)
    print(f"✅ 词表: {len(vocab)} tokens")

    d_model = nf_w['input_proj.weight'].shape[1]  # in_features
    hidden_dim = nf_w['input_proj.weight'].shape[0]  # out_features
    print(f"    d_model={d_model} hidden_dim={hidden_dim}")

    print("\n" + "=" * 60)
    print("🧪 NeuroFlow 推理测试 (Step 5000)")
    print("=" * 60)

    # 测试1
    print("\n📝 贪心解码")
    for prompt in ["人工智能", "中国", "数学"]:
        r = generate(prompt, nf_w, lm_w, vocab, id2token, merge_ranks, max_tokens=15, temp=0.01, top_k=1, seed=42)
        print(f"  '{prompt}' → '{r}'")

    # 测试2
    print("\n📝 温度采样 (temp=0.8, top_k=40)")
    for prompt in ["哲学", "科学", "文化"]:
        r = generate(prompt, nf_w, lm_w, vocab, id2token, merge_ranks, max_tokens=20, temp=0.8, top_k=40, seed=123)
        print(f"  '{prompt}' → '{r}'")

    # 测试3: 完整生成
    print("\n📝 长文本生成")
    r = generate("人工智能是", nf_w, lm_w, vocab, id2token, merge_ranks, max_tokens=50, temp=0.7, top_k=50, seed=42)
    print(f"  结果: '{r}'")

    # 测试4: Logits 分析
    print("\n📊 Logits 分析")
    ids = encode("中", vocab, merge_ranks)
    logits = neuroflow_full_forward(ids[-d_model:], nf_w, lm_w)
    probs = softmax(logits / 0.8)
    top10 = np.argsort(probs)[-10:][::-1]
    print("  Top-10 预测:")
    for idx in top10:
        tok = id2token.get(int(idx), f'<ID{idx}>')
        print(f"    id={idx:6d} {repr(tok):15s} p={probs[idx]:.4f}")

    print(f"\n✅ 测试完成")

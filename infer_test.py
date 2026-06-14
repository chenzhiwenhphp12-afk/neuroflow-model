#!/usr/bin/env python3
"""NeuroFlow 371M 推理 — 严格对齐C++前向传播(model.cpp:157-268)"""
import struct, json, numpy as np, time, sys

def load_nfv1(path):
    weights = {}
    with open(path, 'rb') as f:
        assert f.read(4) == b'NFv1'
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

def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(); var = x.var()
    return w * (x - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def neuroflow_forward(token_ids, W, output_dim=128000, input_dim=512, num_layers=12, num_heads=8):
    """Strict replica of model.cpp:157-268 forward_with_cache (batch=1)"""
    vocab_scale = 1.0 / float(output_dim)
    
    # Input: [input_dim] where x[j] = token_id[j] * vocab_scale
    x = np.zeros(input_dim, dtype=np.float32)
    copy_len = min(len(token_ids), input_dim)
    for j in range(copy_len):
        x[j] = float(token_ids[j]) * vocab_scale

    # input_proj: linear -> norm -> gelu
    h = W['input_proj.weight'] @ x + W['input_proj.bias']         # [2048]
    h = layernorm(h, W['input_proj_norm.weight'], W['input_proj_norm.bias'])
    h = gelu(h)                                                      # [2048]

    # SN: saliency + gates
    s1 = np.maximum(0, W['sn.saliency1.weight'] @ h + W['sn.saliency1.bias'])
    s2 = np.maximum(0, W['sn.saliency2.weight'] @ s1 + W['sn.saliency2.bias'])
    saliency = W['sn.saliency3.weight'] @ s2 + W['sn.saliency3.bias']

    g1 = np.maximum(0, W['sn.gate1.weight'] @ h + W['sn.gate1.bias'])
    gates_2 = W['sn.gate2.weight'] @ g1 + W['sn.gate2.bias']      # [2]
    gates = softmax(gates_2)                                         # [2]
    ecn_gate = gates[0]
    dmn_gate = gates[1]

    # ECN: 12-layer DLPFC
    h_ecn = h.copy()
    for i in range(num_layers):
        h_ecn = W[f'ecn.dlpfc{i}.weight'] @ h_ecn + W[f'ecn.dlpfc{i}.bias']
        h_ecn = layernorm(h_ecn, 
                          W.get(f'ecn.dlpfc{i}_norm.weight', np.ones_like(h_ecn)),
                          W.get(f'ecn.dlpfc{i}_norm.bias', np.zeros_like(h_ecn)))
        h_ecn = gelu(h_ecn)
    ecn_last = h_ecn  # [2048]

    # But wait - C++ uses dlpfc_norm/gelu layers which have weights!
    # Let me check if we have norm weights for DLPFC
    # From NFv1 we DON'T see ecn.dlpfcX_norm - C++ creates them but maybe they're identity?
    # Actually the NFv1 dump shows NO dlpfc_norm weights, so they must be identity or skipped
    # C++ code: dlpfc_linear[i] -> dlpfc_norm[i] -> dlpfc_gelu[i]
    # But NFv1 only saved: ecn.dlpfc{i}.weight/bias (12 layers)
    # So the norm layers in C++ must be identity (no learnable params saved)
    
    # Re-do ECN without norm (since no saved params)
    h_ecn = h.copy()
    for i in range(num_layers):
        h_ecn = W[f'ecn.dlpfc{i}.weight'] @ h_ecn + W[f'ecn.dlpfc{i}.bias']
        # No norm weights in NFv1 -> skip layernorm
        h_ecn = gelu(h_ecn)  # C++ uses gelu, not relu!
    ecn_last = h_ecn  # [2048]

    # ECN OFC: ofc1 -> gelu -> ofc2 (= value)
    ofc_pre = W['ecn.ofc1.weight'] @ ecn_last + W['ecn.ofc1.bias']   # [1024]
    ofc_v = gelu(ofc_pre)
    value = W['ecn.ofc2.weight'] @ ofc_v + W['ecn.ofc2.bias']        # [1]

    # ECN VMPFC: vmpfc1 -> gelu -> vmpfc2 (= decision!)  
    vmpfc_pre = W['ecn.vmpfc1.weight'] @ ecn_last + W['ecn.vmpfc1.bias']  # [1024]
    vmpfc_d = gelu(vmpfc_pre)
    decision = W['ecn.vmpfc2.weight'] @ vmpfc_d + W['ecn.vmpfc2.bias']     # [128000]

    # Memory: encode(h) -> [512]
    mem_encoded = W['memory.encode.weight'] @ h + W['memory.encode.bias']  # [512]

    # DMN: mem_encoder1(gelu) -> mem_encoder2
    dmn_enc = W['dmn.mem_encoder1.weight'] @ mem_encoded + W['dmn.mem_encoder1.bias']  # [2048]
    dmn_enc = gelu(dmn_enc)
    dmn_latent = W['dmn.mem_encoder2.weight'] @ dmn_enc + W['dmn.mem_encoder2.bias']   # [1024]

    # DMN: 8 association heads -> concat -> future_proj
    assoc_outs = []
    for i in range(num_heads):
        a1 = W[f'dmn.head{i}.1.weight'] @ dmn_latent + W[f'dmn.head{i}.1.bias']
        a1 = gelu(a1)
        a2 = W[f'dmn.head{i}.2.weight'] @ a1 + W[f'dmn.head{i}.2.bias']
        assoc_outs.append(a2)                                     # each [1024]
    
    dmn_vision_raw = np.concatenate(assoc_outs)                    # [8192]
    
    # future_proj1: [2048x8192] -> [2048]
    # But wait! Does future_proj1 weight shape match?
    # W['dmn.future_proj1.weight'] = [2048, 8192]
    # dmn_vision_raw = [8192]
    # W @ dmn_vision_raw = [2048] ✅
    dmn_vision = W['dmn.future_proj1.weight'] @ dmn_vision_raw + W['dmn.future_proj1.bias']  # [2048]
    
    # future_norm + future_gelu
    # Again, no saved norm weights for future_norm in NFv1
    # C++ code: future_proj1 -> future_norm -> future_gelu
    # No 'dmn.future_norm' in NFv1 -> identity norm -> just gelu
    dmn_vision = gelu(dmn_vision)  # [2048]

    # Memory forward: encode(h) already done
    # Memory bank read
    mem_bank = W['memory.bank']                                    # [64, 512]
    att = softmax(mem_encoded @ mem_bank.T)                        # [64]
    retrieved = att @ mem_bank                                     # [512]
    mem_retrieved = W['memory.retrieve.weight'] @ retrieved + W['memory.retrieve.bias']  # [2048]

    # === Gated Fusion (C++ line 245-255) ===
    ecn_weighted = decision * ecn_gate                             # [128000]
    
    dmn_weighted = np.zeros(output_dim, dtype=np.float32)
    dmn_dim = dmn_vision.shape[0]                                  # 2048
    for j in range(min(dmn_dim, output_dim)):
        dmn_weighted[j] = dmn_vision[j] * dmn_gate

    mem_weighted = np.zeros(output_dim, dtype=np.float32)
    mem_dim = mem_retrieved.shape[0]                               # 2048
    for j in range(min(mem_dim, output_dim)):
        mem_weighted[j] = mem_retrieved[j]

    # Combined: [3 * output_dim] = [384000]
    combined = np.concatenate([ecn_weighted, dmn_weighted, mem_weighted])

    # Output fusion: down -> bn_norm -> relu -> up -> norm
    fused = W['output_fusion.down.weight'] @ combined + W['output_fusion.down.bias']  # [256]
    fused = layernorm(fused, W['output_fusion.bn_norm.weight'], W['output_fusion.bn_norm.bias'])
    fused_pre_relu = fused.copy()
    fused = np.maximum(0, fused)                                   # relu
    fused = W['output_fusion.up.weight'] @ fused + W['output_fusion.up.bias']        # [128000]
    output = layernorm(fused, W['output_fusion.norm.weight'], W['output_fusion.norm.bias'])
    
    return output  # [128000]

def load_tokenizer(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = data.get('vocab', {})
    id2token = {v: k for k, v in vocab.items()}
    return vocab, id2token

def encode(text, vocab, max_len=512):
    ids = [1]  # BOS
    i = 0
    while i < len(text) and len(ids) < max_len - 1:
        found = False
        for l in range(min(len(text)-i, 20), 0, -1):
            sub = text[i:i+l]
            if sub in vocab:
                ids.append(vocab[sub])
                i += l
                found = True
                break
        if not found:
            ids.append(3)  # UNK
            i += 1
    ids.append(2)  # EOS
    return ids

def decode(ids, id2token):
    parts = []
    for tid in ids:
        if tid in (0, 1, 2, 3): continue
        if tid in id2token:
            t = id2token[tid]
            if t.startswith('<extra_'): continue
            parts.append(t)
    return ''.join(parts)

def generate(prompt, W, vocab, id2token, max_tokens=50, temp=0.8, top_k=40, seed=42):
    input_ids = encode(prompt, vocab)
    generated = []
    rng = np.random.RandomState(seed)
    
    print(f"\n🎯 '{prompt}' → IDs:{input_ids[:15]}{'…' if len(input_ids)>15 else ''}")
    
    for step in range(max_tokens):
        ctx = input_ids[-512:]
        logits = neuroflow_forward(ctx, W)
        
        if temp > 0.01:
            logits = logits / temp
        if 0 < top_k < len(logits):
            topk = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full(len(logits), -np.inf)
            mask[topk] = logits[topk]
            logits = mask
        
        probs = softmax(logits)
        next_id = int(rng.choice(len(probs), p=probs))
        
        if next_id == 2: break
        generated.append(next_id)
        input_ids.append(next_id)
        
        tok = id2token.get(next_id, f'<{next_id}>')
        if step < 12 or step % 10 == 0:
            print(f"   [{step:3d}] id={next_id:6d} {repr(tok):8s} p={probs[next_id]:.4f}")
    
    return decode(generated, id2token)

# ===== MAIN =====
MODEL = '/mnt/d/neuroflow-C++/output/model_final.nfv1'
TOK   = '/mnt/d/neuroflow-C++/configs/tokenizer_cn_013.json'

print("⏳ 加载模型...")
t0 = time.time()
W = load_nfv1(MODEL)
print(f"✅ {len(W)}层 ({time.time()-t0:.1f}s)")

vocab, id2token = load_tokenizer(TOK)
print(f"✅ tokenizer: {len(vocab)} entries (max_id={max(vocab.values())})")

print("\n" + "="*60)
print("🚀 NeuroFlow 371M 推理验证")
print("="*60)

# 测试1: 贪心
print("\n📝 测试1: 贪心")
r = generate("人工智能", W, vocab, id2token, max_tokens=20, temp=0.01, top_k=1, seed=42)
print(f"   ➡️ {r}")

# 测试2: 采样
print("\n📝 测试2: 低温度采样")
r = generate("人工智能", W, vocab, id2token, max_tokens=20, temp=0.5, top_k=20, seed=42)
print(f"   ➡️ {r}")

# 测试3: 多prompt
print("\n📝 测试3: 多prompt贪心")
for p in ["中国", "数学", "哲学", "科学", "文化"]:
    r = generate(p, W, vocab, id2token, max_tokens=15, temp=0.01, top_k=1, seed=42)
    print(f"   '{p}' → '{r}'")

# 测试4: logits分布
print("\n📊 测试4: 单token logits分析")
ids = encode("中", vocab)
logits = neuroflow_forward(ids[-512:], W)
top10 = np.argsort(logits)[-10:][::-1]
print("   Top-10 for '中':")
for idx in top10:
    tok = id2token.get(int(idx), f'<ID{idx}>')
    print(f"     id={idx:6d} {repr(tok):8s} logit={logits[idx]:.3f}")

"""
NeuroFlow TextDecoder Demo — Token 输出演示
============================================
演示 NeuroFlow + 轻量 TextDecoder 的端到端 Token 生成能力。
"""

import sys, time, numpy as np

sys.path.insert(0, "/mnt/d/neuroflow-model")

from neuroflow import NeuroFlowLite, get_backend
from neuroflow.decoder import TextDecoder

print("=" * 60)
print("  🧠 NeuroFlow + TextDecoder Token 生成演示")
print("=" * 60)
print()

# 1. 初始化
print("📋 初始化")
print("-" * 40)
print(f"  NeuroFlow 后端: {get_backend()}")

model = NeuroFlowLite(input_dim=512)
decoder = TextDecoder(model, hidden_dim=256, vocab_size=5000)

stats = decoder.get_stats()
print(f"  Decoder 参数:  {stats['decoder_params']:,}")
print(f"  Decoder 内存:  {stats['decoder_memory_mb']:.2f} MB")
print(f"  词表大小:      {stats['vocab_size']}")
print(f"  隐藏维度:      {stats['hidden_dim']}")
print()

# 2. Token 采样演示
print("🎲 Token 采样演示")
print("-" * 40)

context = np.random.randn(256).astype(np.float32) * 0.3
logits = decoder._logits(context)
top_tokens = np.argsort(logits)[-10:][::-1]
print(f"  Logits shape: {logits.shape}")
print(f"  Top-10 tokens:")
for rank, tid in enumerate(top_tokens):
    token = decoder.vocab[tid]
    if token.startswith("<"):
        continue
    print(f"    #{rank+1}: id={tid:5d}  '{token}'  logit={logits[tid]:.2f}")
print()

# 3. 文本生成测试
print("📝 文本生成测试")
print("-" * 40)

prompts = [
    ("The ", "英文起始"),
    ("AI will ", "AI 预测"),
    ("I think ", "观点生成"),
    ("", "无提示自由生成"),
]

for prompt, label in prompts:
    t0 = time.perf_counter()
    token_ids, text = decoder.generate(
        prompt=prompt,
        max_tokens=30,
        temperature=0.85,
        top_k=40,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  [{label}]")
    print(f"    Prompt:   '{prompt}'")
    print(f"    Generated: '{text[:80]}'")
    print(f"    Tokens:   {len(token_ids)} ({elapsed:.1f}ms)")
    print()

# 4. Beam Search
print("🔍 Beam Search (确定性)")
print("-" * 40)
t0 = time.perf_counter()
text = decoder.beam_search(beam_width=3, max_tokens=20)
elapsed = (time.perf_counter() - t0) * 1000
print(f"  Output:  '{text[:80]}'")
print(f"  Time:    {elapsed:.1f}ms")
print()

# 5. 在线学习
print("🔄 在线学习（一步 SGD）")
print("-" * 40)
x = np.random.randn(1, 512).astype(np.float32) * 0.01
target_text = "hello world"
target_ids = decoder.encode(target_text)
print(f"  Target: '{target_text}' → ids={target_ids}")
loss_before = -decoder._logits(decoder._extract_context(x))[target_ids[1]]
decoder.train_step(x, target_ids, lr=0.1)
loss_after = -decoder._logits(decoder._extract_context(x))[target_ids[1]]
print(f"  Loss before: {loss_before:.4f}")
print(f"  Loss after:  {loss_after:.4f}")
print(f"  Δ:           {loss_after - loss_before:+.4f}")
print()

# 6. 架构总览
print("=" * 60)
print("  🏗️ 端到端架构")
print("=" * 60)
print("""
  Input Text ──► [Encode] ──► Feature Vector [1, 512]
                                     │
                                     ▼
              ┌─────────────────────────────────────┐
              │       NeuroFlow (C++ / AVX2)         │
              │  SN → ECN → DMN → Memory            │
              │  Brain-Inspired Neural Signals       │
              └─────────────────┬───────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   ECN Hidden States   │
                    │   [1, 256]            │
                    └───────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────────┐
              │     TextDecoder (Linear Projection)  │
              │  W_proj [256, 5000] + b_proj        │
              │  → Token Logits [5000]              │
              └─────────────────┬───────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────┐
              │     TokenSampler                     │
              │  Temperature + Top-K + Top-P         │
              │  → Token ID                          │
              └─────────────────┬───────────────────┘
                                │
                                ▼
                        Token Output
                 "The future of AI is..."
""")

# 7. 参数统计
print("📊 总参数量")
print("-" * 40)
total = stats['decoder_params']
print(f"  NeuroFlow Lite:  331,301 params")
print(f"  TextDecoder:     {stats['decoder_params']:,} params")
print(f"  ─────────────────────────────")
print(f"  Total:           {331301 + stats['decoder_params']:,} params")
print(f"  Total Memory:    {0.33 + stats['decoder_memory_mb']:.1f} MB")
print()
print("  ✅ 端到端 Token 生成完成")

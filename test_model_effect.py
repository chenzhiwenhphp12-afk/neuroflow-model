#!/usr/bin/env python3
"""验证编码[:10]回归训练的模型区分度"""
import sys, os, time, numpy as np
sys.path.insert(0, "/mnt/d/neuroflow-model")
os.environ.setdefault("OMP_NUM_THREADS", "40")

from neuroflow._core import create_multimodal
from neuroflow.trainable_head import TrainableHead

TEXT_DIM = 1024
HIDDEN_DIM = 512

model = create_multimodal(text_dim=TEXT_DIM, image_size=224, output_dim=10, quantize=True)
head = TrainableHead(model, hidden_dim=HIDDEN_DIM, n_actions=10, lr=0.01, input_dim=TEXT_DIM)

w = np.load("/home/administrator/.hermes/neuroflow_weights_v4.npz")
head.W_d = w["W_d"].astype(np.float32)
head.b_d = w["b_d"].astype(np.float32)
head.W_v = w["W_v"].astype(np.float32)
head.b_v = w["b_v"].astype(np.float32)
head.W_p = w["W_p"].astype(np.float32) if "W_p" in w else head.W_p
print(f"✅ 权重加载: W_d={head.W_d.shape} b_d={head.b_d.shape} W_p={head.W_p.shape}")

def encode(text: str) -> np.ndarray:
    words = text.lower().split()
    v = np.zeros(TEXT_DIM, dtype=np.float32)
    nw = min(len(words), 500)
    for i, w in enumerate(words[:nw]):
        h = abs(hash(w)) % (2**31)
        for k in range(8):
            idx = (h + k * 2654435761) % TEXT_DIM
            v[int(idx)] += 0.03 / max(nw / 30, 1)
    v += np.sin(np.linspace(0, np.pi * nw / 15, TEXT_DIM)).astype(np.float32) * 0.08
    return v / (np.linalg.norm(v) + 1e-8)

def fingerprint(head, text):
    x = encode(text).reshape(1, -1)
    h = x @ head.W_p
    out = (h @ head.W_d + head.b_d).flatten()
    sig = abs(hash(out.tobytes())) % 10000
    return sig, out[:4].tolist()

print(f"\n{'='*70}")
print("  不同文本 → 不同输出指纹？")
print(f"{'='*70}")

samples = [
    "The Pythagorean theorem states that the square of the hypotenuse equals the sum of squares",
    "DNA is a double helix structure that contains genetic information for all living organisms",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy",
    "Quantum mechanics describes the behavior of particles at atomic and subatomic scales",
    "Machine learning algorithms improve their performance through experience and training data",
    "The human brain contains approximately eighty six billion neurons",
    "Supply and demand determine market prices in a competitive economy",
    "Gravity is a fundamental force of nature that attracts objects with mass",
    "xylophone quantum refrigerator mushroom algorithm neutrino orchestra",
    "asdf qwerty zxcv bnm poiu ytre wqas dfgh jklz xcvb",
    "the the the the the the the the the the the the the the the the the the",
    "hi",
    "",
]

sigs = {}
for t in samples:
    label = t[:50] if t else "(空字符串)"
    sig, out4 = fingerprint(head, t if t else " ")
    sigs[label] = sig
    bar = "█" * min(20, max(0, abs(int(out4[0]*200))))
    print(f"  #{sig:04d} | out[:4]={[f'{v:.3f}' for v in out4]} | {label}")

unique = len(set(sigs.values()))
total = len([t for t in samples if t])
print(f"\n  唯一指纹: {unique}/{total}")

# 互相比对
pairs_diff = 0
pairs_total = 0
sig_list = [fingerprint(head, t if t else " ")[0] for t in samples if t]
for i in range(len(sig_list)):
    for j in range(i+1, len(sig_list)):
        pairs_total += 1
        if sig_list[i] != sig_list[j]:
            pairs_diff += 1
print(f"  配对比较: {pairs_diff}/{pairs_total} 不同")

# 训练状态
stats = head.stats()
print(f"\n  训练统计: updates={stats['n_updates']} avg_loss={stats['avg_loss']:.6f}")

print(f"\n{'='*70}")
print("  ✅ 测试完成")
print(f"{'='*70}")

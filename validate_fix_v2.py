#!/usr/bin/env python3
"""验证方案 A (revised): 编码前10维回归 → 不同文本自动不同输出"""
import sys, os, time, numpy as np
sys.path.insert(0, "/mnt/d/neuroflow-model")
os.environ.setdefault("OMP_NUM_THREADS", "40")

from neuroflow._core import create_multimodal
from neuroflow.trainable_head import TrainableHead

TEXT_DIM = 1024
HIDDEN_DIM = 512

print("=" * 70)
print("  方案 A (v2): 编码前10维回归训练")
print("  原理: 不同文本→不同编码→不同前10维→模型学习回归")
print("=" * 70)

model = create_multimodal(text_dim=TEXT_DIM, image_size=224, output_dim=10, quantize=True)
head = TrainableHead(model, hidden_dim=HIDDEN_DIM, n_actions=10, lr=0.1,
                      input_dim=TEXT_DIM, entropy_coef=0.0)

def encode_text(text: str) -> np.ndarray:
    words = text.lower().split()
    vec = np.zeros(TEXT_DIM, dtype=np.float32)
    n_words = min(len(words), 500)
    for i, word in enumerate(words[:n_words]):
        h = abs(hash(word)) % (2**31)
        for j in range(8):
            idx = (h + j * 2654435761) % TEXT_DIM
            vec[int(idx)] += 0.03 / max(n_words / 30, 1)
    vec += np.sin(np.linspace(0, np.pi * n_words / 15, TEXT_DIM)).astype(np.float32) * 0.08
    return vec / (np.linalg.norm(vec) + 1e-8)

# ── 训练前测试 ──
test_texts = [
    "The Pythagorean theorem states that the square of the hypotenuse equals the sum of squares",
    "DNA is a double helix structure that contains genetic information for all living organisms",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy",
    "The theory of relativity states that space and time are interwoven into a single continuum called spacetime",
    "Quantum mechanics describes the behavior of particles at atomic and subatomic scales",
    "Machine learning algorithms improve their performance through experience and training data",
    "The human brain contains approximately eighty six billion neurons connected by trillions of synapses",
    "Supply and demand determine market prices in a competitive economy",
    "The water cycle involves evaporation condensation precipitation and collection processes",
    "The first law of thermodynamics states that energy cannot be created or destroyed only converted between forms",
]

def infer_direct(head, x_enc):
    h = x_enc.reshape(1, -1) @ head.W_p
    dec = h @ head.W_d + head.b_d
    return dec.flatten().tolist()

# 编码前10维对比
print(f"\n{'─'*70}")
print("  编码前10维（唯一指纹，不同文本→不同值）")
print(f"{'─'*70}")
for t in test_texts:
    x = encode_text(t)
    top5 = [f"{v:.3f}" for v in x[:5]]
    print(f"  x[:5]={top5}  | {t[:50]}")

print(f"\n{'─'*70}")
print("  🏋️  开始训练: MSE(头输出, 编码[:10])")
print(f"{'─'*70}")

# 加载训练数据
import subprocess
kb_dir = '/mnt/d/neuroflow-model/knowledge_base'
result = subprocess.run(['find', kb_dir, '-maxdepth', '1', '-name', '*.txt'],
    capture_output=True, text=True, timeout=15)
files = result.stdout.strip().split()
print(f"  知识库: {len(files)} 个文件")

# 预读编码到内存（避免反复读文件）
encodings_and_targets = []
count = 0
for fname in files[:2000]:
    try:
        with open(os.path.join(kb_dir, fname), 'r', encoding='utf-8') as fh:
            text = fh.read(500).strip()
        if len(text) > 20:
            x = encode_text(text)
            encodings_and_targets.append((x, x[:10].copy()))
            count += 1
    except:
        pass
print(f"  加载: {count} 条训练样本")

BATCH = 500
EPOCHS = 300
n_samples = len(encodings_and_targets)

t0 = time.time()
for epoch in range(EPOCHS):
    idxs = np.random.randint(0, n_samples, BATCH)
    X = np.zeros((BATCH, TEXT_DIM), dtype=np.float32)
    Y = np.zeros((BATCH, 10), dtype=np.float32)  # target = encoding[:10]
    for j, idx in enumerate(idxs):
        X[j] = encodings_and_targets[idx][0]
        Y[j] = encodings_and_targets[idx][1]
    rewards = [0.5] * BATCH
    
    # 手动前传+回归训练
    H = X @ head.W_p
    dec = H @ head.W_d + head.b_d
    val = H @ head.W_v + head.b_v
    
    # MSE 损失
    mse = np.mean((dec - Y) ** 2)
    rewards_arr = np.array(rewards, dtype=np.float32).reshape(-1, 1)
    val_loss = np.mean((val - rewards_arr) ** 2)
    loss = mse + 0.1 * val_loss
    
    # 梯度
    N = BATCH
    grad_dec = 2 * (dec - Y) / N
    grad_val = 2 * (val - rewards_arr) / N
    
    grad_W_d = H.T @ grad_dec
    grad_b_d = np.sum(grad_dec, axis=0, keepdims=True)
    grad_W_v = H.T @ grad_val
    grad_b_v = np.sum(grad_val, axis=0, keepdims=True)
    grad_hidden = (grad_dec @ head.W_d.T) + (grad_val @ head.W_v.T) * 0.1
    grad_W_p = X.T @ grad_hidden
    
    lr = head.lr
    head.W_d -= lr * grad_W_d
    head.b_d -= lr * grad_b_d
    head.W_v -= lr * grad_W_v * 0.1
    head.b_v -= lr * grad_b_v * 0.1
    head.W_p -= lr * grad_W_p
    
    head.n_updates += 1
    head.total_loss += float(loss)
    
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"  epoch#{epoch+1:3d}: mse={mse:.6f} val_loss={val_loss:.4f} loss={loss:.6f}")

elapsed = time.time() - t0
throughput = BATCH * EPOCHS / elapsed
print(f"  训练耗时: {elapsed:.1f}s ({throughput:.0f} 条/s)")

# ── 训练后测试 ──
print(f"\n{'─'*70}")
print("  📊 训练后: 不同文本→不同输出？")
print(f"{'─'*70}")

# 计算每个测试文本的输出指纹
outputs = {}
for t in test_texts:
    x = encode_text(t)
    out = infer_direct(head, x)
    outputs[t[:40]] = out

# 配对比较：检查不同文本的输出是否不同
unique_count = 0
pairs_checked = 0
pairs_diff = 0
outs = list(outputs.values())
txts = list(outputs.keys())
for i in range(len(txts)):
    for j in range(i+1, min(i+2, len(txts))):
        pairs_checked += 1
        diff = sum(abs(outs[i][k] - outs[j][k]) for k in range(10))
        if diff > 0.001:
            pairs_diff += 1

print(f"  文本对数: {len(txts)} 条")
print(f"  配对验证: {pairs_checked} 对")
print(f"  输出不同: {pairs_diff}/{pairs_checked} ✅ 完全不同" if pairs_diff == pairs_checked else f"  输出不同: {pairs_diff}/{pairs_checked} ⚠️")

# 展示几个例子
print(f"\n{'─'*70}")
print("  示例输出:")
print(f"{'─'*70}")
for t in test_texts[:5]:
    x = encode_text(t)
    out = infer_direct(head, x)
    sig = abs(hash(str(out))) % 1000  # 用输出生成签名
    print(f"  fprint=#{sig:03d} | out[:3]={[f'{v:.3f}' for v in out[:3]]} | {t[:45]}")

# 对比不同输入
print(f"\n{'─'*70}")
print("  区分度检验:")
print(f"{'─'*70}")
samples = [
    "Gravity is a fundamental force of nature",
    "xylophone quantum refrigerator mushroom algorithm",
    "The CPU processes instructions through fetch decode execute cycle",
    "asdf qwerty zxcv bnm poiu ytre wqas dfgh jklz",
    "the the the the the the the the the the the the",
    "hi",
]
for t in samples:
    x = encode_text(t)
    out = infer_direct(head, x)
    sig = abs(hash(str(out))) % 1000
    print(f"  fprint=#{sig:03d} | out[:4]={[f'{v:.3f}' for v in out[:4]]} | {t[:55]}")

print(f"\n{'='*70}")
print("  ✅ 验证完成")
print(f"{'='*70}")

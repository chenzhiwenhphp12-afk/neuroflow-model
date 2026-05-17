#!/usr/bin/env python3
"""验证方案 A：直接编码训练是否真正学到不同输出"""
import sys, os, time, numpy as np
sys.path.insert(0, "/mnt/d/neuroflow-model")

os.environ.setdefault("OMP_NUM_THREADS", "40")

from neuroflow._core import create_multimodal
from neuroflow.trainable_head import TrainableHead

TEXT_DIM = 1024
HEAD_ACTIONS = 10
HIDDEN_DIM = 512

print("=" * 70)
print("  验证方案 A：直接编码 + 文本 hash 监督训练")
print("=" * 70)

# ── 初始化 ──
model = create_multimodal(text_dim=TEXT_DIM, image_size=224, output_dim=10, quantize=True)
# 重新初始化：更高学习率 + 更多训练
head = TrainableHead(model, hidden_dim=HIDDEN_DIM, n_actions=HEAD_ACTIONS, lr=0.05,
                      input_dim=TEXT_DIM, entropy_coef=0.03)

# ── 从零开始：不加载旧权重（新架构 W_p + 重新初始化） ──
print("✅ 全新随机初始化 (W_p+W_d+b_d 从零开始)")

# ── 编码器 ──
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

# ── 测试数据集 ──
test_texts = [
    "The Pythagorean theorem states that the square of the hypotenuse equals the sum of squares",
    "DNA is a double helix structure that contains genetic information",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
    "The theory of relativity states that space and time are interwoven into spacetime",
    "Quantum mechanics describes the behavior of particles at atomic scales",
    "Machine learning algorithms improve performance through experience and data",
    "The human brain contains approximately eighty six billion neurons",
    "Supply and demand determine market prices in a competitive economy",
    "The water cycle involves evaporation condensation precipitation and collection",
    "The first law of thermodynamics states that energy cannot be created or destroyed",
    "Shakespeare wrote tragedies comedies and histories exploring human nature",
    "The Renaissance was a period of cultural rebirth in Europe from 14th to 17th century",
    "The Roman Empire dominated the Mediterranean world for over five hundred years",
    "Operating systems manage hardware resources and provide services to applications",
    "The internet is a global network connecting billions of devices through TCP IP",
]

# ── 测试 1: 训练前的输出分布 ──
print(f"\n{'─'*70}")
print("  📊 训练前: 推理各文本的输出")
print(f"{'─'*70}")

def infer_direct(head, x_enc):
    """直接用 head 的 W_p 做推理（不调用C++模型）"""
    h = x_enc.reshape(1, -1) @ head.W_p  # [1, hidden_dim]
    dec = h @ head.W_d + head.b_d
    probs = np.exp(dec - np.max(dec)) / np.sum(np.exp(dec - np.max(dec)))
    return {
        "action": int(np.argmax(dec)),
        "conf": float(np.max(probs)),
        "logits": dec.flatten().tolist(),
        "entropy": float(-np.sum(probs * np.log(probs + 1e-8))),
    }

actions_before = []
for t in test_texts:
    x = encode_text(t)
    r = infer_direct(head, x)
    actions_before.append(r["action"])
    target = abs(hash(t)) % 10
    print(f"  target={target} → pred=action#{r['action']:2d} (conf={r['conf']:.3f} ent={r['entropy']:.3f}) | {t[:45]}")

unique_before = len(set(actions_before))
print(f"\n  训练前: 唯一动作数 = {unique_before}/10 {'⚠️ 坍缩!' if unique_before <= 2 else '✅ 有分布'}")

# ── 测试 2: 小批量训练 ──
print(f"\n{'─'*70}")
print("  🏋️  小批量训练 (batch=400, 20轮)...")
print(f"{'─'*70}")

BATCH = 1000
EPOCHS = 500

# 训练用文本：从知识库中读取更多样化的数据
import subprocess
kb_dir = "/mnt/d/neuroflow-model/knowledge_base"
# 取前 1000 个文件
result = subprocess.run(
    ["find", kb_dir, "-maxdepth", "1", "-name", "*.txt", "-printf", "%f\n"],
    capture_output=True, text=True, timeout=30
)
kb_files = sorted(result.stdout.strip().split("\n"), reverse=True)[:1000]
train_texts = []
count = 0
for fname in kb_files:
    if not fname: continue
    path = os.path.join(kb_dir, fname)
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read(500).strip()
        if len(text) > 20:
            train_texts.append(text)
            count += 1
    except:
        pass
print(f"  📚 训练数据: {len(train_texts)} 条知识文本 (从 {count} 个文件)")

t0 = time.time()
for epoch in range(EPOCHS):
    # 随机选取BATCH条
    idxs = np.random.randint(0, len(train_texts), BATCH)
    X = np.zeros((BATCH, TEXT_DIM), dtype=np.float32)
    targets = []
    for j, idx in enumerate(idxs):
        t = train_texts[idx]
        X[j] = encode_text(t)
        targets.append(abs(hash(t)) % 10)
    rewards = [0.5] * BATCH
    
    result = head.direct_train_batch(X, targets, rewards)
    
    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"  epoch#{epoch+1:2d}: loss={result['loss']:.4f} ce={result['ce_loss']:.4f} "
              f"ent={result.get('entropy',0):.3f} val_loss={result['value_loss']:.4f}")

elapsed = time.time() - t0
print(f"  训练耗时: {elapsed:.1f}s ({BATCH*EPOCHS/elapsed:.0f} 条/s)")

# ── 测试 3: 训练后的输出分布 ──
print(f"\n{'─'*70}")
print("  📊 训练后: 推理各文本的输出")
print(f"{'─'*70}")

actions_after = []
correct = 0
for t in test_texts:
    x = encode_text(t)
    r = infer_direct(head, x)
    actions_after.append(r["action"])
    target = abs(hash(t)) % 10
    hit = r["action"] == target
    if hit: correct += 1
    mark = "✅" if hit else "❌"
    print(f"  {mark} target={target} → pred=action#{r['action']:2d} (conf={r['conf']:.3f} ent={r['entropy']:.3f}) | {t[:45]}")

unique_after = len(set(actions_after))
print(f"\n  训练后: 唯一动作数 = {unique_after}/10 {'✅ 有分布!' if unique_after > 2 else '⚠️ 坍缩!'}")
print(f"  命中率: {correct}/{len(test_texts)} ({correct/len(test_texts)*100:.0f}%)")

# 直观展示对比
print(f"\n{'─'*70}")
print("  训练前后对比")
print(f"{'─'*70}")
print(f"  {'':20s} {'训练前':>12s} {'训练后':>12s}")
print(f"  {'唯一动作数':20s} {unique_before:>5d}/10 {unique_after:>5d}/10")

# ── 测试 4: 区分度检验 ──
print(f"\n{'─'*70}")
print("  区分度检验：不同文本是否产生不同输出？")
print(f"{'─'*70}")
samples = [
    "Gravity is a fundamental force of nature that attracts objects with mass toward each other",
    "xylophone quantum refrigerator mushroom algorithm neutrino orchestra",
    "The CPU processes instructions through fetch decode execute cycle",
    "asdf qwerty zxcv bnm poiu ytre wqas dfgh jklz xcvb",
    "Photosynthesis produces glucose and oxygen from carbon dioxide water and sunlight",
    "the the the the the the the the the the the the the the the the the the the the",
    "hi",
    "Machine learning is a subset of artificial intelligence that enables systems to learn",
]
for t in samples:
    x = encode_text(t)
    r = infer_direct(head, x)
    target = abs(hash(t)) % 10
    bar = "█" * int(r["conf"] * 20) + "░" * (20 - int(r["conf"] * 20))
    print(f"  action#{r['action']:2d} (target={target}) {'✅' if r['action']==target else '❌'} |{bar}| {t[:55]}")

print(f"\n{'='*70}")
print("  ✅ 验证完成")
print(f"{'='*70}")

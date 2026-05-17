#!/usr/bin/env python3
"""NeuroFlow 模型效果测试 — 观察推理过程"""
import sys, os, time, numpy as np

sys.path.insert(0, "/mnt/d/neuroflow-model")
DEPLOY_PATH = "/mnt/d/neuroflow-model"
WEIGHTS_FILE = "/home/administrator/.hermes/neuroflow_weights_v4.npz"

TEXT_DIM = 1024
HIDDEN_DIM = 512
OUTPUT_DIM = 10

os.environ.setdefault("OMP_NUM_THREADS", "40")

# ── 加载模型 ──
print("🧠 加载 NeuroFlow 模型...", end=" ", flush=True)
t0 = time.time()
from neuroflow._core import create_multimodal
from neuroflow.cognition import NeuroSymbolicReasoner
from neuroflow.trainable_head import TrainableHead

model = create_multimodal(text_dim=TEXT_DIM, image_size=224, output_dim=OUTPUT_DIM, quantize=True)
reasoner = NeuroSymbolicReasoner(model)
head = TrainableHead(model, hidden_dim=HIDDEN_DIM, n_actions=OUTPUT_DIM, lr=0.01)

if os.path.exists(WEIGHTS_FILE):
    data = np.load(WEIGHTS_FILE)
    head.load_weights({
        "W_d": data.get("W_d", head.W_d),
        "b_d": data.get("b_d", head.b_d),
        "W_v": data.get("W_v", head.W_v),
        "b_v": data.get("b_v", head.b_v),
    })
    print(f"✅ 权重加载 ({round(os.path.getsize(WEIGHTS_FILE)/1024,1)}KB) {time.time()-t0:.1f}s")
else:
    print("⚠️  未找到权重文件!")
    print(f"   预期位置: {WEIGHTS_FILE}")
    sys.exit(1)

# ── 编码器 ──
def encode_text(text: str, dim: int = TEXT_DIM) -> np.ndarray:
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    n_words = min(len(words), 500)
    for i, word in enumerate(words[:n_words]):
        h = abs(hash(word)) % (2**31)
        for j in range(8):
            idx = (h + j * 2654435761) % dim
            vec[int(idx)] += 0.03 / max(n_words / 30, 1)
    vec += np.sin(np.linspace(0, np.pi * n_words / 15, dim)).astype(np.float32) * 0.08
    return vec / (np.linalg.norm(vec) + 1e-8)

# ── 推理函数 ──
def infer(text: str, verbose: bool = True) -> dict:
    x = encode_text(text).reshape(1, -1)
    trace = reasoner.reason(x, max_steps=5, verbose=verbose)

    result = {
        "conf": trace.final_confidence,
        "steps": len(trace.steps),
        "total_steps": trace.total_steps,
        "elapsed_ms": trace.elapsed_ms,
        "explanation": trace.explanation,
    }

    if trace.final_action is not None:
        result["action"] = int(np.argmax(trace.final_action))
        result["action_dist"] = trace.final_action.flatten().tolist()

    # Head前传
    head_out = head.predict(x)
    result["head_logits"] = head_out.decision.flatten().tolist()
    result["head_value"] = float(head_out.value.flatten()[0])
    result["predicted_action"] = int(np.argmax(head_out.decision.flatten()))

    return result

# ════════════════════════════════════════════
#  测试 1: 各领域知识推理
# ════════════════════════════════════════════
print("\n" + "="*70)
print("  🧪 NEUROFLOW 训练效果测试 — 推理观察")
print("="*70)

test_questions = [
    "The Pythagorean theorem states that the square of the hypotenuse equals the sum of squares of the other two sides",
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

for i, q in enumerate(test_questions):
    r = infer(q, verbose=False)
    bar = "█" * int(r["conf"] * 30) + "░" * (30 - int(r["conf"] * 30))
    top3 = sorted(r["head_logits"], reverse=True)[:3]
    print(f"\n  📝 #{i+1:2d} | ⏱{r['elapsed_ms']:.0f}ms | conf={r['conf']:.3f} |{bar}| action#{r['action']}")
    print(f"        {q[:60]}...")
    print(f"        🧠 头: logits_top3={[round(x,2) for x in top3]} | val={r['head_value']:.3f}")

# ════════════════════════════════════════════
#  测试 2: 对比测试 — 有含义 vs 无意义
# ════════════════════════════════════════════
print(f"\n{'='*70}")
print("  对比测试：有意义 ↔ 无意义/随机文本")
print(f"{'='*70}")

pairs = [
    ("有含义", "Gravity is a fundamental force of nature that attracts objects with mass"),
    ("随机   ", "xylophone quantum refrigerator mushroom algorithm neutrino orchestra"),
    ("有含义", "The CPU processes instructions through fetch decode execute cycle"),
    ("无意义", "asdf qwerty zxcv bnm poiu ytre wqas dfgh jklz xcvb 1234"),
    ("有含义", "Photosynthesis produces glucose and oxygen from carbon dioxide and water"),
    ("重复词", "the the the the the the the the the the the the the the the the the the the the"),
    ("有含义", "Natural selection favors organisms better adapted to their environment"),
    ("短   ", "hi"),
]

for label, txt in pairs:
    r = infer(txt, verbose=False)
    bar = "█" * int(r["conf"] * 30) + "░" * (30 - int(r["conf"] * 30))
    print(f"  [{label} | conf={r['conf']:.3f} |{bar}| {txt[:60]}")

# ════════════════════════════════════════════
#  测试 3: 详细推理过程展示 (verbose模式) 
# ════════════════════════════════════════════
print(f"\n{'='*70}")
print("  详细推理链展示 (verbose=True)")
print(f"{'='*70}")
show_text = "The machine learning algorithm improves performance through experience and training on large datasets"
print(f"  📝 输入: {show_text}")
print()
trace = reasoner.reason(encode_text(show_text).reshape(1, -1), max_steps=5, verbose=True)

# ════════════════════════════════════════════
#  测试 4: 头网络分析
# ════════════════════════════════════════════
print(f"\n{'='*70}")
print("  头网络权重统计")
print(f"{'='*70}")
w = head.get_weights()
for k, v in w.items():
    print(f"    {k}: shape={v.shape} mean={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} max={v.max():.4f}")

stats = head.stats()
print(f"\n  训练统计: n_updates={stats['n_updates']} avg_loss={stats['avg_loss']:.6f}")

# ════════════════════════════════════════════
#  总结
# ════════════════════════════════════════════
confs = []
actions = []
for q in test_questions:
    r = infer(q, verbose=False)
    confs.append(r["conf"])
    actions.append(r["action"])

print(f"\n{'='*70}")
print("  📊 测试总结")
print(f"{'='*70}")
print(f"    测试样本:     {len(test_questions)} 条")
print(f"    平均置信度:   {np.mean(confs):.4f} ± {np.std(confs):.4f}")
print(f"    最高置信度:   {max(confs):.4f}")
print(f"    最低置信度:   {min(confs):.4f}")
print(f"    平均推理耗时: {np.mean([infer(q,verbose=False)['elapsed_ms'] for q in test_questions[:3]]):.1f}ms")
print(f"    平均头价值:   {np.mean([infer(q,verbose=False)['head_value'] for q in test_questions]):.4f}")
print(f"    动作分布:     {np.bincount(np.array(actions), minlength=10)}")
print()
print("  ✅ 模型测试完成")

"""NeuroFlow v4.2 综合评估套件
测试模型各组件能力：重建、记忆、分离、推理、词表、速度、抗噪
"""

import sys, os, time, json, math
sys.path.insert(0, "/mnt/d/neuroflow-model")
os.environ.setdefault("OMP_NUM_THREADS", "40")

import numpy as np
from collections import Counter

DEPLOY_PATH = "/mnt/d/neuroflow-model"
WEIGHTS_FILE = "/home/administrator/.hermes/neuroflow_weights_v4.npz"
VOCAB_FILE = os.path.join(DEPLOY_PATH, "char_vocab.json")
KB_DIR = os.path.join(DEPLOY_PATH, "knowledge_base")

# ── 配置（与 daemon_v3.py 一致）──
TEXT_DIM = 1024
HIDDEN_DIM = 512
HIDDEN2_DIM = 512
OUTPUT_DIM = 1024
MEM_DIM = 256
VOCAB_SIZE = 500
MEM_SLOTS = 24
MEM_DIM_IN = 256
TOP_K = 6
TEMP_ATTN = 8.0
INPUT_NOISE = 0.05  # 与 daemon_v3.py 一致

print("=" * 72)
print("  🧪 NEUROFLOW v4.2 综合评估套件")
print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 72)

# ════════════════════════════════════════
# 1. 加载权重
# ════════════════════════════════════════
t0 = time.time()
print(f"\n📂 加载权重: {WEIGHTS_FILE}")
data = np.load(WEIGHTS_FILE)
print(f"   ✅ {len(data.keys())} 个参数, {os.path.getsize(WEIGHTS_FILE)/1024:.0f} KB")

weights = {k: data[k] for k in data.keys()}
load_time = time.time() - t0

# ════════════════════════════════════════
# 2. 编码器（与 daemon_v3.py 一致）
# ════════════════════════════════════════════
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


def random_encode(dim: int = TEXT_DIM) -> np.ndarray:
    """随机输入（模拟无意义输入）"""
    vec = np.random.randn(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


# ════════════════════════════════════════
# 3. 前向传播（与 daemon_v3.py 一致）
# ════════════════════════════════════════
W_embed = weights["W_embed"]
M_K = weights["M_K"]
M_V = weights["M_V"]
W_q = weights["W_q"]
W_gate = weights["W_gate"]
b_gate = weights["b_gate"]
W_mem_out = weights["W_mem_out"]
W_p = weights["W_p"]
W_m = weights.get("W_m", np.zeros((HIDDEN2_DIM, MEM_DIM)))
b_m = weights.get("b_m", np.zeros((1, MEM_DIM)))
W_gen = weights.get("W_gen", np.zeros((HIDDEN2_DIM, VOCAB_SIZE)))
b_gen = weights.get("b_gen", np.zeros((1, VOCAB_SIZE)))
W_d = weights["W_d"]
b_d = weights["b_d"]
W_v = weights.get("W_v", np.zeros((HIDDEN2_DIM, 1)))
b_v = weights.get("b_v", np.zeros((1, 1)))
V_in = weights.get("V_in", np.zeros((HIDDEN2_DIM, 256)))
V_out = weights.get("V_out", np.zeros((256, VOCAB_SIZE)))
V_bias = weights.get("V_bias", np.zeros((1, VOCAB_SIZE)))


def forward(x: np.ndarray, mask_ratio: float = 0.0):
    """完整前向传播（与 daemon_v3.py 精确一致）
    
    Returns dict with all layer outputs.
    x: shape (N, TEXT_DIM) — 原始编码（不含掩码）
    """
    N = x.shape[0]
    
    # ── 掩码 + 噪声（评估时默认无掩码） ──
    if mask_ratio > 0:
        mask = np.random.rand(N, TEXT_DIM) > mask_ratio
        X_masked = x * mask.astype(np.float32)
        X_masked += np.random.randn(N, TEXT_DIM).astype(np.float32) * INPUT_NOISE
    else:
        X_masked = x.copy()
    
    # ── W_embed 可学习投影（残差，daemon_v3.py L832-834）──
    X_proj = X_masked @ W_embed
    X_proj = np.maximum(X_proj, 0)
    X_in = X_masked + X_proj * 0.1  # 残差
    
    # ── 第一层: X_in → W_p → h1 → ReLU ──
    h1 = X_in @ W_p  # (N, 512)
    h1_relu = np.maximum(h1, 0)
    
    # ── Gated Memory Bank ──
    Q = h1_relu @ W_q  # (N, 256)
    K_norm = M_K / (np.linalg.norm(M_K, axis=1, keepdims=True) + 1e-8)  # (24, 256)
    scores_raw = Q @ K_norm.T  # (N, 24) 原始分数
    
    # 温度缩放 Softmax + Top-6 截断
    temp = 8.0
    scores_max = np.max(scores_raw, axis=1, keepdims=True)
    scores_exp = np.exp(temp * (scores_raw - scores_max))
    topk = 6
    scores_topk = np.partition(scores_exp, -topk, axis=1)[:, -topk:-topk+1].min(axis=1, keepdims=True)
    scores_exp = scores_exp * (scores_exp >= scores_topk)
    attn = scores_exp / (np.sum(scores_exp, axis=1, keepdims=True) + 1e-8)  # (N, 24)
    
    mem_read = attn @ M_V  # (N, 256)
    mem_feat = mem_read @ W_mem_out  # (N, 512)
    
    # 门控融合
    gate = 1.0 / (1.0 + np.exp(-(h1_relu @ W_gate + b_gate)))  # (N, 512)
    h_mem = gate * h1_relu + (1.0 - gate) * mem_feat  # (N, 512)
    h3 = np.maximum(h_mem, 0)  # (N, 512)
    h3_relu = h3
    
    # ── SAE 稀疏瓶颈: top-50 ──
    SPARSE_K = 50
    h3_abs = np.abs(h3_relu)
    h3_thresh = np.partition(h3_abs, -SPARSE_K, axis=1)[:, -SPARSE_K:-SPARSE_K+1]
    sae_mask = (h3_abs >= h3_thresh).astype(np.float32)
    h3_relu = h3_relu * sae_mask  # 仅保留 top-50
    
    # ── 重建头 ──
    recon = h3_relu @ W_d + b_d  # (N, 1024)
    
    # ── 词表头 ──
    word_logits = h3_relu @ W_gen + b_gen  # (N, 500)
    word_sigmoid = 1.0 / (1.0 + np.exp(-word_logits))  # sigmoid
    
    # ── retrieved_mem 头 ──
    mem_pred = h3_relu @ W_m + b_m  # (N, 256)
    
    # ── 价值头 ──
    value = h3_relu @ W_v + b_v  # (N, 1)
    
    # 注意力分析
    topk_indices = np.argsort(-attn, axis=1)[:, :topk]
    
    return {
        "recon": recon,
        "h1_relu": h1_relu,
        "h3_relu": h3_relu,  # SAE 后
        "h3_density": (np.abs(h3_relu) > 1e-6).mean(),
        "sae_mask": sae_mask,
        "mem_read": mem_read,
        "mem_attn": attn,
        "mem_topk": topk_indices,
        "gate_sigmoid": gate.mean(),
        "word_logits": word_logits,
        "word_sigmoid": word_sigmoid,
        "mem_pred": mem_pred,
        "value": value,
    }


# ════════════════════════════════════════
# 测试 1：基础前向测试
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试1: 基础前向传播 & 参数统计")
print("#" * 72)

print(f"\n📊 参数统计:")
total_params = 0
for k, v in weights.items():
    n = v.size
    total_params += n
    print(f"  {k:25s}  {str(v.shape):20s}  {n:>10,d} params  {v.mean():+.4f} ± {v.std():.4f}")

print(f"\n  {'─'*60}")
print(f"  总计: {total_params:,d} 参数 ({total_params*4/1024/1024:.2f} MB @ FP32)")

# 前向测试
test_text = "The scientific method involves observation hypothesis testing and peer review"
x = encode_text(test_text).reshape(1, -1)
fw = forward(x)

print(f"\n🔍 前向测试 (输入: '{test_text[:40]}...'):")
print(f"  recon shape:    {fw['recon'].shape}")
print(f"  recon[0][:5]:   {fw['recon'][0,:5].tolist()}")
print(f"  h3 density:     {fw['h3_density']*100:.1f}% (目标: ≤10%)")
print(f"  gate激活均值:   {fw['gate_sigmoid']:.4f}")
print(f"  value:          {fw['value'][0,0]:.4f}")


# ════════════════════════════════════════
# 测试 2：重建精度
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试2: 重建精度 (Reconstruction MSE)")
print("#" * 72)

# 内置知识（来自 daemon_v3.py 的 BUILTIN_KNOWLEDGE）
test_texts = [
    "The scientific method involves observation hypothesis testing and peer review",
    "Atoms consist of protons neutrons and electrons orbiting the nucleus",
    "DNA is a double helix structure containing genetic information for all life",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
    "The theory of relativity states that space and time are interwoven into spacetime",
    "Quantum mechanics describes the behavior of particles at atomic scales",
    "Evolution by natural selection explains how species adapt over generations",
    "The periodic table organizes elements by atomic number and chemical properties",
    "Newton laws of motion describe the relationship between force mass and acceleration",
    "Entropy measures the disorder in a system and always increases in isolated systems",
    "Algorithms are step by step procedures for solving computational problems",
    "Artificial neural networks are inspired by the structure of biological brains",
    "The internet is a global network connecting billions of devices through TCP IP",
    "Supply and demand determine market prices in a competitive economy",
    "Classical conditioning associates a neutral stimulus with a reflexive response",
    "Cell division produces new cells for growth repair and reproduction",
    "The greenhouse effect traps heat in Earth atmosphere through CO2 and methane",
    "Binary number system uses only zero and one forming the basis of computing",
    "Probability theory quantifies uncertainty and forms the foundation of statistics",
    "The hippocampus is essential for forming new episodic memories",
]

recon_errors = []
for text in test_texts:
    x = encode_text(text).reshape(1, -1)
    fw = forward(x)
    mse = ((fw["recon"] - x) ** 2).mean()
    recon_errors.append(mse)

print(f"\n  测试样本: {len(test_texts)} 条内置知识")
print(f"  平均重建 MSE: {np.mean(recon_errors):.6f}")
print(f"  中位数 MSE:   {np.median(recon_errors):.6f}")
print(f"  最小 MSE:     {min(recon_errors):.6f}")
print(f"  最大 MSE:     {max(recon_errors):.6f}")
print(f"  Std:          {np.std(recon_errors):.6f}")

# 与 daemon 内部报告对比
print(f"\n  📊 与 daemon 内部报告对比:")
print(f"     当前 daemon: recon=0.000775-0.000779")
print(f"     本测试评估:  recon={np.mean(recon_errors):.6f}")


# ════════════════════════════════════════
# 测试 3：Gated Memory Bank 利用率
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试3: Gated Memory Bank 利用率")
print("#" * 72)

all_topk_counts = Counter()
for text in test_texts:
    x = encode_text(text).reshape(1, -1)
    fw = forward(x)
    for idx in fw["mem_topk"][0]:
        all_topk_counts[idx] += 1

active_slots = len(all_topk_counts)
slot_usage = {i: all_topk_counts.get(i, 0) for i in range(MEM_SLOTS)}

print(f"\n  记忆槽总数: {MEM_SLOTS}")
print(f"  活跃记忆槽: {active_slots} ({active_slots/MEM_SLOTS*100:.1f}%)")
print(f"  利用率:     {'✅ 良好' if active_slots >= MEM_SLOTS*0.5 else '⚠️ 偏低'}")
print(f"\n  各槽命中分布:")
for i in range(MEM_SLOTS):
    bar = "█" * all_topk_counts.get(i, 0) + "░" * max(0, 20 - all_topk_counts.get(i, 0))
    print(f"    槽{i:2d}: {all_topk_counts.get(i, 0):3d} 次  {bar}")

# 记忆槽范数检查
M_K_norms = np.linalg.norm(M_K, axis=1)
M_V_norms = np.linalg.norm(M_V, axis=1)
print(f"\n  记忆键范数: mean={M_K_norms.mean():.4f} std={M_K_norms.std():.4f}")
print(f"  记忆值范数: mean={M_V_norms.mean():.4f} std={M_V_norms.std():.4f}")

# 注意力分布均匀性
all_entropy = []
for text in test_texts:
    x = encode_text(text).reshape(1, -1)
    fw = forward(x)
    # 注意力分布（top-6 上的熵）
    attn = fw["mem_attn"][0]
    nonzero_mask = attn > 1e-6
    if nonzero_mask.sum() > 0:
        probs = attn[nonzero_mask]
        probs = probs / probs.sum()
    else:
        probs = np.ones(6) / 6
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    all_entropy.append(entropy)

print(f"\n  注意力熵（top-6）：mean={np.mean(all_entropy):.3f} / max={math.log(TOP_K):.3f}")


# ════════════════════════════════════════
# 测试 4：域间分离度
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试4: 域间分离度 (Domain Separation)")
print("#" * 72)

domains = {
    "science": [
        "Atoms consist of protons neutrons and electrons",
        "DNA is a double helix structure containing genetic information",
        "Photosynthesis converts carbon dioxide into glucose",
        "Quantum mechanics describes the behavior of particles at atomic scales",
        "The speed of light in vacuum is approximately three hundred thousand km/s",
        "Electromagnetic waves include radio microwaves infrared visible",
    ],
    "math": [
        "The Pythagorean theorem square of hypotenuse equals sum of squares",
        "Prime numbers are integers divisible only by themselves and one",
        "The Fibonacci sequence appears in nature from sunflower seeds",
        "Complex numbers consist of real and imaginary parts",
        "Probability theory quantifies uncertainty and statistics",
        "Linear algebra deals with vectors matrices and linear equations",
    ],
    "cs": [
        "Algorithms are step by step procedures for solving problems",
        "Artificial neural networks are inspired by biological brains",
        "Object oriented programming organizes code into classes",
        "Machine learning algorithms improve through experience and data",
        "Database systems organize and retrieve structured data using SQL",
        "Cryptography protects information through encryption techniques",
    ],
    "random_noise": [
        "asdf qwerty zxcv bnm poiu ytre wqas",
        "xylophone quantum refrigerator mushroom algorithm",
        "jfaks lqowpe zmxncbv qwertyuiop asdfghjkl",
        "abcdefghijklmnopqrstuvwxyz qwertyuiop",
        "zzz xxx ccc vvv bbb nnn mmm qqq www eee",
        "random noise meaningless text for testing separation",
    ],
}

def get_h2(text: str):
    x = encode_text(text).reshape(1, -1)
    fw = forward(x)
    return fw["h3_relu"][0]

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

domain_reprs = {}
for domain, texts in domains.items():
    h2s = np.array([get_h2(t) for t in texts])
    domain_reprs[domain] = h2s.mean(axis=0)

print(f"\n  域间余弦相似度矩阵:")
domain_names = list(domains.keys())
print(f"  {'':15s}", end="")
for d in domain_names:
    print(f"  {d:12s}", end="")
print()

intra_sims = []
inter_sims = []
for i, d1 in enumerate(domain_names):
    print(f"  {d1:15s}", end="")
    for j, d2 in enumerate(domain_names):
        sim = cos_sim(domain_reprs[d1], domain_reprs[d2])
        print(f"  {sim:+.4f}    ", end="")
        if i == j:
            # 域内相似度（自我对比）
            texts = domains[d1]
            h2s = np.array([get_h2(t) for t in texts])
            intra = np.mean([cos_sim(h2s[k], h2s[l]) for k in range(len(texts)) for l in range(k+1, len(texts))])
            intra_sims.append(intra)
        elif i < j:
            inter_sims.append(sim)
    print()

avg_intra = np.mean(intra_sims) if intra_sims else 0
avg_inter = np.mean(inter_sims) if inter_sims else 0
sep_score = avg_intra - avg_inter

print(f"\n  域内平均相似度: {avg_intra:+.4f}")
print(f"  域间平均相似度: {avg_inter:+.4f}")
print(f"  分离度得分:     {sep_score:+.4f} {'✅ 良好' if sep_score > 0.1 else '⚠️ 偏低'}")

# 随机 vs 有意义分离
meaningful_h2s = np.array([get_h2(t) for t in test_texts])
random_h2s = np.array([get_h2(f"random text {i} for testing purposes with diverse vocabulary words") for i in range(20)])

meaningful_mean = meaningful_h2s.mean(axis=0)
random_mean = random_h2s.mean(axis=0)
meaningful_random_sim = cos_sim(meaningful_mean, random_mean)

# 随机噪声 vs 有意义
noise_vec = random_encode()
noise_h2 = get_h2(" ".join(["asdf"]*20))
meaningful_h2 = get_h2("The scientific method involves observation hypothesis testing")
noise_to_meaningful = cos_sim(noise_h2, meaningful_h2)
meaningful_to_meaningful = cos_sim(get_h2("Atoms consist of protons neutrons"), get_h2("DNA double helix genetic information"))

print(f"\n  有意义↔有意义的h2相似度: {meaningful_to_meaningful:+.4f}")
print(f"  有意义↔无意义的h2相似度: {noise_to_meaningful:+.4f}")
print(f"  有区分度? {'✅ 是' if meaningful_to_meaningful > noise_to_meaningful + 0.1 else '⚠️ 需要改善'}")


# ════════════════════════════════════════
# 测试 5：词表预测能力
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试5: 词表预测能力 (Vocab Head)")
print("#" * 72)

# 加载字符词表
char_vocab = []
if os.path.exists(VOCAB_FILE):
    with open(VOCAB_FILE, encoding='utf-8') as f:
        char_vocab = json.load(f)
    print(f"  词表大小: {len(char_vocab)} 个字符")
else:
    print(f"  ⚠️ 字符词表未找到: {VOCAB_FILE}")

# 简单字符预测测试
print(f"\n  字符预测样本:")
for text in [
    "energy",
    "algorithm",
    "quantum",
    "entropy",
    "neuron",
    "photosynthesis",
]:
    x = encode_text(text).reshape(1, -1)
    fw = forward(x)
    logits = fw["word_logits"][0]
    top5_indices = np.argsort(-logits)[:5]
    top5_probs = np.exp(logits[top5_indices]) / np.sum(np.exp(logits))
    
    top5_chars = []
    for idx in top5_indices:
        if idx < len(char_vocab):
            top5_chars.append(f"'{char_vocab[idx]}'")
        else:
            top5_chars.append(f"#{idx}")
    
    print(f"    '{text}': top5={top5_chars} probs={[f'{p:.1%}' for p in top5_probs]}")


# ════════════════════════════════════════
# 测试 6：推理速度
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试6: 推理速度 (Inference Speed)")
print("#" * 72)

N_warmup = 100
N_bench = 500

# Warmup
for _ in range(N_warmup):
    _ = forward(encode_text("warmup test").reshape(1, -1))

# Benchmark single
t1 = time.time()
for _ in range(N_bench):
    _ = forward(encode_text("benchmark test for measuring speed").reshape(1, -1))
t_single = (time.time() - t1) / N_bench

# Benchmark batch
batch_x = np.array([encode_text(f"test item number {i} for batch evaluation") for i in range(100)])
t2 = time.time()
for _ in range(10):
    _ = forward(batch_x)
t_batch = (time.time() - t2) / 10
throughput_batch = 100 / t_batch

print(f"\n  单样本推理: {t_single*1000:.2f} ms")
print(f"  批量推理(100): {t_batch*1000:.1f} ms ({throughput_batch:.0f} items/s)")
print(f"  批量推理(单条等效): {t_batch/100*1000:.3f} ms/item")
print(f"  ⚡ 相当于: {throughput_batch*3600:.0f} items/hour")


# ════════════════════════════════════════
# 测试 7：抗噪鲁棒性
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试7: 抗噪鲁棒性 (Noise Robustness)")
print("#" * 72)

noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

for noise in noise_levels:
    mses_noisy = []
    for text in test_texts[:10]:
        x = encode_text(text).reshape(1, -1)
        x_noisy = x + np.random.randn(*x.shape).astype(np.float32) * noise
        x_noisy = x_noisy / (np.linalg.norm(x_noisy) + 1e-8)
        fw_clean = forward(x)
        fw_noisy = forward(x_noisy)
        mse = ((fw_noisy["recon"] - fw_clean["recon"]) ** 2).mean()
        mses_noisy.append(mse)
    
    avg_noise_mse = np.mean(mses_noisy)
    base_recon = np.mean([((forward(encode_text(t).reshape(1,-1))["recon"] - encode_text(t).reshape(1,-1)) ** 2).mean() for t in test_texts[:10]])
    noise_ratio = avg_noise_mse / (base_recon + 1e-10)
    
    quality = "✅ 稳健" if noise_ratio < 5 else ("⚠️ 有影响" if noise_ratio < 20 else "❌ 退化")
    print(f"  噪声 σ={noise:.2f}: 重建偏移 {avg_noise_mse:.6f} (基准 {base_recon:.6f}, 比 {noise_ratio:.1f}x) {quality}")


# ════════════════════════════════════════
# 测试 8: 模型容量验证 — 过拟合测试
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试8: 模型容量验证 (过拟合 vs 泛化)")
print("#" * 72)

# 对同一知识反复测试 — 重建应一致
texts_repeat = test_texts[:5]
repeat_mses = []
for _ in range(3):
    batch_mses = []
    for t in texts_repeat:
        x = encode_text(t).reshape(1, -1)
        fw = forward(x)
        mse = ((fw["recon"] - x) ** 2).mean()
        batch_mses.append(mse)
    repeat_mses.append(np.mean(batch_mses))

print(f"  同数据 3 次重建 MSE 稳定性: {[f'{m:.6f}' for m in repeat_mses]}")
print(f"  标准差: {np.std(repeat_mses):.8f} {'✅ 稳定' if np.std(repeat_mses) < 0.0001 else '⚠️ 波动'}")

# 随机权重: 模型应不记忆随机噪声
random_texts = [f"random_{i}_data_{j}_test_{k}" for i in range(20) for j in range(5) for k in range(3)]
random_mses = []
t0 = time.time()
for t in random_texts:
    x = encode_text(t).reshape(1, -1)
    fw = forward(x)
    mse = ((fw["recon"] - x) ** 2).mean()
    random_mses.append(mse)
random_time = time.time() - t0

print(f"\n  随机文本 (300条) 平均重建 MSE: {np.mean(random_mses):.6f}")
print(f"  随机文本 vs 知识文本 MSE 比: {np.mean(random_mses)/np.mean(recon_errors):.2f}x")
print(f"  模型未记忆噪声? {'✅ 是 (随机MSE > 知识MSE)' if np.mean(random_mses) > np.mean(recon_errors) * 1.2 else '⚠️ 检查'}")


# ════════════════════════════════════════
# 测试 9: SAE 稀疏性分析
# ════════════════════════════════════════
print("\n" + "#" * 72)
print("#  🔬 测试9: SAE 稀疏性分析")
print("#" * 72)

all_h2_nonzero = []
for text in test_texts + random_texts[:50]:
    x = encode_text(text).reshape(1, -1)
    fw = forward(x)
    h3 = fw["h3_relu"][0]
    nonzero = (np.abs(h3) > 1e-6).sum()
    all_h2_nonzero.append(nonzero)

avg_nonzero = np.mean(all_h2_nonzero)
print(f"  隐藏层维度: {HIDDEN_DIM}")
print(f"  平均激活神经元数: {avg_nonzero:.0f} / {HIDDEN_DIM}")
print(f"  稀疏度: {avg_nonzero/HIDDEN_DIM*100:.1f}%")
print(f"  SAE 目标: top-50 = 9.8%，当前 {avg_nonzero/HIDDEN_DIM*100:.1f}%")
print(f"  {'✅ SAE 正常工作' if avg_nonzero <= 55 else '⚠️ SAE 需要调整' if avg_nonzero <= 100 else '❌ SAE 效果不佳'}")


# ════════════════════════════════════════
# 🏁 综合评分
# ════════════════════════════════════════
print("\n" + "=" * 72)
print("  🏁 NEUROFLOW v4.2 综合评估报告")
print("=" * 72)

# 打分标准
scores = {}
scores["加载时间"] = (load_time, f"{load_time:.1f}s", "≤5s" if load_time <= 5 else ">5s")
scores["重建精度"] = (np.mean(recon_errors), f"{np.mean(recon_errors):.6f}", f"≤0.001: {'✅' if np.mean(recon_errors) <= 0.001 else ('⚠️' if np.mean(recon_errors) <= 0.005 else '❌')}")
scores["记忆利用率"] = (active_slots/MEM_SLOTS, f"{active_slots}/{MEM_SLOTS} ({active_slots/MEM_SLOTS*100:.0f}%)", f"{'✅' if active_slots >= MEM_SLOTS*0.5 else '⚠️'}")
scores["域分离度"] = (sep_score, f"{sep_score:+.4f}", f"{'✅' if sep_score > 0.1 else '⚠️'}")
scores["稀疏度"] = (avg_nonzero/HIDDEN_DIM, f"{avg_nonzero/HIDDEN_DIM*100:.1f}%", f"{'✅' if avg_nonzero/HIDDEN_DIM <= 0.1 else '⚠️'}")
scores["推理速度"] = (throughput_batch, f"{throughput_batch:.0f} items/s", f"{'✅' if throughput_batch >= 5000 else '⚠️'}")
scores["抗噪(σ=0.1)"] = (noise_ratio_for_01 := np.mean([((forward(encode_text(t).reshape(1,-1) + np.random.randn(1,1024).astype(np.float32)*0.1)["recon"] - forward(encode_text(t).reshape(1,-1))["recon"])**2).mean() for t in test_texts[:5]]) / (np.mean(recon_errors)+1e-10), f"{noise_ratio_for_01:.1f}x", f"{'✅' if noise_ratio_for_01 < 5 else '⚠️'}")

print(f"\n{'测试项':20s} {'值':>15s} {'评价':>8s}")
print(f"{'─'*50}")
for name, (val, val_str, marker) in scores.items():
    # Color code
    flag = "✅" if "✅" in marker else ("⚠️" if "⚠️" in marker else "❌")
    print(f"  {name:18s}  {val_str:>15s}  {flag}")

# 总体评估
passed = sum(1 for _, _, m in scores.values() if "✅" in m)
warned = sum(1 for _, _, m in scores.values() if "⚠️" in m)
failed = sum(1 for _, _, m in scores.values() if "❌" in m)
total_tests = len(scores)

print(f"\n  {'─'*50}")
print(f"  📊 总体: {passed}/{total_tests} 通过 | {warned} 警告 | {failed} 失败")
print(f"  评分: {'⭐ 优秀' if passed == total_tests else '⭐ 良好' if passed >= total_tests*0.7 else '⭐ 需改进'}")

# 与 daemon 当前状态对比
print(f"\n  📈 与 daemon_v3 训练状态对比:")
print(f"    daemon 汇报: epoch 201, {60920000+40000*2:.0f}+ 条已训练")
print(f"    自动进化:    1524 次")
print(f"    当前重建:    0.000775 (daemon) vs {np.mean(recon_errors):.6f} (本评估)")
print(f"    词表 top5:   22.0% (daemon)")
print(f"    连续运行:    (>12小时，0崩溃)")

print("\n" + "=" * 72)
print("  ✅ 评估完成")
print("=" * 72)

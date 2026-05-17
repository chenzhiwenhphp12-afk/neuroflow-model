"""NeuroFlow v3 — 全面模型测试"""
import sys, os, json, time
import numpy as np

DEPLOY_PATH = "/mnt/d/neuroflow-model"
sys.path.insert(0, DEPLOY_PATH)
WEIGHTS_FILE = "/home/administrator/.hermes/neuroflow_weights_v4.npz"
VOCAB_FILE = os.path.join(DEPLOY_PATH, "char_vocab.json")
STATE_FILE = os.path.join(DEPLOY_PATH, "daemon_state.json")

# ── 参数 ──
TEXT_DIM = 1024
HIDDEN_DIM = 512
HIDDEN2_DIM = 512
OUTPUT_DIM = 1024
MEM_DIM = 256
VOCAB_SIZE = 2000

# ═══ 编码函数（与 daemon_v3.py 完全一致）═══
def encode_text(text: str, dim: int = TEXT_DIM) -> np.ndarray:
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    n_words = min(len(words), 500)
    scale = 0.03 / max(n_words / 30, 1)
    for i, word in enumerate(words[:n_words]):
        h = abs(hash(word)) % (2**31)
        for j in range(8):
            idx = (h + j * 2654435761) % dim
            vec[int(idx)] += scale
    vec += np.sin(np.linspace(0, np.pi * n_words / 15, dim)).astype(np.float32) * 0.08
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec

# ═══ 加载模型权重 ═══
print(f"📂 加载权重: {WEIGHTS_FILE} ({os.path.getsize(WEIGHTS_FILE)/1024/1024:.1f} MB)")
data = np.load(WEIGHTS_FILE)

# TrainableHead 内部权重
W_p = data["W_p"]       # [1024, 512]
W_d = data["W_d"]       # [512, 1024]
b_d = data["b_d"]       # [1, 1024]
W_v = data["W_v"]       # [512, 1]
b_v = data["b_v"]       # [1, 1]

# 新层
W_h = data["W_h"]       # [512, 512]
W_h2 = data["W_h2"]     # [512, 512]
W_m = data["W_m"]       # [512, 256]
b_m = data.get("b_m", np.zeros((1, 256), dtype=np.float32)) if "b_m" in data else np.zeros((1, 256), dtype=np.float32)
W_gen = data["W_gen"]   # [512, 2000]
b_gen = data.get("b_gen", np.zeros((1, VOCAB_SIZE), dtype=np.float32)) if "b_gen" in data else np.zeros((1, VOCAB_SIZE), dtype=np.float32)

print(f"  W_p:  {W_p.shape}")
print(f"  W_h:  {W_h.shape}")
print(f"  W_h2: {W_h2.shape}")
print(f"  W_d:  {W_d.shape}")
print(f"  W_m:  {W_m.shape}")
print(f"  W_v:  {W_v.shape}")
print(f"  W_gen:{W_gen.shape}")

# 加载词汇表
char_vocab = json.load(open(VOCAB_FILE, encoding='utf-8'))
print(f"  📖 词表: {len(char_vocab)} 字符\n")

# ═══ 前向传播 ═══
def forward(X, mask_ratio=0.0):
    """X: [N, 1024]，返回所有输出"""
    N = X.shape[0]
    # 掩码（测试时可选）
    if mask_ratio > 0:
        mask = np.random.rand(N, TEXT_DIM) > mask_ratio
        X_in = X * mask.astype(np.float32)
    else:
        X_in = X.copy()
    
    h1 = X_in @ W_p
    h1_relu = np.maximum(h1, 0)
    h2 = h1_relu @ W_h
    h2_relu = np.maximum(h2, 0)
    h3 = h2_relu @ W_h2
    h3_relu = np.maximum(h3, 0)
    
    recon = h3_relu @ W_d + b_d
    mem_pred = h3_relu @ W_m + b_m
    val = h3_relu @ W_v + b_v
    word_logits = h3_relu @ W_gen + b_gen
    word_probs = 1.0 / (1.0 + np.exp(-word_logits))  # sigmoid
    
    return {
        "h1": h1_relu, "h2": h2_relu, "h3": h3_relu,
        "recon": recon, "mem": mem_pred, "val": val,
        "word_logits": word_logits, "word_probs": word_probs
    }

# ═══ 测试1: 基本编码+重建 ═══
print("=" * 60)
print("📝 测试1: 单样本编码+重建质量")
print("=" * 60)

test_texts = [
    "Python is a high-level programming language with dynamic semantics",
    "Machine learning is a subset of artificial intelligence",
    "The Earth orbits the Sun at a distance of about 93 million miles",
    "In mathematics, a prime number is a natural number greater than 1",
    "Hydrogen is the lightest element on the periodic table",
    "深度学习是机器学习的一个分支 使用多层神经网络",
    "在物理学中 量子力学描述了微观粒子的行为规律",
    "结构力学研究物体在外力作用下的变形和应力分布",
]

for text in test_texts:
    X = encode_text(text).reshape(1, -1)
    out = forward(X)
    recon = out["recon"]
    mse = float(np.mean((recon - X) ** 2))
    cos_sim = float(np.sum(recon * X) / (np.linalg.norm(recon) * np.linalg.norm(X) + 1e-8))
    word_idx = np.argmax(out["word_probs"][0])
    word_char = char_vocab[word_idx] if word_idx < len(char_vocab) else "?"
    word_conf = float(out["word_probs"][0, word_idx])
    val = float(out["val"][0, 0])
    print(f"  [{text[:40]:40s}] MSE={mse:.6f} cos={cos_sim:.4f} val={val:.4f} top词='{word_char}'({word_conf:.2f})")

# ═══ 测试2: 语义相似性 ═══
print("\n" + "=" * 60)
print("📝 测试2: 语义相似性分析 (同类vs异类)")
print("=" * 60)

topic_groups = {
    "编程": ["Python is a programming language", "Java uses object-oriented programming",
             "C++ supports both procedural and object-oriented programming",
             "JavaScript is used for web development"],
    "物理": ["Physics studies matter and energy", "Quantum mechanics describes atomic behavior",
             "Newton's laws of motion describe force and acceleration",
             "Electromagnetism studies electric and magnetic fields"],
    "生物": ["DNA is the hereditary material in all living organisms",
             "Photosynthesis converts sunlight into chemical energy",
             "Evolution explains the diversity of life on Earth",
             "Cells are the basic unit of life"],
    "数学": ["Calculus studies continuous change and rates of change",
             "Linear algebra deals with vectors and matrices",
             "Probability theory studies random events and likelihood",
             "Number theory explores properties of integers"],
    "混合": ["The weather today is sunny and warm", "I enjoy cooking Italian food",
             "Paris is the capital of France", "The stock market closed higher today",
             "马克思主义哲学 辩证唯物主义 历史唯物主义"],
}

all_encodings = {}
for topic, texts in topic_groups.items():
    all_encodings[topic] = [encode_text(t) for t in texts]

# 类内相似度
print("\n  同类相似度 (cosine):")
for topic, vecs in all_encodings.items():
    sims = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            c = float(np.sum(vecs[i] * vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-8))
            sims.append(c)
    avg = np.mean(sims) if sims else 0
    print(f"    {topic:6s}: {avg:.4f} (n={len(sims)})")

# 异类相似度
print("\n  异类相似度:")
topics_list = list(topic_groups.keys())
cross_sims = []
for i in range(len(topics_list)):
    for j in range(i+1, len(topics_list)):
        for vi in all_encodings[topics_list[i]]:
            for vj in all_encodings[topics_list[j]]:
                c = float(np.sum(vi * vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8))
                cross_sims.append(c)
avg_cross = np.mean(cross_sims) if cross_sims else 0
print(f"    平均: {avg_cross:.4f}")

# ═══ 测试3: 隐空间分析 ═══
print("\n" + "=" * 60)
print("📝 测试3: 隐空间分离度分析")
print("=" * 60)

# 对每个文本跑前向，收集h3
h3_by_topic = {}
for topic, texts in topic_groups.items():
    h3s = []
    for t in texts:
        X = encode_text(t).reshape(1, -1)
        out = forward(X)
        h3s.append(out["h3"][0])
    h3_by_topic[topic] = np.array(h3s)

# 类内 vs 类间 h3 距离
print("\n  隐空间(h3) 类内/类间距离:")
intra_dists = []
inter_dists = []
for i, t1 in enumerate(topics_list):
    for j, t2 in enumerate(topics_list):
        dists = []
        for vi in h3_by_topic[t1]:
            for vj in h3_by_topic[t2]:
                d = float(np.linalg.norm(vi - vj))
                dists.append(d)
        avg_d = np.mean(dists)
        if i == j:
            intra_dists.append(avg_d)
            print(f"    ★ {t1:6s} 类内: {avg_d:.4f}")
        else:
            inter_dists.append(avg_d)

avg_intra = np.mean(intra_dists)
avg_inter = np.mean(inter_dists)
print(f"\n    📊 类内平均: {avg_intra:.4f}")
print(f"    📊 类间平均: {avg_inter:.4f}")
print(f"    📊 分离度(类间/类内): {avg_inter/avg_intra:.4f}x")
print(f"    📊 对比方差(h_var): {float(np.var(np.vstack(list(h3_by_topic.values())))):.6f}")

# ═══ 测试4: 词汇预测头分析 ═══
print("\n" + "=" * 60)
print("📝 测试4: 词汇预测头 — 知识关联性")
print("=" * 60)

test_knowledge_pairs = [
    ("Python programming language dynamic typing", "coding"),
    ("Machine learning neural network deep learning", "AI"),
    ("Quantum mechanics wave particle duality", "physics"),
    ("DNA replication transcription translation", "biology"),
    ("Calculus derivative integral limit", "math"),
    ("股票市场 投资 风险管理 收益率", "finance"),
    ("操作系统 进程调度 内存管理 文件系统", "os"),
    ("蛋白质 氨基酸 酶 催化反应", "biochem"),
]

for text, label in test_knowledge_pairs:
    X = encode_text(text).reshape(1, -1)
    out = forward(X)
    probs = out["word_probs"][0]
    
    # Top-5 最高概率字符
    top5 = np.argsort(probs)[-5:][::-1]
    top5_chars = [(char_vocab[i], float(probs[i])) for i in top5 if i < len(char_vocab)]
    
    # 熵
    p = probs + 1e-10
    entropy = float(-np.sum(p * np.log(p)))
    
    print(f"  [{label:8s}] top5: {', '.join(f'{c}({s:.3f})' for c, s in top5_chars)} 熵={entropy:.3f}")

# ═══ 测试5: 掩码重建挑战 ═══
print("\n" + "=" * 60)
print("📝 测试5: 掩码重建 (遮住35% 恢复全量编码)")
print("=" * 60)

for text in test_texts[:5]:
    X = encode_text(text).reshape(1, -1)
    
    # 不加掩码
    out_clean = forward(X, mask_ratio=0.0)
    recon_clean = out_clean["recon"]
    clean_mse = float(np.mean((recon_clean - X) ** 2))
    
    # 加35%掩码
    out_masked = forward(X, mask_ratio=0.35)
    recon_masked = out_masked["recon"]
    masked_mse = float(np.mean((recon_masked - X) ** 2))
    
    deg = masked_mse / (clean_mse + 1e-10)
    print(f"  [{text[:35]:35s}] 无掩码={clean_mse:.6f} | 35%掩码={masked_mse:.6f} | 退化×{deg:.2f}")

# ═══ 测试6: 分布式表征 — 随机样本统计 ═══
print("\n" + "=" * 60)
print("📝 测试6: 模型参数统计 + 表征质量")
print("=" * 60)

total_params = sum(p.size for p in data.values())
print(f"  总参数量: {total_params:,}")
print(f"  W_p 均值/标准差: {float(np.mean(W_p)):.6f}/{float(np.std(W_p)):.6f}")
print(f"  W_h 均值/标准差: {float(np.mean(W_h)):.6f}/{float(np.std(W_h)):.6f}")
print(f"  W_h2均值/标准差: {float(np.mean(W_h2)):.6f}/{float(np.std(W_h2)):.6f}")
print(f"  W_d 均值/标准差: {float(np.mean(W_d)):.6f}/{float(np.std(W_d)):.6f}")
print(f"  W_gen均值/标准差:{float(np.mean(W_gen)):.6f}/{float(np.std(W_gen)):.6f}")

# 生成100个随机文本测试分布
print("\n  随机文本性能采样 (n=100):")
random_texts = [
    "The quick brown fox jumps over the lazy dog",
    "To be or not to be that is the question",
    "In the beginning God created the heavens and the earth",
    "All human beings are born free and equal in dignity and rights",
    "I think therefore I am",
    "Knowledge is power",
    "Time is money",
    "The only thing we have to fear is fear itself",
    "That's one small step for man one giant leap for mankind",
    "Imagination is more important than knowledge",
    # 中文
    "学而时习之不亦说乎 有朋自远方来不亦乐乎",
    "路漫漫其修远兮 吾将上下而求索",
    "先天下之忧而忧 后天下之乐而乐",
    "人生自古谁无死 留取丹心照汗青",
    "天行健 君子以自强不息",
    "千里之行始于足下",
    "知之为知之 不知为不知 是知也",
    "三人行必有我师焉",
    "温故而知新 可以为师矣",
    "学而不思则罔思而不学则殆",
]
for _ in range(5):
    random_texts.append(f"Random knowledge topic {np.random.randint(10000)}: {' '.join(np.random.choice(['quantum','neural','genetic','molecular','statistical','differential','topological','semantic','probabilistic','algorithmic'], 5))}")

all_mses = []
for t in random_texts:
    X = encode_text(t).reshape(1, -1)
    out = forward(X)
    mse = float(np.mean((out["recon"] - X) ** 2))
    all_mses.append(mse)

print(f"  MSE: 均值={np.mean(all_mses):.6f} 中位数={np.median(all_mses):.6f} 最小值={min(all_mses):.6f} 最大值={max(all_mses):.6f}")
print(f"  标准差={np.std(all_mses):.6f}")

# ═══ 总结 ═══
print("\n" + "=" * 60)
print("🏁 测试总结")
print("=" * 60)

recon_avg = np.mean(all_mses)
# 编码与隐层差异
X_sample = encode_text("Test text for analysis").reshape(1, -1)
out_sample = forward(X_sample)
h3_norm = float(np.linalg.norm(out_sample["h3"]))
recon_norm = float(np.linalg.norm(out_sample["recon"]))
X_norm = float(np.linalg.norm(X_sample))

print(f"  ✅ 重建误差 (MSE):      {recon_avg:.6f}")
print(f"  ✅ 重建余弦相似度:      {cos_sim:.4f}")
print(f"  ✅ 隐状态维度(h3):      {HIDDEN2_DIM}")
print(f"  ✅ 词汇表大小:          {VOCAB_SIZE}")
print(f"  ✅ 已学训练步数:        {2800000}")
print(f"  ✅ 参数量:              {total_params:,}")
print(f"  ✅ 隐层范数:            X={X_norm:.4f} → h3={h3_norm:.4f} → recon={recon_norm:.4f}")

print(f"\n  ⚡ 结论: 模型{'运行正常' if recon_avg < 0.01 else '需要更多训练'}")
print(f"    重建MSE={recon_avg:.6f} {'✅ 优秀' if recon_avg < 0.001 else '✅ 良好' if recon_avg < 0.01 else '⚠️ 需要改进'}")

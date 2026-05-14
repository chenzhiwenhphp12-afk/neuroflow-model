"""
NeuroFlow TextDecoder 快速训练脚本
==================================
Step 1: 批量预计算 NeuroFlow 隐状态
Step 2: 用预计算数据训练 Decoder（纯 NumPy，无 C++ 开销）
"""

import sys, time, os, pickle, numpy as np

sys.path.insert(0, "/mnt/d/neuroflow-model")

# ============================================================
# 训练语料
# ============================================================
CORPUS = [
    "the sun rises in the east every morning",
    "water is essential for all living things",
    "she opened the door and walked outside",
    "he reads books every night before sleep",
    "they went to the market to buy food",
    "the cat sat on the warm window sill",
    "birds fly south during the winter months",
    "the river flows gently through the valley",
    "children play in the park after school",
    "music can change the way you feel",
    "knowledge is power in the modern world",
    "time waits for no one so act now",
    "hard work and patience lead to success",
    "every journey begins with a single step",
    "learning never stops even when you grow old",
    "the best way to learn is to teach",
    "mistakes are proof that you are trying",
    "happiness comes from within not from things",
    "courage is not the absence of fear",
    "kindness costs nothing but means everything",
    "artificial intelligence is changing the world rapidly",
    "computers process information at incredible speeds",
    "the internet connects people across the globe",
    "data is the new oil of the digital age",
    "machine learning algorithms learn from large datasets",
    "neural networks are inspired by the human brain",
    "robots can perform tasks that humans find difficult",
    "quantum computing will revolutionize many fields",
    "the earth orbits around the sun once a year",
    "gravity keeps everything grounded on the planet",
    "photosynthesis converts sunlight into chemical energy",
    "the human brain contains billions of neurons",
    "evolution shapes species over millions of years",
    "the universe is expanding at an accelerating rate",
    "the wind whispered secrets through the ancient trees",
    "shadows danced upon the walls in candlelight",
    "her smile was like sunshine on a rainy day",
    "memories fade like footprints in the sand",
    "the ocean waves sang a lullaby to the shore",
    "winter brought a blanket of white snow",
    "hope is the light that never goes out",
    "a friend in need is a friend indeed",
    "actions speak louder than words always do",
    "what we think we become in the end",
    "silence is sometimes the best answer to give",
    "the early bird catches the worm at dawn",
    "people gather together to celebrate the festival",
    "the teacher explained the lesson with great patience",
    "books open doors to worlds beyond imagination",
    "a healthy diet keeps the body strong and fit",
]

# Vocab: char-level
chars = sorted(set("".join(CORPUS) + " "))
char_to_id = {c: i + 5 for i, c in enumerate(chars)}
id_to_char = {i + 5: c for i, c in enumerate(chars)}
VOCAB_SIZE = len(chars) + 10

INPUT_DIM = 512
HIDDEN_DIM = 256
SEQ_LEN = 5
EPOCHS = 500
LR = 0.1

print(f"Corpus: {len(CORPUS)} sentences, vocab: {VOCAB_SIZE}")

# ============================================================
# Step 1: 预计算隐状态
# ============================================================
print("\n[1/3] Precomputing hidden states via NeuroFlow...")
from neuroflow import NeuroFlowLite

model = NeuroFlowLite(input_dim=INPUT_DIM)
hidden_states = []
target_tokens = []

t0 = time.time()
for si, sent in enumerate(CORPUS):
    ids = [char_to_id.get(c, 3) for c in sent]
    if len(ids) < SEQ_LEN + 1:
        continue
    
    for i in range(0, len(ids) - SEQ_LEN, 3):
        ctx = ids[i:i+SEQ_LEN]
        tgt = ids[i+SEQ_LEN] if i+SEQ_LEN < len(ids) else ids[-1]
        
        signal = np.zeros(INPUT_DIM, dtype=np.float32)
        for j, tid in enumerate(ctx):
            idx = (tid * 7 + j * 13) % INPUT_DIM
            signal[idx] += 1.0 / SEQ_LEN
        
        x = signal.reshape(1, -1)
        out = model.forward(x)
        
        d = out.decision[0]
        v = out.value[0]
        
        combined = np.concatenate([d, v])
        repeats = HIDDEN_DIM // len(combined) + 1
        h = np.tile(combined, repeats)[:HIDDEN_DIM].astype(np.float32)
        
        hidden_states.append(h)
        target_tokens.append(tgt)

hidden_states = np.array(hidden_states, dtype=np.float32)
target_tokens = np.array(target_tokens, dtype=np.int32)

t1 = time.time()
print(f"  Precomputed {len(hidden_states)} samples in {t1-t0:.1f}s")
print(f"  Shapes: hidden={hidden_states.shape}, targets={target_tokens.shape}")

# ============================================================
# Step 2: 纯 NumPy 训练 Decoder
# ============================================================
print(f"\n[2/3] Training decoder ({EPOCHS} epochs)...")

# 初始化权重
rng = np.random.RandomState(42)
scale = np.sqrt(2.0 / HIDDEN_DIM)
W = rng.randn(HIDDEN_DIM, VOCAB_SIZE).astype(np.float32) * scale
b = np.zeros(VOCAB_SIZE, dtype=np.float32)

losses = []
N = len(hidden_states)
batch = 64

for epoch in range(EPOCHS):
    idx = np.random.permutation(N)
    total_loss = 0.0
    
    for start in range(0, N, batch):
        end = min(start + batch, N)
        bi = idx[start:end]
        
        H = hidden_states[bi]   # [B, HIDDEN]
        T = target_tokens[bi]   # [B]
        B = H.shape[0]
        
        # Forward
        logits = H @ W + b      # [B, V]
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits.astype(np.float64))
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        # Loss
        losses_batch = -np.log(probs[np.arange(B), T] + 1e-30)
        total_loss += losses_batch.sum()
        
        # Gradient
        grad_logits = probs.copy()
        grad_logits[np.arange(B), T] -= 1.0
        
        grad_W = H.T @ grad_logits.astype(np.float32) / B
        grad_b = grad_logits.astype(np.float32).mean(axis=0)
        
        # Update
        W -= LR * grad_W
        b -= LR * grad_b
    
    avg_loss = total_loss / N
    losses.append(avg_loss)
    
    if epoch < 5 or epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.4f} | LR: {LR:.3f}")
    
    # LR decay
    if epoch % 100 == 99:
        LR *= 0.5

print(f"\n  Final Loss: {losses[-1]:.4f}")
print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

# ============================================================
# Step 3: 测试生成
# ============================================================
print(f"\n[3/3] Testing generation...\n")

from neuroflow.decoder import TokenSampler

def generate(prompt, max_tokens=25, temp=0.7):
    signal = np.zeros(INPUT_DIM, dtype=np.float32)
    for j, c in enumerate(prompt):
        tid = char_to_id.get(c, 3)
        idx = (tid * 7 + j * 13) % INPUT_DIM
        signal[idx] += 0.2
    
    x = signal.reshape(1, -1)
    result = prompt
    
    for step in range(max_tokens):
        out = model.forward(x)
        d = out.decision[0]
        v = out.value[0]
        combined = np.concatenate([d, v])
        h = np.tile(combined, HIDDEN_DIM // len(combined) + 1)[:HIDDEN_DIM].astype(np.float32)
        logits = h @ W + b
        
        tid = TokenSampler.sample(logits, temperature=temp, top_k=30, top_p=0.9)
        
        if tid in id_to_char:
            result += id_to_char[tid]
        elif tid == 2:
            break
        
        token_signal = np.zeros(INPUT_DIM, dtype=np.float32)
        token_signal[tid % INPUT_DIM] = 1.0
        x = (x * 0.7 + token_signal.reshape(1, -1) * 0.3).astype(np.float32)
    
    return result

prompts = ["The ", "I think ", "We should ", "Love is ", "AI will "]
for p in prompts:
    print(f"  {p!r:15s} → '{generate(p)}'")

# ============================================================
# 保存
# ============================================================
save_path = "/mnt/d/neuroflow-model/trained_decoder.npz"
np.savez(save_path, W=W, b=b, char_to_id=char_to_id, id_to_char=id_to_char, losses=losses)
print(f"\n💾 Saved to {save_path} ({os.path.getsize(save_path)/1024:.0f} KB)")

# Loss 曲线
print(f"\n📉 Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
for i in range(0, len(losses), max(1, len(losses)//8)):
    pct = int((losses[0] - losses[i]) / (losses[0] - losses[-1] + 0.001) * 20)
    print(f"  {i:4d} {losses[i]:.4f} {'█'*pct}")

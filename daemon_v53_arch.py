#!/usr/bin/env python3
"""
NeuroFlow v5.3 架构升级版 (ArchUpgrade)
========================================
集成的架构改进:
  1. AdaptiveSAE — 自适应稀疏编码 (k动态调整)
  2. DeepSAEStack — 深层SAE栈 + 残差连接
  3. IterativeMemory — 迭代记忆递归读写
  4. Cosine LR + Plateau扰动 (v5.2已有)
"""
import sys, os, time, json, random, numpy as np
from datetime import datetime
from collections import deque

# 强制stdout行缓冲（确保实时输出）
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
elif hasattr(sys.stdout, 'buffer'):
    # Python <3.7 fallback
    pass

# 自动检测路径
os.environ.setdefault("OMP_NUM_THREADS", "40")
if os.path.exists("/mnt/d/neuroflow-model"):
    BASE_PATH = "/mnt/d/neuroflow-model"
    HOME_PATH = "/home/administrator"
elif os.path.exists("/work/home/acxavb8ge5/neuroflow-model"):
    BASE_PATH = "/work/home/acxavb8ge5/neuroflow-model"
    HOME_PATH = "/work/home/acxavb8ge5"
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    HOME_PATH = os.path.expanduser("~")
sys.path.insert(0, BASE_PATH)

DEPLOY_PATH = BASE_PATH
STATE_FILE = os.path.join(DEPLOY_PATH, "daemon_v53_state.json")
WEIGHTS_FILE = os.path.join(DEPLOY_PATH, "neuroflow_v53_weights.npz")
KB_DIR = os.path.join(DEPLOY_PATH, "knowledge_base")

# 配置
BATCH_SIZE = 50
SEQ_LEN = 128
TRAIN_STEPS_PER_EPOCH = 500

# 架构参数
D_MODEL = 512
D_MEM = 256
MEM_SLOTS = 32
VOCAB_SIZE = 5000
WINDOW_SIZE = 8
GAMMA = 0.85

# 升级开关
ENABLE_ADAPTIVE_SAE = True    # 自适应SAE
ENABLE_DEEP_STACK = True      # 深层SAE栈 (需ENABLE_ADAPTIVE_SAE)
ENABLE_ITERATIVE_MEM = True   # 迭代记忆
SAE_N_LAYERS = 2              # 深层栈层数
MEM_N_ITERS = 2               # 记忆迭代次数

from ops_v5 import *
from ops_v5_arch import (AdaptiveSAE, DeepSAEStack, 
                          iterative_memory_read_write)
from tokenizer_v5_cn import get_tokenizer, PAD_ID, UNK_ID

os.makedirs(KB_DIR, exist_ok=True)


# ── 工具函数 (从ops_v5复制) ──
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def get_pe_table(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(max_len)[:, np.newaxis]
    dim = np.arange(d_model)[np.newaxis, :]
    pe[:, 0::2] = np.sin(pos / (10000 ** (2 * dim[:, 0::2] / d_model)))
    pe[:, 1::2] = np.cos(pos / (10000 ** (2 * dim[:, 1::2] / d_model)))
    return pe

def causal_window_gating_operator(X, W_g, b_g, window_size=WINDOW_SIZE, gamma=GAMMA):
    L, D = X.shape
    W = min(window_size, L)
    X_padded = np.vstack([np.zeros((W - 1, D), dtype=X.dtype), X])
    shape_stride, elem_stride = X_padded.strides
    X_window = np.lib.stride_tricks.as_strided(X_padded, 
        shape=(L, W, D), strides=(shape_stride, shape_stride, elem_stride))
    gamma_decay = (gamma ** np.arange(W))[::-1][np.newaxis, :, np.newaxis]
    C_temporal = np.sum(X_window * gamma_decay, axis=1)
    Gate = sigmoid(X @ W_g + b_g)
    return C_temporal * Gate

def memory_read_write(c, M_Bank, W_read, W_write, W_to_mem,
                      position=None, max_pos=8000, alpha=0.85, damping_strength=0.7):
    r_slot = sigmoid(c @ W_read)
    w_slot = sigmoid(c @ W_write)
    if position is not None and max_pos > 0:
        ratio = position / max_pos
        damp = 1.0 - damping_strength * (1.0 / (1.0 + np.exp(-8.0 * (ratio - 0.5))))
        log_damp = 1.0 / (np.log(np.e + position) ** alpha)
        w_slot = w_slot * damp * log_damp
    mem_context = r_slot @ M_Bank
    c_proj = c @ W_to_mem
    delta = w_slot.T @ c_proj
    M_Bank_new = (1.0 - w_slot.T) * M_Bank + delta
    return mem_context, M_Bank_new

def cross_entropy_loss(logits, targets, mask=None):
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    log_probs = (logits - max_logits) - np.log(np.sum(exp_logits, axis=-1, keepdims=True))
    N = targets.shape[0]
    token_losses = -log_probs[np.arange(N), targets]
    if mask is not None:
        loss = np.sum(token_losses * mask) / (np.sum(mask) + 1e-8)
    else:
        loss = np.mean(token_losses)
    return loss

def vocab_gradient(logits, targets):
    N, V = logits.shape
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-8)
    grad = probs.copy()
    grad[np.arange(N), targets] -= 1.0
    return grad / N

def diagnose_temporal_variance(H_layer, num_chunks=8):
    if len(H_layer.shape) == 3:
        H_seq = H_layer[0]
    else:
        H_seq = H_layer
    L, D = H_seq.shape
    chunk_size = max(1, L // num_chunks)
    v_spectrum = []
    for k in range(num_chunks):
        chunk = H_seq[k * chunk_size:(k + 1) * chunk_size, :]
        v_spectrum.append(float(np.mean(np.var(chunk, axis=1))))
    lvr = v_spectrum[-1] / (v_spectrum[0] + 1e-8)
    return lvr, v_spectrum


class NeuroFlowV53Arch:
    """v5.3 架构升级版"""
    
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.kb_contents = []
        self.kb_idx = 0
        self.epoch = 0
        self.pe_table = get_pe_table(SEQ_LEN, D_MODEL)
        self.train_step_counter = 0
        self._recent_losses = deque(maxlen=50)
        self._recent_vars = deque(maxlen=50)
        
        # 架构升级模块
        self.adaptive_sae = AdaptiveSAE(D_MODEL, base_k=65, k_range=(30, 120))
        self.deep_stack = DeepSAEStack(D_MODEL, n_layers=SAE_N_LAYERS) if ENABLE_DEEP_STACK else None
        
        self._init_model()
        self._load_weights()
        self._reset_pad_head()
        self._load_state()
        self._preload_kb()
        self.last_status = time.time()
        self.arch_stats = {
            "k_history": deque(maxlen=100),
            "layer_vars": deque(maxlen=100),
            "iter_conf": deque(maxlen=100),
            "lvr_history": deque(maxlen=100),
        }
    
    def _init_model(self):
        np.random.seed(42)
        scale = np.sqrt(2.0 / D_MODEL)
        self.W_embed = np.random.randn(VOCAB_SIZE, D_MODEL).astype(np.float32) * 0.01
        self.W_g = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * scale
        self.b_g = np.zeros(D_MODEL, dtype=np.float32)
        K_init = np.random.randn(MEM_SLOTS, D_MEM).astype(np.float32)
        K_init = K_init / (np.linalg.norm(K_init, axis=1, keepdims=True) + 1e-8)
        self.M_K = K_init
        self.M_V = np.random.randn(MEM_SLOTS, D_MEM).astype(np.float32) * 0.01
        self.W_read = np.random.randn(D_MODEL, MEM_SLOTS).astype(np.float32) * 0.01
        self.W_write = np.random.randn(D_MODEL, MEM_SLOTS).astype(np.float32) * 0.01
        self.W_to_mem = np.random.randn(D_MODEL, D_MEM).astype(np.float32) * scale
        self.W_proj = np.random.randn(D_MODEL, D_MEM).astype(np.float32) * scale
        self.W_out = np.random.randn(D_MEM, VOCAB_SIZE).astype(np.float32) * 0.01
        self.b_out = np.zeros(VOCAB_SIZE, dtype=np.float32)
        self.lr = 0.0033
        self.n_updates = 0
        self.total_loss = 0.0
        self.state = {"started": datetime.now().isoformat(), "topics": 0,
                      "evolutions": 0, "fitness": 0.0, "errors": 0,
                      "total_loss": 0.0, "train_steps": 0, "auto_evolutions": 0}
    
    def _reset_pad_head(self):
        """重置PAD输出头，消除14080 batch积累的PAD偏置
        将W_out[:,0]和b_out[0]设为与其他token相近的分布
        """
        # 用其他token的均值±标准差来重置PAD头
        non_pad_cols = np.concatenate([self.W_out[:, :0], self.W_out[:, 1:]], axis=1)
        col_mean = np.mean(non_pad_cols, axis=1)  # (D_MEM,)
        col_std = np.std(non_pad_cols, axis=1)
        self.W_out[:, 0] = np.random.randn(D_MEM).astype(np.float32) * np.mean(col_std) * 0.1
        
        # 重置b_out[0]为其他bias的均值附近
        other_bias = np.concatenate([self.b_out[:0], self.b_out[1:]])
        self.b_out[0] = float(np.mean(other_bias)) + np.random.randn() * float(np.std(other_bias)) * 0.1
        print("  🧹 重置PAD输出头: W_out[:,0] std=%.4f, b_out[0]=%.4f" % (
            float(np.std(self.W_out[:, 0])), float(self.b_out[0])))
    
    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f: self.state.update(json.load(f))
            except: pass
    
    def _save_state(self):
        self.state["total_loss"] = float(self.total_loss)
        self.state["train_steps"] = int(self.n_updates)
        with open(STATE_FILE, "w") as f: json.dump(self.state, f)
    
    def _save_weights(self):
        np.savez_compressed(WEIGHTS_FILE,
            W_embed=self.W_embed, W_g=self.W_g, b_g=self.b_g,
            M_K=self.M_K, M_V=self.M_V, W_read=self.W_read,
            W_write=self.W_write, W_to_mem=self.W_to_mem,
            W_proj=self.W_proj, W_out=self.W_out, b_out=self.b_out)
    
    def _load_weights(self):
        """从之前保存的权重文件恢复（如有）"""
        if os.path.exists(WEIGHTS_FILE):
            try:
                data = np.load(WEIGHTS_FILE)
                for k in ['W_embed', 'W_g', 'b_g', 'M_K', 'M_V',
                          'W_read', 'W_write', 'W_to_mem', 'W_proj', 'W_out', 'b_out']:
                    setattr(self, k, data[k])
                print("  💾 恢复权重: %s" % WEIGHTS_FILE)
                return True
            except Exception as e:
                print("  ⚠️  权重恢复失败: %s" % e)
                return False
        return False
    
    def _preload_kb(self):
        t0 = time.time()
        if not os.path.isdir(KB_DIR): return
        files = sorted(os.listdir(KB_DIR), reverse=True)[:8000]
        contents = []
        for fname in files:
            try:
                with open(os.path.join(KB_DIR, fname), 'r', encoding='utf-8') as f:
                    text = f.read(4000).strip()
                if len(text) > 20: contents.append(text)
            except: pass
        self.kb_contents = contents
        print("  📖 加载 %d 条知识 (%.0fs)" % (len(contents), time.time()-t0))
    
    def _get_next_text(self):
        if not self.kb_contents:
            return "Q: What is 2+2? A: 4"
        text = self.kb_contents[self.kb_idx % len(self.kb_contents)]
        self.kb_idx += 1
        if self.kb_idx >= len(self.kb_contents):
            self.kb_idx = 0
            self.epoch += 1
            random.shuffle(self.kb_contents)
        if random.random() < 0.3:
            return "Q: " + text[:200] + " A: " + text[200:400] if len(text) > 200 else "Q: " + text[:100] + " A: " + text[100:200]
        return text
    
    def forward(self, token_ids):
        N, L = token_ids.shape
        X_embed = self.W_embed[token_ids]
        X = X_embed * np.sqrt(D_MODEL) + self.pe_table[:L]
        h_flat = X.reshape(-1, D_MODEL)
        C_flat = causal_window_gating_operator(h_flat, self.W_g, self.b_g)
        h = C_flat.reshape(N, L, D_MODEL).reshape(-1, D_MODEL)
        h_var = float(np.var(h))
        
        # ── 架构升级1+2: 自适应SAE + 深层栈 ──
        if ENABLE_DEEP_STACK and self.deep_stack is not None:
            h_sparse, layer_k, layer_vars = self.deep_stack.forward(h, h_var)
            self.arch_stats["k_history"].extend(layer_k)
            self.arch_stats["layer_vars"].extend(layer_vars)
        elif ENABLE_ADAPTIVE_SAE:
            h_sparse, k = self.adaptive_sae.forward(h, h_var)
            self.arch_stats["k_history"].append(k)
        else:
            h_abs = np.abs(h)
            thresh = np.partition(h_abs, -65, axis=1)[:, -65:-65+1]
            h_sparse = h * (h_abs >= thresh).astype(np.float32)
        
        # 池化
        h_pooled = h_sparse.reshape(N, L, D_MODEL).mean(axis=1)
        
        # ── 架构升级3: 迭代记忆递归 ──
        if ENABLE_ITERATIVE_MEM:
            mem_out, M_V_new, iter_trace = iterative_memory_read_write(
                h_pooled[:1], self.M_V, self.W_read, self.W_write, self.W_to_mem,
                n_iters=MEM_N_ITERS, position=self.train_step_counter % SEQ_LEN,
                max_pos=SEQ_LEN
            )
            self.arch_stats["iter_conf"].extend(iter_trace)
            self.M_V = M_V_new
            h_mem = h_sparse + 0.1 * np.tile(mem_out @ self.W_to_mem.T, (N * L, 1))
        else:
            mem_out, self.M_V = memory_read_write(h_pooled[:1], self.M_V,
                self.W_read, self.W_write, self.W_to_mem,
                position=self.train_step_counter % SEQ_LEN, max_pos=SEQ_LEN)
            h_mem = h_sparse + 0.1 * np.tile(mem_out @ self.W_to_mem.T, (N * L, 1))
        
        h_proj = np.maximum(h_mem @ self.W_proj, 0)
        logits = h_proj @ self.W_out + self.b_out
        return logits, h_mem, h_var, 1.0
    
    def train_step(self, texts):
        N = min(BATCH_SIZE, len(texts))
        step = self.train_step_counter
        warmup_steps = 500
        total_steps = 3000
        if step < warmup_steps:
            lr = 0.0033 * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = 0.001 + (0.0033 - 0.001) * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        self.lr = lr
        
        if step > 0 and step % 200 == 0 and len(self._recent_losses) >= 50:
            recent = list(self._recent_losses)[-50:]
            plateau = (max(recent) - min(recent)) < 0.3 and len(set([round(x,1) for x in recent])) < 3
            if plateau and np.random.random() < 0.3:
                noise = np.random.randn(*self.W_embed.shape).astype(np.float32) * 0.03
                self.W_embed += noise
                print("  ⚡ 平台期扰动: W_embed += 3% noise", flush=True)
        
        ids_list = [self.tokenizer.encode(text, SEQ_LEN) for text in texts[:N]]
        token_ids = np.stack(ids_list, axis=0)
        targets = np.roll(token_ids, -1, axis=1).ravel()
        # loss_mask: 只对非PAD的target计算loss
        loss_mask = (targets != PAD_ID).astype(np.float32)
        
        logits, h, h_var, _ = self.forward(token_ids)
        loss = cross_entropy_loss(logits, targets, mask=loss_mask)
        grad = vocab_gradient(logits, targets)
        # 梯度mask: 只对非PAD位置更新权重
        grad = grad * loss_mask[:, np.newaxis]
        
        h_proj = np.maximum(h @ self.W_proj, 0)
        d_proj = (grad @ self.W_out.T) * (h_proj > 0).astype(np.float32)
        
        self.W_out -= lr * (h_proj.T @ grad + 0.002 * self.W_out)
        self.b_out -= lr * np.sum(grad, axis=0)
        self.W_proj -= lr * (h.T @ d_proj + 0.002 * self.W_proj)
        
        self.n_updates += 1
        self.total_loss += loss
        self.train_step_counter += 1
        self._recent_losses.append(loss)
        self._recent_vars.append(h_var)
        return loss, h_var, logits, h
    
    def run(self):
        print("\n🧠 NeuroFlow v5.3 架构升级版")
        features = []
        if ENABLE_ADAPTIVE_SAE: features.append("自适应SAE")
        if ENABLE_DEEP_STACK: features.append("%d层SAE栈" % SAE_N_LAYERS)
        if ENABLE_ITERATIVE_MEM: features.append("%d次迭代记忆" % MEM_N_ITERS)
        print("🔧 架构特征: " + ", ".join(features))
        print("📚 知识: %d条" % len(self.kb_contents))
        print("⚙️  batch=%d | seq=%d | dim=%d | vocab=%d" % (BATCH_SIZE, SEQ_LEN, D_MODEL, VOCAB_SIZE))
        
        batch_count = 0
        while True:
            try:
                texts = [self._get_next_text() for _ in range(BATCH_SIZE)]
                loss, h_var, logits, h = self.train_step(texts)
                self.state["topics"] += len(texts)
                batch_count += 1
                
                if batch_count % TRAIN_STEPS_PER_EPOCH == 0:
                    self.epoch += 1
                    fit_val = 1.0 - np.clip(loss / np.log(VOCAB_SIZE), 0, 1)
                    self.state["fitness"] = fit_val
                    self.state["evolutions"] += 1
                    print("  🧬 进化 #%d: fit=%.4f loss=%.4f" % (self.state["evolutions"], fit_val, loss))
                    self._save_weights()
                    self._save_state()
                
                if batch_count % 10 == 0:
                    # Diagnostic
                    if h.size >= D_MODEL:
                        lvr, v_spec = diagnose_temporal_variance(h[:1].reshape(1, -1, D_MODEL)[:, :128, :], 2) if h.shape[1] >= 128 else (0, [])
                    else:
                        lvr, v_spec = 0, []
                    ks = list(self.arch_stats["k_history"])[-5:]
                    k_str = "k=%s" % "-".join(str(int(x)) for x in ks) if ks else ""
                    print("  [%d] 📦 batch#%d e%d loss=%.4f var=%.4f %s |▌" % (
                        self.state["topics"], batch_count, self.epoch, loss, h_var, k_str), flush=True)
                
            except Exception as e:
                print("  ❌ 错误: %s" % e, flush=True)
                self.state["errors"] += 1


if __name__ == "__main__":
    d = NeuroFlowV53Arch()
    d.run()

#!/usr/bin/env python3
"""
NeuroFlow v5.2 中文推理版
==========================
架构: 中文BPE → Embedding → PE → Causal Window → SAE → MemBank → IndexCE
训练: QA格式 (Q:... A:...) + 链式思维CoT
"""
import sys, os, time, json, random, numpy as np
from datetime import datetime
from collections import deque

# Auto-detect environment: WSL vs HPC
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
STATE_FILE = os.path.join(DEPLOY_PATH, "daemon_v5_cn_state.json")
WEIGHTS_FILE = os.path.join(DEPLOY_PATH, "neuroflow_v5_cn_weights.npz")
KB_DIR = os.path.join(DEPLOY_PATH, "knowledge_base")

BATCH_SIZE = 50
SEQ_LEN = 128
TRAIN_STEPS_PER_EPOCH = 500

from ops_v5 import *
from ops_v5 import (AdaptivePositionBlender, EvolutionMonitorV5,
                    diagnose_temporal_variance, DampedGatedMemoryBankV5WithIGR)
from tokenizer_v5_cn import get_tokenizer, PAD_ID, UNK_ID

# Force local seq length (override ops_v5's SEQ_LEN=8000)
SEQ_LEN = 128

os.makedirs(KB_DIR, exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "40")


class NeuroFlowV5CN:
    """v5.2 中文推理版"""
    
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.kb_contents = []
        self.kb_idx = 0
        self.epoch = 0
        self.pe_table = get_pe_table(SEQ_LEN, D_MODEL)
        self.pos_blender = AdaptivePositionBlender(D_MODEL, SEQ_LEN, warmup_steps=1000)
        self.evo_monitor = EvolutionMonitorV5(VOCAB_SIZE, var_target=1.0)
        self.mem_bank = DampedGatedMemoryBankV5WithIGR(mem_dim=D_MEM, base_alpha=0.85)
        self.train_step_counter = 0
        self._recent_losses = deque(maxlen=50)
        self._recent_vars = deque(maxlen=50)
        self._init_model()
        self._load_state()
        self._preload_kb()
        self.last_status = time.time()
    
    def _init_model(self):
        np.random.seed(42)
        scale = np.sqrt(2.0 / D_MODEL)
        self.W_embed = np.random.randn(VOCAB_SIZE, D_MODEL).astype(np.float32) * 0.01
        self.W_g = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * scale
        self.b_g = np.zeros(D_MODEL, dtype=np.float32)
        K_init = np.random.randn(MEM_SLOTS, D_MEM).astype(np.float32)
        K_init = K_init / (np.linalg.norm(K_init, axis=1, keepdims=True) + 1e-8)
        self.M_K = K_init; self.M_V = np.random.randn(MEM_SLOTS, D_MEM).astype(np.float32) * 0.01
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
                      "total_loss": 0.0, "train_steps": 0, "auto_evolutions": 0,
                      "evo_stats": {}}
    
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
    
    def _preload_kb(self):
        t0 = time.time()
        if not os.path.isdir(KB_DIR): return
        files = sorted(os.listdir(KB_DIR), reverse=True)[:8000]  # 8K条(含CoT)
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
        """获取训练文本 QA格式"""
        if not self.kb_contents:
            return "Q: What is 2+2? A: 4"
        
        text = self.kb_contents[self.kb_idx % len(self.kb_contents)]
        self.kb_idx += 1
        if self.kb_idx >= len(self.kb_contents):
            self.kb_idx = 0
            self.epoch += 1
            random.shuffle(self.kb_contents)
        
        # Wrap partial texts in QA format for reasoning training
        if random.random() < 0.3:
            return "Q: " + text[:200] + " A: " + text[200:400] if len(text) > 200 else "Q: " + text[:100] + " A: " + text[100:200]
        return text
    
    def forward(self, token_ids):
        N, L = token_ids.shape
        X_embed = self.W_embed[token_ids]
        _, lambda_t = self.pos_blender.blend(X_embed[0], self.train_step_counter)
        X = X_embed * self.pos_blender.embed_scale + lambda_t * self.pos_blender.PE_table[:L]
        h_flat = X.reshape(-1, D_MODEL)
        C_flat = causal_window_gating_operator(h_flat, self.W_g, self.b_g)
        h = C_flat.reshape(N, L, D_MODEL).reshape(-1, D_MODEL)
        h_abs = np.abs(h)
        thresh = np.partition(h_abs, -65, axis=1)[:, -65:-65+1]
        h = h * (h_abs >= thresh).astype(np.float32)
        h_pooled = h.reshape(N, L, D_MODEL).mean(axis=1)
        mem_out, _ = memory_read_write(h_pooled[:1], self.M_V, self.W_read, self.W_write, self.W_to_mem)
        h = h + 0.1 * np.tile(mem_out @ self.W_to_mem.T, (N * L, 1))
        h_proj = np.maximum(h @ self.W_proj, 0)
        logits = h_proj @ self.W_out + self.b_out
        return logits, h, float(np.var(h)), lambda_t
    
    def train_step(self, texts):
        N = min(BATCH_SIZE, len(texts))
        # Cosine annealing LR: 0.0033 -> 0.001 over 3000 batches
        warmup_steps = 500
        total_steps = 3000
        step = self.train_step_counter
        if step < warmup_steps:
            lr = 0.0033 * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = 0.001 + (0.0033 - 0.001) * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        self.lr = lr
        
        # W_embed noise perturbation at plateau detection (every 200 batches)
        if step > 0 and step % 200 == 0 and len(self._recent_losses) >= 50:
            recent = list(self._recent_losses)[-50:]
            plateau = (max(recent) - min(recent)) < 0.3 and len(set([round(x,1) for x in recent])) < 3
            if plateau and np.random.random() < 0.3:  # 30% chance
                noise = np.random.randn(*self.W_embed.shape).astype(np.float32) * 0.03
                self.W_embed += noise
                print("  ⚡ 平台期扰动: W_embed += 3% noise", flush=True)
        ids_list = [self.tokenizer.encode(text, SEQ_LEN) for text in texts[:N]]
        token_ids = np.stack(ids_list, axis=0)
        targets = np.roll(token_ids, -1, axis=1).ravel()
        targets[targets == PAD_ID] = 0
        targets_mask = (targets > 0) | (np.roll(np.arange(token_ids.size) % SEQ_LEN != SEQ_LEN-1, -1))
        
        logits, h, h_var, lambda_t = self.forward(token_ids)
        loss = cross_entropy_loss(logits, targets)
        grad = vocab_gradient(logits, targets)
        
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
        print("\n🧠 NeuroFlow v5.2 中文推理版启动")
        print("📚 知识: %d条" % len(self.kb_contents))
        print("⚙️  batch=%d | seq=%d | dim=%d | vocab=%d | 中文BPE" % (BATCH_SIZE, SEQ_LEN, D_MODEL, VOCAB_SIZE))
        
        batch_count = 0
        while True:
            try:
                texts = [self._get_next_text() for _ in range(BATCH_SIZE)]
                loss, h_var, logits, h = self.train_step(texts)
                self.state["topics"] += len(texts)
                batch_count += 1
                
                if batch_count % TRAIN_STEPS_PER_EPOCH == 0:
                    self.epoch += 1
                    fit_val, evo_m = self.evo_monitor.evaluate(0, loss, h_var)
                    self.state["fitness"] = fit_val
                    self.state["evolutions"] += 1
                    print("  🧬 进化 #%d: fit=%.4f loss=%.4f" % (self.state["evolutions"], fit_val, loss))
                    self._save_weights()
                    self._save_state()
                
                if batch_count % 10 == 0:
                    lambda_t = self.pos_blender.blend(self.W_embed[:1], self.train_step_counter)[1]
                    lvr, v_spec = diagnose_temporal_variance(h[:1].reshape(1, 1, D_MODEL), 2) if h.size > 0 else (0, [])
                    print("  [%d] 📦 batch#%d e%d loss=%.4f var=%.4f λ=%.2f |▌" % (
                        self.state["topics"], batch_count, self.epoch, loss, h_var, lambda_t), flush=True)
                
            except Exception as e:
                print("  ❌ 错误: %s" % e, flush=True)
                self.state["errors"] += 1


if __name__ == "__main__":
    d = NeuroFlowV5CN()
    d.run()

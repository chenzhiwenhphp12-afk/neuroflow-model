#!/usr/bin/env python3
"""
NeuroFlow v4.6 HPC Standalone — 纯NumPy, 无C++依赖
自适应稀疏度 + 自动进化 + 记忆槽32 + W_embed加速
"""
import numpy as np, time, os, sys, json
from typing import Optional

# ═══ 配置 ═══
TEXT_DIM=1024; HIDDEN_DIM=512; OUTPUT_DIM=1024; MEM_DIM=256
MEM_SLOTS=32; MEM_DIM_IN=256; VOCAB_SIZE=500; BATCH_SIZE=40000
CONTRASTIVE_WEIGHT=1.2; WEIGHT_DECAY=0.001; LR=0.01

class NFv46HPC:
    def __init__(self, weights_path=None):
        self._init_model()
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
            print(f"  Weights loaded: {weights_path}")
    
    def _init_model(self):
        np.random.seed(42)
        scale = np.sqrt(2.0 / TEXT_DIM)
        self.W_embed = np.random.randn(TEXT_DIM, TEXT_DIM).astype(np.float32) * 0.01
        self.W_p = np.random.randn(TEXT_DIM, HIDDEN_DIM).astype(np.float32) * scale
        # Memory Bank
        K = np.random.randn(MEM_SLOTS, MEM_DIM_IN).astype(np.float32)
        self.M_K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
        self.M_V = np.random.randn(MEM_SLOTS, MEM_DIM_IN).astype(np.float32) * 0.01
        self.W_q = np.random.randn(HIDDEN_DIM, MEM_DIM_IN).astype(np.float32) * np.sqrt(2.0/HIDDEN_DIM)
        self.W_gate = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
        self.b_gate = np.random.randn(1, HIDDEN_DIM).astype(np.float32) * 0.01
        self.W_mem_out = np.random.randn(MEM_DIM_IN, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0/MEM_DIM_IN)
        self.W_m = np.random.randn(HIDDEN_DIM, MEM_DIM).astype(np.float32) * np.sqrt(2.0/HIDDEN_DIM)
        self.b_m = np.zeros((1, MEM_DIM), dtype=np.float32)
        self.W_v = np.random.randn(HIDDEN_DIM, 1).astype(np.float32) * np.sqrt(2.0/HIDDEN_DIM)
        self.b_v = np.zeros((1, 1), dtype=np.float32)
        self.W_gen = np.random.randn(HIDDEN_DIM, VOCAB_SIZE).astype(np.float32) * np.sqrt(2.0/HIDDEN_DIM)
        self.b_gen = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        self.W_d = np.random.randn(HIDDEN_DIM, OUTPUT_DIM).astype(np.float32) * np.sqrt(2.0/HIDDEN_DIM)
        self.b_d = np.zeros((1, OUTPUT_DIM), dtype=np.float32)
        self._sae_k = 65  # default k
    
    def load_weights(self, path):
        data = np.load(path)
        for k in [W_embed,W_p,M_K,M_V,W_q,W_gate,b_gate,W_mem_out,
                   W_d,b_d,W_m,b_m,W_v,b_v,W_gen,b_gen]:
            if k in data:
                setattr(self, k, data[k].astype(np.float32))
    
    def forward(self, X):
        """纯NumPy前向传播 — 自适应稀疏度 + Gated MemBank"""
        N = X.shape[0]
        # W_embed
        X_proj = X @ self.W_embed
        Xp_relu = np.maximum(X_proj, 0)
        # W_p
        h1 = Xp_relu @ self.W_p
        h1_relu = np.maximum(h1, 0)
        # MemBank query
        Q = h1_relu @ self.W_q
        Q_norm = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-8)
        attn = Q_norm @ self.M_K.T
        attn_weights = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)
        # Top-6 slot voting
        top6_idx = np.argpartition(attn_weights, -6, axis=-1)[:, -6:]
        mask = np.zeros_like(attn_weights)
        np.put_along_axis(mask, top6_idx, 1.0, axis=-1)
        mem_read = mask[:, :, None] * self.M_V[None, :, :]
        mem_read = np.sum(mem_read, axis=1) / (np.sum(mask, axis=-1, keepdims=True) + 1e-8)
        mem_feat = mem_read @ self.W_mem_out
        gate = 1.0 / (1.0 + np.exp(-(h1_relu @ self.W_gate + self.b_gate)))
        h_mem = gate * h1_relu + (1.0 - gate) * mem_feat
        h3 = np.maximum(h_mem, 0)
        # ═══ Adaptive Sparsity (v4.6) ═══
        K_MIN, K_MAX = 40, 120
        h3_softmax = np.exp(h3 - np.max(h3, axis=-1, keepdims=True))
        h3_softmax /= np.sum(h3_softmax, axis=-1, keepdims=True) + 1e-8
        entropy = -np.sum(h3_softmax * np.log(h3_softmax + 1e-8), axis=-1)
        entropy_norm = entropy / np.log(HIDDEN_DIM)
        k_per = (K_MIN + (K_MAX - K_MIN) * entropy_norm).astype(np.int32)
        k_per = np.clip(k_per, K_MIN, K_MAX)
        K_DYN = int(np.mean(k_per))
        self._sae_k = K_DYN  # for logging
        h3_abs = np.abs(h3)
        thresh = np.partition(h3_abs, -K_DYN, axis=1)[:, -K_DYN:-K_DYN+1]
        mask_sae = (h3_abs >= thresh).astype(np.float32)
        h3_sae = h3 * mask_sae
        # Output heads
        recon = h3_sae @ self.W_d + self.b_d
        return {"recon": recon, "h3": h3_sae, "k": K_DYN}
    
    def train_step(self, X, Y):
        N = X.shape[0]
        out = self.forward(X)
        recon = out["recon"]
        # MSE loss
        loss = np.mean((recon - Y) ** 2)
        # Gradient
        grad_recon = 2 * (recon - Y) / N
        grad_h3 = grad_recon @ self.W_d.T
        # SGD update (pure NumPy)
        lr = LR
        self.W_d -= lr * ((out["h3"].T @ grad_recon) + WEIGHT_DECAY * self.W_d)
        self.b_d -= lr * np.sum(grad_recon, axis=0, keepdims=True)
        return loss

if __name__ == "__main__":
    print("NeuroFlow v4.6 HPC Standalone")
    d = NFv46HPC()
    print(f"  MemBank: {d.M_K.shape[0]} slots x {d.M_K.shape[1]}")
    # Test batch
    X = np.random.randn(4000, TEXT_DIM).astype(np.float32)
    Y = np.random.randn(4000, OUTPUT_DIM).astype(np.float32)
    t0 = time.time()
    loss = d.train_step(X, Y)
    dt = time.time() - t0
    print(f"  Test: loss={loss:.6f} k={d._sae_k} ({4000/dt:.0f} samples/s)")
    print("  OK")

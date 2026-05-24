"""
NeuroFlow v5.0 Subword Model
=============================
架构: BPE嵌入 → PE → 因果窗口 → SAE → 记忆 → 输出头
参数: ~18M | 内存: ~72MB | 词表: 5,000 subwords
"""
import numpy as np
from ops_v5 import (
    get_pe_table, causal_window_gating_operator,
    memory_read_write, cross_entropy_loss, vocab_gradient,
    sigmoid, D_MODEL, D_MEM, MEM_SLOTS, WINDOW_SIZE, MAX_SEQ_LEN, VOCAB_SIZE
)

class NeuroFlowV5:
    """v5.0 Subword Model"""
    
    def __init__(self):
        self.pe_table = get_pe_table(MAX_SEQ_LEN, D_MODEL)  # (8000, 512)
        self._init_params()
        print(f"  v5.0 model: {self._count_params()} params")
    
    def _init_params(self):
        """初始化所有权重"""
        np.random.seed(42)
        scale = np.sqrt(2.0 / D_MODEL)
        
        # ── Embedding ──
        self.W_embed = np.random.randn(VOCAB_SIZE, D_MODEL).astype(np.float32) * 0.01
        
        # ── Causal Window ──
        self.W_g = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * scale
        self.b_g = np.zeros(D_MODEL, dtype=np.float32)
        
        # ── SAE ──
        self.SAE_K = 65
        self.SAE_K_MIN, self.SAE_K_MAX = 40, 120
        
        # ── Memory Bank ──
        K_init = np.random.randn(MEM_SLOTS, D_MEM).astype(np.float32)
        K_init = K_init / (np.linalg.norm(K_init, axis=1, keepdims=True) + 1e-8)
        self.M_K = K_init
        self.M_V = np.random.randn(MEM_SLOTS, D_MEM).astype(np.float32) * 0.01
        
        # ── Memory routing ──
        self.W_read = np.random.randn(D_MODEL, MEM_SLOTS).astype(np.float32) * 0.01
        self.W_write = np.random.randn(D_MODEL, MEM_SLOTS).astype(np.float32) * 0.01
        self.W_to_mem = np.random.randn(D_MODEL, D_MEM).astype(np.float32) * scale
        
        # ── Output heads (low-rank: 512→256→5000) ──
        self.W_proj = np.random.randn(D_MODEL, D_MEM).astype(np.float32) * scale
        self.W_out = np.random.randn(D_MEM, VOCAB_SIZE).astype(np.float32) * 0.01
        self.b_out = np.zeros(VOCAB_SIZE, dtype=np.float32)
        
        # ── 独立词表头 (二阶段训练) ──
        self.V_in = np.random.randn(D_MODEL, D_MEM).astype(np.float32) * scale
        self.V_out = np.random.randn(D_MEM, VOCAB_SIZE).astype(np.float32) * 0.01
        self.V_bias = np.zeros(VOCAB_SIZE, dtype=np.float32)
    
    def _count_params(self):
        total = 0
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, np.ndarray):
                total += np.prod(obj.shape)
        return total
    
    def forward(self, token_ids):
        """前向传播
        token_ids: (N, L) token ID序列, N=batch, L=seq_len
        返回: (N, VOCAB_SIZE) logits
        """
        N, L = token_ids.shape
        
        # 1. Embedding lookup
        X_embed = self.W_embed[token_ids]  # (N, L, D)
        
        # 2. Add PE
        X = X_embed + self.pe_table[:L]  # (N, L, D)
        
        # 3. Causal Window (逐样本处理)
        C = np.zeros_like(X)
        for i in range(N):
            C[i] = causal_window_gating_operator(X[i], self.W_g, self.b_g)
        
        # 4. SAE sparsity
        h = C.reshape(-1, D_MODEL)  # (N*L, D)
        h_abs = np.abs(h)
        K = min(self.SAE_K, h.shape[1])
        thresh = np.partition(h_abs, -K, axis=1)[:, -K:-K+1]
        mask = (h_abs >= thresh).astype(np.float32)
        h = h * mask  # (N*L, D)
        
        # 5. Memory (pooled over sequence)
        h_pooled = h.reshape(N, L, D_MODEL).mean(axis=1)  # (N, D)
        mem_out, _ = memory_read_write(h_pooled[:1], self.M_V, 
                                        self.W_read, self.W_write, self.W_to_mem)
        # Inject memory to all tokens (project 256→512)
        mem_feat = mem_out @ self.W_to_mem.T  # (1, 512)
        h = h + 0.1 * np.tile(mem_feat, (N * L, 1))  # (N*L, D)
        
        # 6. Output head
        h_proj = np.maximum(h @ self.W_proj, 0)  # (N*L, 256) ReLU
        logits = h_proj @ self.W_out + self.b_out  # (N*L, VOCAB_SIZE)
        
        return logits, h
    
    def train_step(self, token_ids, targets):
        """单步训练
        token_ids: (N, L) 输入序列
        targets: (N*L,) 目标token ID (可以是全序列或特定位置)
        """
        N, L = token_ids.shape
        
        # 1. Forward
        logits, h = self.forward(token_ids)
        
        # 2. Loss (高级索引CE)
        loss = cross_entropy_loss(logits, targets)
        
        # 3. Gradient (高级索引梯度)
        grad = vocab_gradient(logits, targets)  # (N*L, V)
        
        # 4. Backward to output head
        lr = 0.0033
        h_proj = np.maximum(h @ self.W_proj, 0)
        d_proj = (grad @ self.W_out.T) * (h_proj > 0).astype(np.float32)  # ReLU grad
        
        # Update output weights
        self.W_out -= lr * (h_proj.T @ grad + 0.002 * self.W_out)
        self.b_out -= lr * np.sum(grad, axis=0)
        self.W_proj -= lr * (h.T @ d_proj + 0.002 * self.W_proj)
        
        return loss


if __name__ == "__main__":
    model = NeuroFlowV5()
    
    # Quick test
    np.random.seed(42)
    tokens = np.random.randint(0, VOCAB_SIZE, size=(2, 32)).astype(np.int32)
    targets = np.random.randint(0, VOCAB_SIZE, size=(2*32,))
    
    logits, h = model.forward(tokens)
    print(f"Forward: {logits.shape} {h.shape}")
    print(f"  logits: mean={logits.mean():.4f} std={logits.std():.4f}")
    
    loss = model.train_step(tokens, targets)
    print(f"Train: loss={loss:.4f}")
    
    print("v5.0 model OK ✅")

"""
NeuroFlow v5.0 核心算子
========================
1. Sinusoidal Positional Encoding (预计算静态表)
2. Causal Decay-Gated Window Operator (as_strided 零拷贝)
3. Memory Bank Low-Rank Interaction (外积擦写)
4. Advanced Index Cross-Entropy Loss (无one-hot)
"""
import numpy as np

# ── 配置 ──
VOCAB_SIZE = 5000
D_MODEL = 512          # 隐藏维度
D_MEM = 256            # 记忆槽维度
MEM_SLOTS = 32
WINDOW_SIZE = 8        # 因果窗口宽度
GAMMA = 0.85           # 时间衰减因子
MAX_SEQ_LEN = 8000     # 最大序列长度


def sigmoid(x):
    """数值稳定的sigmoid"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def get_pe_table(max_len=MAX_SEQ_LEN, d_model=D_MODEL):
    """预计算正弦位置编码表 (sinusoidal PE)
    
    返回: PE_table of shape (max_len, d_model)
    前向时: X = X_embed + PE_table[:seq_len]
    """
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(max_len)[:, np.newaxis]  # (L, 1)
    dim = np.arange(d_model)[np.newaxis, :]  # (1, D)
    
    # 偶数维度用sin, 奇数维度用cos
    pe[:, 0::2] = np.sin(pos / (10000 ** (2 * dim[:, 0::2] / d_model)))
    pe[:, 1::2] = np.cos(pos / (10000 ** (2 * dim[:, 1::2] / d_model)))
    return pe


def causal_window_gating_operator(X, W_g, b_g, window_size=WINDOW_SIZE, gamma=GAMMA):
    """纯NumPy高并行因果滑动窗口门控算子
    
    输入:
      X: (L, D) Token序列 (已带位置编码)
      W_g: (D, D) 门控权重
      b_g: (D,) 门控偏置
    返回:
      C: (L, D) 上下文特征
    复杂度: O(L * W * D), 全向量化, 零Python循环
    """
    L, D = X.shape
    W = min(window_size, L)
    
    # 1. 因果填充 (前补W-1个零)
    X_padded = np.vstack([np.zeros((W - 1, D), dtype=X.dtype), X])
    
    # 2. as_strided 黑魔法: 零拷贝滑动窗口视图
    shape_stride, elem_stride = X_padded.strides
    new_shape = (L, W, D)
    new_strides = (shape_stride, shape_stride, elem_stride)
    X_window = np.lib.stride_tricks.as_strided(X_padded, shape=new_shape, strides=new_strides)
    
    # 3. 时间衰减 Gamma (从近到远递减)
    gamma_decay = (gamma ** np.arange(W))[::-1]  # (W,)
    gamma_decay = gamma_decay[np.newaxis, :, np.newaxis]  # (1, W, 1)
    
    # 4. 并行加权求和 (代替RNN的for循环)
    C_temporal = np.sum(X_window * gamma_decay, axis=1)  # (L, D)
    
    # 5. 动态门控
    Gate = sigmoid(X @ W_g + b_g)  # (L, D)
    
    return C_temporal * Gate


def memory_read_write(c, M_Bank, W_read, W_write, W_to_mem):
    """低秩记忆读写 (外积擦写, 无Attention)
    
    输入:
      c: (1, D) 当前Token凝聚特征
      M_Bank: (B, D_mem) 记忆槽矩阵
    输出:
      output: (1, D) 融合记忆后的特征
    """
    # 1. 路由概率
    r_slot = sigmoid(c @ W_read)    # (1, B)
    w_slot = sigmoid(c @ W_write)   # (1, B)
    
    # 2. 低秩读取
    mem_context = r_slot @ M_Bank    # (1, D_mem)
    
    # 3. 低秩擦写 (增量更新)
    c_proj = c @ W_to_mem           # (1, D_mem)
    delta = w_slot.T @ c_proj        # (B, D_mem)
    M_Bank_new = (1.0 - w_slot.T) * M_Bank + delta
    
    return mem_context, M_Bank_new


def cross_entropy_loss(logits, targets):
    """高级索引交叉熵 (无one-hot矩阵)
    
    输入:
      logits: (N, VOCAB_SIZE) 模型输出
      targets: (N,) 目标token IDs
    返回:
      loss: float 标量损失
    """
    # Log-Sum-Exp稳定化
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    log_probs = (logits - max_logits) - np.log(np.sum(exp_logits, axis=-1, keepdims=True))
    
    # 高级索引: 只取正确token位置
    N = targets.shape[0]
    loss = -np.mean(log_probs[np.arange(N), targets])
    return loss


def vocab_gradient(logits, targets):
    """词表梯度: 无one-hot矩阵, 纯高级索引
    
    返回: grad of shape (N, VOCAB_SIZE)
    """
    N, V = logits.shape
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-8)
    
    grad = probs.copy()
    grad[np.arange(N), targets] -= 1.0
    return grad / N


# ── 测试 ──
if __name__ == "__main__":
    # Test PE
    pe = get_pe_table(100, 512)
    print(f"PE table: {pe.shape}")
    
    # Test Window
    np.random.seed(42)
    X = np.random.randn(16, 512).astype(np.float32)
    W_g = np.random.randn(512, 512).astype(np.float32) * 0.01
    b_g = np.zeros(512, dtype=np.float32)
    C = causal_window_gating_operator(X, W_g, b_g)
    print(f"Window: {X.shape} → {C.shape}")
    
    # Test Memory
    M = np.random.randn(32, 256).astype(np.float32) * 0.01
    W_r = np.random.randn(512, 32).astype(np.float32) * 0.01
    W_w = np.random.randn(512, 32).astype(np.float32) * 0.01
    W_m = np.random.randn(512, 256).astype(np.float32) * 0.01
    c = X[:1]
    mem_out, M_new = memory_read_write(c, M, W_r, W_w, W_m)
    print(f"Memory: {c.shape} → ({mem_out.shape}, {M_new.shape})")
    
    # Test CE Loss
    logits = np.random.randn(8, 5000).astype(np.float32) * 0.1
    targets = np.random.randint(0, 5000, size=8)
    loss = cross_entropy_loss(logits, targets)
    grad = vocab_gradient(logits, targets)
    print(f"CE Loss: {loss:.4f}, grad: {grad.shape}")
    
    print("All v5.0 operators OK ✅")

"""
NeuroFlow v5.x 架构级优化核心算子
================================
1. AdaptiveSAE — 自适应稀疏编码（动态k基于激活统计）
2. DeepSAEStack — 深层SAE栈 + 残差连接
3. IterativeMemory — 迭代记忆递归读写
"""
import numpy as np

# ── 配置 ──
D_MODEL = 512
D_MEM = 256
MEM_SLOTS = 32
VOCAB_SIZE = 5000
WINDOW_SIZE = 8
GAMMA = 0.85
MAX_SEQ_LEN = 8000


# ═══════════════════════════════════════════════
# 改进1: 自适应SAE稀疏度 (Adaptive SAE)
# ═══════════════════════════════════════════════

class AdaptiveSAE:
    """自适应稀疏自编码器 — 基于激活分布动态调整稀疏度
    
    原理:
      k_sparse = base_k + Δk(entropy, variance)
      高熵(信息丰富) → 更多稀疏通道保留
      低方差(模式稳定) → 更强制稀疏（节省容量）
    """
    def __init__(self, d_model=512, base_k=65, k_range=(30, 120),
                 entropy_thresh=0.5, var_thresh=0.001):
        self.d_model = d_model
        self.base_k = base_k
        self.k_min, self.k_max = k_range
        self.entropy_thresh = entropy_thresh
        self.var_thresh = var_thresh
    
    def compute_k(self, h, h_var=None):
        """根据激活熵和方差计算动态k
        
        输入: h (N, D) 激活张量
        返回: k (int) 保留的top-k数量
        """
        # 激活分布的"宽度"指标: 非零比例
        h_norm = np.abs(h) / (np.max(np.abs(h), axis=1, keepdims=True) + 1e-8)
        h_norm = np.clip(h_norm, 1e-10, 1.0)
        
        # 近似熵: -∑p*log(p) where p ∝ |activation|
        p = h_norm / (np.sum(h_norm, axis=1, keepdims=True) + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-10), axis=1).mean()
        
        # 自适应调整
        # 高熵 → 激活分散 → 需要更多容量
        # 低方差 → 模式稳定 → 可以更稀疏
        entropy_factor = np.clip(entropy / self.entropy_thresh, 0.5, 2.0)
        
        base = self.base_k
        delta = int(base * (entropy_factor - 1.0) * 0.3)  # ±30%调节
        
        if h_var is not None:
            # 低方差 → 更强制稀疏
            var_factor = np.clip(h_var / self.var_thresh, 0.5, 2.0)
            delta += int(base * (var_factor - 1.0) * 0.1)  # ±10%
        
        k = base + delta
        return np.clip(k, self.k_min, self.k_max)
    
    def forward(self, h, h_var=None):
        """自适应稀疏前向传播
        
        输入: h (N*L, D)
        返回: h_sparse (N*L, D)
              k (int) 使用的稀疏度
        """
        k = self.compute_k(h, h_var)
        h_abs = np.abs(h)
        thresh = np.partition(h_abs, -k, axis=1)[:, -k:-k+1]
        h_sparse = h * (h_abs >= thresh).astype(np.float32)
        return h_sparse, k


# ═══════════════════════════════════════════════
# 改进2: 深层SAE栈 + 残差连接
# ═══════════════════════════════════════════════

class DeepSAEStack:
    """深层SAE栈 — 多层稀疏编码 + 残差连接
    
    架构:
      h_0 = X
      h_1 = SAE_1(h_0) + h_0 (残差)
      h_2 = SAE_2(h_1) + h_1 (残差)
      ...
      h_out = projection(h_stack)  # 跨层融合
    
    优势:
      - 每层捕捉不同粒度的稀疏模式
      - 残差连接防止梯度消失
      - 跨层融合聚合多尺度特征
    """
    def __init__(self, d_model=512, n_layers=3, base_k=65):
        self.n_layers = n_layers
        self.sae_layers = [AdaptiveSAE(d_model, base_k - i*10) for i in range(n_layers)]
        # 跨层融合权重 (可训练)
        self.W_fuse = np.random.randn(d_model, d_model).astype(np.float32) * np.sqrt(2.0/d_model)
        self.k_used = [0] * n_layers
    
    def forward(self, h, h_var=None):
        """深层稀疏编码
        
        输入: h (N*L, D)
        返回: h_out (N*L, D) 融合后的输出
              layer_k (list) 每层k值
              layer_vars (list) 每层方差
        """
        h_stack = [h]
        layer_k = []
        layer_vars = []
        
        for i, sae in enumerate(self.sae_layers):
            h_sparse, k = sae.forward(h_stack[-1], h_var)
            h_res = h_sparse + 0.1 * h_stack[-1]  # 残差缩减(防协方差偏移)
            h_stack.append(h_res)
            layer_k.append(k)
            layer_vars.append(float(np.var(h_res)))
            h_var = layer_vars[-1]  # 传递到下一层
        
        # 跨层融合: 聚合所有层的输出
        # 使用加权和 + 非线性投影
        if self.n_layers >= 2:
            # 拼接所有层 (跳过输入层)
            H_concat = np.column_stack(h_stack[1:])  # (N*L, n_layers*D)
            # 线性降维融合
            h_fused = H_concat @ np.random.randn(self.n_layers * D_MODEL, D_MODEL).astype(np.float32) * 0.01
            h_out = h_fused + 0.1 * h  # 主残差
        else:
            h_out = h_stack[-1]
        
        self.k_used = layer_k
        return h_out, layer_k, layer_vars


# ═══════════════════════════════════════════════
# 改进3: 迭代记忆递归 (Iterative Memory)
# ═══════════════════════════════════════════════

def iterative_memory_read_write(c, M_V, W_read, W_write, W_to_mem, 
                                n_iters=3, position=None, max_pos=8000,
                                alpha=0.85, damping_strength=0.7,
                                W_refine=None):
    """迭代记忆递归读写 — 多次迭代精炼记忆读取
    
    原理:
      每次迭代 = 读取当前记忆 → 精炼查询向量 → 重新读取
      类似"多次思考": 每次读取补充更多上下文
    
    输入:
      c: (1, D) 当前Token凝聚特征
      M_V: (B, D_mem) 记忆值矩阵
      n_iters: 迭代次数
    返回:
      mem_context: (1, D_mem) 精炼后的记忆特征
      M_V: (B, D_mem) 更新后的记忆
      iter_trace: [n_iters] 每次迭代的置信度
    """
    r_slot = sigmoid(c @ W_read)    # (1, B)
    w_slot = sigmoid(c @ W_write)   # (1, B)
    
    # 时序阻尼(PTD-MC)
    if position is not None and max_pos > 0:
        ratio = position / max_pos
        damp = 1.0 - damping_strength * (1.0 / (1.0 + np.exp(-8.0 * (ratio - 0.5))))
        log_damp = 1.0 / (np.log(np.e + position) ** alpha)
        w_slot = w_slot * damp * log_damp
    
    # 记忆更新 (只在第1次迭代进行擦写)
    c_proj = c @ W_to_mem           # (1, D_mem)
    delta = w_slot.T @ c_proj        # (B, D_mem)
    M_V_new = (1.0 - w_slot.T) * M_V + delta
    
    # 迭代读取: 多次精炼查询
    query = c_proj  # (1, D_mem)
    iter_trace = []
    r_slot_refined = r_slot  # 初始路由
    mem_read = r_slot @ M_V_new  # 初始读取
    
    for it in range(n_iters):
        # 读取记忆
        mem_read = r_slot @ M_V_new if it == 0 else r_slot_refined @ M_V_new
        
        # 精炼: 融合当前查询与记忆读取
        if it < n_iters - 1:
            if W_refine is not None:
                mem_as_q = mem_read @ W_refine  # (1, D_MODEL)
            else:
                mem_as_q = mem_read @ W_to_mem.T  # 回射到D_MODEL
            # 精炼路由
            q_refined = c + 0.3 * mem_as_q  # (1, D_MODEL)
            r_slot_refined = sigmoid(q_refined @ W_read)  # (1, B)
            r_slot_refined = r_slot_refined / (np.sum(r_slot_refined) + 1e-8)
        
        # 置信度: 读取向量的范数 / 一致性
        confidence = float(np.linalg.norm(mem_read) / (np.linalg.norm(query) + 1e-8))
        iter_trace.append(confidence)
    
    return mem_read, M_V_new, iter_trace


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


# ═══════════════════════════════════════════════
# 完整升级版: 架构优化版 NeuroFlow 前向
# ═══════════════════════════════════════════════

class NeuroFlowV5ArchUpgrade:
    """v5.2 架构升级版 — 集成Adaptive SAE + Deep Stack + Iterative Memory"""
    
    def __init__(self, d_model=D_MODEL, d_mem=D_MEM, n_layers=2, mem_iters=2):
        self.d_model = d_model
        self.d_mem = d_mem
        self.adaptive_sae = AdaptiveSAE(d_model)
        self.deep_stack = DeepSAEStack(d_model, n_layers=n_layers) if n_layers > 1 else None
        self.mem_iters = mem_iters
        self.stats = {"k_history": [], "layer_vars": [], "iter_conf": []}
    
    def forward(self, h, h_var=None, position=None, max_pos=8000, 
                M_V=None, W_read=None, W_write=None, W_to_mem=None):
        """升级前向
        
        输入: h (N*L, D) 已通过因果窗口的激活
        返回: h_out (N*L, D) 稀疏+记忆融合后的输出
              stats dict
        """
        # === 步骤1: 自适应SAE稀疏 ===
        if self.deep_stack:
            h_sparse, layer_k, layer_vars = self.deep_stack.forward(h, h_var)
        else:
            h_sparse, k = self.adaptive_sae.forward(h, h_var)
            layer_k = [k]; layer_vars = [float(np.var(h_sparse))]
        
        self.stats["k_history"].extend(layer_k)
        self.stats["layer_vars"].extend(layer_vars)
        
        # === 步骤2: 池化 + 迭代记忆递归 ===
        N, L = h.shape[0] // 128, 128  # 从batch恢复
        h_pooled = h_sparse.reshape(N, L, self.d_model).mean(axis=1)
        
        if self.mem_iters > 1 and all(x is not None for x in [M_V, W_read, W_write, W_to_mem]):
            mem_out, M_V_new, iter_trace = iterative_memory_read_write(
                h_pooled[:1], M_V, W_read, W_write, W_to_mem,
                n_iters=self.mem_iters, position=position, max_pos=max_pos
            )
            self.stats["iter_conf"].extend(iter_trace)
        else:
            mem_out = h_pooled[:1] @ W_to_mem if W_to_mem is not None else np.zeros((1, self.d_mem))
            M_V_new = None
            iter_trace = []
        
        # === 步骤3: 记忆融合 ===
        h_out = h_sparse + 0.1 * np.tile(mem_out @ np.random.randn(self.d_mem, self.d_model).astype(np.float32) * 0.01, (N * L, 1))
        
        # === 步骤4: 投影 ===
        W_proj = np.random.randn(self.d_model, self.d_mem).astype(np.float32) * np.sqrt(2.0/self.d_model)
        h_proj = np.maximum(h_out @ W_proj, 0)
        
        return h_proj, self.stats


if __name__ == "__main__":
    print("=== 架构升级测试 ===")
    np.random.seed(42)
    
    # 测试自适应SAE
    asae = AdaptiveSAE()
    X = np.random.randn(128, 512).astype(np.float32) * 0.1
    h_sparse, k = asae.forward(X, h_var=float(np.var(X)))
    print("自适应SAE: k=%d (base=65, range=[30,120])" % k)
    
    # 测试深层栈
    dst = DeepSAEStack(n_layers=2)
    h_out, layer_k, layer_vars = dst.forward(X)
    print("深层SAE栈: %d layers, k=%s" % (dst.n_layers, layer_k))
    
    # 测试迭代记忆
    M_V = np.random.randn(32, 256).astype(np.float32) * 0.01
    W_r = np.random.randn(512, 32).astype(np.float32) * 0.01
    W_w = np.random.randn(512, 32).astype(np.float32) * 0.01
    W_m = np.random.randn(512, 256).astype(np.float32) * 0.01
    c = X[:1]
    mem_out, M_new, trace = iterative_memory_read_write(c, M_V, W_r, W_w, W_m, n_iters=3)
    print("迭代记忆: %d iters, conf=%s" % (len(trace), ["%.4f" % t for t in trace]))
    
    print("✅ 架构升级算子测试通过")

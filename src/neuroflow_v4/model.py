"""
NeuroFlow v4 — 纯 NumPy 核心模型
====================================
架构:
  X → W_embed → ReLU → 残差 → h1 → ReLU → GatedMemBank → h3 → SAE → 输出头

组件:
  - W_embed: 可学习输入特征重排 (1024×1024)
  - W_p: 第一层投影 (1024→512)
  - Gated Memory Bank: 32槽记忆键值对 + 门控融合
  - SAE: 自适应稀疏编码 (k=40~120)
  - 输出头: 重建 / retrieved_mem / 价值 / 词汇预测

依赖: 仅 numpy
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict

from . import config as C


# ═══════════════════════════════════════════════════════════════
# 门控温度锐化器
# ═══════════════════════════════════════════════════════════════
class DynamicGateSharper:
    """门控温度自适应余弦退火补丁 — numpy版
    
    当 M_V 范数 ≥ 0.5 且门控标准差 < 0.015 时触发激活。
    通过除法 τ 锐化 Sigmoid，打破门控均质化。
    
    τ(t) = τ_target + 0.5 * (τ_start - τ_target) * (1 + cos(π · t / T))
    """
    
    def __init__(self, start_tau: float = C.GATE_SHARPEN_START_TAU,
                 target_tau: float = C.GATE_SHARPEN_TARGET_TAU,
                 duration_topics: int = C.GATE_SHARPEN_DURATION):
        self.start_tau = start_tau
        self.target_tau = target_tau
        self.duration_topics = duration_topics
        self.current_tau = start_tau
    
    def step(self, global_topics: int) -> float:
        """余弦退火更新温度
        
        Args:
            global_topics: 当前已训练 topics 数
        
        Returns:
            当前温度系数 τ
        """
        if global_topics >= self.duration_topics:
            self.current_tau = self.target_tau
        else:
            progress = global_topics / self.duration_topics
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            self.current_tau = self.target_tau + (self.start_tau - self.target_tau) * cosine_decay
        return self.current_tau
    
    def forward(self, gate_logits: np.ndarray) -> np.ndarray:
        """带温度的门控前向: sigmoid(x / τ)
        
        Args:
            gate_logits: [..., D] 门控 logits
        
        Returns:
            门控值 [0, 1]
        """
        tau = max(self.current_tau, 1e-4)
        scaled = np.clip(gate_logits / tau, -50, 50)
        return 1.0 / (1.0 + np.exp(-scaled))
    
    def backward(self, dgate: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """带温度的门控反向: dσ/dx = σ(1-σ) / τ
        
        Args:
            dgate: 上游梯度
            gate: 门控值
        
        Returns:
            修正后的梯度
        """
        tau = max(self.current_tau, 1e-4)
        return (dgate * gate * (1.0 - gate)) / tau
    
    def __repr__(self) -> str:
        return (f"DynamicGateSharper(τ={self.current_tau:.3f}, "
                f"start={self.start_tau}→target={self.target_tau}, "
                f"duration={self.duration_topics:,} topics)")


# ═══════════════════════════════════════════════════════════════
# 词表梯度热身控制器
# ═══════════════════════════════════════════════════════════════
class VocabGradientController:
    """词表梯度热身控制器
    
    防止词表 BCE 梯度砸碎刚恢复的流形。
    两阶段策略:
      Phase 1: W_gen 学习率体温 (wgen_lr_mult 从 0 → 1)
      Phase 2: h3 词表梯度渐进注入 (h3_weight 从 0 → 0.2)
    """
    
    def __init__(self, target_h3_weight: float = C.VOCAB_TARGET_H3_WEIGHT,
                 target_wgen_lr_mult: float = 1.0,
                 warm_up_steps: int = C.VOCAB_WARMUP_STEPS,
                 start_var_threshold: float = C.VOCAB_START_VAR_THRESHOLD):
        self.target_h3_weight = target_h3_weight
        self.target_wgen_lr_mult = target_wgen_lr_mult
        self.warm_up_steps = warm_up_steps
        self.start_var_threshold = start_var_threshold
        self.warm_up_counter = 0
        self.is_warming_up = False
        self.active = False
        self.h3_weight = 0.0
        self.wgen_lr_mult = 0.0
    
    def update(self, current_var: float) -> Tuple[float, float]:
        """更新热身状态
        
        Args:
            current_var: 当前隐状态方差
        
        Returns:
            (h3_word_weight, wgen_lr_mult)
        """
        if not self.active and current_var >= self.start_var_threshold:
            self.active = True
            self.is_warming_up = True
            self.warm_up_counter = 0
        
        if not self.active:
            return 0.0, 0.0
        
        if self.is_warming_up:
            self.warm_up_counter += 1
            x = 12.0 * (self.warm_up_counter / self.warm_up_steps) - 6.0
            sigmoid_factor = 1.0 / (1.0 + np.exp(-x))
            
            self.wgen_lr_mult = sigmoid_factor * self.target_wgen_lr_mult
            self.h3_weight = max(0.0, sigmoid_factor - 0.5) * 2.0 * self.target_h3_weight
            
            if self.warm_up_counter >= self.warm_up_steps:
                self.is_warming_up = False
                self.h3_weight = self.target_h3_weight
                self.wgen_lr_mult = self.target_wgen_lr_mult
        
        return self.h3_weight, self.wgen_lr_mult


# ═══════════════════════════════════════════════════════════════
# NeuroFlow v4 核心模型
# ═══════════════════════════════════════════════════════════════
class NeuroFlowV4:
    """NeuroFlow v4 — Gated Memory Bank + SAE 模型
    
    架构:
        X → W_embed → ReLU → 残差
          → W_p → ReLU → h1
          → Gated Memory Bank (M_K/M_V + W_q + gate)
          → h3 → SAE (top-k mask)
          ↕ 输出头: W_d(重建) + W_m(mem) + W_v(价值) + W_gen(词汇)
    
    特性:
        - 纯 NumPy 实现 (3.29M 参数)
        - 自适应稀疏 SAE (输入熵驱动)
        - 门控温度锐化 (余弦退火)
        - 记忆能量泵 (M_V 范数 + 多样性)
        - 独立词表头 (V_in → V_out)
    """
    
    def __init__(self, weights: Optional[Dict[str, np.ndarray]] = None):
        """初始化模型
        
        Args:
            weights: 可选预加载权重字典 (npz 格式)
                     如为 None, 需要用 load_weights() 或 load_pretrained() 加载
        """
        # 门控温控器
        self.gate_sharper = DynamicGateSharper()
        self.gate_tau_active = False  # 默认关闭, 由 M_V 范数自动触发
        
        # 词表热身控制器
        self.vocab_controller = VocabGradientController()
        
        # 初始化参数
        self._init_parameters()
        
        # 如果提供了权重, 直接加载
        if weights is not None:
            self.load_weights(weights)
    
    def _init_parameters(self):
        """初始化所有参数 (小随机 + 正交初始化)"""
        D = C.TEXT_DIM              # 1024
        H = C.HIDDEN_DIM            # 512
        M = C.MEM_DIM_IN            # 256
        S = C.MEM_SLOTS             # 32
        V = C.VOCAB_SIZE            # 500
        
        # ── 输入投影 ──
        self.W_embed = np.random.randn(D, D).astype(np.float32) * 0.01
        
        # ── 第一层投影 ──
        self.W_p = np.random.randn(D, H).astype(np.float32) * np.sqrt(2.0 / D)
        
        # ── Gated Memory Bank ──
        # 记忆键 (L2 归一化, 单位球面)
        K_init = np.random.randn(S, M).astype(np.float32)
        K_init = K_init / (np.linalg.norm(K_init, axis=1, keepdims=True) + 1e-8)
        self.M_K = K_init
        # 记忆值 (小随机)
        self.M_V = np.random.randn(S, M).astype(np.float32) * 0.01
        # 查询投影
        self.W_q = np.random.randn(H, M).astype(np.float32) * np.sqrt(2.0 / H)
        # 门控
        self.W_gate = np.random.randn(H, H).astype(np.float32) * 0.01
        self.b_gate = np.random.randn(1, H).astype(np.float32) * 0.01
        # 记忆读出投影
        self.W_mem_out = np.random.randn(M, H).astype(np.float32) * np.sqrt(2.0 / M)
        
        # ── retrieved_mem 头 ──
        self.W_m = np.random.randn(H, M).astype(np.float32) * np.sqrt(2.0 / H)
        self.b_m = np.zeros((1, M), dtype=np.float32)
        
        # ── 解码器 (重建) ──
        self.W_d = np.random.randn(H, D).astype(np.float32) * np.sqrt(2.0 / H)
        self.b_d = np.zeros((1, D), dtype=np.float32)
        
        # ── 价值头 ──
        self.W_v = np.random.randn(H, 1).astype(np.float32) * np.sqrt(2.0 / H)
        self.b_v = np.zeros((1, 1), dtype=np.float32)
        
        # ── 词汇预测头 (正交初始化) ──
        Q, _ = np.linalg.qr(np.random.randn(H, H).astype(np.float32))
        self.W_gen = Q[:, :V].astype(np.float32) * 1.5
        self.b_gen = np.zeros((1, V), dtype=np.float32)
        
        # ── 独立词表头 ──
        scale_v = np.sqrt(2.0 / H)
        self.V_in = np.random.randn(H, 256).astype(np.float32) * scale_v
        self.V_out = np.random.randn(256, V).astype(np.float32) * scale_v
        self.V_bias = np.zeros((1, V), dtype=np.float32)
    
    # ═══════════════════════════════════════════════
    # 权重加载
    # ═══════════════════════════════════════════════
    
    def load_weights(self, data: Dict[str, np.ndarray]):
        """从字典加载权重 (npz 格式)
        
        Args:
            data: 包含所有权重的字典, 键名如 'W_embed', 'M_K', 'M_V', 'W_gate' 等
        """
        for attr, saved_key in [
            ("W_embed", "W_embed"),
            ("W_p", "W_p"),
            ("M_K", "M_K"), ("M_V", "M_V"),
            ("W_q", "W_q"), ("W_gate", "W_gate"),
            ("b_gate", "b_gate"), ("W_mem_out", "W_mem_out"),
            ("W_m", "W_m"), ("b_m", "b_m"),
            ("W_gen", "W_gen"), ("b_gen", "b_gen"),
            ("W_d", "W_d"), ("b_d", "b_d"),
            ("W_v", "W_v"), ("b_v", "b_v"),
            ("V_in", "V_in"), ("V_out", "V_out"), ("V_bias", "V_bias"),
        ]:
            if saved_key in data:
                saved = data[saved_key]
                current = getattr(self, attr)
                if saved.shape == current.shape:
                    setattr(self, attr, saved.astype(np.float32))
    
    def _get_weights_dict(self) -> Dict[str, np.ndarray]:
        """获取当前所有权重的字典 (用于保存)"""
        return {
            "W_embed": self.W_embed.copy(),
            "M_K": self.M_K.copy(), "M_V": self.M_V.copy(),
            "W_q": self.W_q.copy(), "W_gate": self.W_gate.copy(),
            "b_gate": self.b_gate.copy(), "W_mem_out": self.W_mem_out.copy(),
            "W_m": self.W_m.copy(), "b_m": self.b_m.copy(),
            "W_gen": self.W_gen.copy(), "b_gen": self.b_gen.copy(),
            "V_in": self.V_in.copy(), "V_out": self.V_out.copy(),
            "V_bias": self.V_bias.copy(),
        }
    
    # ═══════════════════════════════════════════════
    # 前向传播 (推理)
    # ═══════════════════════════════════════════════
    
    def forward(self, X: np.ndarray,
                return_intermediates: bool = False,
                tau_active: bool = False) -> Dict:
        """完整前向传播
        
        Args:
            X: 输入 [N, 1024] — 文本编码向量 (可使用 encode_text 生成)
            return_intermediates: 是否返回中间隐层
            tau_active: 是否使用门控温度锐化
        
        Returns:
            dict 包含所有输出:
              - recon: [N, 1024] 编码重建
              - mem_pred: [N, 256] retrieved_mem 预测
              - value: [N, 1] 价值评估
              - word_logits: [N, 500] 词汇预测 logits
              - h3: [N, 512] 最终隐层状态
              - gate: [N, 512] 门控值
              - attn: [N, 32] 记忆注意力权重
              - h_var: float 隐状态方差
              - k_active: int SAE 激活数
              (return_intermediates=True 时额外返回 h1, mem_read, mem_feat, h_mem, h3_normed)
        """
        N = X.shape[0]
        
        # ── 输入投影 ──
        X_proj = X @ self.W_embed
        X_proj = np.maximum(X_proj, 0)
        X_in = X + X_proj * 0.1  # 残差
        
        # ── 第一层 ──
        h1 = X_in @ self.W_p
        h1_relu = np.maximum(h1, 0)
        
        # ── Gated Memory Bank ──
        Q = h1_relu @ self.W_q
        K_norm = self.M_K / (np.linalg.norm(self.M_K, axis=1, keepdims=True) + 1e-8)
        scores = Q @ K_norm.T
        
        # Top-K 注意力
        temp = C.ATTN_TEMPERATURE
        scores_max = np.max(scores, axis=1, keepdims=True)
        scores_exp = np.exp(temp * (scores - scores_max))
        topk = C.ATTN_TOPK
        scores_topk = np.partition(scores_exp, -topk, axis=1)[:, -topk:-topk+1].min(axis=1, keepdims=True)
        scores_exp = scores_exp * (scores_exp >= scores_topk)
        attn = scores_exp / (np.sum(scores_exp, axis=1, keepdims=True) + 1e-8)
        
        mem_read = attn @ self.M_V
        mem_feat = mem_read @ self.W_mem_out
        
        # 门控融合
        gate_logits = h1_relu @ self.W_gate + self.b_gate
        if tau_active:
            gate = self.gate_sharper.forward(gate_logits)
        else:
            gate = 1.0 / (1.0 + np.exp(-gate_logits))
        
        h_mem = gate * h1_relu + (1.0 - gate) * mem_feat
        h3 = np.maximum(h_mem, 0)
        
        # ── SAE LayerNorm + 自适应稀疏 ──
        h3_mean = np.mean(h3, axis=1, keepdims=True)
        h3_std = np.std(h3, axis=1, keepdims=True) + 1e-5
        h3_normed = (h3 - h3_mean) / h3_std
        
        # 熵驱动的自适应 top-k
        h3_softmax = np.exp(h3_normed - np.max(h3_normed, axis=-1, keepdims=True))
        h3_softmax = h3_softmax / (np.sum(h3_softmax, axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(h3_softmax * np.log(h3_softmax + 1e-8), axis=-1)
        entropy_norm = entropy / np.log(C.HIDDEN_DIM)
        
        k_per_sample = (C.SAE_K_MIN + (C.SAE_K_MAX - C.SAE_K_MIN) * entropy_norm).astype(np.int32)
        k_per_sample = np.clip(k_per_sample, C.SAE_K_MIN, C.SAE_K_MAX)
        K_DYNAMIC = int(np.mean(k_per_sample))
        
        h3_normed_abs = np.abs(h3_normed)
        h3_thresh = np.partition(h3_normed_abs, -K_DYNAMIC, axis=1)[:, -K_DYNAMIC:-K_DYNAMIC+1]
        sae_mask = (h3_normed_abs >= h3_thresh).astype(np.float32)
        h3_masked = h3 * sae_mask  # 推理时应用 SAE mask
        
        # ── 输出头 ──
        recon = h3_masked @ self.W_d + self.b_d
        mem_pred = h3_masked @ self.W_m + self.b_m
        value = h3_masked @ self.W_v + self.b_v
        word_logits = h3_masked @ self.W_gen + self.b_gen
        
        # ── 隐状态方差 ──
        h_centered = h3_masked - np.mean(h3_masked, axis=0, keepdims=True)
        h_var = float(np.mean(h_centered ** 2))
        
        result = {
            "recon": recon,
            "mem_pred": mem_pred,
            "value": value,
            "word_logits": word_logits,
            "h3": h3_masked,
            "h3_raw": h3,
            "gate": gate,
            "attn": attn,
            "h_var": h_var,
            "k_active": K_DYNAMIC,
        }
        
        if return_intermediates:
            result.update({
                "h1": h1,
                "mem_read": mem_read,
                "mem_feat": mem_feat,
                "h_mem": h_mem,
                "h3_normed": h3_normed,
                "sae_mask": sae_mask,
            })
        
        return result
    
    # ═══════════════════════════════════════════════
    # 词汇预测 (推理)
    # ═══════════════════════════════════════════════
    
    def predict_vocab(self, h3: np.ndarray) -> np.ndarray:
        """独立词表头预测
        
        Args:
            h3: 隐层状态 [N, 512]
        
        Returns:
            probs: [N, 500] 词汇概率
        """
        v_hidden = h3 @ self.V_in
        v_hidden_r = np.maximum(v_hidden, 0)
        v_logits = v_hidden_r @ self.V_out + self.V_bias
        return 1.0 / (1.0 + np.exp(-v_logits))
    
    # ═══════════════════════════════════════════════
    # 模型分析
    # ═══════════════════════════════════════════════
    
    def analyze(self) -> Dict:
        """返回模型运行状态分析
        
        Returns:
            dict 包含各层统计量
        """
        stats = {}
        
        # M_V 范数
        mv_norms = np.linalg.norm(self.M_V, axis=1)
        stats["M_V"] = {
            "mean_norm": float(np.mean(mv_norms)),
            "std_norm": float(np.std(mv_norms)),
            "min_norm": float(np.min(mv_norms)),
            "max_norm": float(np.max(mv_norms)),
        }
        
        # M_K 正交性
        K_norm = self.M_K / (np.linalg.norm(self.M_K, axis=1, keepdims=True) + 1e-8)
        K_cos = K_norm @ K_norm.T
        stats["M_K"] = {
            "mean_self_cos": float(np.mean(K_cos - np.eye(K_cos.shape[0]))),
            "mean_norm": float(np.mean(np.linalg.norm(self.M_K, axis=1))),
        }
        
        # 门控偏置范围
        stats["gate"] = {
            "bias_range": [float(self.b_gate.min()), float(self.b_gate.max())],
            "bias_std": float(np.std(self.b_gate)),
        }
        
        # W_embed 有效秩
        _, S, _ = np.linalg.svd(self.W_embed, full_matrices=False)
        stats["W_embed"] = {
            "effective_rank": int(np.sum(S > 0.01)),
            "singular_top5": [float(s) for s in S[:5]],
        }
        
        # 总参数量
        total = sum(v.nbytes for v in self.__dict__.values()
                    if isinstance(v, np.ndarray))
        stats["memory_mb"] = total / 1024 / 1024
        
        return stats
    
    def __repr__(self) -> str:
        return (f"NeuroFlowV4(\n"
                f"  W_embed: {self.W_embed.shape}  |  "
                f"M_K: {self.M_K.shape}  |  "
                f"M_V: {self.M_V.shape}\n"
                f"  gate_τ_active: {self.gate_tau_active}  |  "
                f"W_gen: {self.W_gen.shape}\n"
                f")")

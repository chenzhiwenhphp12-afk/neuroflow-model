"""
NeuroFlow 可训练决策层
=======================
"固定身体，训练皮层" — 冻结 SN/DMN/CrossModal 随机权重，
仅训练 Hidden[256] → decision[10] + value[1] 线性投影。

训练参数: 256*10 + 10 + 256*1 + 1 = 2,827 个
内存占用: ~11KB
每步计算: <1μs on CPU
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainableOutput:
    """可训练决策层输出 — 与 C++ Output 接口兼容"""
    decision: np.ndarray
    value: np.ndarray
    saliency: np.ndarray
    anomaly: np.ndarray
    ecn_gate: Optional[np.ndarray] = None
    # 附加: 暴露隐状态
    hidden: Optional[np.ndarray] = None


class TrainableHead:
    """
    可训练线性决策层。
    
    冻结前传 (model.forward_text) → 取 h = output.retrieved_mem
    → decision = h @ W_d + b_d
    → value    = h @ W_v + b_v
    
    Args:
        hidden_dim: 隐状态维度 (256)
        n_actions: 动作/决策维度 (10)
        lr: 学习率
    """
    
    def __init__(self, model, hidden_dim: int = 256, n_actions: int = 10, lr: float = 0.01):
        self.model = model
        self.lr = lr
        
        # Xavier 初始化
        scale = np.sqrt(2.0 / hidden_dim)
        self.W_d = np.random.randn(hidden_dim, n_actions).astype(np.float32) * scale
        self.b_d = np.zeros((1, n_actions), dtype=np.float32)
        self.W_v = np.random.randn(hidden_dim, 1).astype(np.float32) * scale * 0.1
        self.b_v = np.zeros((1, 1), dtype=np.float32)
        
        self.n_updates = 0
        self.total_loss = 0.0
    
    def _forward(self, x: np.ndarray):
        """前向传播 (冻结部分)"""
        if hasattr(self.model, 'forward_text'):
            return self.model.forward_text(x)
        return self.model.forward(x)
    
    def predict(self, x: np.ndarray) -> TrainableOutput:
        """
        推理: 前传 + 可训练投影。
        
        Args:
            x: 输入 [batch, input_dim]
        
        Returns:
            TrainableOutput: decision, value, saliency, hidden
        """
        # 冻结前传
        output = self._forward(x)
        
        # 提取隐状态
        h = output.retrieved_mem.astype(np.float32)  # [batch, 256]
        if h.size == 0:
            h = np.random.randn(x.shape[0], self.W_d.shape[0]).astype(np.float32) * 0.01
        
        # 可训练投影
        decision = h @ self.W_d + self.b_d  # [batch, n_actions]
        value = h @ self.W_v + self.b_v    # [batch, 1]
        
        return TrainableOutput(
            decision=decision,
            value=value,
            saliency=output.saliency if hasattr(output, 'saliency') else np.zeros((x.shape[0], 1), dtype=np.float32),
            anomaly=output.anomaly if hasattr(output, 'anomaly') else np.zeros(1, dtype=np.float32),
            ecn_gate=output.gates if hasattr(output, 'gates') else None,
            hidden=h,
        )
    
    def train_step(self, x: np.ndarray, target_action: int, reward: float) -> dict:
        """
        单步 SGD 训练。
        
        Args:
            x: 输入 [1, input_dim]
            target_action: 目标动作索引
            reward: 环境奖励
        
        Returns:
            dict: {loss, decision_before, decision_after, value}
        """
        batch = x.shape[0]
        
        # 冻结前传
        output = self._forward(x)
        h = output.retrieved_mem.astype(np.float32)
        if h.size == 0:
            h = np.random.randn(batch, self.W_d.shape[0]).astype(np.float32) * 0.01
        
        # 前向投影
        decision = h @ self.W_d + self.b_d   # [batch, n_actions]
        value = h @ self.W_v + self.b_v       # [batch, 1]
        
        # Softmax 概率
        decision_shifted = decision - np.max(decision, axis=1, keepdims=True)
        exp_d = np.exp(decision_shifted)
        probs = exp_d / (np.sum(exp_d, axis=1, keepdims=True) + 1e-8)
        
        # 损失: 交叉熵 (决策) + MSE (价值)
        target_onehot = np.zeros_like(decision)
        target_onehot[0, target_action] = 1.0
        ce_loss = -np.log(probs[0, target_action] + 1e-8)
        value_loss = (value[0, 0] - reward) ** 2
        loss = ce_loss + 0.1 * value_loss
        
        # 梯度: W_d, b_d
        grad_d_logits = (probs - target_onehot) / batch  # [batch, n_actions]
        grad_W_d = h.T @ grad_d_logits                     # [256, n_actions]
        grad_b_d = np.sum(grad_d_logits, axis=0, keepdims=True)  # [1, n_actions]
        
        # 梯度: W_v, b_v
        grad_v = 2 * (value - reward) / batch             # [batch, 1]
        grad_W_v = h.T @ grad_v                            # [256, 1]
        grad_b_v = np.sum(grad_v, axis=0, keepdims=True)  # [1, 1]
        
        # SGD 更新
        self.W_d -= self.lr * grad_W_d
        self.b_d -= self.lr * grad_b_d
        self.W_v -= self.lr * grad_W_v * 0.1  # 价值学习率降低
        self.b_v -= self.lr * grad_b_v * 0.1
        
        self.n_updates += 1
        self.total_loss += float(loss)
        
        # 更新后重新计算
        decision_after = h @ self.W_d + self.b_d
        
        return {
            "loss": float(loss),
            "ce_loss": float(ce_loss),
            "value_loss": float(value_loss),
            "decision_idx": int(np.argmax(decision[0])),
            "best_action_after": int(np.argmax(decision_after[0])),
            "value": float(value[0, 0]),
            "predicted_action": int(np.argmax(probs[0])),
        }
    
    def get_weights(self) -> dict:
        """导出可训练权重"""
        return {
            "W_d": self.W_d.copy(),
            "b_d": self.b_d.copy(),
            "W_v": self.W_v.copy(),
            "b_v": self.b_v.copy(),
        }
    
    def load_weights(self, weights: dict):
        """加载权重"""
        self.W_d = weights["W_d"].astype(np.float32)
        self.b_d = weights["b_d"].astype(np.float32)
        self.W_v = weights["W_v"].astype(np.float32)
        self.b_v = weights["b_v"].astype(np.float32)
    
    def stats(self) -> dict:
        return {
            "n_updates": self.n_updates,
            "avg_loss": self.total_loss / max(self.n_updates, 1),
            "W_d_norm": float(np.linalg.norm(self.W_d)),
            "W_v_norm": float(np.linalg.norm(self.W_v)),
        }

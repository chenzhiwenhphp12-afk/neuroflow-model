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
    
    def __init__(self, model, hidden_dim: int = 256, n_actions: int = 10, lr: float = 0.01,
                 hidden_scale: float = 1.0, input_dim: int = 1024, entropy_coef: float = 0.05):
        self.model = model
        self.lr = lr
        self.hidden_scale = hidden_scale
        self.entropy_coef = entropy_coef
        
        # Xavier 初始化 (scaled for normalized hidden)
        scale = np.sqrt(2.0 / hidden_dim)
        self.W_d = np.random.randn(hidden_dim, n_actions).astype(np.float32) * scale
        self.b_d = np.zeros((1, n_actions), dtype=np.float32)
        self.W_v = np.random.randn(hidden_dim, 1).astype(np.float32) * scale * 0.1
        self.b_v = np.zeros((1, 1), dtype=np.float32)
        
        # ── 方案 A: 直接编码投影 (1024 → hidden_dim) ──
        proj_scale = np.sqrt(2.0 / input_dim)
        self.W_p = np.random.randn(input_dim, hidden_dim).astype(np.float32) * proj_scale
        
        self.n_updates = 0
        self.total_loss = 0.0
    
    def _forward(self, x: np.ndarray):
        """前向传播 (冻结部分)"""
        if hasattr(self.model, 'forward_text'):
            output = self.model.forward_text(x)
            # 拼接多个有信息的输出作为隐状态
            parts = []
            if hasattr(output, 'decision') and output.decision.size > 0:
                parts.append(output.decision.flatten().astype(np.float32))
            if hasattr(output, 'value') and output.value.size > 0:
                parts.append(output.value.flatten().astype(np.float32))
            if hasattr(output, 'saliency') and output.saliency.size > 0:
                parts.append(output.saliency.flatten().astype(np.float32))
            if hasattr(output, 'gates') and output.gates.size > 0:
                parts.append(output.gates.flatten().astype(np.float32))
            if hasattr(output, 'retrieved_mem') and output.retrieved_mem.size > 0:
                parts.append(output.retrieved_mem.flatten().astype(np.float32))
            if parts:
                h = np.concatenate(parts)
            else:
                h = np.zeros(self.W_d.shape[0], dtype=np.float32)
            # Pad/truncate to hidden_dim
            if len(h) < self.W_d.shape[0]:
                h = np.pad(h, (0, self.W_d.shape[0] - len(h)))
            else:
                h = h[:self.W_d.shape[0]]
            h = h.reshape(1, -1)
            # Store for predict/train_step to use
            self._last_hidden = h
            self._last_output = output
            return output
        return self.model.forward(x)
    
    def predict(self, x: np.ndarray) -> TrainableOutput:
        """
        推理: 前传 + 可训练投影。
        
        Args:
            x: 输入 [batch, input_dim]
        
        Returns:
            TrainableOutput: decision, value, saliency, hidden
        """
        # 冻结前传 (内部提取多源隐状态到 self._last_hidden)
        output = self._forward(x)
        h = self._last_hidden.astype(np.float32)
        
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
        
        # 冻结前传 (内部提取多源隐状态)
        self._forward(x)
        h = self._last_hidden.astype(np.float32)
        
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
    
    def train_batch(self, x_batch: np.ndarray, target_actions: list, rewards: list) -> dict:
        """
        批量梯度下降 — 累积所有样本梯度后一次更新。
        
        Args:
            x_batch: 输入 [N, input_dim]
            target_actions: 目标动作列表 [N]
            rewards: 奖励列表 [N]
        
        Returns:
            dict: {loss, ce_loss, value_loss, n_samples}
        """
        N = x_batch.shape[0]
        hidden_dim = self.W_d.shape[0]
        n_actions = self.W_d.shape[1]
        
        # 逐个前传收集隐状态
        H = np.zeros((N, hidden_dim), dtype=np.float32)
        decisions = np.zeros((N, n_actions), dtype=np.float32)
        values = np.zeros((N, 1), dtype=np.float32)
        
        for i in range(N):
            self._forward(x_batch[i:i+1])
            H[i] = self._last_hidden.flatten().astype(np.float32)
        
        # 批量前向
        decisions = H @ self.W_d + self.b_d
        values = H @ self.W_v + self.b_v
        
        # Softmax
        d_shifted = decisions - np.max(decisions, axis=1, keepdims=True)
        exp_d = np.exp(d_shifted)
        probs = exp_d / (np.sum(exp_d, axis=1, keepdims=True) + 1e-8)
        
        # 批量损失
        ce_losses = np.zeros(N, dtype=np.float32)
        target_onehot = np.zeros((N, n_actions), dtype=np.float32)
        for i in range(N):
            target_onehot[i, target_actions[i]] = 1.0
            ce_losses[i] = -np.log(probs[i, target_actions[i]] + 1e-8)
        
        rewards_arr = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        value_losses = (values - rewards_arr) ** 2
        value_loss = np.mean(value_losses)
        
        avg_ce = np.mean(ce_losses)
        avg_loss = avg_ce + 0.1 * value_loss
        
        # 批量梯度 (平均)
        grad_d_logits = (probs - target_onehot) / N
        grad_W_d = H.T @ grad_d_logits
        grad_b_d = np.sum(grad_d_logits, axis=0, keepdims=True)
        
        grad_v = 2 * (values - rewards_arr) / N
        grad_W_v = H.T @ grad_v
        grad_b_v = np.sum(grad_v, axis=0, keepdims=True)
        
        # 单次批量更新
        self.W_d -= self.lr * grad_W_d
        self.b_d -= self.lr * grad_b_d
        self.W_v -= self.lr * grad_W_v * 0.1
        self.b_v -= self.lr * grad_b_v * 0.1
        
        self.n_updates += 1
        self.total_loss += float(avg_loss)
        
        return {
            "loss": float(avg_loss),
            "ce_loss": float(avg_ce),
            "value_loss": float(value_loss),
            "n_samples": N,
        }
    
    def direct_train_batch(self, x_batch: np.ndarray, target_actions: list, rewards: list) -> dict:
        """
        方案 A: 直接编码训练 — 跳过 C++ 模型，用 W_p 投影编码到隐状态
        
        x_batch: 文本编码 [N, input_dim] (1024-dim hash编码, 已归一化)
        target_actions: 文本 hash 目标 [N]
        rewards: 奖励 [N]
        """
        N = x_batch.shape[0]
        input_dim = self.W_p.shape[0]
        hidden_dim = self.W_d.shape[0]
        n_actions = self.W_d.shape[1]
        
        # 投影: 编码 → 隐状态 [N, input_dim] @ [input_dim, hidden_dim] = [N, hidden_dim]
        H = x_batch @ self.W_p  # [N, hidden_dim]
        
        # 决策和价值
        decisions = H @ self.W_d + self.b_d  # [N, n_actions]
        values = H @ self.W_v + self.b_v      # [N, 1]
        
        # Softmax + 熵
        d_shifted = decisions - np.max(decisions, axis=1, keepdims=True)
        exp_d = np.exp(d_shifted)
        probs = exp_d / (np.sum(exp_d, axis=1, keepdims=True) + 1e-8)
        
        # 熵正则: H(p) = -Σ p*log(p) — 越大越均匀，防止坍缩
        eps = 1e-8
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)  # [N]
        avg_entropy = float(np.mean(entropy))
        
        # 损失: 交叉熵 + 价值MSE - 熵正则
        target_onehot = np.zeros((N, n_actions), dtype=np.float32)
        ce_losses = np.zeros(N, dtype=np.float32)
        for i in range(N):
            ta = target_actions[i]
            if ta < 0 or ta >= n_actions:
                ta = 0
            target_onehot[i, ta] = 1.0
            ce_losses[i] = -np.log(probs[i, ta] + eps)
        
        rewards_arr = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        value_losses = (values - rewards_arr) ** 2
        
        avg_ce = float(np.mean(ce_losses))
        avg_value_loss = float(np.mean(value_losses))
        avg_loss = avg_ce + 0.1 * avg_value_loss - self.entropy_coef * avg_entropy
        
        # ── 批量梯度 (平均梯度) ──
        # 1) 决策层梯度 dL/d(decision)
        grad_logits = (probs - target_onehot) / N  # [N, n_actions]
        
        grad_W_d = H.T @ grad_logits                     # [hidden_dim, n_actions]
        grad_b_d = np.sum(grad_logits, axis=0, keepdims=True)  # [1, n_actions]
        
        # 2) 价值层梯度
        grad_value = 2 * (values - rewards_arr) / N      # [N, 1]
        grad_W_v = H.T @ grad_value                       # [hidden_dim, 1]
        grad_b_v = np.sum(grad_value, axis=0, keepdims=True)  # [1, 1]
        
        # 3) 投影层梯度: dL/d(W_p) = x.T @ grad_logits @ W_d.T + x.T @ grad_value @ W_v.T
        #    = x.T @ (grad_logits @ W_d.T + grad_value @ W_v.T * 0.1)
        grad_hidden = (grad_logits @ self.W_d.T) + (grad_value @ self.W_v.T) * 0.1
        grad_W_p = x_batch.T @ grad_hidden  # [input_dim, hidden_dim]
        
        # SGD 更新
        self.W_d -= self.lr * grad_W_d
        self.b_d -= self.lr * grad_b_d
        self.W_v -= self.lr * grad_W_v * 0.1
        self.b_v -= self.lr * grad_b_v * 0.1
        self.W_p -= self.lr * grad_W_p
        
        self.n_updates += 1
        self.total_loss += float(avg_loss)
        
        return {
            "loss": float(avg_loss),
            "ce_loss": avg_ce,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "n_samples": N,
            "action_dist": [int(np.bincount(target_actions, minlength=n_actions).tolist()[i])
                           if i < n_actions else 0 for i in range(n_actions)],
        }
    
    def get_weights(self) -> dict:
        """导出可训练权重"""
        w = {
            "W_d": self.W_d.copy(),
            "b_d": self.b_d.copy(),
            "W_v": self.W_v.copy(),
            "b_v": self.b_v.copy(),
        }
        if hasattr(self, 'W_p'):
            w["W_p"] = self.W_p.copy()
        return w
    
    def load_weights(self, weights: dict):
        """加载权重"""
        self.W_d = weights["W_d"].astype(np.float32)
        self.b_d = weights["b_d"].astype(np.float32)
        self.W_v = weights["W_v"].astype(np.float32)
        self.b_v = weights["b_v"].astype(np.float32)
        if "W_p" in weights:
            self.W_p = weights["W_p"].astype(np.float32)
    
    def stats(self) -> dict:
        return {
            "n_updates": self.n_updates,
            "avg_loss": self.total_loss / max(self.n_updates, 1),
            "W_d_norm": float(np.linalg.norm(self.W_d)),
            "W_v_norm": float(np.linalg.norm(self.W_v)),
            "b_d_norm": float(np.linalg.norm(self.b_d)),
            "b_v_norm": float(np.linalg.norm(self.b_v)),
        }

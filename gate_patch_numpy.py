"""
NeuroFlow Gate Temperature Annealing Patch (numpy版)
=====================================================
余弦退火锐化门控Sigmoid，打破均质化

公式: τ(t) = τ_target + 0.5*(τ_start - τ_target)*(1 + cos(π·t/T))

注入后daemon自动在训练循环中更新τ
"""

import math
import numpy as np

class DynamicGateSharper:
    """门控温度自适应余弦退火补丁 — numpy版"""
    
    def __init__(self, start_tau=0.2, target_tau=1.0, duration_topics=500000):
        self.start_tau = start_tau
        self.target_tau = target_tau
        self.duration_topics = duration_topics
        self.current_tau = start_tau
        
    def step(self, global_topics):
        """
        根据全局topics进度，计算当前温度系数τ（余弦退火）
        """
        if global_topics >= self.duration_topics:
            self.current_tau = self.target_tau
        else:
            progress = global_topics / self.duration_topics
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            self.current_tau = self.target_tau + (self.start_tau - self.target_tau) * cosine_decay
        return self.current_tau
    
    def patch_forward(self, gate_logits):
        """
        替代原生门控计算: sigmoid(gate_logits / τ)
        """
        tau = max(self.current_tau, 1e-4)
        scaled = np.clip(gate_logits / tau, -50, 50)
        return 1.0 / (1.0 + np.exp(-scaled))
    
    def patch_backward(self, dgate, gate):
        """
        反向梯度修正：链式法则 → 除以τ
        原生: dgate_logits = dgate * gate * (1 - gate)
        修正: dgate_logits = dgate * gate * (1 - gate) / τ
        """
        tau = max(self.current_tau, 1e-4)
        return (dgate * gate * (1.0 - gate)) / tau
    
    def __repr__(self):
        return (f"DynamicGateSharper(τ={self.current_tau:.3f}, "
                f"start={self.start_tau}→target={self.target_tau}, "
                f"duration={self.duration_topics:,}topics)")


# ═══════════════════════════════════════
# 注入 daemon 的代码模板
# ═══════════════════════════════════════
#
# 在 daemon.__init__ 末尾添加：
#   self.gate_sharper = DynamicGateSharper()
#
# 在训练循环（_train_vocab_separately）中
# 替换 line 986:
#   原: gate = 1.0 / (1.0 + np.exp(-(h1_relu @ self.W_gate + self.b_gate)))
#   新: gate_logits = h1_relu @ self.W_gate + self.b_gate
#      tau = self.gate_sharper.step(self.total_topics)
#      gate = self.gate_sharper.patch_forward(gate_logits)

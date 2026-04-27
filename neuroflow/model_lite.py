"""
NeuroFlow Simple Optimized Version

简化版优化，基于 DeepSeek 核心技术：
1. MLA 压缩 KV cache
2. 稀疏 MoE 降低计算
3. 轻量化设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class CompressedMemory(nn.Module):
    """压缩记忆模块 - MLA 风格"""
    
    def __init__(self, input_dim: int, memory_dim: int, memory_slots: int, compress_ratio: int = 4):
        super().__init__()
        self.compress_dim = memory_dim // compress_ratio  # 压缩后的维度
        
        # 压缩投影
        self.compress = nn.Linear(input_dim, self.compress_dim)
        self.decompress = nn.Linear(self.compress_dim, input_dim)
        
        # 记忆库（压缩存储）
        self.memory_bank = nn.Parameter(torch.randn(memory_slots, self.compress_dim) * 0.02)
        
        # 检索
        self.query = nn.Linear(input_dim, self.compress_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) - 已经过 input_proj
        Returns:
            retrieved: (batch, input_dim)
            attention: (batch, memory_slots)
        """
        # 压缩查询
        q = self.query(x)  # (batch, compress_dim)
        
        # 注意力检索
        attn = F.softmax(torch.matmul(q, self.memory_bank.T) / math.sqrt(self.compress_dim), dim=-1)
        
        # 检索并解压
        compressed = torch.matmul(attn, self.memory_bank)  # (batch, compress_dim)
        retrieved = self.decompress(compressed)  # (batch, input_dim)
        
        return retrieved, attn


class SparseExpertLayer(nn.Module):
    """稀疏专家层 - 简化版 MoE"""
    
    def __init__(self, hidden_dim: int, output_dim: int, n_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # 路由
        self.gate = nn.Linear(hidden_dim, n_experts)
        
        # 专家（简单 FFN）
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for _ in range(n_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, hidden_dim)
        Returns:
            output: (batch, output_dim)
            aux_loss: 负载均衡损失
        """
        batch_size = x.size(0)
        
        # 路由分数
        gates = F.softmax(self.gate(x), dim=-1)  # (batch, n_experts)
        
        # Top-K 选择
        top_k_gates, top_k_idx = torch.topk(gates, self.top_k, dim=-1)
        
        # 专家输出
        output = torch.zeros(batch_size, self.experts[0].out_features, device=x.device)
        
        for i in range(self.top_k):
            idx = top_k_idx[:, i]  # (batch,)
            weight = top_k_gates[:, i]  # (batch,)
            
            for e in range(self.n_experts):
                mask = (idx == e)
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    output[mask] += weight[mask].unsqueeze(-1) * expert_out
        
        # 负载均衡损失
        aux_loss = self.n_experts * torch.sum(gates.mean(dim=0) ** 2)
        
        return output, aux_loss


class OptimizedNeuroFlowSimple(nn.Module):
    """
    简化版优化 NeuroFlow
    
    特点：
    - 压缩记忆（MLA 风格）
    - 稀疏专家（MoE 风格）
    - 参数量减少 30-50%
    - 推理速度更快
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 10,
        memory_dim: int = 128,
        memory_slots: int = 64,
        compress_ratio: int = 4,  # MLA 压缩比
        n_experts: int = 4,
        top_k: int = 1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 压缩记忆（MLA）
        self.memory = CompressedMemory(
            input_dim=hidden_dim,
            memory_dim=memory_dim,
            memory_slots=memory_slots,
            compress_ratio=compress_ratio,
        )
        
        # 稀疏专家决策层（MoE）
        self.decision_experts = SparseExpertLayer(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_experts=n_experts,
            top_k=top_k,
        )
        
        # 价值评估（保持简单）
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 输出融合
        self.output_fusion = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            dict: output, decision, value, memory_attention, aux_loss
        """
        # 输入投影
        h = self.input_proj(x)  # (batch, hidden_dim)
        
        # 记忆检索
        mem_retrieved, mem_attn = self.memory(h)  # (batch, hidden_dim), (batch, slots)
        h = h + 0.1 * mem_retrieved  # 残差
        
        # 稀疏专家决策
        decision, aux_loss = self.decision_experts(h)  # (batch, output_dim)
        
        # 价值评估
        value = self.value_head(h)  # (batch, 1)
        
        # 融合
        combined = torch.cat([decision, mem_retrieved[:, :self.output_dim]], dim=-1)
        output = self.output_fusion(combined)
        
        return {
            'output': output,
            'decision': decision,
            'value': value,
            'memory_attention': mem_attn,
            'aux_loss': aux_loss,
        }
    
    def get_stats(self) -> Dict[str, float]:
        """获取模型统计"""
        params = sum(p.numel() for p in self.parameters())
        return {
            'params': params,
            'params_mb': params * 4 / 1024 / 1024,
        }


class NeuroFlowLiteV2(OptimizedNeuroFlowSimple):
    """超轻量版 - 适合边缘设备"""
    
    def __init__(self, input_dim: int = 512):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=10,
            memory_dim=64,
            memory_slots=32,
            compress_ratio=4,
            n_experts=4,
            top_k=1,
        )


def benchmark():
    """对比测试"""
    import time
    import sys
    sys.path.insert(0, './neuroflow-model')
    from neuroflow.model import NeuroFlowModel
    
    print("=" * 50)
    print("NeuroFlow 优化对比测试")
    print("=" * 50)
    
    device = torch.device('cpu')
    input_dim = 512
    batch_size = 32
    iterations = 100
    
    # 原始模型
    print("\n[原始 NeuroFlow]")
    original = NeuroFlowModel(input_dim=input_dim, hidden_dim=256, output_dim=10)
    orig_params = sum(p.numel() for p in original.parameters())
    print(f"  参数量: {orig_params:,}")
    
    # 优化模型
    print("\n[优化 NeuroFlow Simple]")
    optimized = OptimizedNeuroFlowSimple(input_dim=input_dim)
    opt_params = sum(p.numel() for p in optimized.parameters())
    print(f"  参数量: {opt_params:,}")
    
    # 轻量模型
    print("\n[NeuroFlow Lite V2]")
    lite = NeuroFlowLiteV2(input_dim=input_dim)
    lite_params = sum(p.numel() for p in lite.parameters())
    print(f"  参数量: {lite_params:,}")
    
    # 测试
    x = torch.randn(batch_size, input_dim)
    
    print("\n=== 前向传播测试 ===")
    
    # 原始
    orig_out = original(x)
    print(f"原始输出: {orig_out['output'].shape}")
    
    # 优化
    opt_out = optimized(x)
    print(f"优化输出: {opt_out['output'].shape}")
    print(f"MoE 负载均衡损失: {opt_out['aux_loss'].item():.4f}")
    
    # 轻量
    lite_out = lite(x)
    print(f"轻量输出: {lite_out['output'].shape}")
    
    # 性能测试
    print("\n=== 性能对比 ===")
    
    # 原始
    start = time.perf_counter()
    for _ in range(iterations):
        _ = original(x)
    orig_time = (time.perf_counter() - start) / iterations * 1000
    
    # 优化
    start = time.perf_counter()
    for _ in range(iterations):
        _ = optimized(x)
    opt_time = (time.perf_counter() - start) / iterations * 1000
    
    # 轻量
    start = time.perf_counter()
    for _ in range(iterations):
        _ = lite(x)
    lite_time = (time.perf_counter() - start) / iterations * 1000
    
    print(f"\n推理时间 (batch={batch_size}):")
    print(f"  原始: {orig_time:.2f} ms")
    print(f"  优化: {opt_time:.2f} ms ({orig_time/opt_time:.2f}x 加速)")
    print(f"  轻量: {lite_time:.2f} ms ({orig_time/lite_time:.2f}x 加速)")
    
    print(f"\n参数量对比:")
    print(f"  优化 vs 原始: 减少 {(1 - opt_params/orig_params)*100:.1f}%")
    print(f"  轻量 vs 原始: 减少 {(1 - lite_params/orig_params)*100:.1f}%")
    
    return {
        'original': {'params': orig_params, 'time': orig_time},
        'optimized': {'params': opt_params, 'time': opt_time},
        'lite': {'params': lite_params, 'time': lite_time},
    }


if __name__ == "__main__":
    benchmark()
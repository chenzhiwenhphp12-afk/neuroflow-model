"""
NeuroFlow — 类脑模块化神经网络框架

灵感来源于2026年神经科学前沿研究 + DeepSeek V4 核心技术，实现了：
1. 多模块协作架构（执行网络、默认模式网络、显著性网络）
2. 神经流形（Neural Manifolds）低维表征学习
3. 动态门控与跨区域信息整合
4. 类海马体记忆巩固机制
5. DeepSeek V4 集成：
   - Engram Memory (百万级上下文)
   - MLA 压缩注意力
   - Sparse MoE 稀疏专家
   - Flash-Attention 4
   - Muon 优化器
   - 自我纠错推理

Author: Chen Zhiwen <chenzhiwenhphp12@gmail.com>
License: MIT
"""

from neuroflow.model import NeuroFlowModel
from neuroflow.modules import (
    ExecutiveControlNetwork,
    DefaultModeNetwork,
    SalienceNetwork,
    MemoryConsolidationModule,
)

# DeepSeek V3 优化模块
from neuroflow.deepseek_optimizations import (
    LatentKVCompression,
    SparseMoE,
    RotaryPositionalEmbedding,
    QuantizedLinear,
    quantize_model,
    EfficientMemoryModule,
    OptimizedECN,
)

# DeepSeek V4 高级优化模块
from neuroflow.deepseek_v4_optimizations import (
    EngramMemory,
    EngramConfig,
    MuonOptimizer,
    FlashAttentionV4,
    RotaryPositionalEmbeddingV4,
    MultiHyperConnection,
    IterativeSelfCorrection,
    EAGLESpeculativeDecoding,
    NeuroFlowV4,
)

__version__ = "0.2.0"
__all__ = [
    # 原始模块
    "NeuroFlowModel",
    "ExecutiveControlNetwork",
    "DefaultModeNetwork",
    "SalienceNetwork",
    "MemoryConsolidationModule",
    # DeepSeek V3 优化
    "LatentKVCompression",
    "SparseMoE",
    "RotaryPositionalEmbedding",
    "QuantizedLinear",
    "quantize_model",
    "EfficientMemoryModule",
    "OptimizedECN",
    # DeepSeek V4 优化
    "EngramMemory",
    "EngramConfig",
    "MuonOptimizer",
    "FlashAttentionV4",
    "RotaryPositionalEmbeddingV4",
    "MultiHyperConnection",
    "IterativeSelfCorrection",
    "EAGLESpeculativeDecoding",
    "NeuroFlowV4",
]

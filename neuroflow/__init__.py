"""
NeuroFlow — 类脑模块化神经网络框架

灵感来源于2026年神经科学前沿研究，实现了：
1. 多模块协作架构（执行网络、默认模式网络、显著性网络）
2. 神经流形（Neural Manifolds）低维表征学习
3. 动态门控与跨区域信息整合
4. 类海马体记忆巩固机制

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

__version__ = "0.1.0"
__all__ = [
    "NeuroFlowModel",
    "ExecutiveControlNetwork",
    "DefaultModeNetwork",
    "SalienceNetwork",
    "MemoryConsolidationModule",
]

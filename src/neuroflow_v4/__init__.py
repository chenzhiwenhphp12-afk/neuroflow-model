"""
NeuroFlow v4 — 自主进化的类脑记忆模型
======================================
纯 NumPy 实现 · 3.29M 参数 · 13 MB 权重

核心特性:
  - Gated Memory Bank (32槽 × 256维)
  - 自适应稀疏 SAE (k=40~120, 基于输入熵)
  - 门控温度锐化 (余弦退火)
  - 记忆能量泵 (M_V 范数→多样性)
  - VICReg 方差正则化
  - 自蒸馏 + 对比学习 + 线性进化

快速开始:
  >>> from neuroflow_v4 import NeuroFlowV4
  >>> model = NeuroFlowV4()
  >>> h3, metrics = model.forward(X)  # X: [N, 1024]
"""

from .model import NeuroFlowV4
from .encoder import encode_text, encode_batch
from .inference import Predictor
from . import config

__version__ = "4.0.0"
__author__ = "Chen Zhiwen <chenzhiwenhphp12@gmail.com>"
__license__ = "MIT"

__all__ = [
    "NeuroFlowV4",
    "Predictor",
    "encode_text",
    "encode_batch",
    "config",
]

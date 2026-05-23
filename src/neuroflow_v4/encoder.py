"""
NeuroFlow v4 — 文本编码器
============================
基于 hash + sinusoid 的确定性编码器 (零外部依赖)

编码过程:
  1. 分词 → hash 投影 (每个词 8 次 hash 到 1024 维)
  2. 位置 sinusoid 注入
  3. L2 归一化

注意: 这是轻量级编码器, 用于训练时的自重建目标。
实际部署时可替换为更强大的编码器 (如 BERT / Sentence-T5) 提取的 embedding。
"""

import numpy as np
from typing import List, Optional


def encode_text(text: str, dim: int = 1024) -> np.ndarray:
    """将文本编码为 1024 维特征向量
    
    使用确定性 hash 投影 + 位置 sinusoid 编码。
    相同输入 → 相同输出 (确定性)。
    
    Args:
        text: 输入文本
        dim: 输出维度 (默认 1024)
    
    Returns:
        [dim] 归一化特征向量
    """
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    
    n_words = min(len(words), 500)
    for i, word in enumerate(words[:n_words]):
        h = abs(hash(word)) % (2**31)
        for j in range(8):
            idx = (h + j * 2654435761) % dim
            vec[int(idx)] += 0.03 / max(n_words / 30, 1)
    
    vec += np.sin(np.linspace(0, np.pi * n_words / 15, dim)).astype(np.float32) * 0.08
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


def encode_batch(texts: List[str], dim: int = 1024) -> np.ndarray:
    """批量编码文本
    
    Args:
        texts: 文本列表
        dim: 输出维度
    
    Returns:
        [N, dim] 归一化特征矩阵
    """
    n = len(texts)
    X = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        X[i] = encode_text(texts[i], dim)
    return X


def compute_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """计算两个编码矩阵之间的余弦相似度
    
    Args:
        X: [N, D]
        Y: [M, D]
    
    Returns:
        [N, M] 余弦相似度矩阵
    """
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return X_norm @ Y_norm.T

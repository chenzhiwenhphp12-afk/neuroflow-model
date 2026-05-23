"""
NeuroFlow v4 — 推理 API
==========================
零外部依赖的推理接口。

自动处理:
  - 权重下载 (Hugging Face / 本地)
  - 文本编码
  - 完整前向传播
  - 词汇预测

用法:
  >>> from neuroflow_v4 import Predictor
  >>> predictor = Predictor()
  >>> result = predictor("What is the meaning of life?")
  >>> print(result["h_var"])   # 隐状态方差
  >>> print(result["top5_chars"])  # Top-5 预测字符
"""

import numpy as np
from typing import Optional, Dict, List, Union

from .model import NeuroFlowV4
from .encoder import encode_text, encode_batch
from .weights import load_pretrained
from . import config as C


class Predictor:
    """NeuroFlow v4 推理接口
    
    用法:
      >>> pred = Predictor()          # 自动加载权重
      >>> out = pred("hello world")   # 单文本推理
      >>> out = pred(["a", "b"])      # 批量推理
    
    返回:
      dict 包含:
        - h3: [N, 512] 隐层状态
        - h_var: float 方差异常度 (越高越好, 信号丰富)
        - value: [N, 1] 价值评估
        - gate_mean: float 门控均值
        - gate_std: float 门控标准差
        - attn_top_slot: int 最高注意力的记忆槽
        - k_active: int SAE 激活数
        - top5_chars: List[str] Top-5 预测字符
        - word_probs: [N, 500] 词汇概率
        - recon_mse: float 重建误差
    """
    
    def __init__(self, weights_path: Optional[str] = None,
                 auto_download: bool = True,
                 char_vocab: Optional[List[str]] = None):
        """初始化预测器
        
        Args:
            weights_path: 权重文件路径 (None = 自动查找/下载)
            auto_download: 自动从 Hugging Face 下载
            char_vocab: 字符词表 (默认使用 ASCII + 基本标点)
        """
        # 加载权重
        if weights_path == "random":
            weights = None
        else:
            weights = load_pretrained(weights_path, auto_download)
        
        self.model = NeuroFlowV4(weights=weights)
        
        # 字符词表
        if char_vocab is None:
            self.char_vocab = list(
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789 .,!?;:'\"-()[]{}"
            )
        else:
            self.char_vocab = char_vocab
        self.char_to_idx = {c: i for i, c in enumerate(self.char_vocab)}
        
        print(f"🧠 NeuroFlow v4 就绪")
        print(f"   参数: {C.ARCHITECTURE['total_params']:,}  |  权重: {C.ARCHITECTURE['weight_size_mb']} MB")
    
    def __call__(self, texts: Union[str, List[str]],
                 return_details: bool = False) -> Dict:
        """推理入口
        
        Args:
            texts: 单文本 (str) 或文本列表
            return_details: 是否返回完整中间结果
        
        Returns:
            包含推理结果的 dict
        """
        single = isinstance(texts, str)
        if single:
            texts_list = [texts]
        else:
            texts_list = texts
        
        # 编码
        X = encode_batch(texts_list, dim=C.TEXT_DIM)
        
        # 模型前向
        output = self.model.forward(
            X,
            return_intermediates=return_details,
            tau_active=self.model.gate_tau_active
        )
        
        # 词汇预测
        word_probs = self.model.predict_vocab(output["h3"])
        
        # 构建结果
        result = {
            "h3": output["h3"],
            "h_var": output["h_var"],
            "value": output["value"],
            "gate_mean": float(np.mean(output["gate"])),
            "gate_std": float(np.std(output["gate"])),
            "attn_top_slot": int(np.argmax(np.mean(output["attn"], axis=0))),
            "k_active": output["k_active"],
            "word_probs": word_probs,
            "recon_mse": float(np.mean((output["recon"] - X) ** 2)),
        }
        
        # Top-5 字符预测 (每样本)
        top5_probs = np.sort(word_probs, axis=1)[:, -5:]
        top5_indices = np.argsort(word_probs, axis=1)[:, -5:]
        result["top5_chars"] = []
        for i in range(len(texts_list)):
            chars = []
            for idx in top5_indices[i]:
                if int(idx) < len(self.char_vocab):
                    chars.append(self.char_vocab[int(idx)])
                else:
                    chars.append("?")
            result["top5_chars"].append(chars)
        
        if return_details:
            result["model_output"] = output
        
        # 单文本输入时返回单结果
        if single:
            result["top5_chars"] = result["top5_chars"][0]
        
        return result
    
    def analyze(self) -> Dict:
        """获取模型分析报告"""
        stats = self.model.analyze()
        stats["vocab_size"] = len(self.char_vocab)
        stats["architecture"] = C.ARCHITECTURE["name"]
        stats["version"] = "4.0.0"
        return stats

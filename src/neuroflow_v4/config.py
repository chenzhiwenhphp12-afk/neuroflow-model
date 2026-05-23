"""
NeuroFlow v4 配置 — 模型超参数与架构定义
===========================================
参考: daemon_v3.py 中的架构常量
"""

import numpy as np

# ── 模型维度 ──
TEXT_DIM = 1024           # 输入文本编码维度
HIDDEN_DIM = 512          # 隐藏层维度 (h1, h3)
MEM_DIM_IN = 256          # 记忆槽键/值维度
MEM_SLOTS = 32            # Gated Memory Bank 槽数
VOCAB_SIZE = 500          # 字符词表大小

# ── SAE (Sparse Autoencoder) ──
SAE_K_BASE = 65           # 基础 top-k 激活数
SAE_K_MIN = 40            # 最小 top-k
SAE_K_MAX = 120           # 最大 top-k

# ── 记忆注意力 ──
ATTN_TEMPERATURE = 8.0    # 注意力温度
ATTN_TOPK = 6             # 保留 Top-K 记忆槽

# ── 门控 ──
GATE_SHARPEN_START_TAU = 0.2     # 温控起始 τ
GATE_SHARPEN_TARGET_TAU = 1.0    # 温控目标 τ
GATE_SHARPEN_DURATION = 500000   # 温控退火周期 (topics)
M_V_NORM_TARGET = 1.0            # M_V 范数泵目标
M_V_NORM_PUMP_WEIGHT = 1.0       # M_V 范数泵强度
M_V_DIVERSITY_WEIGHT = 0.5       # M_V 多样性强度

# ── 损失权重 ──
MEM_LOSS_WEIGHT = 0.5     # retrieved_mem MSE 权重
CONTRASTIVE_WEIGHT = 0.8  # 对比损失权重（默认）
VICREG_VAR_WEIGHT = 0.5   # VICReg 方差正则化
VICREG_GAMMA = 0.05       # VICReg 铰链阈值
VOCAB_LOSS_WEIGHT = 0.1   # 词表 BCE 损失权重
VOCAB_POS_WEIGHT = 3.0    # 正类加权系数
WEIGHT_DECAY = 1e-5       # 权重衰减

# ── 训练 ──
LEARNING_RATE = 0.01      # 基础学习率
SHARED_LR_RATIO = 0.2     # 共享层 = 输出头 LR × 此值
MASK_RATIO = 0.15         # 输入掩码比例 (15%)
INPUT_NOISE = 0.01        # 输入高斯噪声 std

# ── 词表热度 ──
VOCAB_WARMUP_STEPS = 3000         # 词表热身步数
VOCAB_TARGET_H3_WEIGHT = 0.2      # h3 词表梯度权重
VOCAB_START_VAR_THRESHOLD = 0.1   # 启动热身的 var 阈值

# ── 权重文件 ──
WEIGHT_FILENAME = "neuroflow_weights_v4.npz"
# Hugging Face 权重下载链接
HF_REPO = "chenzhiwenhphp12/neuroflow-v4"
HF_WEIGHT_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{WEIGHT_FILENAME}"

# ── 架构元信息 ──
ARCHITECTURE = {
    "version": "4.0.0",
    "name": "NeuroFlow v4 — GatedMemBank + SAE",
    "total_params": 3_287_273,
    "weight_size_mb": 12.54,
    "backend": "numpy (CPU only)",
    "parameters": {
        "W_embed": (1024, 1024),   # 可学习输入投影
        "W_p": (1024, 512),        # 第一层投影 (head)
        "W_q": (512, 256),         # 记忆查询投影
        "M_K": (32, 256),          # 记忆键 (L2归一化)
        "M_V": (32, 256),          # 记忆值
        "W_mem_out": (256, 512),   # 记忆读出投影
        "W_gate": (512, 512),      # 门控网络
        "b_gate": (512,),          # 门控偏置
        "W_m": (512, 256),         # retrieved_mem 头
        "b_m": (256,),             # retrieved_mem 偏置
        "W_d": (512, 1024),        # 解码器 (重建)
        "b_d": (1024,),            # 解码器偏置
        "W_v": (512, 1),           # 价值头
        "b_v": (1,),               # 价值偏置
        "W_gen": (512, 500),       # 词汇生成器
        "b_gen": (500,),           # 词汇生成偏置
        "V_in": (512, 256),        # 独立词表输入
        "V_out": (256, 500),       # 独立词表输出
        "V_bias": (500,),          # 独立词表偏置
    }
}

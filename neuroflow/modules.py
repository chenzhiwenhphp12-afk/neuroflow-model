"""
NeuroFlow 核心模块实现

基于2026年神经科学报告的三大核心网络：
1. ExecutiveControlNetwork (ECN) — 背外侧前额叶 (dlPFC) 模拟
2. DefaultModeNetwork (DMN) — 内侧前额叶 + 后扣带回 模拟
3. SalienceNetwork (SN) — 前岛叶 + 前扣带回 模拟
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExecutiveControlNetwork(nn.Module):
    """
    高级执行与决策系统 — 模拟 Prefrontal Suite (dlPFC, OFC, vmPFC)

    功能：
    - dlPFC: 在线信息处理、逻辑推理
    - OFC: 价值评估
    - vmPFC: 社交决策与同理心
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.dlpfc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            for i in range(num_layers)
        ])
        # OFC: 价值评估分支
        self.ofc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # vmPFC: 决策分支
        self.vmpfc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)
        Returns:
            decision: (batch, output_dim) — vmPFC 决策输出
            value: (batch, 1) — OFC 价值评估
            hidden: list — 中间层激活（用于流形分析）
        """
        hidden_states = []
        h = x
        for layer in self.dlpfc:
            h = layer(h)
            hidden_states.append(h)

        value = self.ofc(h)
        decision = self.vmpfc(h)
        return decision, value, hidden_states


class DefaultModeNetwork(nn.Module):
    """
    内部意识与自我系统 — 模拟 DMN (PCC, mPFC)

    功能：
    - 自传体记忆检索
    - 未来愿景规划
    - 创造性联想生成
    """

    def __init__(self, memory_dim: int, latent_dim: int, num_associations: int = 8):
        super().__init__()
        self.memory_dim = memory_dim
        self.latent_dim = latent_dim

        # 记忆检索编码器
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # 联想生成器（多头注意力模拟创造性联想）
        self.num_associations = num_associations
        self.association_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
            for _ in range(num_associations)
        ])

        # 未来愿景投影
        self.future_projection = nn.Sequential(
            nn.Linear(latent_dim * num_associations, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
        )

    def forward(self, memory_input):
        """
        Args:
            memory_input: (batch, memory_dim) — 记忆输入
        Returns:
            vision: (batch, latent_dim * 2) — 未来愿景表征
            associations: list — 创造性联想输出
            latent: (batch, latent_dim) — 潜在记忆表征
        """
        latent = self.memory_encoder(memory_input)

        associations = [head(latent) for head in self.association_heads]
        combined = torch.cat(associations, dim=-1)
        vision = self.future_projection(combined)

        return vision, associations, latent


class SalienceNetwork(nn.Module):
    """
    环境监测与切换系统 — 模拟 SN (Anterior Insula, ACC)

    功能：
    - 显著性信号检测（过滤无关输入）
    - ECN ↔ DMN 状态切换门控
    - 异常检测与注意力重定向
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # 显著性评分器
        self.saliency_scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 门控权重生成器（控制 ECN/DMN 切换）
        self.gate_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [ecn_gate, dmn_gate]
            nn.Softmax(dim=-1),
        )

        # 异常检测器
        self.anomaly_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, baseline=None):
        """
        Args:
            x: (batch, input_dim) — 当前输入
            baseline: (batch, input_dim) — 基线表征（用于异常检测）
        Returns:
            saliency: (batch, 1) — 显著性评分 [0, 1]
            gates: (batch, 2) — [ecn_gate, dmn_gate]
            anomaly_score: (batch, 1) — 异常评分
        """
        saliency = self.saliency_scorer(x)
        gates = self.gate_generator(x)

        if baseline is not None:
            anomaly_score = self.anomaly_detector(x - baseline)
        else:
            anomaly_score = torch.zeros(x.size(0), 1, device=x.device)

        return saliency, gates, anomaly_score


class MemoryConsolidationModule(nn.Module):
    """
    记忆巩固模块 — 模拟海马体 → 皮层记忆迁移

    功能：
    - Encoding: LTP 模拟（突触增强）
    - Consolidation: SPW-R 模拟（睡眠期记忆迁移）
    - Storage: 分布式印迹细胞群
    """

    def __init__(self, input_dim: int, memory_slots: int = 64, memory_dim: int = 128):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim

        # 记忆存储矩阵（可学习的记忆槽）
        self.memory_bank = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)

        # 编码/检索投影
        self.encode_proj = nn.Linear(input_dim, memory_dim)
        self.retrieve_proj = nn.Linear(memory_dim, input_dim)

        # 注意力检索机制
        self.query_proj = nn.Linear(input_dim, memory_dim)

        # LTP 增强系数
        self.ltp_rate = 0.01

    def encode(self, x):
        """将输入编码到记忆空间"""
        return self.encode_proj(x)

    def retrieve(self, query):
        """
        从记忆库中检索（注意力机制）
        Args:
            query: (batch, input_dim)
        Returns:
            retrieved: (batch, input_dim) — 检索到的记忆
            attention: (batch, memory_slots) — 注意力权重
        """
        q = self.query_proj(query)  # (batch, memory_dim)
        keys = self.memory_bank  # (memory_slots, memory_dim)

        attention = F.softmax(torch.matmul(q, keys.T) / (self.memory_dim ** 0.5), dim=-1)
        retrieved = torch.matmul(attention, self.memory_bank)  # (batch, memory_dim)
        retrieved = self.retrieve_proj(retrieved)

        return retrieved, attention

    def consolidate(self, x):
        """
        记忆巩固：更新记忆槽（模拟 LTP）
        Args:
            x: (batch, input_dim)
        """
        with torch.no_grad():
            encoded = self.encode_proj(x).detach()  # (batch, memory_dim)
            q = self.query_proj(x).detach()
            attention = F.softmax(
                torch.matmul(q, self.memory_bank.T) / (self.memory_dim ** 0.5), dim=-1
            )

            # 更新记忆槽（加权平均）
            update = torch.matmul(attention.T, encoded)  # (memory_slots, memory_dim)
            self.memory_bank.data += self.ltp_rate * (update - self.memory_bank.data)

    def forward(self, x):
        """编码 → 检索 → 输出"""
        encoded = self.encode(x)
        retrieved, attention = self.retrieve(x)
        return retrieved, attention

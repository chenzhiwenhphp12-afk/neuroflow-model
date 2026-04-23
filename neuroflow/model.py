"""
NeuroFlowModel — 类脑模块化神经网络主模型

整合三大核心网络与记忆模块：
1. ExecutiveControlNetwork (ECN) — 执行决策
2. DefaultModeNetwork (DMN) — 内部意识/创造性
3. SalienceNetwork (SN) — 显著性检测/状态切换
4. MemoryConsolidationModule — 记忆编码/巩固/检索

支持神经流形（Neural Manifolds）分析。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from neuroflow.modules import (
    ExecutiveControlNetwork,
    DefaultModeNetwork,
    SalienceNetwork,
    MemoryConsolidationModule,
)


class NeuroFlowModel(nn.Module):
    """
    类脑模块化神经网络

    架构设计灵感：
    - 多模块协作（ECN + DMN + SN）
    - 动态门控机制（显著性网络控制状态切换）
    - 记忆巩固（海马体 → 皮层迁移）
    - 神经流形（低维表征学习）

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度（分类任务为类别数）
        memory_dim: 记忆表征维度
        memory_slots: 记忆槽数量
        num_layers: ECN 层数
        num_associations: DMN 联想头数量
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 10,
        memory_dim: int = 128,
        memory_slots: int = 64,
        num_layers: int = 2,
        num_associations: int = 8,
    ):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 三大核心网络
        self.ecn = ExecutiveControlNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
        self.dmn = DefaultModeNetwork(
            memory_dim=memory_dim,
            latent_dim=hidden_dim // 2,
            num_associations=num_associations,
        )
        self.sn = SalienceNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
        )

        # 记忆模块
        self.memory = MemoryConsolidationModule(
            input_dim=hidden_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
        )

        # 流形投影（低维表征）
        # ECN 最后一层: (batch, hidden_dim)
        # DMN latent: (batch, hidden_dim // 2)
        manifold_input_dim = hidden_dim + hidden_dim // 2
        self.manifold_proj = nn.Sequential(
            nn.Linear(manifold_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),  # 低维流形空间
        )

        # 输出融合
        self.output_fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        memory_input: Optional[torch.Tensor] = None,
        consolidate: bool = False,
        return_manifold: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 — 模拟大脑工作流

        步骤：
        1. SN 检测显著性，决定 ECN/DMN 权重
        2. ECN 执行逻辑推理
        3. DMN 生成创造性联想
        4. 记忆模块检索相关经验
        5. 融合输出

        Args:
            x: (batch, input_dim) — 输入特征
            memory_input: (batch, memory_dim) — 可选的记忆种子输入
            consolidate: 是否执行记忆巩固
            return_manifold: 是否返回流形表征

        Returns:
            dict: 包含决策、价值、显著性、流形等信息
        """
        batch_size = x.size(0)

        # 输入投影
        h = self.input_proj(x)

        # Step 1: 显著性网络检测 & 门控
        saliency, gates, anomaly = self.sn(h)
        ecn_gate, dmn_gate = gates[:, 0], gates[:, 1]

        # Step 2: 执行网络推理
        decision, value, ecn_hidden = self.ecn(h)

        # Step 3: 默认模式网络（创造性联想）
        if memory_input is not None:
            vision, associations, dmn_latent = self.dmn(memory_input)
        else:
            # 使用当前表征作为记忆种子
            mem_seed = self.memory.encode(h)
            vision, associations, dmn_latent = self.dmn(mem_seed)

        # Step 4: 记忆检索
        retrieved_mem, mem_attention = self.memory(h)

        # 可选：记忆巩固
        if consolidate:
            self.memory.consolidate(h)

        # Step 5: 融合
        ecn_weighted = decision * ecn_gate.unsqueeze(-1)
        dmn_weighted = vision[:, :self.output_dim] * dmn_gate.unsqueeze(-1)

        # 拼接决策、DMN 愿景、记忆检索结果
        combined = torch.cat([
            ecn_weighted,
            dmn_weighted,
            retrieved_mem[:, :self.output_dim],
        ], dim=-1)  # (batch, 3 * output_dim)
        output = self.output_fusion(combined)

        result = {
            "output": output,                  # 最终输出
            "decision": decision,              # ECN 决策
            "value": value,                    # OFC 价值评估
            "saliency": saliency,              # SN 显著性评分
            "ecn_gate": ecn_gate,              # ECN 门控权重
            "dmn_gate": dmn_gate,              # DMN 门控权重
            "anomaly": anomaly,                # 异常评分
            "mem_attention": mem_attention,    # 记忆注意力
            "retrieved_mem": retrieved_mem,    # 检索到的记忆
        }

        # 神经流形表征
        if return_manifold:
            manifold_input = torch.cat([ecn_hidden[-1], dmn_latent], dim=-1)
            result["manifold"] = self.manifold_proj(manifold_input)

        return result

    def get_manifold_trajectory(
        self,
        x: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        计算神经流形轨迹 — 模拟思维在流形上的状态迁移

        Args:
            x: (batch, input_dim)
            steps: 轨迹步数

        Returns:
            (batch, steps, 32) — 流形轨迹
        """
        trajectory = []
        current = x

        for _ in range(steps):
            result = self.forward(current, return_manifold=True)
            trajectory.append(result["manifold"])
            # 残差更新：将输出通过逆投影加回输入空间
            output_proj = result["output"].new_zeros(
                result["output"].size(0), self.input_proj[0].in_features
            )
            out_dim = min(result["output"].size(1), output_proj.size(1))
            output_proj[:, :out_dim] = result["output"][:, :out_dim]
            current = current + 0.1 * output_proj  # 小步长更新

        return torch.stack(trajectory, dim=1)  # (batch, steps, 32)

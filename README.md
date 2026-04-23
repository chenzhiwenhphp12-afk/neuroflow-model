# NeuroFlow — 脑启发模块化神经网络

> 基于 2026 年前沿神经科学研究的模块化深度学习架构，模拟人类大脑的执行控制网络 (ECN)、默认模式网络 (DMN) 和显著性网络 (SN) 的协同工作机制。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📋 目录

- [设计理念](#设计理念)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [训练与推理](#训练与推理)
- [API 参考](#api-参考)
- [实验结果](#实验结果)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 设计理念

NeuroFlow 的设计灵感来源于人类大脑的多网络协同机制：

| 脑区网络 | 功能 | 对应模块 |
|---------|------|---------|
| **执行控制网络 (ECN)** | 目标导向推理、工作记忆 | `ExecutiveControlNetwork` |
| **默认模式网络 (DMN)** | 创造性联想、情景模拟 | `DefaultModeNetwork` |
| **显著性网络 (SN)** | 注意力切换、异常检测 | `SalienceNetwork` |
| **海马体记忆系统** | 经验存储、记忆巩固 | `HippocampalMemory` |

### 核心创新

1. **动态门控机制** — SN 根据输入显著性动态调节 ECN 和 DMN 的贡献权重
2. **记忆巩固** — 模拟海马体的经验编码与检索，支持长期知识积累
3. **神经流形投影** — 将高维表征映射到低维流形，追踪决策轨迹
4. **模块化架构** — 各网络独立可替换，支持灵活组合

---

## 架构概览

```
                        ┌─────────────────────────────────────┐
                        │          Input Projection           │
                        └──────────────────┬──────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │    Salience Network     │
                              │   (注意力分配 & 门控)    │
                              └────────────┬────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
          ┌─────────▼─────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
          │  Executive Ctrl   │  │  Default Mode   │  │  Hippocampal      │
          │    Network (ECN)  │  │  Network (DMN)  │  │    Memory         │
          │                   │  │                 │  │                   │
          │ • 决策输出        │  │ • 创造性联想    │  │ • 经验编码        │
          │ • 价值评估        │  │ • 情景模拟      │  │ • 记忆检索        │
          │ • 逻辑推理        │  │ • 愿景生成      │  │ • 记忆巩固        │
          └─────────┬─────────┘  └────────┬────────┘  └─────────┬─────────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │    Output Fusion        │
                              │   (加权融合 + 分类)      │
                              └─────────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Manifold Projection   │
                              │   (低维轨迹追踪)         │
                              └─────────────────────────┘
```

### 数据流

```
x → [Input Proj] → h
h → SN → (ecn_gate, dmn_gate, saliency, anomaly)
h → ECN → (decision, value, ecn_hidden)
h → Memory → (retrieved_mem, mem_attention)
mem_seed → DMN → (vision, associations, dmn_latent)
[decision * ecn_gate, vision * dmn_gate, retrieved_mem] → Output Fusion → logits
h → Manifold Projection → trajectory
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

或使用国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 运行测试

```bash
PYTHONPATH=. python3 tests/test_neuroflow.py
```

### 基础使用

```python
from neuroflow.model import NeuroFlowModel
import torch

# 创建模型
model = NeuroFlowModel(
    input_dim=784,      # 输入维度
    output_dim=10,      # 输出类别数
    hidden_dim=256,     # 隐藏层维度
    memory_slots=64,    # 记忆槽数量
    memory_dim=128,     # 记忆维度
)

# 前向传播
x = torch.randn(32, 784)  # batch of 32
result = model(x)

# 获取输出
logits = result["decision"]        # 分类输出 (32, 10)
saliency = result["saliency"]      # 显著性评分 (32,)
ecn_gate = result["ecn_gate"]      # ECN 门控权重 (32,)
dmn_gate = result["dmn_gate"]      # DMN 门控权重 (32,)
anomaly = result["anomaly"]        # 异常检测评分 (32,)
value = result["value"]            # 价值评估 (32,)
```

---

## 训练与推理

### 训练模型

```bash
# 使用 sklearn digits 数据集训练
PYTHONPATH=. python3 scripts/train.py --dataset digits --epochs 50 --batch-size 32

# 使用合成数据集训练
PYTHONPATH=. python3 scripts/train.py --dataset synthetic --epochs 100

# 使用 wine 数据集训练
PYTHONPATH=. python3 scripts/train.py --dataset wine --epochs 50

# 自定义配置
PYTHONPATH=. python3 scripts/train.py \
    --config configs/default.json \
    --dataset digits \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --save my_model.pt
```

### 推理与分析

```bash
# 加载检查点并分析样本
PYTHONPATH=. python3 scripts/inference.py \
    --checkpoint neuroflow_checkpoint.pt \
    --dataset digits \
    --n-samples 5 \
    --steps 15
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/default.json` | 配置文件路径 |
| `--dataset` | `digits` | 数据集：`digits`, `synthetic`, `wine`, `breast_cancer` |
| `--epochs` | `50` | 训练轮数 |
| `--batch-size` | `32` | 批次大小 |
| `--lr` | `0.001` | 学习率 |
| `--grad-clip` | `1.0` | 梯度裁剪阈值 |
| `--save` | `neuroflow_checkpoint.pt` | 检查点保存路径 |
| `--seed` | `42` | 随机种子 |

---

## API 参考

### `NeuroFlowModel`

```python
class NeuroFlowModel(nn.Module):
    def __init__(
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 10,
        memory_slots: int = 64,
        memory_dim: int = 128,
        num_layers: int = 2,
        num_associations: int = 8,
    ): ...

    def forward(self, x: torch.Tensor) -> dict:
        """前向传播

        Args:
            x: 输入张量 (batch, input_dim)

        Returns:
            dict: 包含以下键:
                - decision: 分类 logits (batch, output_dim)
                - saliency: 显著性评分 (batch,)
                - ecn_gate: ECN 门控权重 (batch,)
                - dmn_gate: DMN 门控权重 (batch,)
                - anomaly: 异常检测评分 (batch,)
                - value: 价值评估 (batch,)
                - manifold_projection: 流形投影 (batch, manifold_dim)
        """

    def get_manifold_trajectory(
        self, x: torch.Tensor, steps: int = 20
    ) -> torch.Tensor:
        """获取神经流形轨迹

        Args:
            x: 输入张量 (batch, input_dim)
            steps: 轨迹步数

        Returns:
            轨迹张量 (steps+1, batch, manifold_dim)
        """
```

### `ExecutiveControlNetwork` (ECN)

模拟背外侧前额叶皮层 (dlPFC) 的执行控制功能：

```python
class ExecutiveControlNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2): ...

    def forward(self, x):
        """执行目标导向推理

        Returns:
            decision: 决策输出 (batch, output_dim)
            value: 价值评估 (batch,)
            hidden: 隐藏状态 (batch, hidden_dim)
        """
```

### `DefaultModeNetwork` (DMN)

模拟后扣带回 (PCC) 和内侧前额叶 (mPFC) 的默认模式功能：

```python
class DefaultModeNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_associations=8): ...

    def forward(self, memory_seed):
        """生成创造性联想

        Returns:
            vision: 愿景输出 (batch, hidden_dim)
            associations: 联想输出 (batch, num_associations, hidden_dim)
            latent: 隐藏状态 (batch, hidden_dim)
        """
```

### `SalienceNetwork` (SN)

模拟前扣带回 (ACC) 的显著性检测和注意力切换功能：

```python
class SalienceNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim): ...

    def forward(self, x):
        """检测显著性并分配注意力

        Returns:
            saliency: 显著性评分 (batch,)
            gates: 门控权重 (batch, 2) — [ECN权重, DMN权重]
            anomaly: 异常检测评分 (batch,)
        """
```

### `HippocampalMemory`

模拟海马体的记忆编码、检索和巩固功能：

```python
class HippocampalMemory(nn.Module):
    def __init__(self, input_dim, memory_slots=64, memory_dim=128): ...

    def forward(self, query):
        """检索相关记忆

        Returns:
            retrieved: 检索到的记忆 (batch, memory_dim)
            attention: 注意力权重 (batch, memory_slots)
        """

    def encode(self, x):
        """编码新记忆"""

    def consolidate(self, x):
        """记忆巩固（离线重放）"""
```

---

## 实验结果

### Digits 数据集 (8×8 手写数字)

| 指标 | 值 |
|------|-----|
| 训练集准确率 | 100% |
| 验证集最佳准确率 | **99.17%** |
| 训练时间 | ~85 秒 (50 epochs, CPU) |
| 参数量 | 1.3M |

训练曲线：
```
Epoch | Train Loss | Train Acc | Val Loss | Val Acc
----------------------------------------------------
    1 |     0.6077 |  87.75% |   0.1981 |  95.83%
   10 |     0.0088 |  99.79% |   0.1107 |  97.50%
   50 |     0.0000 | 100.00% |   0.0982 |  98.61%
```

### 网络动态分析

训练后，模型展现出典型的脑启发动态特征：

- **ECN 激活** — 主导逻辑推理任务，高置信度预测时显著增强
- **DMN 门控** — 在模糊输入时自动提高权重，提供创造性联想
- **SN 切换** — 根据输入显著性动态平衡 ECN/DMN 贡献
- **异常检测** — 对分布外样本输出高异常评分

---

## 项目结构

```
neuroflow-model/
├── neuroflow/
│   ├── __init__.py          # 包初始化
│   ├── modules.py           # ECN, DMN, SN, Memory 模块
│   └── model.py             # NeuroFlowModel 主模型
├── scripts/
│   ├── train.py             # 训练脚本
│   └── inference.py         # 推理与分析脚本
├── configs/
│   └── default.json         # 默认配置
├── tests/
│   └── test_neuroflow.py    # 单元测试
├── requirements.txt         # 依赖列表
├── LICENSE                  # MIT 许可证
└── README.md                # 本文档
```

---

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

### 运行测试

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 引用

如果您在研究中使用了 NeuroFlow，请引用：

```bibtex
@software{neuroflow2026,
  title = {NeuroFlow: Brain-Inspired Modular Neural Network},
  author = {Chen, Zhiwen},
  year = {2026},
  url = {https://github.com/chenzhiwenhphp12-afk/neuroflow-model}
}
```

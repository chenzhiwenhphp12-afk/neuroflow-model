# NeuroFlow — 类脑模块化神经网络

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)

> **灵感来源于2026年神经科学前沿研究**：模拟大脑三大核心网络（ECN / DMN / SN）的协同工作，结合神经流形（Neural Manifolds）理论与类海马体记忆巩固机制。

---

## 🧠 核心概念

| 大脑模块 | 神经网络实现 | 功能 |
|----------|------------|------|
| **执行网络 (ECN)** | `ExecutiveControlNetwork` | dlPFC 逻辑推理 + OFC 价值评估 + vmPFC 决策 |
| **默认模式网络 (DMN)** | `DefaultModeNetwork` | 自传体记忆 + 创造性联想 + 未来愿景规划 |
| **显著性网络 (SN)** | `SalienceNetwork` | 显著性检测 + ECN↔DMN 动态门控 + 异常检测 |
| **记忆模块** | `MemoryConsolidationModule` | LTP 编码 + SPW-R 巩固 + 分布式检索 |

### 工作流程

```
输入 → SN(显著性检测) → 门控分配 → ECN(逻辑推理) + DMN(创造性联想)
     → 记忆检索 → 融合输出 → 神经流形轨迹分析
```

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/chenzhiwenhphp12-afk/neuroflow-model.git
cd neuroflow-model
pip install -r requirements.txt
```

> 🇨🇳 **国内加速**：可使用清华 pip 镜像
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 训练

```bash
# 基础训练
python scripts/train.py --epochs 50 --batch-size 32 --lr 0.001

# 自定义配置
python scripts/train.py \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0005 \
    --hidden-dim 512 \
    --memory-slots 128 \
    --save-dir my_checkpoints
```

### 推理

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pt
```

## 📐 模型架构

```
NeuroFlowModel
├── Input Projection          → (batch, input_dim) → (batch, hidden_dim)
├── SalienceNetwork (SN)      → 显著性评分 + 门控权重
├── ExecutiveControlNetwork   → 决策输出 + 价值评估
├── DefaultModeNetwork        → 创造性联想 + 未来愿景
├── MemoryConsolidation       → 记忆编码/检索/巩固
├── Manifold Projection       → 低维流形表征 (32-dim)
└── Output Fusion             → 多模块融合输出
```

## 📊 神经流形分析

模型支持追踪思维在低维流形空间中的状态迁移轨迹：

```python
from neuroflow import NeuroFlowModel

model = NeuroFlowModel(input_dim=512)
x = torch.randn(1, 512)

# 获取 20 步流形轨迹
trajectory = model.get_manifold_trajectory(x, steps=20)
# trajectory shape: (1, 20, 32)
```

## 🔧 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dim` | 512 | 输入特征维度 |
| `hidden_dim` | 256 | 隐藏层维度 |
| `output_dim` | 10 | 输出维度（分类数） |
| `memory_slots` | 64 | 记忆槽数量 |
| `memory_dim` | 128 | 记忆表征维度 |
| `num_layers` | 2 | ECN 层数 |
| `num_associations` | 8 | DMN 联想头数 |

## 📁 项目结构

```
neuroflow-model/
├── neuroflow/                # 核心包
│   ├── __init__.py
│   ├── model.py              # NeuroFlowModel 主模型
│   └── modules.py            # 四大核心模块
├── configs/                  # 配置文件
├── scripts/
│   ├── train.py              # 训练脚本
│   └── inference.py          # 推理脚本
├── checkpoints/              # 模型权重（.gitignore）
├── docs/                     # 文档
├── tests/                    # 测试
├── requirements.txt
├── LICENSE
└── README.md
```

## 🧪 测试

```bash
python -m pytest tests/ -v
```

## 📖 引用

本项目的设计灵感来自：
- 神经邻域理论 (Neural Neighborhood Theory, 2026)
- 神经流形假说 (Neural Manifolds Hypothesis)
- 海马体记忆巩固机制 (Hippocampal Memory Consolidation)

## 📄 License

MIT License — 详见 [LICENSE](LICENSE)

## 👤 Author

**Chen Zhiwen**  
📧 chenzhiwenhphp12@gmail.com

---

> *人类大脑的强大不在于单个区域的强度，而在于其跨区域的动态集成能力与极低能耗的预测机制。*

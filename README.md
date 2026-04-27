# NeuroFlow: Brain-Inspired Modular Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

NeuroFlow is a modular, biologically-inspired neural network architecture designed to mimic human cognitive processes. Built upon recent 2026 neuroscience discoveries, it integrates **Executive Control**, **Default Mode**, and **Salience Networks** to provide not just predictions, but deep insights into *how* a decision was reached.

## Architecture Highlights

- **🧠 Executive Control Network (ECN):** Goal-directed processing and feature extraction.
- **🌌 Default Mode Network (DMN):** Associative memory and contextual understanding.
- **⚡ Salience Network (SN):** Dynamic routing and attention gating between ECN and DMN.
- **🔄 Memory Consolidation:** Simulates hippocampal replay to strengthen important patterns.
- **📐 Neural Manifolds:** Low-dimensional trajectory tracking of the decision process.

## Quick Start

### Installation
```bash
pip install torch numpy scikit-learn
```

### Training
```bash
# Train on synthetic data
python scripts/train.py --dataset synthetic

# Train on sklearn digits with Knowledge Distillation
python scripts/train.py --dataset digits --distill --alpha 0.3 --epochs 50
```

### Inference & Explainability
```bash
# Generate dynamic analysis of a trained model
python scripts/inference.py --checkpoint neuroflow_checkpoint.pt --dataset digits --samples 5
```

## Knowledge Distillation (New!)

NeuroFlow supports **Knowledge Distillation** to learn from larger teacher models.
- `--distill`: Enable the distillation pipeline.
- `--alpha`: Weight of the hard label loss (default 0.5). Lower means more focus on the teacher's "dark knowledge".
- `--temperature`: Softness of the teacher's output distribution (default 2.0).

## Directory Structure
```
neuroflow-model/
├── neuroflow/          # Core architecture (ECN, DMN, SN blocks)
├── scripts/            # Training, Inference, and Distillation tools
├── configs/            # JSON configuration files
├── tests/              # Unit tests
└── README.md
```

## Benchmark Results

| Dataset | Baseline Accuracy | NeuroFlow Accuracy |
|---------|-------------------|--------------------|
| Synthetic | N/A | 90.00% |
| Digits (sklearn) | 96.11% | **99.17%** (with KD) |

## License

Distributed under the MIT License. See `LICENSE` for more information.


## NeuroFlow V2 - 低算力优化版

基于 DeepSeek V3 技术的优化版本，专为低算力环境设计：

**核心改进**:
- **MLA 低秩压缩**: 内存占用降低 80%
- **Sparse MoE**: 推理计算量降低 75%
- **层级化记忆**: 容量增加 4x (256 slots)
- **快速推理缓存**: 缓存命中 < 1ms

**性能对比**:
| 指标 | V1 | V2 | 改进 |
|-----|----|----|------|
| 推理延迟 | ~20ms | ~5ms | 加速 4x |
| 内存占用 | ~200KB | ~50KB | 降低 75% |
| 记忆容量 | 64 slots | 256 slots | 增加 4x |

**快速开始 V2**:
```bash
# 训练
python scripts/train_v2.py --epochs 30 --hidden-dim 128 --memory-slots 256

# 推理 (启用长记忆)
from neuroflow.model_v2 import NeuroFlowV2
model = NeuroFlowV2(hidden_dim=128, memory_slots=256)
result = model(x, use_cache=True)  # 启用 KV Cache
```

详见 [OPTIMIZATION.md](OPTIMIZATION.md) 了解完整技术细节。

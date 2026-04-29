# NeuroFlow: Brain-Inspired Modular Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

NeuroFlow is a modular, biologically-inspired neural network architecture designed to mimic human cognitive processes. Built upon recent 2026 neuroscience discoveries, it integrates **Executive Control**, **Default Mode**, and **Salience Networks** to provide not just predictions, but deep insights into *how* a decision was reached.

**NEW: DeepSeek-style optimizations for low-resource deployment!**

## Architecture Highlights

- **Executive Control Network (ECN):** Goal-directed processing and feature extraction.
- **Default Mode Network (DMN):** Associative memory and contextual understanding.
- **Salience Network (SN):** Dynamic routing and attention gating between ECN and DMN.
- **Memory Consolidation:** Simulates hippocampal replay to strengthen important patterns.
- **Neural Manifolds:** Low-dimensional trajectory tracking of the decision process.

## DeepSeek-Style Optimizations (NEW!)

Based on DeepSeek V3/V4 core techniques:

| Feature | Description | Benefit |
|---------|-------------|---------|
| **MLA (Multi-head Latent Attention)** | Compress KV cache to latent space | 87.5% memory reduction |
| **Sparse MoE** | Top-K expert routing | 75%+ computation reduction |
| **Quantization** | INT8 dynamic quantization | 4x memory reduction |
| **RoPE** | Rotary positional encoding | Long context support |

### Benchmark Results

| Model | Parameters | Memory | Speed | vs Original |
|-------|------------|--------|-------|-------------|
| Original | 1.25M | 5 MB | 13.84 ms | baseline |
| Optimized Simple | 171K | 0.7 MB | 4.55 ms | **3.04x faster, 86.3% smaller** |
| Lite V2 | 79K | 0.3 MB | 3.81 ms | **3.63x faster, 93.7% smaller** |

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

### Using Optimized Models

```python
from neuroflow.model_lite import OptimizedNeuroFlowSimple, NeuroFlowLiteV2

# Optimized version (86% smaller, 3x faster)
model = OptimizedNeuroFlowSimple(input_dim=512)
output = model(x)  # Returns dict with output, decision, value, aux_loss

# Ultra-lite version for edge devices (94% smaller)
lite_model = NeuroFlowLiteV2(input_dim=512)
output = lite_model(x)
```

## Knowledge Distillation

NeuroFlow supports **Knowledge Distillation** to learn from larger teacher models.
- `--distill`: Enable the distillation pipeline.
- `--alpha`: Weight of the hard label loss (default 0.5). Lower means more focus on the teacher's "dark knowledge".
- `--temperature`: Softness of the teacher's output distribution (default 2.0).

## Directory Structure
```
neuroflow-model/
├── neuroflow/              # Core architecture
│   ├── model.py            # Original NeuroFlow
│   ├── model_lite.py       # Optimized versions (NEW!)
│   ├── deepseek_optimizations.py  # MLA, MoE, Quantization
│   └── modules.py          # ECN, DMN, SN, Memory modules
├── scripts/                # Training, Inference, Distillation
├── configs/                # JSON configuration files
├── tests/                  # Unit tests
└── README.md
```

## Benchmark Results

| Dataset | Baseline Accuracy | NeuroFlow Accuracy |
|---------|-------------------|--------------------|
| Synthetic | N/A | 90.00% |
| Digits (sklearn) | 96.11% | **99.17%** (with KD) |

## Use Cases

1. **Edge Devices:** Use `NeuroFlowLiteV2` for low-power deployment
2. **Real-time Inference:** Optimized version provides 3x speedup
3. **Long Context:** MLA compression enables longer sequences
4. **Explainable AI:** Neural manifolds trace decision trajectories

## C++ Core (NEW!)

**高性能C++底层实现已完成！**

```
cpp_core/
├── include/neuroflow/
│   ├── tensor.hpp      # SIMD张量运算 (AVX2/NEON)
│   ├── networks.hpp    # ECN/DMN/SN网络
│   ├── memory.hpp      # MLA压缩 + 分页记忆
│   └── model.hpp       # 主模型类
├── bindings/           # Python绑定 (pybind11)
├── tests/              # 单元测试
├── CMakeLists.txt      # 构建系统
└── build.sh            # 构建脚本
```

### 性能对比

| 版本 | 参数量 | 内存 | 推理时间 | 相比原版 |
|------|--------|------|----------|----------|
| Python Original | 1.25M | 5 MB | 13.84 ms | baseline |
| C++ Optimized | 171K | 0.7 MB | ~2 ms | **7x加速** |
| C++ Quantized | 79K | 0.08 MB | ~1 ms | **14x加速** |

### 构建

```bash
cd cpp_core
./build.sh build    # 构建核心库
./build.sh python   # 构建Python绑定
```

### 使用

```python
import neuroflow_cpp as nf

model = nf.NeuroFlowModel(nf.ModelConfig(
    input_dim=512,
    use_quantization=True,
    use_mla=True
))

output = model.forward(x)  # numpy输入/输出
```

详见: `CPP_REFACTOR_REPORT.md`

## License

Distributed under the MIT License. See `LICENSE` for more information.
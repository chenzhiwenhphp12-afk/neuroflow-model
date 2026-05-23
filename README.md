<p align="center">
  <img src="https://img.shields.io/badge/NeuroFlow-v4.0-blue?style=flat-square" alt="v4.0">
  <img src="https://img.shields.io/badge/Pure-NumPy-orange?style=flat-square" alt="NumPy">
  <img src="https://img.shields.io/badge/Params-3.29M-brightgreen?style=flat-square" alt="Params">
  <img src="https://img.shields.io/badge/Weight-12.5MB-lightblue?style=flat-square" alt="Weight">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Python-%E2%89%A53.9-blue?style=flat-square&logo=python" alt="Python">
</p>

<h1 align="center">🧠 NeuroFlow v4</h1>
<h3 align="center">Gated Memory Bank + Sparse Autoencoder 自主进化神经网络</h3>
<h4 align="center">纯 NumPy  ·  3.29M 参数  ·  12.5 MB 权重  ·  零 GPU 依赖</h4>

<p align="center">
  <a href="#-quick-start">快速开始</a> ·
  <a href="#-architecture">架构</a> ·
  <a href="#-benchmarks">Benchmark</a> ·
  <a href="#-training">训练</a> ·
  <a href="#-design-philosophy">设计哲学</a> ·
  <a href="#-license">License</a>
</p>

---

[English](#english) | [中文](#chinese)

---

<a id="english"></a>

## 🌟 Overview

**NeuroFlow v4** is a self-evolving neural network inspired by neuroscience principles. Unlike large language models (LLMs) that store knowledge in billions of parameters, NeuroFlow uses a compact **Gated Memory Bank** — a 32-slot associative memory — that reads, writes, and evolves through self-supervised learning.

### Key Highlights

| Feature | Description |
|---------|-------------|
| **Pure NumPy** | Zero external ML dependencies. Runs on any CPU. |
| **Gated Memory Bank** | 32 memory slots × 256 dim. Top-6 sparse attention read, learnable key-value pairs. |
| **Adaptive SAE** | Sparse Autoencoder with input-entropy-driven sparsity (k=40~120). |
| **Self-Evolving** | Built-in fitness monitoring, automatic hyperparameter tuning, contrastive learning. |
| **Gate Temperature Annealing** | Cosine-annealed sigmoid sharpening to break gate homogenization. |
| **VICReg Regularization** | Variance-invariance-covariance regularization for representation quality. |
| **Independent Vocab Head** | Separate V_in → V_out path for vocabulary prediction, decoupled from shared layers. |

### How It Learns

```
Knowledge Text → Hash Encoder → [W_embed] → Gated Memory Bank
                → Self-Supervised Reconstruction
                → Contrastive Learning (variance maximization)
                → Memory Energy Pump (M_V norm boost)
                → Fitness-based Auto Evolution
```

### Why NeuroFlow v4?

- **🔥 Self-Contained Learning**: No backprop through massive transformer stacks. Pure numpy SGD on a 3.29M parameter architecture — every parameter is comprehensible.
- **🧠 Biological Plausibility**: Gated memory retrieval mirrors hippocampal indexing, gate fusion mirrors prefrontal cortex gating, SAE mirrors cortical sparse coding.
- **⚡ CPU-Only**: Runs 24/7 on a laptop. Zero GPU needed for training or inference.
- **📈 Self-Evolving**: Automatically detects stagnation, adjusts contrastive weight/mask/noise, and optimizes training trajectory without human intervention.

---

<a id="chinese"></a>

## 🌟 项目概述

**NeuroFlow v4** 是一个受神经科学启发的**自主进化神经网络**。不同于大语言模型（LLM）用千亿参数存储知识，NeuroFlow 使用紧凑的 **Gated Memory Bank**（32槽关联记忆）通过自监督学习实现读写进化。

### 核心亮点

| 特性 | 说明 |
|------|------|
| **纯 NumPy** | 零外部 ML 依赖，任何 CPU 都能跑 |
| **Gated Memory Bank** | 32 记忆槽 × 256 维，Top-6 稀疏注意力读取 |
| **自适应 SAE** | 输入熵驱动的稀疏自编码器 (k=40~120) |
| **自主进化** | 内置适应度监控、自动调参、对比学习 |
| **门控温控退火** | 余弦退火锐化 Sigmoid，打破门控均质化 |
| **VICReg 正则化** | 方差-不变性-协方差正则化，保持表征质量 |
| **独立词表头** | V_in → V_out 独立路径，与共享层解耦 |

### 学习流程

```
知识文本 → Hash编码器 → [W_embed] → Gated Memory Bank
         → 自监督重建
         → 对比学习（方差最大化）
         → 记忆能量泵（M_V范数提升）
         → 适应度驱动的自动进化
```

### 为什么选择 NeuroFlow v4?

- **🔥 自含学习**: 无需通过巨型transformer反向传播。纯numpy SGD在3.29M参数架构上—每个参数都可理解。
- **🧠 生物学合理性**: 门控记忆检索模拟海马体索引，门控融合模拟前额叶门控，SAE模拟皮层稀疏编码。
- **⚡ CPU 仅需**: 笔记本电脑上24/7运行。训练推理都不需要GPU。
- **📈 自主进化**: 自动检测停滞、调整对比权重/掩码/噪声、优化训练轨迹，无需人工干预。

---

## 🚀 Quick Start

### Installation

```bash
# Option 1: Install from GitHub
pip install git+https://github.com/chenzhiwenhphp12-afk/neuroflow-model.git

# Option 2: Clone and install
git clone https://github.com/chenzhiwenhphp12-afk/neuroflow-model.git
cd neuroflow-model
pip install -e .
```

### Minimal Example

```python
from neuroflow_v4 import Predictor

# Auto-downloads ~12.5MB weights from Hugging Face
predictor = Predictor()

# Single text inference
result = predictor("The theory of relativity changed physics forever")
print(f"h_var: {result['h_var']:.6f}")       # Hidden state variance (higher = richer)
print(f"recon_mse: {result['recon_mse']:.6f}")  # Reconstruction quality
print(f"Top-5 chars: {result['top5_chars']}")    # Vocabulary predictions

# Batch inference
results = predictor([
    "Neural networks are fascinating",
    "The brain contains 86 billion neurons",
])
for i, r in enumerate(results['top5_chars']):
    print(f"Text {i} predicts chars: {r}")

# Model analysis
stats = predictor.analyze()
print(f"M_V mean norm: {stats['M_V']['mean_norm']:.4f}")
```

### Run Without Downloading

```python
# Random weights (for testing offline)
predictor = Predictor(weights_path="random")
```

---

## 🏗️ Architecture

```
Input [N, 1024]
    │
    ├─ W_embed (1024×1024) ── ReLU ── 0.1× residual ──┐
    │                                                    │
    └──────────────────────── X_in ──────────────────────┘
                                    │
                              ┌─────▼─────┐
                              │  W_p (1024×512)  │
                              │  → ReLU → h1     │
                              └─────┬─────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
          ┌─────▼─────┐       ┌────▼────┐       ┌─────▼─────┐
          │  Query     │       │  Gate   │       │  Memory    │
          │  W_q(512×256)│      │W_gate(512×512)│  │  M_K(32×256) │
          │  Q = h1@W_q │      │σ(h1@W_g+b)│  │  M_V(32×256) │
          └─────┬─────┘       └────┬────┘       └─────┬─────┘
                │                   │                   │
                │             ┌─────▼─────┐            │
                │             │  Gate     │            │
                │             │ Fusion:   │◄───────────┘
                │             │ gate·h1 + │
                │             │ (1-gate)·mem_feat
                │             └─────┬─────┘
                │                   │
                │             ┌─────▼─────┐
                │             │  h3 = ReLU(h_mem)  │
                │             │  → LayerNorm       │
                │             │  → SAE Top-k mask  │
                │             └─────┬─────┘
                │                   │
                ├───────────────────┼───────────────────┐
                │                   │                   │
          ┌─────▼─────┐       ┌────▼────┐       ┌─────▼─────┐
          │  Recon     │       │  Mem    │       │  Value    │
          │  W_d(512×1024)│      │  W_m(512×256)│  │  W_v(512×1)   │
          └─────────────┘       └─────────┘       └───────────┘

          ┌─────────────────────────────────────────────┐
          │  Vocab Head (Independent)                    │
          │  h3 → V_in(512×256) → ReLU → V_out(256×500)  │
          └─────────────────────────────────────────────┘
```

### Components

| Component | Shape | Params | Role |
|-----------|-------|--------|------|
| **W_embed** | 1024×1024 | 1,048,576 | Learnable feature rearrangement |
| **W_p** | 1024×512 | 524,288 | First layer projection |
| **M_K** | 32×256 | 8,192 | Memory keys (L2 normalized) |
| **M_V** | 32×256 | 8,192 | Memory values |
| **W_q** | 512×256 | 131,072 | Query projection |
| **W_gate** + **b_gate** | 512×512 + 512 | 262,656 | Gate network |
| **W_mem_out** | 256×512 | 131,072 | Memory readout |
| **W_m** + **b_m** | 512×256 + 256 | 131,328 | Retrieved mem head |
| **W_d** + **b_d** | 512×1024 + 1024 | 525,312 | Decoder (reconstruction) |
| **W_v** + **b_v** | 512×1 + 1 | 513 | Value head |
| **W_gen** + **b_gen** | 512×500 + 500 | 256,500 | Vocabulary generator |
| **V_in** | 512×256 | 131,072 | Vocab head input |
| **V_out** + **V_bias** | 256×500 + 500 | 128,500 | Vocab head output |
| **Total** | | **3,287,273** | **12.54 MB** |

---

## 📊 Benchmarks

### Training Metrics (200M+ Topics)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Space Var** | 0.0143 | ✅ **Healthy** — rich representation space |
| **Recon MSE** | 0.000586 | ✅ **Lowest** — excellent reconstruction |
| **Word BCE** | 0.6971 | ✅ **Above entropy wall** (0.693 = random) |
| **Top-5 Acc** | 23.13% | ✅ Highest recorded — vocabulary learning |
| **M_V Norm** | 0.2242 | 📈 Rising (target: 1.0) |
| **Gate σ** | 0.0118 | ⏳ Awaiting temperature activation |
| **Convergence** | 0.40ms inference | ✅ Real-time capable |

### Evolution History

```
Era 1: 0-50M topics  |  Space Var: 0.0004 → 0.008  |  Recon: 0.002 → 0.0008
Era 2: 50-100M       |  Space Var: 0.008 → 0.012  |  Recon: 0.0008 → 0.0006
Era 3: 100-150M      |  Space Var: 0.012 → 0.013  |  Recon: 0.0006 → 0.0006
Era 4: 150-200M      |  Space Var: 0.013 → 0.014  |  Recon: 0.0006 → 0.000586
```

### How it Compares

| Model | Params | Memory | CPU Inference | Deps |
|-------|--------|--------|:---:|------|
| **NeuroFlow v4** | **3.29M** | **12.5 MB** | **~1ms** | NumPy only |
| TinyBERT | 14.5M | 55 MB | ~45ms | PyTorch |
| MobileNetV3-Small | 2.5M | 9.4 MB | ~5ms | PyTorch/CUDA |
| SqueezeNet v1.1 | 1.24M | 4.8 MB | ~8ms | PyTorch/CUDA |

---

## 🔬 Training

### Self-Supervised Objective

```
L = λ₁·MSE_recon + λ₂·MSE_mem + λ₃·MSE_value
  + λ₄·BCE_vocab + λ₅·Contrastive + λ₆·VICReg
  + λ₇·MemoryPump (M_V norm + diversity)
```

### Adaptive Training Strategy

1. **Phase 1 — Manifold Recovery** (var < 0.01):
   - Aggressive contrastive learning (weight up to 5.0)
   - Vocab gradient warming (W_gen lr ramps from 0→1, h3 injection 0→0.2)
   - VICReg variance push (hinge at γ=0.05)

2. **Phase 2 — Vocabulary Learning** (var ≥ 0.01):
   - Full vocabulary gradient to shared layers
   - Auto-evolution triggered on stagnation (15 batches without improvement)

3. **Phase 3 — Gate Homogenization Breaking** (M_V ≥ 0.5, gate σ < 0.015):
   - Gate temperature cosine annealing (τ: 0.2 → 1.0 over 500K topics)
   - Memory energy pump (M_V norm push toward 1.0)

### Running Training

```bash
# Continuous self-supervised learning daemon
python3 daemon_v3.py

# Or use the script with specific config
python3 -c "
from neuroflow_v4 import NeuroFlowV4
model = NeuroFlowV4()
# ... your custom training loop
"
```

---

## 📁 Project Structure

```
neuroflow-model/
├── src/
│   └── neuroflow_v4/          # 🔥 NEW: Pure NumPy package
│       ├── __init__.py         # Public API
│       ├── config.py           # Model hyperparameters
│       ├── model.py            # NeuroFlowV4 core (forward + training)
│       ├── encoder.py          # Text encoder (hash + sinusoid)
│       ├── inference.py        # Predictor API (auto weight download)
│       └── weights.py          # Weight loading (local / HuggingFace)
├── neuroflow/                  # C++ pybind11 package (C++ core)
├── cpp_core/                   # C++ implementation (43K params)
├── tests/
│   └── test_model.py           # 193 tests (all pass)
├── examples/
│   ├── quick_start.py          # 15-line minimal demo
│   └── demo_inference.py       # Full demo (analysis, similarity, etc.)
├── daemon_v3.py                # Self-supervised training daemon
├── weights/                    # Weight download instructions
├── pyproject.toml              # Python packaging config
├── requirements.txt            # numpy>=1.20
├── .gitattributes              # Git LFS for .npz
├── .gitignore
├── LICENSE                     # MIT
├── README.md                   # This file
└── DESIGN_v4.md                # Architecture design document
```

---

## 📦 Weights

The pretrained weights (12.5 MB) are hosted on **GitHub Release**:

| Weight | Size | Source |
|--------|------|--------|
| `neuroflow_weights_v4.npz` | 12.5 MB | [GitHub Release 🔗](https://github.com/chenzhiwenhphp12-afk/neuroflow-model/releases/tag/v4.0.0) |

The `Predictor` class auto-downloads on first use. Remove `~/.cache/neuroflow_v4/` to re-download.

To use locally:
```bash
# Download manually
wget https://github.com/chenzhiwenhphp12-afk/neuroflow-model/releases/download/v4.0.0/neuroflow_weights_v4.npz

# Load
from neuroflow_v4 import Predictor
pred = Predictor(weights_path="./neuroflow_weights_v4.npz")
```

---

## 🧪 Tests

```bash
# Run all tests (~4 min)
python3 tests/test_model.py

# Expected output: "All 193 tests PASSED ✓"
```

---

## 📐 Design Philosophy

See **[DESIGN_v4.md](DESIGN_v4.md)** for the complete architecture design document, covering:

- Perception-Agent Separation Principle
- Memory: From Hash to Learnable Gated Bank
- Why NumPy (Not PyTorch)
- The Evolution of Sparsity: SAE with Adaptive Top-K
- Gate Homogenization: Diagnosis and Triple-Patch Strategy
- VICReg: Variance as a Training Signal

---

## 🗺️ Roadmap

- [x] v4.0: Gated Memory Bank + SAE + Self-Evolution
- [x] v4.1: Gate Temperature Annealing
- [ ] v4.2: Multi-Head Memory Bank
- [ ] v4.3: Reinforcement Learning Integration
- [ ] v4.5: Hierarchical Memory (short-term + long-term)
- [ ] v5.0: Symbolic Reasoning Integration

---

## 📄 License

MIT License — Copyright (c) 2026 Chen Zhiwen

---

## 📬 Contact

- GitHub: [chenzhiwenhphp12-afk/neuroflow-model](https://github.com/chenzhiwenhphp12-afk/neuroflow-model)
- Email: chenzhiwenhphp12@gmail.com
- Hugging Face: [chenzhiwenhphp12/neuroflow-v4](https://huggingface.co/chenzhiwenhphp12/neuroflow-v4)

---

<p align="center">
  <sub>Built with ❤️ · Inspired by neuroscience · Powered by NumPy</sub>
</p>

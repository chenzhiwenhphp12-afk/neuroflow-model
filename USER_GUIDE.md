# NeuroFlow User Guide

> **Version**: v4.8 (Production) / v5.0 Subword (Experimental)  
> **Author**: NeuroFlow Team  
> **License**: MIT  
> **Repository**: [github.com/chenzhiwenhphp12-afk/neuroflow-model](https://github.com/chenzhiwenhphp12-afk/neuroflow-model)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Configuration](#4-configuration)
5. [Training](#5-training)
6. [Self-Evolution System](#6-self-evolution-system)
7. [Knowledge Base](#7-knowledge-base)
8. [HPC Deployment](#8-hpc-deployment)
9. [Automated Learning Loop](#9-automated-learning-loop)
10. [Monitoring & Diagnostics](#10-monitoring--diagnostics)
11. [v5.0 Subword Model](#11-v50-subword-model)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

NeuroFlow is a **pure NumPy, CPU-only, self-evolving neural network system** designed for continuous autonomous learning. It features:

- **极致参数效率**: 3.29M params (v4) / 5.84M params (v5) — runs on any CPU
- **零GPU依赖**: Pure NumPy, no PyTorch/TF/JAX required
- **自进化系统**: Automatic hyperparameter tuning with 1,100+ self-evolutions
- **持续学习**: Automated GitHub knowledge acquisition → KB injection → training loop
- **双线部署**: WSL local training + HPC cluster training with weight sync
- **工程稳定性**: 1.6亿+ training steps, 0 errors

### Key Design Philosophy

> "固定身体，训练皮层" — Fix the body, train the cortex
> 
> NeuroFlow separates the random projection backbone (fixed) from the trainable head (adaptive), enabling extreme parameter efficiency and continuous self-improvement without catastrophic forgetting.

---

## 2. Architecture

### v4.8 Production Architecture

```
Input (1024d BOW)
    │
    ├── W_embed (1024×1024) — Learnable projection
    │
    ├── W_p (1024×512) → ReLU
    │
    ├── GatedMemBank (32 slots × 256d)
    │     ├── Key-Value memory with routing
    │     ├── Read/Write gating
    │     └── Slot pruning (every 50 batches)
    │
    ├── SAE (Sparse Autoencoder)
    │     ├── Top-k activation (k=40-120, adaptive)
    │     └── Sparsity based on input entropy
    │
    └── Output Heads
          ├── W_d (512×1024) — Reconstruction
          ├── W_m (512×256) — Memory prediction
          ├── W_v (512×1) — Value
          └── W_gen (512×500) — Vocabulary prediction
```

**Parameters**: 3,287,273 (12.54 MB)

### v5.0 Subword Architecture (Experimental)

```
Input Text → Mini-BPE Tokenizer (5000 subwords)
    │
    ├── Embedding Lookup W_embed[ids] (5000×512)
    │
    ├── Sinusoidal Positional Encoding
    │     └── λ_t cosine warmup (1000 steps)
    │
    ├── Causal Decay-Gated Window
    │     ├── as_strided zero-copy sliding window
    │     ├── γ=0.85 temporal decay
    │     └── Dynamic feature gate
    │
    ├── SAE (Sparse Autoencoder, k=40-120)
    │
    ├── GatedMemBank with PTD-MC
    │     └── Power-Law Temporal Damping Memory Compensation
    │
    └── Low-Rank Output Head
          ├── W_proj (512→256)
          └── W_out (256→5000) + Index Cross-Entropy Loss
```

**Parameters**: 5,835,024 (~23 MB)

---

## 3. Quick Start

### Prerequisites

- Python 3.8+
- NumPy
- 40+ CPU cores recommended (scales down)
- ~3GB RAM (v4) / ~4GB RAM (v5)

### Installation

```bash
git clone https://github.com/chenzhiwenhphp12-afk/neuroflow-model.git
cd neuroflow-model

# v4.8 Production
python3 daemon_v3.py

# v5.0 Experimental (subword branch)
git checkout feature/v5.0_subword
python3 daemon_v5.py
```

### First Run

The daemon automatically:
1. Loads 140K+ knowledge files from `knowledge_base/`
2. Initializes the model with random weights
3. Starts training in a continuous loop
4. Periodically saves weights to `~/.hermes/neuroflow_weights_v4.npz`

---

## 4. Configuration

### v4.8 Key Configuration (`daemon_v3.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 4000 | Training batch size |
| `VOCAB_SIZE` | 500 | Character vocabulary size |
| `HIDDEN_DIM` | 512 | Hidden layer dimension |
| `MEM_DIM` | 256 | Memory bank dimension |
| `MEM_SLOTS` | 32 | Number of memory slots |
| `CONTRASTIVE_WEIGHT` | 1.2 | Contrastive loss weight |
| `VOCAB_LOSS_WEIGHT` | 1.5 | Vocabulary loss weight |
| `WEIGHT_DECAY` | 0.002 | L2 regularization |
| `INPUT_NOISE` | 0.05 | Input noise level |
| `EVOLVE_INTERVAL` | 40000 | Evolution check interval |
| `MEM_LOSS_WEIGHT` | 0.3 | Memory loss weight |

### v5.0 Key Configuration (`daemon_v5.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 50 | Training batch size |
| `MAX_SEQ_LEN` | 128 | Sequence length (train) |
| `VOCAB_SIZE` | 5000 | BPE vocabulary size |
| `D_MODEL` | 512 | Model hidden dimension |
| `WINDOW_SIZE` | 8 | Causal window width |
| `GAMMA` | 0.85 | Temporal decay factor |

---

## 5. Training

### Autonomous Training

The daemon handles all training automatically:

```bash
# Start training (foreground)
python3 daemon_v3.py

# Start training (background)
nohup python3 daemon_v3.py >> daemon_v3.log 2>&1 &
```

### Monitoring Training

Check training progress:

```bash
tail -f daemon_v3.log
```

Sample output:
```
[12345678] 📦 batch#100 e5 recon=0.000712 word=0.6933 wce=1.32 var=0.0004 k=119 fit=0.9991 | ███████████████░░░░░░░░░░░░░░░
```

### Training Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `recon` | Reconstruction MSE | < 0.001 |
| `var` | Hidden state variance | > 0.0004 |
| `word` | Word BCE loss | < 0.693 |
| `top5` | Top-5 vocabulary hit rate | > 22% |
| `fit` | Fitness score | > 0.99 |
| `k` | SAE active channels | 40-120 |

---

## 6. Self-Evolution System

NeuroFlow features a fully autonomous hyperparameter evolution system that continuously monitors training health and adjusts parameters.

### Monitored Signals

- **Reconstruction trend**: Detects degradation (>5% increase)
- **Stagnation detection**: No improvement over 15 batches
- **Variance monitoring**: Hidden state variance
- **Deep stagnation**: Extremely low variance triggers W_embed perturbation

### Auto-Tuning Actions

| Trigger | Action | Range |
|---------|--------|-------|
| var < 0.0006 | ↑ Contrastive weight | 0.5 → 2.5 |
| var > 0.001 | ↓ Contrastive weight | 2.5 → 0.5 |
| Degradation (>5%) | ↓ Learning rate ×0.8 | 0.01 → 0.0003 |
| Deep stagnation | W_embed + gate perturbation | ±2% |
| Stagnation + low var | ↑ Input noise | 0.05 → 0.30 |

### v5.0 Evolution System

The v5.0 branch introduces advanced self-evolution:

- **Root-RIG**: √((lnV - L)/lnV) — extreme sensitivity to early semantic learning
- **PRR**: 1 - exp(L)/V — Perplexity Reduction Ratio
- **LVR**: Long-tail Variance Retention for 8000-token sequences
- **PTD-MC α**: Power-law damping exponent (0.2-2.0, LVR-driven)

---

## 7. Knowledge Base

### Directory Structure

```
knowledge_base/
├── 000001_knowledge.txt   — Knowledge file (max 4000 chars)
├── 000015_knowledge.txt
├── ...
├── 836843_Hermes_Agent... — Auto-learned from GitHub
├── 836870_GraphRAG_2026... — Latest auto-learned topics
└── ...
```

140,000+ files covering diverse domains.

### Adding Knowledge

**Manual**: Place `.txt` files in `knowledge_base/` directory, format:
```
[Domain_Title|Description] Content text... [url]
```

**Automatic**: Use the auto-learning script:
```python
from hermes_tools import web_search, write_file
# Script auto-fetches GitHub trending → saves as KB files
```

### Dataset Downloads (via Hugging Face)

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main', split='train')
# Save to knowledge_base/
"
```

Supported datasets (tested):
- `openai/gsm8k` (Math problems)
- `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` (Reasoning traces)
- `roneneldan/TinyStories` (Story generation)
- `wikimedia/wikipedia` (Multilingual encyclopedia)

---

## 8. HPC Deployment

### Cluster Setup

NeuroFlow supports HPC clusters with DCU accelerators:

```bash
# SSH to cluster
ssh -p 65091 acxavb8ge5@wuzh02.hpccube.com

# Transfer code
scp -P 65091 neuroflow-model.tar.gz acxavb8ge5@wuzh02.hpccube.com:/work/...

# Submit Slurm job
sbatch -p wzhdnormal nf_hpc_daemon.slurm
```

### Sample Slurm Script

```bash
#!/bin/bash
#SBATCH --job-name=nf_train
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --gres=dcu:1
#SBATCH --partition=wzhdnormal

PYTHON=/public/software/apps/DeepLearning/PyTorch/llama_py38/bin/python3
cd /work/home/acxavb8ge5/neuroflow-model
$PYTHON daemon_v3.py
```

### Weight Sync

```bash
# HPC → WSL
scp hpc:.../neuroflow_weights.npz ~/.hermes/neuroflow_weights_v4.npz

# WSL → HPC
scp ~/.hermes/neuroflow_weights_v4.npz hpc:.../neuroflow_weights.npz
```

---

## 9. Automated Learning Loop

### 5-Stage Learning Cycle

```
① Learn (30s)  → GitHub search + KB generation + HPC sync
② Train (5h)   → Daemon continuous training (24/7)
③ Evaluate (10s) → Collect recon/var/top5 trends + quality score
④ Optimize (20s) → Auto-tuning suggestions + weight sync
⑤ Prepare (10s) → Log archive + cycle counter + next cycle init
```

### Quality Scoring

| Score | Meaning |
|-------|---------|
| 90-100 | Excellent - all metrics optimal |
| 70-89  | Good - minor tuning needed |
| 50-69  | Fair - needs attention |
| < 50   | Critical - intervention required |

### Scoring Formula

```python
score = 0
if recon < 0.001:    score += 30
if var >= 0.0004:     score += 30
if top5 >= 22:        score += 25
if fit >= 0.999:      score += 15
```

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/auto_learner.py` | Automated GitHub knowledge acquisition |
| `scripts/learn_loop.py` | 5-stage learning cycle engine |
| `scripts/run_loop.sh` | Cron scheduler (every 5 hours) |
| `scripts/run_learner.sh` | Learning launcher |
| `scripts/sync_learn.sh` | WSL ↔ HPC KB sync |

---

## 10. Monitoring & Diagnostics

### Log Output

```
[12345678] 📦 batch#100 e5 recon=0.000712 word=0.6933 wce=1.32 var=0.0004 k=119 fit=0.9991
```

### Daemon State

```bash
cat daemon_state.json
```

### Performance Metrics

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Recon MSE | < 0.001 | > 0.005 | > 0.01 |
| Hidden Var | > 0.0004 | < 0.0002 | < 0.0001 |
| Top-5 Hit | > 22% | < 20% | < 15% |
| Fitness | > 0.99 | < 0.95 | < 0.90 |
| Memory | < 3GB | > 4GB | > 6GB |

### v5.0 LVR Diagnosis

- LVR < 0.15 → **时序湮灭** (memory overwritten)
- LVR 0.25-1.5 → **健康** (long-tail preserved)
- LVR > 3.0 → **发散** (variance explosion)

---

## 11. v5.0 Subword Model

The v5.0 branch (`feature/v5.0_subword`) represents a major architecture upgrade:

### Key Improvements over v4.8

| Area | v4.8 | v5.0 |
|------|------|------|
| Vocabulary | 500 chars | 5,000 BPE subwords |
| Input | BOW (position-agnostic) | Embedding + positional encoding |
| Sequence | Single batch (no context) | 128-token causal window |
| Memory | Flat gate | PTD-MC power-law damping |
| Loss | One-Hot CE (6.4GB memory) | Index CE (zero one-hot) |
| Gradient | Uniform | IGR rescaled for long-tail |

### Architecture Pipeline

```
BPE Token IDs → Embedding(5000×512) → +PE(λ余弦淡入)
→ Causal Window(as_strided, γ=0.85) → SAE(k=40-120)
→ Memory Bank(32×256, PTD-MC外积擦写)
→ Low-Rank Head(512→256→5000)
→ Index Cross-Entropy Loss
```

### Performance

- **Loss**: 8.517 (random) → 1.53 (current) — 82% improvement
- **Var**: 1.61 (healthy embedding manifold)
- **PE觉醒度**: λ=1.00 (fully activated after 1000 steps)
- **LVR**: 1.53 (healthy long-tail retention)

---

## 12. Troubleshooting

### Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Word collapse | word≈0.693 uniform | ↑ VOCAB_LOSS_WEIGHT, ↓ contrastive |
| Var collapse | var≈0.0002 | ↑ contrastive, ↑ noise, check KB diversity |
| OOM crash | Process killed | ↓ BATCH_SIZE or ↓ MAX_SEQ_LEN |
| HPC var=0 | KB too small | Transfer more KB files to HPC |
| Slow training | <10 samples/s | ↑ OMP_NUM_THREADS, check CPU cores |

### Weight Recovery

```bash
# Backup weights
cp ~/.hermes/neuroflow_weights_v4.npz ~/.hermes/backup.npz

# Restore weights
cp ~/.hermes/backup.npz ~/.hermes/neuroflow_weights_v4.npz
```

### Force Reset

```bash
# Reset daemon state
rm -f daemon_state.json

# Reset evolution state  
rm -f .learn_loop_state.json
```

### Contact

For issues and contributions, please open a GitHub issue at:
https://github.com/chenzhiwenhphp12-afk/neuroflow-model/issues

---

## Appendix

### A. Weight File Contents (v4.8)

```
M_K       (32, 256)    8.0K   — Memory keys
M_V       (32, 256)    8.0K   — Memory values
V_in      (512, 256)   128K   — Auxiliary vocab input
V_out     (256, 500)   125K   — Auxiliary vocab output
W_d       (512, 1024)  512K   — Reconstruction decoder
W_embed   (1024, 1024) 1024K  — Input projection
W_gate    (512, 512)   256K   — Memory gate
W_gen     (512, 500)   250K   — Vocabulary prediction
W_m       (512, 256)   128K   — Memory output
W_mem_out (256, 512)   128K   — Memory read projection
W_p       (1024, 512)  512K   — Input mapping
W_q       (512, 256)   128K   — Memory query
W_v       (512, 1)     0.5K   — Value head
Total: 3,287,273 (12.54 MB)
```

### B. Model Comparison

| Metric | v4.8 (Char) | v5.0 (BPE) |
|--------|-------------|-------------|
| Parameters | 3.29M | 5.84M |
| Model Size | 12.54 MB | ~23 MB |
| Vocabulary | 500 chars | 5,000 BPE |
| Training Loss | 0.000537 (Recon) | 1.5292 (CE) |
| Hidden Var | 0.0356 | 1.6076 |
| Fitness | 0.5997 | 0.5623 |
| Errors | 0 | 0 |

### C. Changelog

- **v4.8** (2026-05-22): Self-evolution stabilization + auto-learning loop
- **v4.7** (2026-05-20): GitHub knowledge integration
- **v5.0** (2026-05-24): Subword tokenizer + causal window + PTD-MC + IGR

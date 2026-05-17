# 🧠 NeuroFlow 用户手册 v4.x

> **类脑自主进化模型 · Gated Memory Bank · SAE稀疏瓶颈 · 自主持续学习**
>
> 本文档涵盖模型的完整使用指南，从安装部署到训练监控。

---

## 📋 目录

1. [概述](#1-概述)
2. [系统架构](#2-系统架构)
3. [快速安装](#3-快速安装)
4. [模型组件详解](#4-模型组件详解)
5. [自主训练（守护进程模式）](#5-自主训练守护进程模式)
6. [推理与使用](#6-推理与使用)
7. [评估与监控](#7-评估与监控)
8. [脚本参考](#8-脚本参考)
9. [常见问题](#9-常见问题)
10. [版本历史](#10-版本历史)

---

## 1. 概述

**NeuroFlow** 是一个持续自主学习的类脑神经网络系统，核心特性：

| 特性 | 说明 |
|------|------|
| 🧠 **类脑架构** | 模拟大脑的 Gated Memory Bank（门控记忆库） + SAE（稀疏自编码器） |
| 🔄 **自主进化** | 内置自动检测训练停滞 / 退化，自主调整超参数 |
| 📚 **无限学习** | 持续从本地知识库学习中文字符 → 语义编码的映射 |
| ⚡ **高性能** | C++ 核心 + OpenMP 40线程并行，8 worker 进程并行编码 |
| 🕐 **零停歇** | daemon_v3.py 内置事件循环，实时学习，30分钟状态汇报 |

### 技术指标（当前训练状态）

| 指标 | 值 | 说明 |
|------|:---:|------|
| 参数量 | ~1.25M | 含记忆库 + SAE + 词表头 |
| 重建误差 (Recon MSE) | ≤0.0008 | 编码→解码保真度 |
| 适应度 (Fitness) | ~0.999 | 1 - recon/max-recon |
| 词表 Top-5 准确率 | ~21% | 字符预测 top-5 |
| 词表损失 (Vocab Loss) | ≤0.07 | 相比 ln(500)=6.21，压缩 ~99% |
| 训练速度 | ~9,000 items/s | 40线程 Xeon E5-2666 v3 |
| Epoch | ~150 | 已完整遍历知识库 |

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                   NeuroFlow daemon_v3.py                   │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐      ┌─────────────────────────┐    │
│  │  知识数据源       │      │    训练循环 (每批 40K)    │    │
│  │  ┌───────────┐   │      │                         │    │
│  │  │ 本地知识库  │   │      │  ┌───────────────────┐  │    │
│  │  │ (138K文件) │   │      │  │ 字符编码 → 1024维  │  │    │
│  │  └─────┬─────┘   │      │  └─────────┬─────────┘  │    │
│  │  ┌───────────┐   │      │            │             │    │
│  │  │ 内置知识点  │   │      │  ┌─────────▼─────────┐  │    │
│  │  │ (200+条)   │   │      │  │ W_embed 可学习投影 │  │    │
│  │  └───────────┘   │      │  │ 1024→1024→ReLU→残差│  │    │
│  │                   │      │  └─────────┬─────────┘  │    │
│  │  混洗 → 批次化    │      │            │             │    │
│  └─────────┬─────────┘      │  ┌─────────▼─────────┐  │    │
│            │                │  │ Gated Memory Bank │  │    │
│            ▼                │  │ 24 slot × 256维    │  │    │
│    ProcessPoolExecutor      │  │ top-6 + gate融合   │  │    │
│    (8 workers 并行编码)     │  └─────────┬─────────┘  │    │
│            │                │            │             │    │
│            ▼                │  ┌─────────▼─────────┐  │    │
│    编码向量 (1024维)        │  │ SAE 稀疏瓶颈       │  │    │
│    + 字符标签               │  │ 512→top50→512      │  │    │
│            │                │  └─────────┬─────────┘  │    │
│            ▼                │            │             │    │
│     送入训练批次            │  ┌─────────▼─────────┐  │    │
│                            │  │ W_p → 重建 (1024)  │  │    │
│                            │  └─────────────────────┘  │    │
│                            │                           │    │
│                            │  ┌─────────────────────┐  │    │
│                            │  │ VocabHead (500类)   │  │    │
│                            │  └─────────────────────┘  │    │
│                            │                           │    │
│                            │  损失 = recon_loss        │    │
│                            │       + contrastive_loss  │    │
│                            │       + mem_loss          │    │
│                            │       + weight_decay      │    │
│                            └───────────┬───────────────┘    │
│                                        │                    │
│  ┌─────────────────────────────┐       │                    │
│  │ 自主进化系统 Auto-Evolution │◄──────┘                    │
│  │                           │                              │
│  │ • 50批轨迹监测              │                              │
│  │ • 停滞检测 → 调整超参数     │                              │
│  │ • 退化检测 → 降低学习率     │                              │
│  │ • 对比权重自动调优          │                              │
│  │ • 噪声注入 → 跳出局部最优   │                              │
│  └─────────────────────────────┘                              │
│                                                               │
│  ┌─────────────────────────────────────┐                      │
│  │ 监控系统 (每30分钟)                  │                      │
│  │ recon · fitness · vocab · epoch     │                      │
│  │ → 推送到微信 (ntfy)                 │                      │
│  └─────────────────────────────────────┘                      │
└───────────────────────────────────────────────────────────────┘
```

### 组件关系

| 组件 | 文件 | 功能 |
|------|------|------|
| 守护进程 | `daemon_v3.py` | 主循环：数据加载 → 编码 → 训练 → 进化 → 汇报 |
| C++ 核心 | `cpp_core/` | 高性能推理引擎（零外部依赖） |
| 评估 | `eval_full.py` | MTEB 风格综合评估套件 |
| 监控 | `watchdoc_v2.py` | 状态提取 + ntfy 推送 |
| 词表 | `char_vocab.json` | 500个高频中文字符映射 |

---

## 3. 快速安装

### 环境要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Linux / WSL2 (推荐) / macOS |
| Python | ≥ 3.8 |
| CPU | 支持 OpenMP（多核加速） |
| 内存 | ≥ 16 GB（建议 32 GB） |
| 磁盘 | ≥ 10 GB（含知识库） |

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/chenzhiwenhphp12-afk/neuroflow-model.git
cd neuroflow-model

# 2. 安装 Python 依赖（最小依赖）
pip install numpy requests

# 3. 编译 C++ 核心（可选，纯 Python 也能跑）
cd cpp_core && bash build.sh && cd ..

# 4. 创建词表（首次运行）
python daemon_v3.py --build-vocab
```

> 💡 **WSL 用户注意**：确保在 `/mnt/d/` 等性能较好磁盘上工作。`OMP_NUM_THREADS` 建议设为物理核心数。

---

## 4. 模型组件详解

### 4.1 W_embed — 可学习输入投影

将哈希编码（146维）投影到1024维语义空间：

```
hash(x) [146] → W_embed → ReLU → residual → h [1024]
```

- 输入：字符级别的 one-hot / multi-hot 编码
- 输出：1024维语义向量
- 特点：带残差连接的 ReLU 非线性，比纯哈希编码降低 15% 重建误差

### 4.2 Gated Memory Bank — 门控记忆库

可寻址的联想记忆系统，替换传统的 MLP 隐藏层：

```
查询: q = h · W_q                    # 1024→256
注意力: scores = q · M_K^T / √256   # 对24个记忆槽打分
top-6: 筛选最高分的6个槽
门控: gate = σ(q · W_gate + b_gate) # 控制读入量
读取: read = gate ⊙ (scores · M_V)  # 门控融合
输出: h' = h + read · W_mem_out     # 残差连接
```

| 参数 | 值 | 说明 |
|------|:---:|------|
| 记忆槽数 | 24 | 同时存储24个原型模式 |
| 槽维度 | 256 | 每个原型的特征维度 |
| Top-K | 6 | 每次查询激活最多6个槽 |
| 温度 | 8 | 注意力分布平滑度 |

### 4.3 SAE — 稀疏自编码器瓶颈

强制学习稀疏化表示，防止过拟合：

```
h2 [512] → ReLU → top-50 → gate_mask → SAE_out [512]
```

- 512维隐藏层中仅保留激活值最大的 **50个神经元**
- 其余462个神经元梯度被掩码（gate_mask=0），不参与训练
- 效果：域间分离度提升 174%（0.053 → 0.144）

### 4.4 Vocab Head — 独立词表预测头

预测输入字符的 top-5 候选：

```
h_avg [1024] → W_vocab → 500个字符分数 → softmax
```

- 每5批次训练一次（vocab_train_freq=5）
- 独立于主训练流，避免干扰编码质量

### 4.5 损失函数

```
总损失 = recon_loss × 1.0
       + contrastive_loss × 0.8
       + mem_loss × 0.3
       + weight_decay × 0.002
       + (vocab_loss × 0.0)  # 默认为0，独立训练
```

| 损失项 | 说明 |
|--------|------|
| **Recon MSE** | 重建输入编码的均方误差 |
| **对比损失** | 同一性 pull / 异性 push，域间分离 |
| **Mem 损失** | 记忆槽回归到目标编码 |
| **Weight Decay** | L2 正则化，防止权重膨胀 |

---

## 5. 自主训练（守护进程模式）

### 5.1 启动

```bash
# 前台运行（调试用）
python3 daemon_v3.py

# 后台运行（生产）
nohup python3 daemon_v3.py &> daemon_v3.log &
```

### 5.2 配置参数（daemon_v3.py 顶部）

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `MASK_RATIO` | 0.35 | 输入掩码比例，控制自监督难度 |
| `BATCH_SIZE` | 40000 | 每批样本数 |
| `PARALLEL_WORKERS` | 8 | 并行编码进程数 |
| `LEARNING_INTERVAL` | 0 | 批次间延迟（0=实时） |
| `SAVE_EVERY` | 40000 | 每N条保存一次权重 |
| `CONTRASTIVE_WEIGHT` | 0.8 | 对比损失系数 |
| `VOCAB_SIZE` | 500 | 词表大小 |
| `OMP_NUM_THREADS` | 40 | OpenMP 线程数 |

### 5.3 训练循环

```
每批 (40,000 条):
  1. 加载本地知识库文件 (混洗)
  2. 进程池编码 → 1024维向量 + 字符标签
  3. 前向传播：W_embed → MemoryBank → SAE → W_p
  4. 重建对比混合损失 → 反向传播 → Adam更新
  5. auto_evolve() 检查 (每批后)
  6. 每40K条保存一次 checkpoint
  7. 每30分钟生成状态报告
```

### 5.4 自主进化系统

系统持续监测最后 **50 批** 的训练轨迹：

| 检测机制 | 触发条件 | 动作 |
|----------|----------|------|
| **停滞检测** | 连续15批 recon 无改善 | 增加对比权重 +0.1，注入5%噪声 |
| **退化检测** | recon 突增 >5% | 学习率 ×0.5 |
| **对比权重自动调优** | 方差 <0.0003 | 增加权重至 1.5~2.0 |
| **噪声注入** | 仍在停滞 | 5%~20% 高斯噪声 |
| **长期停滞** | >30批无改善 | 重置对比权重至 0.5 |

自系统启动以来，已触发 **172次** 自动进化。

### 5.5 权重文件

训练权重保存在 `weights_initial.npz`，包含：

| 键 | 形状 | 说明 |
|----|:----:|------|
| W_embed | (1024,1024) | 可学习投影 |
| M_K, M_V | (24,256) | 记忆库键/值 |
| W_q | (1024,256) | 查询投影 |
| W_gate, b_gate | (256,), (256,) | 门控参数 |
| W_mem_out | (256,512) | 记忆读出 |
| W_h1, b_h1 | (1024,512), (512,) | SAE前权重 |
| W_h2, b_h2 | (512,1024), (1024,) | SAE后权重 |
| W_vocab, b_vocab | (1024,500), (500,) | 词表头 |
| V_in, V_out, V_bias | 普通/零初始化 | 替代attention |

---

## 6. 推理与使用

### 6.1 通过 daemon_v3.py 载入模型

```python
import numpy as np
from daemon_v3 import NeuroFlowV3, load_weights

# 加载权重
weights = np.load("weights_initial.npz")
model = NeuroFlowV3()
load_weights(model, weights)

# 推理一个样本
x = np.random.randn(1024).astype(np.float32)  # 模拟编码输入
output = model.forward(x)
reconstructed, vocab_logits = output
```

### 6.2 现有 Demo 脚本

| 脚本 | 功能 |
|------|------|
| `daemon.py` | v2 守护进程（旧版） |
| `daemon_local.py` | 纯本地训练（不联网） |
| `demo_evolution.py` | 展示进化过程 |
| `demo_multimodal.py` | 多模态推理演示 |
| `demo_token.py` | token 化演示 |
| `test_inference.py` | 推理测试 |
| `test_model.py` | 模型基础测试 |
| `test_model_v3.py` | v3 模型专项测试 |

### 6.3 C++ 核心推理

```cpp
#include <neuroflow/model.hpp>
using namespace neuroflow;

NeuroFlow::Config cfg;
cfg.input_dim = 1024;
cfg.hidden_dim = 512;
cfg.output_dim = 1024;

NeuroFlow model(cfg);
Tensor input({1, 1024});
// 填充输入...
auto output = model.forward(input);
// output.decision, output.value, output.saliency
```

---

## 7. 评估与监控

### 7.1 综合评估

```bash
python3 eval_full.py
```

评估项目：

| 测试项 | 说明 | 当前值 |
|--------|------|:------:|
| Reconstruction MSE | 重建精度 | 0.00078 |
| Memory Utilization | 记忆槽利用率 | 75% (18/24) |
| Domain Separation | 域间分离度 | 0.1442 |
| Robustness (Noise) | 抗噪能力 | >90% |
| Speed | 推理速度 | ~9000 items/s |
| Vocab Top-5 | 字符预测 | ~21% |

### 7.2 监控看板（Watchdog）

每30分钟自动推送状态报告到微信，格式：

```
🧠 NeuroFlow 巡检报告
━━━━━━━━━━━━━━━
epoch 150 | 已训练 48.2M 条
recon 0.000782 (↓0.3% vs 上轮) ✅
fitness 0.9992 | var 0.0004
vocab loss 0.0686 | top5 21.1%
记忆槽活跃: 18/24 | 域分离: 0.144
自动进化: 172次 | 错误: 5
CPU: 12.8% | 内存: 3.2/31.0 GB
```

### 7.3 状态文件

`daemon_state.json` 保存运行状态：

```json
{
    "total_trained": 48200000,
    "epoch": 150,
    "batch_count": 1205,
    "last_recon": 0.000782,
    "last_fitness": 0.9992,
    "evolution_count": 172,
    "error_count": 5
}
```

---

## 8. 脚本参考

### 全部可执行脚本一览

| 脚本 | 分类 | 用途 | 运行方式 |
|------|------|------|----------|
| `daemon_v3.py` | **核心** | 自主训练守护进程（v4架构） | `python3 daemon_v3.py` |
| `daemon.py` | 核心 | 旧版守护进程 | `python3 daemon.py` |
| `daemon_local.py` | 核心 | 离线训练（不连网） | `python3 daemon_local.py` |
| `eval_full.py` | 评估 | MTEB综合评估 | `python3 eval_full.py` |
| `diagnose_model.py` | 诊断 | 模型健康检查 | `python3 diagnose_model.py` |
| `test_inference.py` | 测试 | 推理正确性验证 | `python3 test_inference.py` |
| `test_model.py` | 测试 | 基础模型测试 | `python3 test_model.py` |
| `test_model_v3.py` | 测试 | v3模型测试 | `python3 test_model_v3.py` |
| `demo_evolution.py` | Demo | 进化过程可视化 | `python3 demo_evolution.py` |
| `demo_multimodal.py` | Demo | 多模态推理演示 | `python3 demo_multimodal.py` |
| `demo_token.py` | Demo | 分词演示 | `python3 demo_token.py` |
| `build_vocab.py` | 工具 | 构建字词表 | `python3 build_vocab.py` |
| `check_vocab.py` | 工具 | 检查词表质量 | `python3 check_vocab.py` |
| `rebuild_vocab.py` | 工具 | 重建词表 | `python3 rebuild_vocab.py` |
| `learn_batch.py` | 工具 | 单批训练测试 | `python3 learn_batch.py` |
| `turbo_train.py` | 工具 | 快速训练脚本 | `python3 turbo_train.py` |
| `extract_knowledge.py` | 工具 | 提取本地知识 | `python3 extract_knowledge.py` |
| `extract_github_knowledge.py` | 工具 | 提取GitHub知识 | `python3 extract_github_knowledge.py` |
| `extract_new_repos.py` | 工具 | 发现新仓库 | `python3 extract_new_repos.py` |
| `validate_fix_a.py` | 修复 | 模型修复A | `python3 validate_fix_a.py` |
| `validate_fix_v2.py` | 修复 | 模型修复v2 | `python3 validate_fix_v2.py` |
| `train_cartpole_linear.py` | Gym | CartPole线性训练 | `python3 train_cartpole_linear.py` |
| `train_cartpole_rl.py` | Gym | CartPole RL训练 | `python3 train_cartpole_rl.py` |
| `train_gym.py` | Gym | Gym环境训练 | `python3 train_gym.py` |
| `benchmark_gym.py` | Gym | Gym基准测试 | `python3 benchmark_gym.py` |

### 辅助脚本

| 脚本 | 位置 | 用途 |
|------|------|------|
| `scripts/deploy.sh` | 部署 | 一键部署脚本 |
| `scripts/train_cognition.py` | 训练 | 认知训练 |
| `scripts/train_decoder.py` | 训练 | 解码器训练 |
| `scripts/train_education.py` | 训练 | 教育训练 |

---

## 9. 常见问题

### Q: 训练变慢怎么办？
- 检查 OMP_NUM_THREADS 是否设为物理核心数
- 降低 `PARALLEL_WORKERS`（过多进程导致上下文切换）
- 检查磁盘 I/O（知识库太大导致加载瓶颈）

### Q: 重建误差不下降？
- 系统会通过 auto_evolve() 自动调整，等待即可
- 尝试增大 `CONTRASTIVE_WEIGHT`（当前 0.8）
- 降低 `MASK_RATIO`（从 0.35 降到 0.25）

### Q: GPU 可以用吗？
- 当前训练引擎使用 NumPy + OpenMP（纯 CPU）
- 如需 GPU 加速，可以安装 `cupy` 替换 numpy
- C++ 核心支持 CUDA（需自行编译）

### Q: 如何增加新知识？
- 将 .txt 文件放入 `knowledge_base/` 目录
- 守护进程会自动加载新文件
- 也可直接编辑 `BUILTIN_KNOWLEDGE` 列表（daemon_v3.py 第68行）

### Q: 怎么恢复训练？
- 守护进程自动保存 `weights_initial.npz` 和 `daemon_state.json`
- 重启后会从 checkpoint 恢复
- 手动恢复：`python3 daemon_v3.py --restore`

### Q: 在哪里看实时日志？
```bash
tail -f daemon_v3.log
```

---

## 10. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| **v4.2** | 2026-05 | W_embed 可学习投影 + Gated Memory Bank + SAE top-50 + 自主进化 + 看门狗 |
| **v4.1** | 2026-05 | 独立 VocabHead + 课程学习 + 自动超参数调整 |
| **v4.0** | 2026-05 | daemon_v3.py 重写：内置事件循环、混合数据源、TrainableHead SGD |
| **v3.x** | 2026 | Neuro-Symbolic 推理 + Gym benchmark |
| **v2.x** | 2026 | 多模态支持 + INT8 量化 + MLA KV Cache |
| **v1.x** | 2026 | C++ 核心 + SIMD + 类脑三网络 |
| **v0.1** | 2026 | Python 原型 |

---

## 附：文件清单

| 文件 | 大小 | 说明 |
|------|:----:|------|
| `daemon_v3.py` | 62 KB | ⭐ 主守护进程 |
| `eval_full.py` | ~50 KB | 综合评估套件 |
| `watchdoc_v2.py` | ~3 KB | 监控报告脚本 |
| `weights_initial.npz` | 80 KB | 当前训练权重 |
| `vocab.json` | 21 KB | 500字词表(含频率) |
| `char_vocab.json` | 3 KB | 简约字符词表 |
| `trained_decoder.npz` | 80 KB | 已训练解码器 |
| `README.md` | 10 KB | 项目首页 |
| `DEPLOYMENT.md` | 12 KB | 部署手册 |
| `DESIGN.md` | 7 KB | 架构设计 |
| `OPTIMIZATION.md` | 4 KB | 优化说明 |

---

> 💡 **提示**：守护进程已稳定运行超过 150 轮，累计训练近 5000 万条知识，0 次崩溃。
>
> 遇到问题请查看 `daemon_v3.log`，或在 GitHub 提 Issue。

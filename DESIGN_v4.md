# NeuroFlow v4 架构设计文档

> **版本**: 4.0.0
> **架构**: Gated Memory Bank + Sparse Autoencoder
> **参数量**: 3,287,273 (12.54 MB)
> **后端**: 纯 NumPy (CPU)

---

## 1. 核心理念：记忆即学习

### 1.1 从 LLM 范式到 NeuroFlow

传统 LLM 将世界知识编码在数十亿参数中，依赖海量数据和 GPU 进行训练。NeuroFlow v4 采用完全不同的路径：

> **NeuroFlow 不是知识存储器，而是学习机器。它通过记忆读写来理解世界。**

| 对比维度 | LLM 范式 | NeuroFlow 范式 |
|---------|----------|----------------|
| 知识存储 | 参数内隐式编码 (bias) | 显式记忆槽 + 可学习投影 |
| 学习方式 | 海量数据预训练 + Fine-tune | 持续自监督学习 |
| 泛化 | 上下文学习 (In-Context) | 记忆检索 + 门控融合 |
| 硬件需求 | GPU 集群 | 任何 CPU |
| 可解释性 | 黑箱 | 每层可理解 (注意力/门控/SAE) |

### 1.2 生物学映射

```
大脑结构          NeuroFlow 组件
────────────────────────────────
海马体              M_K/M_V (记忆键值对)
前额叶门控          W_gate (自适应门控)
皮层稀疏编码        SAE (Top-K 选择性激活)
工作记忆            32 槽 Gated Memory Bank
记忆巩固            自监督重建 + 对比学习
注意力选择          Top-K 稀疏注意力
```

---

## 2. 架构设计

### 2.1 数据流

```
X [N, 1024]
  │
  ├── W_embed (1024×1024) ── ReLU ────┐
  │    (可学习特征重排)                  │
  │                                    │
  ├──────── 残差连接 (0.1×) ──────────┘
  │         X_in = X + 0.1·ReLU(X·W_embed)
  │
  ├── W_p (1024×512) ── ReLU ── h1 [N, 512]
  │
  ├── Gated Memory Bank
  │    ├── Query:    Q = h1·W_q                 [N, 256]
  │    ├── Keys:     K_norm = M_K / ||M_K||     [32, 256]
  │    ├── Scores:   S = Q·K_norm^T             [N, 32]
  │    ├── Top-6:    attn = softmax(temp·S)     [N, 32]
  │    ├── Read:     mem_read = attn·M_V        [N, 256]
  │    ├── Proj:     mem_feat = mem_read·W_mem_out [N, 512]
  │    └── Gate:     gate = σ(h1·W_gate + b_g)  [N, 512]
  │
  ├── Fusion: h_mem = gate·h1 + (1-gate)·mem_feat
  │
  ├── h3 = ReLU(h_mem)      [N, 512]
  │
  ├── LayerNorm: h3_normed = (h3 - μ) / σ
  │
  ├── SAE: Top-K mask (k = f(entropy))
  │    ├── entropy_norm ∈ [0, 1]
  │    ├── k = 40 + 80·entropy_norm  → [40, 120]
  │    └── h3 *= (abs(h3_normed) ≥ threshold)
  │
  ├── Output Heads
  │    ├── recon = h3·W_d + b_d     [N, 1024] 重建
  │    ├── mem   = h3·W_m + b_m     [N, 256]  记忆预测
  │    ├── value = h3·W_v + b_v     [N, 1]    价值
  │    └── word  = h3·W_gen + b_gen [N, 500]  词汇
  │
  └── Independent Vocab Head
       └── h3 → V_in → ReLU → V_out → sigmoid [N, 500]
```

### 2.2 记忆系统设计

#### Gated Memory Bank

记忆系统是 NeuroFlow v4 的核心创新。32 个可学习记忆槽，每个 256 维：

```
概念: 每个记忆槽 = 一个"原型概念"
键 M_K: 该概念的"触发器" (L2归一化到单位球面)
值 M_V: 该概念存储的"知识内容"

读取过程:
  1. 输入 h1 通过 W_q 生成查询向量 Q
  2. Q 与所有 32 个键计算余弦相似度 (注意: M_K 已被 L2 归一化)
  3. 仅保留 Top-6 最高相似度的槽 (稀疏注意力)
  4. 加权读出: mem_read = attn·M_V
  5. 投影到隐层: mem_feat = mem_read·W_mem_out
  6. 门控融合: h_mem = gate·h1 + (1-gate)·mem_feat
```

#### M_V 能量泵 (Memory Energy Pump)

当 M_V 范数过低时 (< 0.5)，记忆值能量不足，门控完全依赖原始特征 (gate ≈ 1.0)：

```
能量泵梯度:
  ∇_norm = -W_pump · M_V / ||M_V||     (范数提升)
  ∇_div  = -2·W_div · (M_V - μ) / 32   (槽间多样性)
  ∇_total = ∇_norm + ∇_div

自动降级:
  M_V ≥ 0.5  → 停止泵, 转为维护模式
  M_V < 0.3  → 重启泵 (能量危机检测)
```

#### 门控均质化与温控策略

NeuroFlow v4 面临的关键问题：**门控均质化 (Gate Homogenization)** — 所有门控单元输出接近 0.5，无法有效区分原始特征和记忆。

```
三重补丁策略:

层 1: M_V 能量泵 (范数+多样性提升)
      目标: M_V 范数从 0.2 → 0.5+
      手段: 梯度注入 M_V 更新

层 2: 门控温控退火 (Gate Temperature Annealing)
      触发条件: M_V ≥ 0.5 且 gate σ < 0.015
      τ(t) = τ_target + 0.5·(τ_start - τ_target)·(1 + cos(π·t/T))
      τ: 0.2 → 1.0 (余弦退火, 500K topics)
      前向: σ(x/τ)  (锐化)
      反向: σ(1-σ)/τ  (正确的链式法则)

层 3: b_gate 定向扰动 (备用方案)
      向门控偏置注入结构化噪声 σ=0.05
      仅 512 维, 最后手段
```

---

## 3. 为什么选择 NumPy (而非 PyTorch)

| 考量 | 选择 NumPy | 若用 PyTorch |
|------|-----------|-------------|
| 依赖 | 标准库 | 需 2GB+ CUDA 工具链 |
| 安装 | `pip install numpy` | `pip install torch` (1.2GB) |
| 训练 | 自己写 SGD/反向传播 | 自动微分 |
| 速度 | 3M 参数, CPU <1ms | 更快, 但需 GPU |
| 可移植性 | 任何 Python 环境 | 环境依赖 |
| 可读性 | 梯度显式可见 | 自动微分黑箱 |

**决策**: NeuroFlow v4 的 3.29M 参数足够小，纯 NumPy 手动反向传播完全可行。更重要的是，**手写反向传播让每个梯度步骤都清晰可理解** — 这对研究性质的项目至关重要。

---

## 4. 训练策略

### 4.1 多目标损失

```
L_total = λ₁·L_recon + λ₂·L_mem + λ₃·L_value
        + λ₄·L_vocab + λ₅·L_contrastive + λ₆·L_vicreg
        + λ₇·L_pump

默认权重:
  λ₁ = 1.0 (重建 MSE)
  λ₂ = 0.5 (记忆预测 MSE)
  λ₃ = 0.1 (价值 MSE)
  λ₄ = 0.1 (词表 BCE)
  λ₅ = 0.8 (对比损失, 可调)
  λ₆ = 0.5 (VICReg 方差)
  λ₇ = 1.0 (记忆能量泵, 自动)
```

### 4.2 自适应训练调度

```
Phase 0: 恢复期 (Space Var < 1e-6)
  - 跳过 SAE mask (全 512 维通过)
  - 对比度 5.0
  - 直接噪声注入 h3

Phase 1: 流形恢复 (Space Var < 0.01)
  - 高强度对比损失 (weight = 5.0)
  - 词表梯度热身: W_gen 学习率渐启
  - VICReg 方差推散

Phase 2: 词汇学习 (Space Var ≥ 0.01)
  - 词表梯度注入 h3 (weight: 0 → 0.2)
  - 自动进化: 检测停滞, 调参

Phase 3: 门控突破 (M_V ≥ 0.5, gate σ < 0.015)
  - 门控温控退火激活
  - 余弦退火 τ: 0.2 → 1.0
```

### 4.3 自动进化 (Auto-Evolution)

每 15 batch 评估:

```
停滞检测:
  - 最近 15 batch 的 recon 改善 < 1% → stagnation++
  - stagnation ≥ 5 → 自动调参

调参策略:
  - recon 退化 (>5%↑): 学习率 ×0.8
  - 对比损失停滞: contrastive_weight += 0.2
  - var 崩溃 (<1e-8): mask_ratio += 0.05, noise += 0.01
  - word_bce ≈ 0.693 (熵壁): 增加 VICREG_VAR_WEIGHT
  - 深度停滞: 全部推高 (contrastive+mask+noise)

适应度:
  fitness = 1/(1+avg_mse) × collapse_penalty
  collapse_penalty: var, word_bce 检测
```

---

## 5. 评测体系

### 5.1 核心指标

| 指标 | 含义 | 健康范围 | 当前值 |
|------|------|---------|--------|
| Space Var | 隐状态方差 (表征丰富度) | > 0.005 | **0.0143** ✅ |
| Recon MSE | 编码重建误差 | < 0.001 | **0.000586** ✅ |
| Word BCE | 词汇预测交叉熵 | < 0.693 | **0.6971** ✅ |
| Word Var Ratio | word_var / var (词汇相关性) | > 200 | **442** ✅ |
| M_V Norm | 记忆值平均范数 | > 0.5 | **0.2242** ⏳ |
| Gate σ | 门控标准差 | > 0.05 | **0.0118** ⏳ |
| h3 CV | 变异系数 (跨维度) | > 0.05 | **0.037** ⏳ |
| Top-5 Acc | 词汇预测 Top-5 命中率 | > 20% | **23.13%** ✅ |
| KB 利用率 | 记忆槽利用百分比 | > 80% | **~70%** |
| 适应度 | 综合健康度 | > 0.25 | **0.2998** ✅ |

### 5.2 指标联动分析

```
Scene 1: var ↓ + recon ↓ + word ≈ 0.693
  → 表征坍塌！模型学习退化为常量编码

Scene 2: var ↑ + recon ↓ + gate σ ↓
  → 表征恢复中, 门控尚未活跃, 需能量泵

Scene 3: var ↑ + recon ↓ + gate σ ↑ + h3 CV ↑
  → 完美！全指标正向, 门控已分化

Scene 4: var ↑ + recon ↔ + word ↓
  → 流形稳定, 词表学习启动
```

---

## 6. 部署约束

### 6.1 性能预算

| 操作 | 复杂度 | 实测 (N=1) |
|------|--------|-----------|
| W_embed | O(D²) | ~0.02ms |
| W_p | O(D·H) | ~0.01ms |
| Attention (32 slots) | O(S·M) | ~0.005ms |
| SAE Top-K | O(H log H) | ~0.01ms |
| 总推理 | | **< 1ms** |

### 6.2 知识库

- 词汇表: 500 字符 (ASCII + 基本中文+标点)
- 编码方式: Hash + Sinusoid
- 知识来源: 200+ 内置跨领域知识点
- 在线学习: 可选 HN/NPR/Wikipedia (GFW 受限时自动降级)
- 记忆容量: 140K+ 知识库文件 (约 250MB)

---

## 7. 版本历史

| 版本 | 参数量 | 关键变化 |
|------|--------|---------|
| v1.0 | 43K | C++ 核心 (SN/ECN/DMN) |
| v2.0 | 232K | 多模态 + SIMD + MLA |
| v3.0 | 1.25M | Python 原型 + TrainableHead |
| **v4.0** | **3.29M** | **GatedMemBank + SAE + 纯NumPy** |

---

## 参考文档

- [daemon_v3.py](../daemon_v3.py) — 自主进化守护进程 (完全版)
- [gate_patch_numpy.py](../gate_patch_numpy.py) — 门控温控补丁
- [eval_neuroflow_v2.py](../eval_neuroflow_v2.py) — 完整评测脚本
- [HuggingFace Weights](https://huggingface.co/chenzhiwenhphp12/neuroflow-v4)

---

<p align="center">
  <sub>Design v4.0 · 2026 · Chen Zhiwen</sub>
</p>

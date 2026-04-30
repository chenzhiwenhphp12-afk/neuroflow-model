# NeuroFlow 1.5B 训练计划

## 目标概述

训练一个 **1.5B参数** 的类脑神经网络模型，专精于：
- **中文理解能力** (60%)
- **编程能力优化** (40%)

---

## 1. 模型配置

### 1.1 基础架构

| 参数 | 数值 |
|-----|------|
| 总参数量 | 1.53B |
| Vocab Size | 60,000 (中文50K + 代码15K) |
| Hidden Dim | 2048 |
| Intermediate Dim | 8192 |
| Attention Heads | 32 |
| Transformer Layers | 24 |

### 1.2 类脑网络配置

| 组件 | 配置 |
|-----|------|
| ECN (执行控制网络) | 10层 |
| DMN (默认模式网络) | 16个联想头 |
| SN (显著性网络) | 1024隐藏维度 |
| Memory Slots | 512 |
| Memory Dim | 768 |

### 1.3 MLA记忆压缩

| 参数 | 数值 |
|-----|------|
| MLA Latent Dim | 192 |
| Max Cache Length | 8192 |
| Memory Saving | 87.5% |

### 1.4 多模态配置

| 参数 | 数值 |
|-----|------|
| Vision Hidden Dim | 1024 |
| Vision Layers | 10 |
| Fusion Dim | 2048 |

---

## 2. 训练数据集

### 2.1 中文数据 (60%)

| 数据集 | 权重 | 说明 |
|--------|------|------|
| Skywork/Skywork-CN-corpus | 30% | 大规模中文语料 |
| CLUE Corpus | 20% | 中文语言理解基准 |
| 中文技术文档 | 15% | API文档、技术博客 |
| 中文问答数据 | 10% | 问答理解训练 |
| 中文算法题解 | 5% | 编程相关中文内容 |

### 2.2 编程数据 (40%)

| 数据集 | 权重 | 说明 |
|--------|------|------|
| bigcode/the-stack | 30% | 多语言代码数据 |
| Python专项数据 | 15% | Python代码训练 |
| HumanEval/MBPP | 10% | 代码生成基准 |
| CodeAlpaca | 10% | 代码指令微调 |

### 2.3 中文编程混合数据

| 数据集 | 说明 |
|--------|------|
| LeetCode-CN | 力扣中文题解 |
| CSDN编程博客 | 中文技术文章 |
| 知乎技术问答 | 中文编程问答 |
| GitHub中文项目 | 中文代码项目 |

---

## 3. 训练参数

### 3.1 LoRA微调配置

| 参数 | 数值 |
|-----|------|
| LoRA r | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target Modules | 7个投影层 |

### 3.2 训练超参数

| 参数 | 数值 |
|-----|------|
| Epochs | 3 |
| Batch Size | 16 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 64 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 10% |
| LR Scheduler | Cosine |
| Optimizer | AdamW 8bit |

### 3.3 可训练参数

- 可训练参数: 约 100M (LoRA)
- 占总参数比例: ~6.5%
- 大幅降低训练成本

---

## 4. 硬件需求

### 4.1 GPU推荐配置

| 级别 | GPU | 内存 | 预估时间 | 预估费用 |
|------|-----|------|---------|----------|
| 最小 | RTX 3090/4090 | 24GB | 72小时 | $200-300 |
| 推荐 | A100 40GB | 40GB | 24-36小时 | $300-500 |
| 最优 | H100 80GB | 80GB | 12-18小时 | $400-600 |

---

## 5. 云平台选择

### 5.1 Modal (推荐 - 最简单)

```bash
# 安装
pip install modal

# 配置Token
modal token new

# 部署训练
modal deploy train_modal.py
```

优点:
- 无服务器，无需管理基础设施
- 按秒计费，自动管理GPU
- 支持A100/H100等高端GPU

### 5.2 AWS EC2 (大规模训练)

```bash
# 创建实例 (p4d.24xlarge - 8x A100)
aws ec2 run-instances --instance-type p4d.24xlarge

# 安装依赖
pip install torch transformers datasets peft trl unsloth

# 运行训练
python train_1.5b_unsloth.py
```

预估费用: $32.77/小时 × 24-36小时 = $800-1200

### 5.3 阿里云PAI (国内优化)

优点:
- 国内节点，中文数据加载快
- 预装深度学习环境
- 支持中文文档

预估费用: ¥20/小时 × 48-72小时 = ¥1000-1500

---

## 6. 评估基准

### 6.1 中文评估

| 基准 | 说明 |
|------|------|
| CMMLU | 中文多任务理解 |
| C-Eval | 中文综合评估 |
| CLUE | 中文语言理解基准 |

### 6.2 编程评估

| 基准 | 说明 |
|------|------|
| HumanEval | Python代码生成 |
| MBPP | Python编程基准 |
| CodeContests | 代码竞赛评估 |

---

## 7. 执行步骤

### Step 1: 准备环境

```bash
# 安装依赖
pip install unsloth transformers datasets peft trl accelerate

# 配置HF Token
export HF_TOKEN=your_token
```

### Step 2: 选择平台

- 快速开始: Modal
- 大规模: AWS
- 国内: 阿里云

### Step 3: 启动训练

```bash
# 本地训练 (需要GPU)
python train_1.5b_unsloth.py --epochs 3 --batch_size 16

# Modal训练
modal deploy train_modal.py

# Docker训练
docker run --gpus all neuroflow-training:latest
```

### Step 4: 评估模型

```bash
# 中文评估
python eval_chinese.py --model ./checkpoints/neuroflow-1.5b

# 编程评估
python eval_code.py --model ./checkpoints/neuroflow-1.5b
```

### Step 5: 导出部署

```bash
# 导出GGUF (本地推理)
python export_gguf.py --model ./checkpoints/neuroflow-1.5b

# 导出C++ (NeuroFlow核心)
python export_cpp.py --model ./checkpoints/neuroflow-1.5b
```

---

## 8. 生成的文件

| 文件 | 说明 |
|------|------|
| configs/model_1.5b_config.py | 模型配置 |
| configs/dataset_config.py | 数据集配置 |
| configs/cloud_gpu_config.py | 云GPU配置 |
| train_1.5b_unsloth.py | Unsloth训练脚本 |
| train_modal.py | Modal部署脚本 |

---

## 9. 预期成果

训练完成后，模型将具备:

1. **中文理解能力**
   - 语义理解
   - 情感分析
   - 文本生成
   - 问答系统
   - 阅读理解

2. **编程能力**
   - 代码生成 (Python, JS, Java, C++等)
   - 代码补全
   - 代码解释
   - 错误修复
   - 单元测试生成

3. **性能指标**
   - 参数量: 1.53B
   - MLA压缩: 87.5%内存节省
   - 推理速度: 预计50-100ms/token

---

**计划生成**: Hermes Agent
**日期**: 2026-04-30
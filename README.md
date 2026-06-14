# NeuroFlow — 自研认知架构语言模型

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

NeuroFlow 是一个基于多网络认知架构（SN/ECN/DMN）的语言模型，融合了神经科学启发的注意力机制与经典深度学习技术。

## 架构概览

```
输入 → Input Proj → ┌── SN (显著性网络) ──── 门控 ──┐
                     ├── ECN (执行控制网络) ── 12层 ─┤→ Fusion → Output
                     ├── DMN (默认模式网络) ── 联想 ─┤
                     └── Memory (记忆巩固) ──────────┘
```

| 组件 | 说明 |
|:----|:----|
| SN (Salience Network) | 检测输入显著性，输出 ECN/DMN 门控权重 |
| ECN (Executive Control) | 12层 DLPFC 网络 + OFC/vmPFC 决策与价值评估 |
| DMN (Default Mode Network) | 记忆编码、联想检索、未来投射 |
| Memory | NTM式记忆槽 + 注意力检索 + LTP巩固 |
| Output Fusion | 三流加权融合 → 瓶颈降维 → 输出 |

## 版本

### v2 (当前 — 128K BPE 分支)

- **参数**: 371M (bottleneck输出融合)
- **词表**: 128K BPE (44,499 merges，从语料训练)
- **架构**: d_model=512, hidden_dim=2048, 12层
- **训练**: 混合课程训练 + 经验回放 (replay buffer)
- **优化**: OpenBLAS GEMM + OpenMP + NUMA绑定

### 训练语料

| 数据 | 学科 | 规模 |
|:----|:---:|:----:|
| 小学 | 14科 | 1GB |
| 初中 | 19科 | 17GB |
| 高中 | 20科 | 9.5GB |
| 大学 | 14学科 | ~500K样本 |

## 快速开始

```bash
# 编译
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNEUROFLOW_USE_BLAS=ON
make -j$(nproc)

# 训练
./neuroflow_train_v2 \
  --config configs/config.json \
  --tokenizer configs/tokenizer_128k.json \
  --data /path/to/corpus \
  --output ./output \
  --lr 3e-5 --batch-size 64 \
  --replay-buffer 10000 --replay-ratio 0.25 \
  --epochs 20

# 推理 (测试)
g++ -std=c++17 -O2 -I include -I src \
  scripts/test_generate.cpp src/tensor.cpp src/model.cpp src/weight_io.cpp \
  -o test_generate -fopenmp -lopenblas
./test_generate configs/config.json output/model_final.nfv1
```

## 社区

- **GitHub Issues**: 报告Bug/提议功能
- **Discussions**: 技术讨论与协作
- **欢迎贡献**: PR 前请先开 Issue 讨论

## 致谢

- Hermes Agent (Nous Research) — Agent框架
- OpenBLAS — 高性能GEMM

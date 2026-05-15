# NeuroFlow 架构设计文档

> **核心理念**：NeuroFlow 不是 LLM 的竞争者，而是 LLM 的驾驶者。
> 它是低功耗、可进化的决策中枢，而非海量知识的存储器。

---

## 1. 系统边界：感知代理分离原则

### 问题

CLIP (88M)、Whisper (39M+) 等感知模型一旦加载进 NeuroFlow 进程，核心 10MB 内存优势将被彻底淹没。

### 原则

```
┌─────────────────────────────────────────────────────┐
│                 感知代理层 (Sensory Proxies)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  CLIP    │  │ Whisper  │  │ 其他感知  │          │
│  │ (视觉)   │  │ (语音)   │  │ 模块     │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │ 512-dim     │ 512-dim     │ 低维特征        │
│       └──────────────┼─────────────┘                │
│                      │                              │
│              ┌───────▼────────┐                      │
│              │  特征降维/缓冲  │  ← 可选，降低传输带宽 │
│              └───────┬────────┘                      │
├──────────────────────┼──────────────────────────────┤
│                      │ 低维 embedding               │
│  ┌───────────────────▼────────────────────────────┐ │
│  │          NeuroFlow Core (决策中枢)              │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌───────────┐  │ │
│  │  │ SN   │  │ ECN  │  │ DMN  │  │ SelfEvolve│  │ │
│  │  └──────┘  └──────┘  └──────┘  └───────────┘  │ │
│  │  43K 参数 · 0.2MB · 0.40ms · 纯C++17         │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  部署目标：边缘设备 / MCU / 机器人控制器 / 量化终端    │
└─────────────────────────────────────────────────────┘
```

**硬约束**：
- 感知模型（CLIP/Whisper/YOLO/…）运行在**独立进程或远程服务**中
- NeuroFlow 只接收降维后的 **embedding 向量**（512-dim float32）
- NeuroFlow 的依赖列表永远不包含 `torch`、`transformers`、`whisper`、`clip`
- 核心内存上限：2MB (C++) / 10MB (Python 原型)

---

## 2. 推理架构：Neuro-Symbolic 双系统

### 理论基础

丹尼尔·卡尼曼《思考，快与慢》双系统理论：

| 系统 | 功能 | NeuroFlow 映射 |
|------|------|----------------|
| **系统 1** | 直觉、快速、并行 | SN + ECN 的 Value/Saliency 评估（神经计算） |
| **系统 2** | 逻辑、慢速、串行 | ReasoningLoop 的规则引擎（符号计算） |

### 当前问题

`ReasoningLoop` 本质是带噪声的随机投影迭代 → 固定点收敛，不是真正的推理。
置信度与随机权重无关，证明推理结果由数值收敛驱动而非认知过程。

### 目标架构

```python
class NeuroSymbolicReasoner:
    """
    系统 2 = 符号规则引擎 + 系统 1 的直觉评估
    
    推理循环:
    1. 规则引擎根据当前状态选择推理路径
    2. 神经 Value 网络评估候选决策的"好坏"
    3. 神经 Saliency 网络决定注意力分配
    4. 符号规则控制收敛条件和安全边界
    """
    
    def reason(self, x, context=None):
        # 系统 1: 直觉评估（神经部分）
        output = self.model.forward(x)
        intuition_value = output.value
        intuition_saliency = output.saliency
        
        # 系统 2: 逻辑推理（符号部分）
        state = self.rule_engine.init(x, context)
        for step in range(max_steps):
            candidates = self.rule_engine.expand(state)
            # 系统 1 为系统 2 提供启发式引导
            best = self.select_best(candidates, intuition_value)
            state = self.rule_engine.apply(state, best)
            if self.rule_engine.is_terminal(state):
                break
        
        return state.decision, state.confidence
```

---

## 3. 学习策略：固定身体，训练皮层

### 原则

- **冻结**：SN/DMN/CrossModal Fusion 的随机权重保持不变（作为非线性特征映射）
- **训练**：仅对最终的 Hidden[256] → decision[10] + value[10] 线性投影层做在线 SGD
- **成本**：每步梯度计算 < 1KB，完全可以在边缘设备上做实时 RL

### 数学

```
h = f_frozen(x)           # 冻结的特征提取器
d = W_d h + b_d            # 可训练决策层
v = W_v h + b_v            # 可训练价值层

loss = (d - d_target)^2 + (v - r)^2
W_d -= lr * ∂loss/∂W_d    # 仅更新这两层
W_v -= lr * ∂loss/∂W_v
```

### 理由

- 保留随机投影的数学优势（高维空间中的压缩映射）
- 决策层极轻量（~2600 个可训练参数 vs 1.6M 总量）
- 杜绝灾难性遗忘（冻结部分不会改变）
- 训练速度：<1μs/step on CPU

---

## 4. 应用场景矩阵

| 场景 | 适合度 | 理由 |
|------|:---:|------|
| 量化交易 (vnpy) | ⭐⭐⭐⭐⭐ | 规则明确、高频决策、低延迟、可进化 |
| 机器人控制 | ⭐⭐⭐⭐⭐ | 低维输入、实时性要求高、在线适应 |
| IoT 传感器融合 | ⭐⭐⭐⭐ | 极低功耗、边缘推理 |
| 游戏 AI | ⭐⭐⭐ | 可用但需自定义环境 |
| NLP / 聊天 | ⭐ | 知识容量不足，不如用小 LLM |
| 知识问答 | ⭐ | 1.6M 参数无法存储世界知识 |

---

## 5. 设计公理

1. **不要试图记住知识** — 那是 LLM 的事。NeuroFlow 只需要学会**如何决策**。
2. **不要接入大模型权重** — 感知模型是外部 API，不是系统组件。
3. **不要假装推理** — 如果规则引擎驱动了逻辑，坦然承认这是 Neuro-Symbolic 架构。
4. **保持内核纯粹** — 核心 C++ 代码库永远不引入 torch/transformers/opencv 等重型依赖。
5. **用标准基准说话** — Gym/MuJoCo/vnpy 回测的 Reward 曲线比自定指标更有说服力。

---

## 版本

| 版本 | 日期 | 作者 |
|------|------|------|
| v1.0 | 2026-05-15 | NeuroFlow Team |

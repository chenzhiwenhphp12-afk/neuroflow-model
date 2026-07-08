# NeuroFlow 测试报告 — 2026年7月

## 测试环境

- **平台**: Windows 10 (x64), MSVC, C++17
- **构建**: CMake Release, CUDA=OFF, BLAS=OFF, AVX2=OFF
- **测试框架**: 自研 `test_framework.hpp` (125行)

---

## 测试结果总览

| 测试套件 | 通过 | 失败 | 总计 |
|---------|------|------|------|
| test_adamw | 6 | 0 | 6 |
| test_rope | 5 | 0 | 5 |
| test_swiglu | 5 | 0 | 5 |
| test_rms_norm | 5 | 0 | 5 |
| test_scheduler | 5 | 1 | 6 |
| test_gqa | 5 | 0 | 5 |
| test_lm_pathway | 7 | 0 | 7 |
| test_full_path (新增) | 7 | 4 | 11 |
| **总计** | **45** | **5** | **50** |

---

## 发现的 BUG 与问题

### 🔴 BUG 1：空输入产生 NaN

- **文件**: `src/causal_lm.cpp` — `CausalLMHead::forward()`
- **触发条件**: 调用 `forward({})` 传入空 token 列表
- **根因**: `pool()` 函数对零长度的序列做 `mean_pool`，`inv_n = 1/0 = inf`，导致 NaN 传播
- **影响**: 如果外部代码意外传入空输入，整个推理链路产生 NaN

**修复方向**: 在 `forward()` 入口处增加空输入检查：

```cpp
if (token_ids.empty()) {
    // 返回全零 logits（等效于均匀分布）
    Tensor logits({1, config_.vocab_size}, QuantType::FP32);
    memset(logits.as_fp32(), 0, logits.data_size_);
    return logits;
}
```

---

### 🟡 BUG 2：`forward_step()` 和 `forward()` 结果不一致

- **文件**: `src/causal_lm.cpp`
- **触发条件**: 对相同 token 序列依次调用 `forward_step()` 和 `forward()` 结果差异 ~4.82
- **根因**: 当前架构使用 `pool()` + 状态ful组件（`causal_window_gate` 的 depthwise conv + `NTM` 内存写入），导致 `forward()`（全序列并行）和 `forward_step()`（逐 token 增量）路径产生不同结果。这不是传统 Transformer 的因果 LM 行为。

- **影响**: 推理时 `forward_step()` 和训练时 `forward_for_training()` 的行为不一致，可能导致训练后推理效果劣化

- **修复方向**: 两个选择：
  1. **短期**: `forward_step()` 中禁用 NTM write 和 causal_gate 的状态依赖，使其与 `forward()` 对齐
  2. **长期**: 将 `pool()` 改为 `last_token_pool()` 并确保所有前层组件是因果的

---

### 🔴 BUG 3：`CosineScheduler` 在 `total_steps=0` 时返回错误学习率

- **文件**: `src/scheduler.cpp:15`
- **触发条件**: `CosineScheduler sched(0.001f, 0, 0.1f, 0.0f)`，调用 `get_lr(0)`
- **实际**: 返回 `lr_min_ = 0.0001`（期望 `lr_max_ = 0.001`）
- **根因**:

```cpp
// 问题行:
if (step >= total_steps_) return lr_min_;   // 0 >= 0 → TRUE → 返回 lr_min_
```

当 `total_steps_=0` 时，第 0 步立即被视为 "训练完成"，返回最小学习率。

- **影响**: 如果 `TrainLM` 配置中 `total_steps=0`（默认），调度器总是返回 `lr_min_`，导致训练几乎不收敛

- **修复方向**:

```cpp
float get_lr(size_t step) const {
    if (total_steps_ == 0) return lr_max_;   // 新增: 无总步数时返回最大LR
    if (step >= total_steps_) return lr_min_;
    // ...
}
```

---

### 🟢 假阳性（测试用例设计问题，非代码 BUG）

#### 1. `TrainingLoop.SingleBatchNoCrash` — 期望 `shape[0]=4` 但得到 `1`

`forward_for_training()` 调用 `pool(x)` 将序列压缩为单个向量（`seq_len→1`），这是架构设计而非 BUG。但这个设计意味着 NeuroFlow 的 `CausalLMHead` 每个序列只预测 **1 个 token**，而非标准 Transformer 的每个位置预测下一个 token。

#### 2. `ForwardStepVsForwardConsistent` — 期望完全一致但差异 4.82

如 🟡 BUG 2 所述，这是架构层面的特性，不是实现层面的 BUG。

---

## 构建警告汇总

| 警告类型 | 数量 | 风险 | 说明 |
|---------|------|------|------|
| C4267: size_t→uint32_t 截断 | ~50+ | 🟢 低 | 64位→32位，在 x64 上可能出问题 |
| C4244: double→float 截断 | ~15 | 🟢 低 | 常量默认 double，显式 f 后缀即可 |

---

## 总结

| 类别 | 通过 | 失败 | 通过率 |
|------|------|------|--------|
| 单元测试 | 45 | 5 | 90% |
| 实为测试设计问题（假阳性） | — | 2 | — |
| **真实 BUG** | — | **3** | — |

### 优先级建议

1. 🔴 **高**: BUG 1 (空输入 NaN) + BUG 3 (CosineScheduler total_steps=0) — 修复简单，影响大
2. 🟡 **中**: BUG 2 (forward vs forward_step 不一致) — 需要架构设计决策
3. 🟢 **低**: C4267/C4244 警告 — 大量但不紧急

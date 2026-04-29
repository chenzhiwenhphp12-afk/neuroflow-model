# NeuroFlow C++ Core - Memory Crash Fix Report

## 修复日期
2026-04-29

## 问题描述
`neuroflow_model_test` 在 forward pass 时崩溃：`malloc(): invalid size (unsorted)`

## 修复状态
**已成功修复并推送到 GitHub**

Commit: `f91263b` - https://github.com/chenzhiwenhphp12-afk/neuroflow-model

## 已修复问题

### 1. tensor.hpp - 添加 as_fp32_const() 方法
**位置**: 第113行
**问题**: `online_learning.hpp` 调用了 `as_fp32_const()` 但该方法不存在
**修复**: 添加了兼容性别名方法

```cpp
// 别名，兼容性
const float* as_fp32_const() const {
    return as_fp32();
}
```

### 2. model.hpp - forward() 函数内存问题
**位置**: 第189-216行
**问题**: `reshape()` 创建的张量 `owns_data=false`，后续加权操作直接修改共享数据导致内存损坏
**修复**: 
- 创建新的独立张量 `ecn_weighted` 和 `dmn_weighted_full`
- 将数据复制到新张量后再进行加权操作
- 避免 reshape 后的张量被修改

**修改前**:
```cpp
Tensor ecn_weighted = out.decision.clone();
Tensor dmn_weighted = dmn_out.vision.reshape({...});
ew[i * config.output_dim + j] *= eg[i];  // 修改共享数据
```

**修改后**:
```cpp
Tensor ecn_weighted({batch, config.output_dim}, QuantType::FP32);
Tensor dmn_weighted_full({batch, dmn_out.vision.shape[1]}, QuantType::FP32);
ew[i * config.output_dim + j] = ed[i * config.output_dim + j] * eg[i];  // 新数据
```

### 3. online_learning.hpp - const指针类型修复
**位置**: 多处
**问题**: const Tensor& 参数调用 `as_fp32()` 返回 `const float*`，但用 `float*` 接收
**修复**: 将所有 `float*` 改为 `const float*`

修复位置：
- cross_entropy(): 第33-34行
- mse(): 第69-70行  
- linear_backward(): 第138-140行
- layernorm_backward(): 第182-183行
- gelu_backward(): 第226-227行
- sgd_step(): 第257行
- adam_step(): 第288行

## 测试结果

```
Test project /home/admin/neuroflow-model/cpp_core/build
    Start 1: tensor_test
1/4 Test #1: tensor_test ......................   Passed    0.08 sec
    Start 2: model_test
2/4 Test #2: model_test .......................   Passed    0.63 sec
    Start 3: multimodal_test
3/4 Test #3: multimodal_test ..................   Passed    0.44 sec
    Start 4: online_test
4/4 Test #4: online_test ......................   Passed    0.08 sec

100% tests passed, 0 tests failed out of 4
Total Test time (real) =   1.23 sec
```

### 性能数据 (model_test)

| 模型 | 参数量 | 内存 | 推理时间 |
|------|--------|------|----------|
| Original | 1,244,133 | 4.75 MB | 50.08 ms |
| Lite (量化) | 265,253 | - | 0.34 ms |
| **加速** | 78.7%减少 | - | **147.7x** |

| 特性 | 数据 |
|------|------|
| MLA内存节省 | 87.5% |
| 量化比例 | 100% |

## 技术总结

### 根本原因
Tensor 的 `reshape()` 方法返回 `owns_data=false` 的张量，共享原始数据指针。
当后续代码对这些 reshape 后的张量进行 `*= ` 或 `+=` 等修改操作时，可能破坏内存分配器的内部状态，触发 `malloc(): invalid size (unsorted)` 错误。

### 最佳实践
1. `reshape()` 后的张量应只用于读取，不应修改
2. 需要修改数据时，应创建新的独立张量
3. 使用 `clone()` 创建独立数据副本后再修改
4. 注意 reshape 返回张量的生命周期

### 代码审查建议
- 检查所有 `.reshape()` 后的赋值/修改操作
- const Tensor& 参数应使用 const 指针接收
- 考虑添加 reshape 后张量的"只读"标记

## 下一步计划

1. Python绑定构建 (需要pybind11)
2. 完整的内存泄漏检测
3. 添加边界条件测试
4. 实现完整反向传播链
5. 性能基准测试和优化

## 相关文件

- 修复报告: `/home/admin/neuroflow-model/cpp_core/MEMORY_FIX_REPORT.md`
- 进度更新: `/home/admin/neuroflow-model/EMAIL_UPDATE.txt`
- GitHub: https://github.com/chenzhiwenhphp12-afk/neuroflow-model
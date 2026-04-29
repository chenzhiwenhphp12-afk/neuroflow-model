# NeuroFlow C++ Core - Memory Crash Fix Report

## 修复日期
2026-04-29

## 问题描述
`neuroflow_model_test` 在 forward pass 时崩溃：`malloc(): invalid size (unsorted)`

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
// 直接修改共享数据
ew[i * config.output_dim + j] *= eg[i];
```

**修改后**:
```cpp
Tensor ecn_weighted({batch, config.output_dim}, QuantType::FP32);
Tensor dmn_weighted_full({batch, dmn_out.vision.shape[1]}, QuantType::FP32);
// 复制并加权
ew[i * config.output_dim + j] = ed[i * config.output_dim + j] * eg[i];
```

### 3. online_learning.hpp - 方法调用修复
**位置**: 第353行
**问题**: 调用了不存在的 `as_fp32_const()` 方法
**修复**: 改为使用 `as_fp32()` (const 版本)

## 待验证

执行以下命令验证修复：

```bash
cd /home/admin/neuroflow-model/cpp_core
./build.sh build
./build.sh test
```

或手动编译：

```bash
cd /home/admin/neuroflow-model/cpp_core/build
cmake ..
make -j$(nproc)
./neuroflow_model_test
./neuroflow_tensor_test
```

## 技术总结

### 根本原因
Tensor 的 `reshape()` 方法返回 `owns_data=false` 的张量，共享原始数据。当后续代码对这些 reshape 后的张量进行修改操作时，可能导致内存分配器状态不一致，触发 `malloc(): invalid size` 错误。

### 最佳实践
1. `reshape()` 后的张量应只用于读取，不应修改
2. 需要修改数据时，应创建新的独立张量
3. 使用 `clone()` 创建独立数据副本后再修改
4. 注意 `reshape()` 返回的张量的生命周期

## 下一步

1. 编译并运行测试验证修复
2. 如果仍有问题，检查 `memory.hpp` 中的 `LatentKVCache::forward()`
3. 完成后推送到 GitHub
4. 构建 Python 绑定
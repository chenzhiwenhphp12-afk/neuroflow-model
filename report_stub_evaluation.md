# NeuroFlow 空壳源文件评估报告

## 空壳文件分析

- `src\model.cpp`: 13行, 有效7行, ❌ 非空壳
- `src\neuroflow_gpu.cpp`: 234行, 有效169行, ❌ 非空壳
- `src\tensor.cpp`: 12行, 有效6行, ❌ 非空壳
- `src\train_v2.cpp`: 211行, 有效164行, ❌ 非空壳
- `src\weight_io.cpp`: 146行, 有效118行, ❌ 非空壳

## Header-Only vs 编译分离

### Header-Only 优点

- 无需编译.cpp文件，包含即可用
- 模板代码可内联优化
- 分发简单(单头文件)
- 适合小型模板库
- 避免链接顺序问题

### Header-Only 缺点

- 编译时间随包含次数线性增长
- 二进制体积膨胀(重复实例化)
- 循环依赖风险高
- 调试困难(模板展开复杂)
- IDE代码补全和跳转受限
- 修改头文件触发全量重编译

### 编译分离 优点

- 编译时间可控(修改cpp仅重编译单文件)
- 二进制体积小(单次实例化)
- 可隐藏实现细节(Pimpl模式)
- 循环依赖易解(前向声明)
- 调试友好(独立编译单元)
- 增量编译高效

### 编译分离 缺点

- 需维护hpp/cpp文件对
- 模板代码仍需在头文件
- 构建系统更复杂
- 分发需同时提供头文件和库文件
- 链接顺序可能出错

## 混合模式决策: **MIXED**

- `backprop.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离
- `generative.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离
- `memory.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离
- `model.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离
- `multimodal.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离
- `multimodal_model.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离
- `networks.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离
- `online_learning.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离
- `tensor.hpp`: **COMPILED_SEP** — 模板占比低，迁移到编译分离

## CMakeLists.txt 修改方案

```cmake
# === NeuroFlow 文件完整性补丁 ===
# 在neuroflow_core源文件列表中添加:
#   src/weight_io.cpp
# 新增训练可执行目标:
#   add_executable(neuroflow_train_v2 src/train_v2.cpp)
#   target_link_libraries(neuroflow_train_v2 PRIVATE neuroflow_core OpenMP::OpenMP_CXX)
```
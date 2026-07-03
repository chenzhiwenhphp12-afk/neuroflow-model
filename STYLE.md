# NeuroFlow C++ 代码风格规范

> 基于 Google C++ Style Guide (2025 版，目标 C++20)
> 配置: `.clang-format` / `.clang-tidy`
> 最后更新: 2026-06-14

---

## 🎯 一句话原则

**风格服务于正确性和可维护性。** 不追求完美统一 — 现有代码可以渐进迁移，但新代码必须遵守本规范。

---

## 一、文件命名

| 规则 | ✅ 正确 | ❌ 错误 |
|------|---------|---------|
| 头文件: `snake_case.h` | `backprop.h` | `Backprop.hpp`, `back-prop.h` |
| 源文件: `snake_case.cpp` | `train_v2.cpp` | `trainV2.cpp`, `Train.cpp` |
| 测试文件: `test_*` 或 `*_test` | `test_model.cpp` | `ModelTest.cpp` |

> ⚠️ **迁移说明**: 当前使用 `.hpp`，考虑逐步改为 `.h`。新文件请用 `.h`。

---

## 二、命名规则速查

```
类型:                     PascalCase    NeuroFlowModel, LayerConfig
函数/方法:                PascalCase    ForwardPass(), ComputeLoss()
变量:                     snake_case    hidden_size, batch_size
类成员变量:               snake_case_    weight_, bias_, grad_
结构体成员:               snake_case    num_layers, vocab_size
常量 (static constexpr):  kPascalCase   kMaxSeqLen, kDefaultEpsilon
枚举值:                   kPascalCase   kFp32, kInt8
命名空间:                 snake_case    neuroflow::training
宏:                       UPPER_SNAKE   NEUROFLOW_CHECK(cond)
模板参数:                 PascalCase    typename LayerType
```

### 命名注意

- **避免缩写** — `learning_rate` 不写 `lr`，除非是公认缩写 (`num_`, `idx`)
- **函数名动词开头** — `ComputeLoss()` 不写 `LossComputer()`
- **布尔变量用 is/has 前缀** — `is_training`, `has_bias`

---

## 三、头文件纪律

### 3.1 自包含
每个 `.h` 必须能独立编译。测试方法:
```bash
g++ -std=c++20 -fsyntax-only -Iinclude include/neuroflow/my_header.h
```

### 3.2 头文件保护
```cpp
#ifndef NEUROFLOW_SRC_MODEL_MODEL_H_
#define NEUROFLOW_SRC_MODEL_MODEL_H_
// ...
#endif  // NEUROFLOW_SRC_MODEL_MODEL_H_
```

> 也接受 `#pragma once`（实际使用中更简洁），但格式统一: 一个项目只选一种。

### 3.3 Include 顺序
```cpp
// 1. 关联头文件（同名 .h）
#include "my_class.h"

// 2. C 系统头文件
#include <fcntl.h>
#include <unistd.h>

// 3. C++ 标准库
#include <algorithm>
#include <memory>
#include <vector>

// 4. 第三方库
#include <cblas.h>

// 5. 项目头文件
#include "neuroflow/tensor.hpp"
#include "neuroflow/networks.hpp"
```

> `.clang-format` 的 `SortIncludes: true` 会自动排序。

### 3.4 禁止事项
- ❌ `.h` 内用 `using namespace xxx`
- ❌ `.h` 内定义匿名命名空间或 `static` 函数
- ❌ 依赖辗转包含（每个文件显式 include 自己需要的）
- ❌ `-inl.h` 文件
- ❌ inline 命名空间

---

## 四、类

### 4.1 struct vs class
- `struct` — 纯数据聚合，无不变量
- `class` — 有封装、有不变量

### 4.2 构造/析构
```cpp
class Model {
public:
    explicit Model(const Config& cfg);        // 单参构造 = explicit
    Model(const Model&) = delete;             // 显式声明
    Model(Model&&) = default;                 // 或 delete
    virtual ~Model() = default;
    
    // 构造函数内禁止调用虚函数
};
```

### 4.3 成员变量声明顺序
```cpp
class MyClass {
public:
    // 类型定义、常量
    static constexpr int kMaxLayers = 24;
    using Config = MyConfig;
    
    // 公共方法
    void Forward();
    
private:
    // 私有方法
    void InternalHelper();
    
    // 成员变量（尾部下划线）
    int num_layers_;
    float learning_rate_;
};
```

---

## 五、智能指针

| 场景 | 类型 |
|------|------|
| 独占所有权 | `std::unique_ptr<T>` |
| 共享所有权 | `std::shared_ptr<T>` |
| 非拥有观察 | `T*` 或 `const T&` |
| 避免 | `std::weak_ptr<T>` |
| 废弃 | `std::auto_ptr<T>` |

创建用 `std::make_unique` / `std::make_shared`，不用 `new`:
```cpp
auto model = std::make_unique<NeuroFlowModel>(config);       // ✅
auto model = std::unique_ptr<NeuroFlowModel>(new NeuroFlowModel(config)); // ❌
```

---

## 六、命名空间

```cpp
// .h 文件
namespace neuroflow {
class MyClass { /* ... */ };
}  // namespace neuroflow

// .cpp 文件 — 不写 using namespace
namespace neuroflow {
void MyFunction() { /* ... */ }
}  // namespace neuroflow
```

> ❌ 禁止在 `.h` 和 `.cpp` 文件级写 `using namespace neuroflow;`
> ✅ `.cpp` 内部函数/方法内可以 `using neuroflow::Tensor;`

---

## 七、全局/静态变量

**❌ 禁止非平凡析构的全局对象。**

快速判断:
```cpp
constexpr int kMax = 100;                              // ✅ trivially destructible
static const char* const kName = "NeuroFlow";          // ✅
static const std::string kPath = "/data";              // ❌ 非平凡析构!
```

若确实需要:
```cpp
static const auto& kPath = *new std::string("/data");  // 永不析构模式
```

---

## 八、整数类型

统一用 `<cstdint>`:
```cpp
int64_t steps;        // ✅
unsigned long long x; // ❌
long y;               // ❌
```

---

## 九、异常与 RTTI

- ❌ **不使用 C++ 异常** — 不用 `try`/`catch`/`throw`
- ❌ **不使用 RTTI** — 不用 `dynamic_cast` / `typeid`
- 错误处理: 返回错误码、`std::optional`、断言

---

## 十、auto

仅在以下情况使用:
```cpp
auto it = vec.begin();                           // ✅ 迭代器类型显而易见
auto result = std::make_unique<Model>(cfg);      // ✅ 右侧已明确类型
std::vector<int> data = {1, 2, 3};               // ✅ 简单类型不用 auto
```

---

## 十一、注释

```cpp
// 用 // 行注释，不用 /* */ 块注释
// 中英文混用: 函数级注释可用中文（面向团队），API 用英文
//
// 变量注释紧贴声明上方:
// 学习率，控制每次参数更新的步幅
float learning_rate_ = 3e-5f;

// TODO(chenzhiwen): 将来用 Adam 替换朴素 SGD
```
> `.clang-format` 的 `ReflowComments: false` 防止破坏中文注释。

---

## 十二、格式化工具

### clang-format
```bash
clang-format -i src/myfile.cpp             # 单个文件
find src/ include/ -name '*.cpp' -o -name '*.hpp' \
  | xargs clang-format -i                   # 整个项目
clang-format --dry-run src/myfile.cpp       # 仅检查，不修改
```

### clang-tidy
```bash
clang-tidy src/myfile.cpp -- -Iinclude -std=c++20
# 通过 CMake (建议):
cmake -DCMAKE_CXX_CLANG_TIDY="clang-tidy" ..
make
```

> 当前 `.clang-tidy` 只启用了**低侵入性**检查 — 不阻塞现有代码编译。

---

## 十三、新旧代码共存策略

| 场景 | 规则 |
|------|------|
| **新文件** | 严格遵守本规范 |
| **修改旧文件** | 修改部分遵守新规范；不改无关行 |
| **全量重命名** | 另立专项，不在功能 PR 里混入风格改动 |
| **CI 检查** | 当前不强制 — 等核心训练稳定后再开 `WarningsAsErrors` |

---

## 十四、快速检查清单（Code Review 用）

- [ ] 头文件自包含？ (`-fsyntax-only` 能过？)
- [ ] `#include` 顺序正确？
- [ ] 无 `using namespace` 在文件级/头文件？
- [ ] 成员变量有尾下划线 `_`？
- [ ] 单参构造有 `explicit`？
- [ ] 智能指针用 `make_unique`/`make_shared`？
- [ ] 无 C 风格类型转换 `(int)x`（用 `static_cast<int>(x)`）？
- [ ] 无全局非平凡析构对象？
- [ ] 整数用 `<cstdint>` 类型？

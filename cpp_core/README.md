# NeuroFlow C++ Core

高性能C++底层实现，提供Python绑定接口。

## 特点

- **轻量化**: 相比原Python版本减少80%+内存占用
- **SIMD优化**: AVX2/ARM NEON加速矩阵运算
- **量化支持**: INT8/FP8量化，4x内存压缩
- **MLA技术**: DeepSeek KV压缩，87.5%+内存节省
- **长记忆**: 分页记忆系统，支持磁盘溢出
- **零依赖**: 单静态库，无外部依赖

## 构建

```bash
cd cpp_core
chmod +x build.sh
./build.sh build    # 构建核心库
./build.sh test     # 构建并测试
./build.sh python   # 构建Python绑定
```

## 目录结构

```
cpp_core/
├── include/neuroflow/
│   ├── tensor.hpp      # 张量运算引擎 (SIMD)
│   ├── networks.hpp    # ECN/DMN/SN网络
│   ├── memory.hpp      # MLA/记忆模块
│   └── model.hpp       # 主模型类
├── src/
│   ├── tensor.cpp
│   └── model.cpp
├── bindings/
│   └── python_bindings.cpp  # pybind11绑定
├── tests/
│   ├── test_tensor.cpp
│   └── test_model.cpp
├── CMakeLists.txt
└── build.sh
```

## Python使用

```python
import neuroflow_cpp as nf

# 创建模型
config = nf.ModelConfig(
    input_dim=512,
    hidden_dim=256,
    output_dim=10,
    use_quantization=True
)
model = nf.NeuroFlowModel(config)

# 前向传播
import numpy as np
x = np.random.randn(32, 512).astype(np.float32)
output = model.forward(x)

print(output.output.shape)  # (32, 10)
print(output.decision.shape)
print(output.manifold.shape)  # 如果return_manifold=True

# 性能对比
stats = nf.benchmark()
print(f"Size reduction: {stats['size_reduction']*100:.1f}%")
```

## 性能对比

| 版本 | 参数量 | 内存 | 推理时间 | 相比原版 |
|------|--------|------|----------|----------|
| C++ Original | 1.25M | 5 MB | 2.5 ms | baseline |
| C++ Optimized | 171K | 0.7 MB | 1.2 ms | 2x加速 |
| C++ Lite | 79K | 0.3 MB | 0.8 ms | 3x加速 |
| C++ Quantized | 79K | 0.08 MB | 0.6 ms | 4x加速 |
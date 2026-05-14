# 🚀 NeuroFlow 本地部署手册

> 完整覆盖 Linux / macOS / Windows / Docker 四大平台的部署流程。
> 支持 **CPU (AVX2/NEON)** 和 **GPU (CUDA)** 双模式推理。

---

## 📋 目录

1. [系统要求](#1-系统要求)
2. [工具链安装](#2-工具链安装)
3. [快速部署](#3-快速部署)
4. [C++ 源码编译](#4-c-源码编译)
5. [Python 绑定安装](#5-python-绑定安装)
6. [GPU 加速部署](#6-gpu-加速部署)
7. [Docker 容器化部署](#7-docker-容器化部署)
8. [验证部署](#8-验证部署)
9. [性能调优](#9-性能调优)
10. [常见问题](#10-常见问题)

---

## 1. 系统要求

### 最低配置

| 组件 | CPU 模式 | GPU 模式 |
|------|----------|----------|
| 内存 | 256 MB | 512 MB |
| 磁盘 | 10 MB | 50 MB |
| CPU | x86_64 (AVX2) 或 ARM64 (NEON) | 任意 |
| GPU | 不需要 | NVIDIA CUDA 11.0+ (≥2GB VRAM) |

### 支持平台

| 平台 | CPU | GPU | 已验证 |
|------|:---:|:---:|:---:|
| Ubuntu 22.04+ | ✅ AVX2 + NEON | ✅ CUDA 11/12 | ✅ |
| macOS 13+ (Intel) | ✅ AVX2 | ❌ | ✅ |
| macOS 14+ (Apple Silicon) | ✅ NEON | ❌ (MPS计划中) | ✅ |
| Windows 11+ | ✅ AVX2 | ✅ CUDA | ✅ |
| Debian 12 / RHEL 9 | ✅ AVX2 + NEON | ✅ CUDA | ✅ |
| Raspberry Pi 5 | ✅ NEON | ❌ | ⚠️ 测试中 |

---

## 2. 工具链安装

### 2.1 Linux (Ubuntu/Debian)

```bash
# 编译器 + 构建工具
sudo apt update
sudo apt install -y build-essential cmake ninja-build git

# Python 开发包
sudo apt install -y python3-dev python3-pip

# pybind11（二选一）
pip install pybind11                # 方式1: pip
# sudo apt install -y pybind11-dev   # 方式2: apt

# GPU 支持（可选）
sudo apt install -y nvidia-cuda-toolkit
```

### 2.2 macOS

```bash
# Xcode 命令行工具
xcode-select --install

# Homebrew 工具链
brew install cmake ninja python@3.11

# pybind11
pip install pybind11
```

### 2.3 Windows

```powershell
# 方式1：Visual Studio 2022 Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools

# 方式2：MinGW-w64
winget install -e --id MSYS2.MSYS2
# 在 MSYS2 中: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja

# Python
winget install Python.Python.3.11

# pybind11
pip install pybind11
```

### 2.4 一键安装脚本

```bash
# Linux / macOS 自动安装
curl -fsSL https://raw.githubusercontent.com/chenzhiwenhphp12-afk/neuroflow-model/main/scripts/install_tools.sh | bash
```

---

## 3. 快速部署

### 方式一：pip 安装（推荐）

```bash
# 直接从 GitHub 安装（自动编译 C++ 核心）
pip install git+https://github.com/chenzhiwenhphp12-afk/neuroflow-model.git

# 或者本地安装
git clone https://github.com/chenzhiwenhphp12-afk/neuroflow-model.git
cd neuroflow-model
pip install -e .

# 验证
python -c "import neuroflow; print(neuroflow.get_backend())"
# 输出: C++
```

### 方式二：预编译 wheel（即将提供）

```bash
# 下载对应平台的 .whl 文件
pip install neuroflow-2.1.0-cp311-cp311-linux_x86_64.whl
```

---

## 4. C++ 源码编译

### 4.1 基础编译（CPU 模式）

```bash
git clone https://github.com/chenzhiwenhphp12-afk/neuroflow-model.git
cd neuroflow-model/cpp_core

# 配置
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Ninja"

# 编译
ninja -j$(nproc)     # Linux/macOS
ninja -j%NUMBER_OF_PROCESSORS%   # Windows

# 产物
#   libneuroflow_core.a          — 静态库
#   neuroflow_tensor_test         — 张量测试
#   neuroflow_model_test          — 模型测试
#   neuroflow_multimodal_test     — 多模态测试
#   neuroflow_python.so           — Python 绑定
```

### 4.2 CMake 编译选项

| 选项 | 默认值 | 说明 |
|------|:---:|------|
| `CMAKE_BUILD_TYPE` | Release | Debug / Release / RelWithDebInfo |
| `NEUROFLOW_ENABLE_AVX2` | ON | x86 AVX2 SIMD 加速 |
| `NEUROFLOW_ENABLE_NEON` | ON | ARM NEON SIMD 加速 |
| `NEUROFLOW_ENABLE_TESTS` | ON | 编译单元测试 |
| `NEUROFLOW_BUILD_PYTHON` | ON | 编译 pybind11 Python 绑定 |
| `NEUROFLOW_ENABLE_CUDA` | OFF | CUDA GPU 加速 |

```bash
# 自定义编译示例
cmake .. -G "Ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEUROFLOW_ENABLE_AVX2=ON \
  -DNEUROFLOW_ENABLE_TESTS=ON \
  -DNEUROFLOW_BUILD_PYTHON=ON
```

### 4.3 平台特定参数

```bash
# ARM64 (树莓派 / Apple Silicon)
cmake .. -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DNEUROFLOW_ENABLE_NEON=ON

# Apple Silicon 交叉编译
cmake .. -DCMAKE_OSX_ARCHITECTURES=arm64

# Windows MinGW
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
```

---

## 5. Python 绑定安装

### 5.1 标准安装

```bash
cd neuroflow-model
pip install -e .
```

### 5.2 手动编译 + 安装

```bash
# 1. 编译 C++ 绑定
cd cpp_core/build
cmake .. -DNEUROFLOW_BUILD_PYTHON=ON
ninja

# 2. 复制 .so/.pyd 到包目录
# Linux/macOS:
cp neuroflow_python.*.so ../neuroflow/_core.*.so
# Windows:
copy neuroflow_python.*.pyd ..\neuroflow\_core.*.pyd

# 3. 测试
cd ../..
python -c "from neuroflow._core import NeuroFlowLite; print('OK')"
```

### 5.3 多 Python 版本

```bash
# 指定 Python 版本编译
cmake .. -DPython3_EXECUTABLE=/usr/bin/python3.11
cmake .. -DPython3_EXECUTABLE=/usr/bin/python3.12
```

---

## 6. GPU 加速部署

### 6.1 CUDA 编译

```bash
# 前提：已安装 CUDA Toolkit 11.0+
nvcc --version   # 验证

cd cpp_core/build
cmake .. \
  -DNEUROFLOW_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"   # RTX 20/30/40/50 系列

ninja
```

### 6.2 GPU 推理示例

```python
import neuroflow
from neuroflow import ModelConfig, NeuroFlowModel

# 创建 GPU 模型
cfg = ModelConfig()
cfg.input_dim = 512
cfg.hidden_dim = 256
cfg.output_dim = 10
cfg.device = "cuda"         # GPU 推理
cfg.use_quantization = True

model = NeuroFlowModel(cfg)

import numpy as np
x = np.random.randn(32, 512).astype(np.float32)  # 批次推理
output = model.forward(x)
print(f"GPU batch inference: {output.decision.shape}")
```

### 6.3 GPU 性能对比

| 模式 | 批次大小 | 推理时间 | 吞吐量 |
|------|:---:|------|:---:|
| CPU (AVX2) | 1 | 1.69 ms | 591 img/s |
| CPU (AVX2) | 32 | 18.2 ms | 1758 img/s |
| GPU (CUDA) | 32 | 1.24 ms | **25,800 img/s** |
| GPU (CUDA) | 256 | 4.87 ms | **52,500 img/s** |

---

## 7. Docker 容器化部署

### 7.1 Dockerfile

```dockerfile
FROM ubuntu:22.04

RUN apt update && apt install -y \
    build-essential cmake ninja-build git \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pybind11 numpy

WORKDIR /app
COPY . /app/neuroflow-model

RUN cd /app/neuroflow-model/cpp_core \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -G "Ninja" \
    && ninja -j$(nproc) \
    && cp neuroflow_python.*.so /app/neuroflow-model/neuroflow/_core.*.so

ENV PYTHONPATH=/app/neuroflow-model:$PYTHONPATH

CMD ["python3", "-c", "import neuroflow; print(f'NeuroFlow {neuroflow.__version__} — {neuroflow.get_backend()} backend')"]
```

### 7.2 构建与运行

```bash
# 构建
docker build -t neuroflow:latest .

# 运行
docker run --rm neuroflow:latest

# GPU 版本
docker run --rm --gpus all neuroflow:latest
```

### 7.3 Docker Compose

```yaml
# docker-compose.yml
version: "3.8"
services:
  neuroflow:
    build: .
    image: neuroflow:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 8. 验证部署

### 8.1 C++ 测试

```bash
cd cpp_core/build

# 全部测试
./neuroflow_tensor_test       # 10项张量测试
./neuroflow_model_test        # 10项模型测试
./neuroflow_multimodal_test   # 10项多模态测试
./neuroflow_online_test       # 在线学习测试
```

预期输出：
```
========================================
All tests PASSED!
========================================
```

### 8.2 Python 验证

```bash
python -c "
import neuroflow, numpy as np, time

print(f'NeuroFlow v{neuroflow.__version__}')
print(f'Backend: {neuroflow.get_backend()}')

# 模型统计
from neuroflow import NeuroFlowLite
model = NeuroFlowLite(input_dim=512)

# 推理测试
x = np.random.randn(100, 512).astype(np.float32)
for _ in range(10): _ = model.forward(x)

t0 = time.perf_counter()
for _ in range(100): _ = model.forward(x)
elapsed = time.perf_counter() - t0

print(f'Inference: {elapsed/100*1000:.2f}ms avg')
print('✅ Deployment verified!')"
```

### 8.3 基准测试工具

```bash
# 内置基准测试
neuroflow-bench --mode lite --iterations 1000
neuroflow-bench --mode full --iterations 100

# Python API 基准测试
python -c "from neuroflow import benchmark; print(benchmark())"
```

---

## 9. 性能调优

### 9.1 CPU 优化

```bash
# 1. 启用编译器优化
cmake .. -DCMAKE_CXX_FLAGS="-march=native -mtune=native -O3 -flto"

# 2. 多线程 GEMM
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 3. 大页内存
echo 2048 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```

### 9.2 INT8 量化调优

```python
from neuroflow import NeuroFlowModel, ModelConfig

cfg = ModelConfig()
cfg.use_quantization = True        # 启用 INT8
cfg.quantization_scheme = "sym"    # 对称量化（更快）
# cfg.quantization_scheme = "asym" # 非对称量化（更准）

model = NeuroFlowModel(cfg)
stats = model.get_stats()
print(f"Quantization ratio: {stats.quantization_ratio:.1%}")
```

### 9.3 批次推理

```python
# 大批次提高吞吐量
x = np.random.randn(256, 512).astype(np.float32)  # 批次大小 256
output = model.forward(x)
```

---

## 10. 常见问题

### Q1: `cmake: command not found`
```bash
pip install cmake          # Python 方式
# 或 sudo apt install cmake
```

### Q2: `pybind11 not found`
```bash
pip install pybind11
# 然后指定路径
cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
```

### Q3: AVX2 不可用
```bash
# 检查 CPU 支持
grep -o 'avx2' /proc/cpuinfo | head -1   # Linux
sysctl -a | grep AVX2                      # macOS

# 不支持则禁用
cmake .. -DNEUROFLOW_ENABLE_AVX2=OFF
```

### Q4: Python 版本不匹配
```bash
# 指定 Python 版本
cmake .. -DPython3_EXECUTABLE=/path/to/python3.11
```

### Q5: 导入时报 `_core` 模块未找到
```bash
# 确保 .so 文件在正确位置
ls neuroflow/_core*.so   # Linux/macOS
ls neuroflow\_core*.pyd  # Windows

# 手动复制
cp cpp_core/build/neuroflow_python.*.so neuroflow/_core.*.so
```

### Q6: GPU 模式报 CUDA 未找到
```bash
nvcc --version               # 检查 CUDA
nvidia-smi                   # 检查驱动
cmake .. -DNEUROFLOW_ENABLE_CUDA=OFF   # 回退到 CPU
```

---

## 📊 部署验证清单

| 步骤 | 命令 | 预期结果 |
|------|------|----------|
| 1. 编译 | `cmake .. && ninja` | Build OK |
| 2. 测试 | `./neuroflow_model_test` | All PASSED |
| 3. Python | `import neuroflow` | `C++` backend |
| 4. 推理 | `model.forward(x)` | < 5ms |
| 5. 量化 | `cfg.use_quantization=True` | ratio > 70% |
| 6. GPU | `cfg.device="cuda"` | 可用时报成功 |

---

## 📁 部署产出物

| 文件 | 类型 | 用途 |
|------|------|------|
| `cpp_core/build/libneuroflow_core.a` | 静态库 | C++ 项目链接 |
| `cpp_core/build/neuroflow_python.*.so` | 动态库 | Python 绑定 |
| `neuroflow/_core.*.so` | 动态库 | Python 包内绑定 |
| `cpp_core/build/neuroflow_*_test` | 可执行文件 | 单元测试 |

---

## 🔗 相关文档

- [README](README.md) — 项目概述
- [CPP_REFACTOR_REPORT.md](CPP_REFACTOR_REPORT.md) — C++ 重构报告
- [OPTIMIZATION.md](OPTIMIZATION.md) — 优化详解
- [cpp_core/README_MULTIMODAL.md](cpp_core/README_MULTIMODAL.md) — 多模态文档

---

<p align="center">
  <sub>NeuroFlow v2.1 · MIT License · <a href="https://github.com/chenzhiwenhphp12-afk/neuroflow-model">GitHub</a></sub>
</p>

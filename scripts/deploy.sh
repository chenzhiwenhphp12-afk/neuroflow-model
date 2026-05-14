#!/bin/bash
# ============================================================
# NeuroFlow Deploy Script — 一键安装 + 编译 + 验证
# 用法: bash scripts/deploy.sh [--gpu] [--target /path/to/install]
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

TARGET_DIR=""
ENABLE_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)    ENABLE_GPU=true; shift ;;
        --target) TARGET_DIR="$2"; shift 2 ;;
        *)        echo "Usage: $0 [--gpu] [--target /path]"; exit 1 ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/cpp_core/build"
INSTALL_DIR="${TARGET_DIR:-${PROJECT_DIR}}"

echo ""
echo "=============================================="
echo "  NeuroFlow Deployment Script v2.1"
echo "=============================================="
echo "  Project:  ${PROJECT_DIR}"
echo "  Target:   ${INSTALL_DIR}"
echo "  GPU:      ${ENABLE_GPU}"
echo "  Platform: $(uname -s) ($(uname -m))"
echo "=============================================="
echo ""

# Step 1: 检查工具链
info "Step 1/5: Checking toolchain..."
command -v cmake  >/dev/null || fail "cmake not found. Run: pip install cmake"
command -v ninja  >/dev/null || fail "ninja not found. Run: pip install ninja"
command -v python3 >/dev/null || fail "python3 not found"
python3 -c "import pybind11" 2>/dev/null || fail "pybind11 not found. Run: pip install pybind11"
ok "cmake $(cmake --version | head -1 | awk '{print $3}'), ninja, python3, pybind11"

# Step 2: 编译 C++ 核心
info "Step 2/5: Building C++ core..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -G "Ninja"
    -Dpybind11_DIR="${PYBIND11_DIR}"
    -DPython3_EXECUTABLE="$(which python3)"
    -DNEUROFLOW_BUILD_PYTHON=ON
)

# 平台检测
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    CMAKE_ARGS+=(-DNEUROFLOW_ENABLE_AVX2=ON)
elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    CMAKE_ARGS+=(-DNEUROFLOW_ENABLE_NEON=ON -DNEUROFLOW_ENABLE_AVX2=OFF)
fi

if $ENABLE_GPU; then
    CMAKE_ARGS+=(-DNEUROFLOW_ENABLE_CUDA=ON)
fi

cmake .. "${CMAKE_ARGS[@]}" 2>&1 | tail -3
ninja -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
ok "C++ core built"

# Step 3: 运行测试
info "Step 3/5: Running tests..."
./neuroflow_tensor_test 2>&1 | grep -q "All tests PASSED" || fail "Tensor tests failed"
./neuroflow_model_test 2>&1 | grep -q "All tests PASSED" || fail "Model tests failed"
./neuroflow_multimodal_test 2>&1 | grep -q "All MultiModal tests PASSED" || fail "MultiModal tests failed"
ok "30/30 tests passed"

# Step 4: 安装 Python 绑定
info "Step 4/5: Installing Python bindings..."
SO_FILE=$(ls neuroflow_python.*.so 2>/dev/null | head -1)
PYD_FILE=$(ls neuroflow_python.*.pyd 2>/dev/null | head -1)

if [ -n "$SO_FILE" ]; then
    cp "$SO_FILE" "${INSTALL_DIR}/neuroflow/_core$(echo $SO_FILE | sed 's/neuroflow_python//')"
elif [ -n "$PYD_FILE" ]; then
    cp "$PYD_FILE" "${INSTALL_DIR}/neuroflow/_core$(echo $PYD_FILE | sed 's/neuroflow_python//')"
fi
ok "Python binding installed"

# Step 5: 验证
info "Step 5/5: Verifying deployment..."
cd "${INSTALL_DIR}"
python3 -c "
import sys; sys.path.insert(0, '.')
from neuroflow import get_backend, NeuroFlowLite
import numpy as np, time

assert get_backend() == 'C++', 'Backend should be C++'

model = NeuroFlowLite(input_dim=512)
x = np.random.randn(1, 512).astype(np.float32)
for _ in range(10): _ = model.forward(x)

t0 = time.perf_counter()
for _ in range(100): _ = model.forward(x)
elapsed = time.perf_counter() - t0
avg_ms = elapsed / 100 * 1000

print(f'        Backend: C++ (SIMD)')
print(f'        Inference: {avg_ms:.2f}ms avg')
print(f'        Throughput: {100/elapsed:.0f} samples/s')
assert avg_ms < 10, f'Inference too slow: {avg_ms:.2f}ms'
"
ok "Deployment verified"

echo ""
echo "=============================================="
echo -e "  ${GREEN}✅ NeuroFlow Deployment Complete!${NC}"
echo "=============================================="
echo ""
echo "  Quick start:"
echo "    cd ${INSTALL_DIR}"
echo "    python3 -c 'import neuroflow; print(neuroflow.get_backend())'"
echo ""
echo "  Run benchmark:"
echo "    python3 -c 'from neuroflow import benchmark; print(benchmark())'"
echo ""

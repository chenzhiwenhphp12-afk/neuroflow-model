#!/bin/bash
# NeuroFlow C++ 编译脚本 — 自动暂停/恢复 cron 防止冲突
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DEST="/mnt/d/neuroflow-model/neuroflow/_core.cpython-311-x86_64-linux-gnu.so"
VENV_PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python3"
PYBIND11_DIR="$($VENV_PYTHON -c 'import pybind11; print(pybind11.get_cmake_dir())')"
CRON_LEARN="bd007dc2fe14"
CRON_REPORT="5d4dac90ff0d"

echo "=== NeuroFlow C++ Build ==="

# 1. 暂停 cron 任务
echo "[1/5] Pausing cron jobs..."
hermes cron pause "$CRON_LEARN" 2>/dev/null || echo "  ($CRON_LEARN already paused)"
hermes cron pause "$CRON_REPORT" 2>/dev/null || echo "  ($CRON_REPORT already paused)"
sleep 2  # 等正在运行的任务结束

# 2. 编译
echo "[2/5] Building with ninja..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEUROFLOW_BUILD_TESTS=OFF \
  -DNEUROFLOW_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$VENV_PYTHON" \
  -Dpybind11_DIR="$PYBIND11_DIR"
ninja -j$(nproc)

# 3. 部署 so 到 neuroflow 包目录
echo "[3/5] Deploying .so to neuroflow package..."
cp "$BUILD_DIR/_core.cpython-311-x86_64-linux-gnu.so" "$DEST"

# 4. 验证
echo "[4/5] Verifying..."
"$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '/mnt/d/neuroflow-model')
from neuroflow._core import create_multimodal
m = create_multimodal()
import numpy as np
x = np.random.randn(1, 512).astype(np.float32)
out = m.forward_text(x)
print(f'  forward_text OK, output shape: {out.decision.shape}')
m.save('/tmp/nf_build_test.bin')
m2 = create_multimodal()
m2.load('/tmp/nf_build_test.bin')
print('  save/load roundtrip OK')
print('  ✅ Verification passed')
"

# 5. 恢复 cron
echo "[5/5] Resuming cron jobs..."
hermes cron resume "$CRON_LEARN" 2>/dev/null || echo "  ($CRON_LEARN already running)"
hermes cron resume "$CRON_REPORT" 2>/dev/null || echo "  ($CRON_REPORT already running)"

echo "=== Build complete ==="

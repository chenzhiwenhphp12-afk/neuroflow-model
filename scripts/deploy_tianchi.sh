#!/bin/bash
# ════════════════════════════════════════════════════════
# NeuroFlow — 阿里天池 V100 一键部署脚本
# 用法: 在天池实验室 Notebook 终端里运行:
#   bash deploy_tianchi.sh
# ════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════════╗"
echo "║     NeuroFlow 天池 V100 自动部署脚本             ║"
echo "╚══════════════════════════════════════════════════╝"

# ── 0. 检查 GPU ──
echo ""
echo "🔍 [1/6] 检查 GPU 环境..."
nvidia-smi
echo "✅ GPU 就绪"

# ── 1. 安装依赖 ──
echo ""
echo "📦 [2/6] 安装编译依赖..."
apt-get update -qq
apt-get install -y -qq cmake build-essential libomp-dev 2>/dev/null

# 检查 CUDA
if command -v nvcc &>/dev/null; then
    echo "   CUDA $(nvcc --version | grep 'release' | awk '{print $6}' | tr -d ',')"
else
    echo "   ⚠️ nvcc 未找到，尝试安装 CUDA..."
fi

# ── 2. 获取代码 ──
echo ""
echo "📥 [3/6] 获取 NeuroFlow 源码..."
# 二选一:
#   方案 A: 从 GitHub 克隆 (推荐)
#   方案 B: 从上传的文件复制
if [ -d "neuroflow-C++" ]; then
    echo "   检测到已有目录，跳过克隆"
else
    # === 修改这里为你的仓库地址 ===
    GIT_REPO="https://github.com/你的用户名/neuroflow-C++.git"
    
    echo "   正在克隆: $GIT_REPO"
    git clone "$GIT_REPO" 2>/dev/null || {
        echo "   ⚠️ GitHub 克隆失败，尝试从本地上传..."
        echo "   请手动上传 neuroflow-C++ 到当前目录"
        echo "   然后重新运行本脚本"
        exit 1
    }
fi
cd neuroflow-C++

# ── 3. 准备训练数据 ──
echo ""
echo "📝 [4/6] 准备训练数据..."
if [ ! -f "data/train.txt" ]; then
    mkdir -p data
    echo "   生成小规模测试数据..."
    python3 -c "
import os
# 生成 1000 条中文训练样本
samples = []
for i in range(1000):
    samples.append(f'这是第{i}条训练数据，用于NeuroFlow模型的预训练和微调任务。')
with open('data/train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(samples))
print(f'   已生成 {len(samples)} 条训练样本 → data/train.txt')
"
else
    echo "   训练数据已存在: data/train.txt"
fi

# ── 4. 编译 ──
echo ""
echo "🔧 [5/6] 编译 NeuroFlow (CUDA 模式)..."
# 检查是否有 CUDA
HAS_CUDA=false
if [ -f /usr/local/cuda/include/cuda.h ] || [ -d /usr/local/cuda ]; then
    HAS_CUDA=true
fi

if [ "$HAS_CUDA" = true ]; then
    echo "   使用 CUDA 后端编译..."
    cmake -B build_cuda \
        -DNEUROFLOW_USE_CUDA=ON \
        -DNEUROFLOW_USE_AVX2=ON \
        -DNEUROFLOW_USE_BLAS=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build_cuda -j$(nproc)
    BUILD_DIR="build_cuda"
else
    echo "   ⚠️ 未检测到 CUDA，使用 CPU OpenMP 编译..."
    cmake -B build_cpu \
        -DNEUROFLOW_USE_CUDA=OFF \
        -DNEUROFLOW_USE_AVX2=ON \
        -DNEUROFLOW_USE_BLAS=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build_cpu -j$(nproc)
    BUILD_DIR="build_cpu"
fi

echo "✅ 编译完成！二进制文件在 ./$BUILD_DIR/"

# ── 5. 运行训练 ──
echo ""
echo "🚀 [6/6] 开始训练..."
echo "════════════════════════════════════════════════════"
echo "  配置: configs/config_distill.json"
echo "  数据: data/train.txt"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  显存: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "════════════════════════════════════════════════════"

# 先用 distill 小配置测试
./$BUILD_DIR/neuroflow_train_v2 \
    --config configs/config_distill.json \
    --data data/train.txt \
    --output output_tianchi \
    --epochs 5 \
    --batch-size 64 \
    --lr 0.001 \
    --use-cuda \
    --adam \
    --log-interval 10

echo ""
echo "🎉 训练完成！"
echo "   模型保存在: output_tianchi/"
echo ""
echo "   🔜 下一步: 修改 configs/config.json 跑全量训练"
echo "   ./$BUILD_DIR/neuroflow_train_v2 \\"
echo "        --config configs/config.json \\"
echo "        --data data/train.txt \\"
echo "        --output output_full \\"
echo "        --epochs 100 \\"
echo "        --batch-size 64 \\"
echo "        --lr 0.0001 \\"
echo "        --use-cuda --adam"

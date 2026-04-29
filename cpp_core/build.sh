#!/bin/bash
# NeuroFlow C++ Core Build Script

set -e

echo "========================================"
echo "NeuroFlow C++ Core Build"
echo "========================================"

# 检查依赖
check_dependencies() {
    echo "Checking dependencies..."
    
    # CMake
    if ! command -v cmake &> /dev/null; then
        echo "ERROR: cmake not found. Please install cmake >= 3.14"
        exit 1
    fi
    
    # C++编译器
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        echo "ERROR: C++ compiler not found"
        exit 1
    fi
    
    echo "  cmake: $(cmake --version | head -1)"
    echo "  compiler: ${CXX:-g++}"
}

# 构建C++核心
build_cpp() {
    echo ""
    echo "Building C++ core..."
    
    mkdir -p build
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DNEUROFLOW_BUILD_TESTS=ON \
        -DNEUROFLOW_BUILD_PYTHON=OFF
    
    make -j$(nproc)
    
    echo "  Build complete!"
}

# 运行测试
run_tests() {
    echo ""
    echo "Running tests..."
    
    cd build
    
    if [ -f neuroflow_tests ]; then
        ./neuroflow_tests
    else
        echo "  Tests not built"
    fi
}

# 构建Python绑定 (可选)
build_python() {
    echo ""
    echo "Building Python bindings..."
    
    # 检查pybind11
    python3 -c "import pybind11" 2>/dev/null || {
        echo "  pybind11 not found, installing..."
        pip3 install pybind11
    }
    
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DNEUROFLOW_BUILD_PYTHON=ON
    
    make -j$(nproc)
    
    echo "  Python bindings built!"
    
    # 测试Python导入
    python3 -c "import neuroflow_cpp; print('  Import OK')" || {
        echo "  Warning: Python import failed, check library path"
    }
}

# 安装
install() {
    echo ""
    echo "Installing..."
    
    cd build
    sudo make install
    
    echo "  Installed to /usr/local"
}

# 清理
clean() {
    echo "Cleaning..."
    rm -rf build
    echo "  Clean complete"
}

# 帮助
help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build C++ core"
    echo "  test        Build and run tests"
    echo "  python      Build with Python bindings"
    echo "  install     Install to system"
    echo "  clean       Clean build directory"
    echo "  all         Build, test, and install"
    echo "  help        Show this help"
}

# 主入口
case "${1:-build}" in
    build)
        check_dependencies
        build_cpp
        ;;
    test)
        check_dependencies
        build_cpp
        run_tests
        ;;
    python)
        check_dependencies
        build_python
        ;;
    install)
        check_dependencies
        build_cpp
        install
        ;;
    clean)
        clean
        ;;
    all)
        check_dependencies
        build_cpp
        run_tests
        install
        ;;
    help|--help|-h)
        help
        ;;
    *)
        echo "Unknown command: $1"
        help
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
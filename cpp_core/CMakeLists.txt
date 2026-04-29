cmake_minimum_required(VERSION 3.14)
project(neuroflow_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 编译选项
option(NEUROFLOW_BUILD_TESTS "Build unit tests" ON)
option(NEUROFLOW_BUILD_PYTHON "Build Python bindings" ON)
option(NEUROFLOW_USE_AVX2 "Enable AVX2 SIMD" ON)
option(NEUROFLOW_USE_NEON "Enable ARM NEON SIMD" ON)
option(NEUROFLOW_USE_QUANTIZATION "Enable quantization support" ON)

# 检测SIMD支持
if(NEUROFLOW_USE_AVX2)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        message(STATUS "AVX2 SIMD enabled")
    else()
        message(STATUS "AVX2 not supported by compiler")
    endif()
endif()

if(NEUROFLOW_USE_NEON AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    message(STATUS "ARM NEON SIMD enabled")
endif()

# 优化选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# 核心库
add_library(neuroflow_core STATIC
    src/tensor.cpp
    src/model.cpp
)

target_include_directories(neuroflow_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(neuroflow_core PRIVATE
    $<$<CONFIG:DEBUG>:DEBUG>
)

# Python绑定 (可选)
if(NEUROFLOW_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # 查找pybind11
    find_package(pybind11 CONFIG)
    if(NOT pybind11_FOUND)
        # 尝试通过pip安装的pybind11
        execute_process(
            COMMAND Python3::Interpreter -m pip show pybind11
            OUTPUT_VARIABLE PYBIND11_INFO
            RESULT_VARIABLE PYBIND11_RESULT
        )
        if(PYBIND11_RESULT EQUAL 0)
            # 找到pybind11位置
            string(REGEX MATCH "Location: ([^\n]+)" PYBIND11_LOCATION "${PYBIND11_INFO}")
            set(pybind11_DIR "${CMAKE_MATCH_1}/pybind11/share/cmake/pybind11")
            find_package(pybind11 CONFIG REQUIRED)
        else()
            message(WARNING "pybind11 not found, skipping Python bindings")
            set(NEUROFLOW_BUILD_PYTHON OFF)
        endif()
    endif()
    
    if(NEUROFLOW_BUILD_PYTHON)
        pybind11_add_module(neuroflow_python bindings/python_bindings.cpp)
        target_link_libraries(neuroflow_python PRIVATE neuroflow_core)
        target_compile_definitions(neuroflow_python PRIVATE VERSION_INFO=${PROJECT_VERSION})
        message(STATUS "Python bindings enabled")
    endif()
endif()

# 测试 (可选)
if(NEUROFLOW_BUILD_TESTS)
    enable_testing()
    
    # 分开的测试文件，避免重复 main
    add_executable(neuroflow_tensor_test tests/test_tensor.cpp)
    target_link_libraries(neuroflow_tensor_test PRIVATE neuroflow_core)
    add_test(NAME tensor_test COMMAND neuroflow_tensor_test)
    
    add_executable(neuroflow_model_test tests/test_model.cpp)
    target_link_libraries(neuroflow_model_test PRIVATE neuroflow_core)
    add_test(NAME model_test COMMAND neuroflow_model_test)
    
    # 多模态测试
    add_executable(neuroflow_multimodal_test tests/test_multimodal.cpp)
    target_link_libraries(neuroflow_multimodal_test PRIVATE neuroflow_core)
    add_test(NAME multimodal_test COMMAND neuroflow_multimodal_test)
    
    message(STATUS "Unit tests enabled")
endif()

# 安装规则
install(TARGETS neuroflow_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/neuroflow
    DESTINATION include
)

# 导出配置 (简化)
# install(EXPORT neuroflow_core-export
#     FILE neuroflow_core-config.cmake
#     DESTINATION lib/cmake/neuroflow_core
# )

# 打印配置信息
message(STATUS "")
message(STATUS "NeuroFlow Core Build Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  AVX2:           ${NEUROFLOW_USE_AVX2}")
message(STATUS "  NEON:           ${NEUROFLOW_USE_NEON}")
message(STATUS "  Tests:          ${NEUROFLOW_BUILD_TESTS}")
message(STATUS "  Python:         ${NEUROFLOW_BUILD_PYTHON}")
message(STATUS "")
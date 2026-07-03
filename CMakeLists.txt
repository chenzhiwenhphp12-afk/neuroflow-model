cmake_minimum_required(VERSION 3.14)

option(NEUROFLOW_USE_CUDA "Enable CUDA/cuBLAS backend" OFF)

if(NEUROFLOW_USE_CUDA)
    project(neuroflow_core VERSION 1.0.0 LANGUAGES CXX CUDA)
else()
    project(neuroflow_core VERSION 1.0.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NEUROFLOW_BUILD_TESTS "Build unit tests" OFF)
option(NEUROFLOW_BUILD_PYTHON "Build Python bindings" OFF)
option(NEUROFLOW_USE_AVX2 "Enable AVX2 SIMD" ON)
option(NEUROFLOW_USE_BLAS "Use external BLAS (OpenBLAS/MKL) for GEMM" ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /GR /EHsc /utf-8")
    if(NEUROFLOW_USE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        message(STATUS "AVX2 SIMD enabled (MSVC)")
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /GL")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG")
    find_package(OpenMP)
else()
    find_package(OpenMP REQUIRED)

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

    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -funroll-loops -ffast-math -ftree-vectorize")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
endif()


add_library(neuroflow_core STATIC
    src/tensor.cpp
    src/model.cpp
    src/weight_io.cpp
)

if(NEUROFLOW_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_VERSION VERSION_LESS "11.0")
        message(FATAL_ERROR "CUDA >= 11.0 required, found ${CUDAToolkit_VERSION}")
    endif()
    target_compile_definitions(neuroflow_core PUBLIC USE_CUDA)
    target_include_directories(neuroflow_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(neuroflow_core PUBLIC CUDA::cudart CUDA::cublas)
    target_sources(neuroflow_core PRIVATE src/cuda_context.cpp)
    set_source_files_properties(src/cuda_context.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(neuroflow_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_86,code=[sm_86,compute_86]>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    )
    set(NEUROFLOW_CUDA_FOUND TRUE)
    message(STATUS "CUDA backend enabled: ${CUDAToolkit_VERSION}")
else()
    set(NEUROFLOW_CUDA_FOUND FALSE)
endif()

target_include_directories(neuroflow_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(neuroflow_core PRIVATE
    $<$<CONFIG:DEBUG>:DEBUG>
)

if(NEUROFLOW_USE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        target_compile_definitions(neuroflow_core PUBLIC USE_CBLAS)
        target_link_libraries(neuroflow_core PUBLIC ${BLAS_LIBRARIES})
        if(BLAS_INCLUDE_DIRS)
            target_include_directories(neuroflow_core PUBLIC ${BLAS_INCLUDE_DIRS})
        endif()
        set(NEUROFLOW_BLAS_FOUND TRUE)
        message(STATUS "External BLAS found: ${BLAS_LIBRARIES}")
        message(STATUS "  NOTE: Set OPENBLAS_NUM_THREADS=1 at runtime to avoid thread oversubscription with OpenMP")
    else()
        set(OPENBLAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas")
        if(EXISTS "${OPENBLAS_DIR}/include/cblas.h" AND EXISTS "${OPENBLAS_DIR}/lib/libscipy_openblas.lib")
            target_compile_definitions(neuroflow_core PUBLIC USE_CBLAS)
            target_include_directories(neuroflow_core PUBLIC "${OPENBLAS_DIR}/include")
            target_link_libraries(neuroflow_core PUBLIC "${OPENBLAS_DIR}/lib/libscipy_openblas.lib")
            set(NEUROFLOW_BLAS_FOUND TRUE)
            message(STATUS "Using bundled OpenBLAS from: ${OPENBLAS_DIR}")
            message(STATUS "  NOTE: Set OPENBLAS_NUM_THREADS=1 at runtime to avoid thread oversubscription with OpenMP")
        else()
            set(NEUROFLOW_BLAS_FOUND FALSE)
            message(STATUS "External BLAS not found, using built-in GEMM")
        endif()
    endif()
else()
    set(NEUROFLOW_BLAS_FOUND FALSE)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(neuroflow_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(neuroflow_train_v2 src/train_v2.cpp)
target_link_libraries(neuroflow_train_v2 PRIVATE neuroflow_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neuroflow_train_v2 PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(neuroflow_test_lm tests/test_lm_pathway.cpp)
target_link_libraries(neuroflow_test_lm PRIVATE neuroflow_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neuroflow_test_lm PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(neuroflow_infer_v2 src/infer_v2.cpp)
target_link_libraries(neuroflow_infer_v2 PRIVATE neuroflow_core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neuroflow_infer_v2 PRIVATE OpenMP::OpenMP_CXX)
endif()

if(NEUROFLOW_BUILD_TESTS)
    enable_testing()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor.cpp)
        add_executable(neuroflow_tensor_test tests/test_tensor.cpp)
        target_link_libraries(neuroflow_tensor_test PRIVATE neuroflow_core)
        add_test(NAME tensor_test COMMAND neuroflow_tensor_test)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model.cpp)
        add_executable(neuroflow_model_test tests/test_model.cpp)
        target_link_libraries(neuroflow_model_test PRIVATE neuroflow_core)
        add_test(NAME model_test COMMAND neuroflow_model_test)
    endif()
    message(STATUS "Unit tests enabled (if test files exist)")
endif()

install(TARGETS neuroflow_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/neuroflow
    DESTINATION include
)

message(STATUS "")
message(STATUS "NeuroFlow Core Build Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  AVX2:           ${NEUROFLOW_USE_AVX2}")
message(STATUS "  Tests:          ${NEUROFLOW_BUILD_TESTS}")
message(STATUS "  Python:         ${NEUROFLOW_BUILD_PYTHON}")
message(STATUS "  OpenMP:         ${OpenMP_CXX_FOUND}")
message(STATUS "  BLAS:           ${NEUROFLOW_BLAS_FOUND}")
message(STATUS "  CUDA:           ${NEUROFLOW_CUDA_FOUND}")
message(STATUS "")

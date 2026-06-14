/**
 * NeuroFlow Core Tests - Tensor Operations
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include "../include/neuroflow/tensor.hpp"

using namespace neuroflow;

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;
    
    Tensor t1;
    assert(t1.data_size_ == 0);
    
    Tensor t2({32, 512});
    assert(t2.shape_.size() == 2);
    assert(t2.shape_[0] == 32);
    assert(t2.shape_[1] == 512);
    assert(t2.numel() == 32 * 512);
    assert(t2.data_size_ == 32 * 512 * 4);
    
    std::cout << "  PASSED: tensor creation" << std::endl;
}

void test_tensor_reshape() {
    std::cout << "Testing tensor reshape..." << std::endl;
    
    Tensor t({32, 512});
    Tensor r = t.reshape({16, 1024});
    
    assert(r.shape_[0] == 16);
    assert(r.shape_[1] == 1024);
    assert(r.numel() == t.numel());
    assert(!r.owns_data_);  // 零拷贝
    
    std::cout << "  PASSED: tensor reshape (zero-copy)" << std::endl;
}

void test_tensor_clone() {
    std::cout << "Testing tensor clone..." << std::endl;
    
    Tensor t({10, 20});
    float* data = t.as_fp32();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor c = t.clone();
    assert(c.owns_data_);
    assert(c.numel() == t.numel());
    
    float* cdata = c.as_fp32();
    for (size_t i = 0; i < c.numel(); ++i) {
        assert(std::abs(cdata[i] - static_cast<float>(i)) < 1e-6);
    }
    
    std::cout << "  PASSED: tensor clone" << std::endl;
}

void test_gemm() {
    std::cout << "Testing GEMM (matrix multiplication)..." << std::endl;
    
    // 简单测试: A(2,3) @ B(3,2) = C(2,2)
    Tensor A({2, 3});
    Tensor B({3, 2});
    Tensor C({2, 2});
    
    float* a = A.as_fp32();
    float* b = B.as_fp32();
    
    // A = [[1,2,3], [4,5,6]]
    a[0] = 1; a[1] = 2; a[2] = 3;
    a[3] = 4; a[4] = 5; a[5] = 6;
    
    // B = [[7,8], [9,10], [11,12]]
    b[0] = 7;  b[1] = 8;
    b[2] = 9;  b[3] = 10;
    b[4] = 11; b[5] = 12;
    
    TensorOps::gemm(A, B, C);
    
    float* c = C.as_fp32();
    
    // C = [[58,64], [139,154]]
    assert(std::abs(c[0] - 58) < 1e-4);
    assert(std::abs(c[1] - 64) < 1e-4);
    assert(std::abs(c[2] - 139) < 1e-4);
    assert(std::abs(c[3] - 154) < 1e-4);
    
    std::cout << "  PASSED: GEMM basic" << std::endl;
}

void test_gemm_performance() {
    std::cout << "Testing GEMM performance..." << std::endl;
    
    size_t M = 256, K = 512, N = 256;
    
    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C({M, N});
    
    // 填充随机数据
    float* a = A.as_fp32();
    float* b = B.as_fp32();
    for (size_t i = 0; i < A.numel(); ++i) a[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    for (size_t i = 0; i < B.numel(); ++i) b[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    
    // 预热
    TensorOps::gemm(A, B, C);
    
    // 性能测试
    int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        TensorOps::gemm(A, B, C);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ms = duration.count() / 1000.0 / iterations;
    double gflops = 2.0 * M * K * N / (ms / 1000.0) / 1e9;
    
    std::cout << "  GEMM (256x512x256): " << ms << " ms per iteration, " << gflops << " GFLOPS" << std::endl;
    std::cout << "  PASSED: GEMM performance" << std::endl;
}

void test_layer_norm() {
    std::cout << "Testing LayerNorm..." << std::endl;
    
    Tensor x({2, 4});
    Tensor weight({4});
    Tensor bias({4});
    
    float* data = x.as_fp32();
    data[0] = 1; data[1] = 2; data[2] = 3; data[3] = 4;
    data[4] = 5; data[5] = 6; data[6] = 7; data[7] = 8;
    
    float* w = weight.as_fp32();
    float* b = bias.as_fp32();
    for (size_t i = 0; i < 4; ++i) {
        w[i] = 1.0f;
        b[i] = 0.0f;
    }
    
    TensorOps::layer_norm(x, weight, bias);
    
    // 验证均值接近0，方差接近1
    float* out = x.as_fp32();
    
    // 第一行均值
    float mean1 = 0;
    for (size_t i = 0; i < 4; ++i) mean1 += out[i];
    mean1 /= 4;
    assert(std::abs(mean1) < 1e-4);
    
    std::cout << "  PASSED: LayerNorm" << std::endl;
}

void test_gelu() {
    std::cout << "Testing GELU..." << std::endl;
    
    Tensor x({5});
    float* data = x.as_fp32();
    data[0] = -1; data[1] = 0; data[2] = 1; data[3] = 2; data[4] = 3;
    
    TensorOps::gelu(x);
    
    // GELU(0) ≈ 0
    assert(std::abs(data[1]) < 1e-4);
    
    // GELU(1) ≈ 0.841
    assert(std::abs(data[2] - 0.841f) < 0.01);
    
    std::cout << "  PASSED: GELU" << std::endl;
}

void test_softmax() {
    std::cout << "Testing Softmax..." << std::endl;
    
    Tensor x({2, 4});
    float* data = x.as_fp32();
    data[0] = 1; data[1] = 2; data[2] = 3; data[3] = 4;
    data[4] = 0; data[5] = 0; data[6] = 0; data[7] = 0;
    
    TensorOps::softmax(x);
    
    // 验证每行和为1
    float sum1 = 0;
    for (size_t i = 0; i < 4; ++i) sum1 += data[i];
    assert(std::abs(sum1 - 1.0f) < 1e-4);
    
    float sum2 = 0;
    for (size_t i = 4; i < 8; ++i) sum2 += data[i];
    assert(std::abs(sum2 - 1.0f) < 1e-4);
    
    std::cout << "  PASSED: Softmax" << std::endl;
}

void test_quantization() {
    std::cout << "Testing INT8 quantization..." << std::endl;
    
    Tensor fp32({4, 8});
    float* data = fp32.as_fp32();
    for (size_t i = 0; i < fp32.numel(); ++i) {
        data[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 10;
    }
    
    Tensor int8({4, 8}, QuantType::INT8);
    Tensor scale({4});
    
    TensorOps::quantize_int8(fp32, int8, scale);
    
    // 反量化
    Tensor dequant({4, 8});
    TensorOps::dequantize_int8(int8, dequant, scale);
    
    // 验证误差小于量化精度
    float* original = fp32.as_fp32();
    float* restored = dequant.as_fp32();
    
    float max_error = 0;
    for (size_t i = 0; i < fp32.numel(); ++i) {
        float err = std::abs(original[i] - restored[i]);
        max_error = std::max(max_error, err);
    }
    
    std::cout << "  Max quantization error: " << max_error << std::endl;
    std::cout << "  PASSED: INT8 quantization" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "NeuroFlow Core - Tensor Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_tensor_creation();
    test_tensor_reshape();
    test_tensor_clone();
    test_gemm();
    test_gemm_performance();
    test_layer_norm();
    test_gelu();
    test_softmax();
    test_quantization();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
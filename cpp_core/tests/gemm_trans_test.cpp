#include <iostream>
#include "../include/neuroflow/tensor.hpp"

using namespace neuroflow;

int main() {
    std::cout << "GEMM transpose test..." << std::endl;
    
    // 简化场景
    Tensor A({1, 64});  // input {batch, d_model}
    Tensor B({64, 64}); // weight {d_model, d_model}
    Tensor C({1, 64});  // output {batch, d_model}
    
    float* a = A.as_fp32();
    float* b = B.as_fp32();
    for (size_t i = 0; i < A.numel(); ++i) a[i] = 0.1f * i;
    for (size_t i = 0; i < B.numel(); ++i) b[i] = 0.01f * i;
    
    std::cout << "A shape: [" << A.shape[0] << ", " << A.shape[1] << "]" << std::endl;
    std::cout << "B shape: [" << B.shape[0] << ", " << B.shape[1] << "]" << std::endl;
    std::cout << "C shape: [" << C.shape[0] << ", " << C.shape[1] << "]" << std::endl;
    
    // transB=true 时的参数
    bool transA = false;
    bool transB = true;
    
    size_t M = transA ? A.shape[1] : A.shape[0];  // 1
    size_t K = transA ? A.shape[0] : A.shape[1];  // 64
    size_t N = transB ? B.shape[0] : B.shape[1];  // 64 (B.shape[0])
    
    std::cout << "M=" << M << ", K=" << K << ", N=" << N << std::endl;
    std::cout << "transA=" << transA << ", transB=" << transB << std::endl;
    
    std::cout << "Calling gemm..." << std::endl;
    TensorOps::gemm(A, B, C, transA, transB);
    
    std::cout << "C values: ";
    float* c = C.as_fp32();
    for (size_t i = 0; i < 5; ++i) std::cout << c[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Success!" << std::endl;
    return 0;
}

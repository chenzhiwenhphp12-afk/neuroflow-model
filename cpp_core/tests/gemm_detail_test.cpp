#include <iostream>
#include "../include/neuroflow/tensor.hpp"

using namespace neuroflow;

int main() {
    std::cout << "GEMM detail test..." << std::endl;
    
    Tensor A({2, 64});  // input
    Tensor B({32, 64}); // weight
    Tensor C({2, 32});  // output
    
    std::cout << "A shape: [" << A.shape[0] << ", " << A.shape[1] << "]" << std::endl;
    std::cout << "B shape: [" << B.shape[0] << ", " << B.shape[1] << "]" << std::endl;
    std::cout << "C shape: [" << C.shape[0] << ", " << C.shape[1] << "]" << std::endl;
    
    // Fill data
    float* a = A.as_fp32();
    float* b = B.as_fp32();
    for (size_t i = 0; i < A.numel(); ++i) a[i] = 0.1f * i;
    for (size_t i = 0; i < B.numel(); ++i) b[i] = 0.01f * i;
    
    std::cout << "Calling gemm..." << std::endl;
    TensorOps::gemm(A, B, C);  // C = A @ B^T ?
    
    std::cout << "C numel: " << C.numel() << std::endl;
    std::cout << "C data_size: " << C.data_size << std::endl;
    
    float* c = C.as_fp32();
    std::cout << "First 5 C values: ";
    for (size_t i = 0; i < 5; ++i) std::cout << c[i] << " ";
    std::cout << std::endl;
    
    std::cout << "After gemm, trying to allocate new tensor..." << std::endl;
    Tensor new_t({10});
    
    std::cout << "Success!" << std::endl;
    return 0;
}

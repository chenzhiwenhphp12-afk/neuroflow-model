#include <iostream>
#define HAS_AVX2 0  // 强制禁用 AVX2
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    std::cout << "GEMM without AVX2 test..." << std::endl;
    
    size_t d_model = 64;
    Linear W_q(d_model, d_model, false);
    
    Tensor input({1, d_model});
    float* d = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) d[i] = 0.1f * i;
    
    Tensor x_flat = input.reshape({1, d_model});
    std::cout << "x_flat shape: [" << x_flat.shape_[0] << ", " << x_flat.shape_[1] << "]" << std::endl;
    
    std::cout << "Calling W_q.forward..." << std::endl;
    Tensor q = W_q.forward(x_flat);
    
    std::cout << "q shape: [" << q.shape_[0] << ", " << q.shape_[1] << "]" << std::endl;
    std::cout << "Success!" << std::endl;
    return 0;
}

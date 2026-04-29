#include <iostream>
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Linear full test..." << std::endl;
    
    Linear linear(64, 64, true);  // use_bias = true
    
    std::cout << "weight shape: [" << linear.weight.shape[0] << ", " << linear.weight.shape[1] << "]" << std::endl;
    std::cout << "bias shape size: " << linear.bias.shape.size() << std::endl;
    std::cout << "bias shape[0]: " << linear.bias.shape[0] << std::endl;
    
    Tensor input({1, 64});
    float* d = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) d[i] = 0.1f * i;
    
    std::cout << "input shape: [" << input.shape[0] << ", " << input.shape[1] << "]" << std::endl;
    std::cout << "input numel: " << input.numel() << std::endl;
    
    // 手动执行 forward 的步骤
    Tensor output({input.shape[0], linear.weight.shape[0]});
    std::cout << "output shape: [" << output.shape[0] << ", " << output.shape[1] << "]" << std::endl;
    
    std::cout << "Calling gemm..." << std::endl;
    TensorOps::gemm(input, linear.weight, output, false, true);
    std::cout << "gemm done" << std::endl;
    
    float* out = output.as_fp32();
    float* b = linear.bias.as_fp32();
    
    std::cout << "Adding bias..." << std::endl;
    std::cout << "output.shape[0]=" << output.shape[0] << std::endl;
    std::cout << "output.shape[1]=" << output.shape[1] << std::endl;
    
    for (size_t i = 0; i < output.shape[0]; ++i) {
        for (size_t j = 0; j < output.shape[1]; ++j) {
            out[i * output.shape[1] + j] += b[j];
        }
    }
    
    std::cout << "First 5 output values: ";
    for (size_t i = 0; i < 5; ++i) std::cout << out[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Success!" << std::endl;
    return 0;
}

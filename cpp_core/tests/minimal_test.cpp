#include <iostream>
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Minimal test..." << std::endl;
    
    // Test basic tensor reshape
    Tensor t({2, 64});
    float* data = t.as_fp32();
    for (size_t i = 0; i < t.numel(); ++i) data[i] = 0.1f * i;
    
    std::cout << "Original shape: [" << t.shape[0] << ", " << t.shape[1] << "]" << std::endl;
    
    // Reshape
    Tensor reshaped = t.reshape({128});
    std::cout << "Reshaped: [" << reshaped.shape[0] << "]" << std::endl;
    
    // Test Linear layer
    std::cout << "Testing Linear..." << std::endl;
    Linear linear(64, 32);
    Tensor input({2, 64});
    for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
    
    Tensor output = linear.forward(input);
    std::cout << "Linear output: [" << output.shape[0] << ", " << output.shape[1] << "]" << std::endl;
    
    // Test LayerNorm
    std::cout << "Testing LayerNorm..." << std::endl;
    LayerNorm norm(32);
    Tensor norm_out = norm.forward(output);
    std::cout << "LayerNorm output: [" << norm_out.shape[0] << ", " << norm_out.shape[1] << "]" << std::endl;
    
    std::cout << "Success!" << std::endl;
    return 0;
}

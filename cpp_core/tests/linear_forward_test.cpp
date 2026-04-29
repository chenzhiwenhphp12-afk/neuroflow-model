#include <iostream>
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Linear forward test..." << std::endl;
    
    Linear linear(64, 64, false);  // use_bias = false
    
    std::cout << "weight shape: [" << linear.weight.shape[0] << ", " << linear.weight.shape[1] << "]" << std::endl;
    std::cout << "quantized: " << linear.quantized << std::endl;
    
    Tensor input({1, 64});
    std::cout << "input shape: [" << input.shape[0] << ", " << input.shape[1] << "]" << std::endl;
    
    float* d = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) d[i] = 0.1f * i;
    
    std::cout << "Calling linear.forward(input)..." << std::endl;
    Tensor output = linear.forward(input);
    
    std::cout << "output shape: [" << output.shape[0] << ", " << output.shape[1] << "]" << std::endl;
    std::cout << "Success!" << std::endl;
    return 0;
}

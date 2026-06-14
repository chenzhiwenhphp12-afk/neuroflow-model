#include <iostream>
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    std::cout << "LayerNorm class test..." << std::endl;
    
    // Create LayerNorm
    std::cout << "Creating LayerNorm(32)..." << std::endl;
    LayerNorm norm(32);
    
    std::cout << "weight shape size: " << norm.weight.shape_.size() << std::endl;
    std::cout << "weight shape[0]: " << norm.weight.shape_[0] << std::endl;
    std::cout << "weight numel: " << norm.weight.numel() << std::endl;
    std::cout << "weight data_size: " << norm.weight.data_size_ << std::endl;
    
    std::cout << "bias shape size: " << norm.bias.shape_.size() << std::endl;
    std::cout << "bias shape[0]: " << norm.bias.shape_[0] << std::endl;
    std::cout << "bias numel: " << norm.bias.numel() << std::endl;
    std::cout << "bias data_size: " << norm.bias.data_size_ << std::endl;
    
    // Create input
    Tensor input({2, 32});
    float* id = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) id[i] = 0.1f * i;
    
    std::cout << "Calling norm.forward(input)..." << std::endl;
    Tensor output = norm.forward(input);
    
    std::cout << "output shape: [" << output.shape_[0] << ", " << output.shape_[1] << "]" << std::endl;
    
    std::cout << "Success!" << std::endl;
    return 0;
}

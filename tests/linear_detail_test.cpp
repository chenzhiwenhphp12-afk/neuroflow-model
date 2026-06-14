#include <iostream>
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Linear detail test..." << std::endl;
    
    Linear linear(64, 32);
    
    Tensor input({2, 64});
    for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
    
    std::cout << "input shape: [" << input.shape_[0] << ", " << input.shape_[1] << "]" << std::endl;
    std::cout << "input numel: " << input.numel() << std::endl;
    std::cout << "input data_size: " << input.data_size_ << std::endl;
    std::cout << "input owns_data: " << input.owns_data_ << std::endl;
    
    std::cout << "Linear weight shape: [" << linear.weight.shape_[0] << ", " << linear.weight.shape_[1] << "]" << std::endl;
    std::cout << "Linear weight numel: " << linear.weight.numel() << std::endl;
    std::cout << "Linear weight data_size: " << linear.weight.data_size_ << std::endl;
    
    std::cout << "Calling linear.forward..." << std::endl;
    Tensor output = linear.forward(input);
    
    std::cout << "output shape: [" << output.shape_[0] << ", " << output.shape_[1] << "]" << std::endl;
    std::cout << "output numel: " << output.numel() << std::endl;
    std::cout << "output data_size: " << output.data_size_ << std::endl;
    std::cout << "output owns_data: " << output.owns_data_ << std::endl;
    
    // Check data pointer
    float* od = output.as_fp32();
    std::cout << "output data ptr: " << (void*)od << std::endl;
    std::cout << "First 5 output values: ";
    for (size_t i = 0; i < 5; ++i) std::cout << od[i] << " ";
    std::cout << std::endl;
    
    // Now do something with output
    std::cout << "Copying output..." << std::endl;
    Tensor copied = output.clone();
    
    std::cout << "Success!" << std::endl;
    return 0;
}

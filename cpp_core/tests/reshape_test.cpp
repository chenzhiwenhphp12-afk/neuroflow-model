#include <iostream>
#include "../include/neuroflow/tensor.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Reshape test..." << std::endl;
    
    Tensor input({1, 64});
    std::cout << "input shape: [" << input.shape[0] << ", " << input.shape[1] << "]" << std::endl;
    std::cout << "input strides: [" << input.strides[0] << ", " << input.strides[1] << "]" << std::endl;
    std::cout << "input numel: " << input.numel() << std::endl;
    std::cout << "input data_size: " << input.data_size << std::endl;
    
    // 添加数据
    float* d = input.as_fp32();
    std::cout << "Writing data..." << std::endl;
    for (size_t i = 0; i < input.numel(); ++i) d[i] = 0.1f * i;
    std::cout << "Data written" << std::endl;
    
    // reshape
    std::cout << "Calling reshape..." << std::endl;
    Tensor reshaped = input.reshape({1, 64});
    
    std::cout << "reshaped shape: [" << reshaped.shape[0] << ", " << reshaped.shape[1] << "]" << std::endl;
    std::cout << "reshaped strides: [" << reshaped.strides[0] << ", " << reshaped.strides[1] << "]" << std::endl;
    std::cout << "reshaped numel: " << reshaped.numel() << std::endl;
    std::cout << "reshaped owns_data: " << reshaped.owns_data << std::endl;
    
    // 访问 reshaped 数据
    std::cout << "Accessing reshaped data..." << std::endl;
    float* rd = reshaped.as_fp32();
    std::cout << "First 5 values: ";
    for (size_t i = 0; i < 5; ++i) std::cout << rd[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Success!" << std::endl;
    return 0;
}

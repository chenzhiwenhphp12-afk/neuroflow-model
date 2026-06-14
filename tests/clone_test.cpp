#include <iostream>
#include "../include/neuroflow/tensor.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Clone test..." << std::endl;
    
    Tensor x({2, 32});
    float* xd = x.as_fp32();
    for (size_t i = 0; i < x.numel(); ++i) xd[i] = 0.1f * i;
    
    std::cout << "Cloning..." << std::endl;
    Tensor cloned = x.clone();
    
    std::cout << "cloned shape: [" << cloned.shape_[0] << ", " << cloned.shape_[1] << "]" << std::endl;
    std::cout << "cloned numel: " << cloned.numel() << std::endl;
    std::cout << "cloned data_size: " << cloned.data_size_ << std::endl;
    
    float* cd = cloned.as_fp32();
    std::cout << "First 5 values: ";
    for (size_t i = 0; i < 5; ++i) std::cout << cd[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Success!" << std::endl;
    return 0;
}

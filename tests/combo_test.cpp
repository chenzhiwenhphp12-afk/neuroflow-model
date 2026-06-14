#include <iostream>
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Combo test..." << std::endl;
    
    Linear linear(64, 32);
    LayerNorm norm(32);
    GELU gelu;
    
    Tensor input({2, 64});
    for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
    
    std::cout << "Linear..." << std::endl;
    Tensor h1 = linear.forward(input);
    std::cout << "h1: [" << h1.shape_[0] << ", " << h1.shape_[1] << "]" << std::endl;
    
    std::cout << "LayerNorm..." << std::endl;
    Tensor h2 = norm.forward(h1);
    std::cout << "h2: [" << h2.shape_[0] << ", " << h2.shape_[1] << "]" << std::endl;
    
    std::cout << "GELU..." << std::endl;
    Tensor h3 = gelu.forward(h2);
    std::cout << "h3: [" << h3.shape_[0] << ", " << h3.shape_[1] << "]" << std::endl;
    
    std::cout << "Success!" << std::endl;
    return 0;
}

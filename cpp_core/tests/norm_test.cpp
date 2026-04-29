#include <iostream>
#include "../include/neuroflow/tensor.hpp"

using namespace neuroflow;

int main() {
    std::cout << "LayerNorm test..." << std::endl;
    
    Tensor x({2, 32});
    Tensor weight({32});
    Tensor bias({32});
    
    float* xd = x.as_fp32();
    float* wd = weight.as_fp32();
    float* bd = bias.as_fp32();
    
    for (size_t i = 0; i < x.numel(); ++i) xd[i] = 0.1f * i;
    for (size_t i = 0; i < 32; ++i) wd[i] = 1.0f;
    for (size_t i = 0; i < 32; ++i) bd[i] = 0.0f;
    
    std::cout << "x shape: [" << x.shape[0] << ", " << x.shape[1] << "]" << std::endl;
    std::cout << "x numel: " << x.numel() << std::endl;
    std::cout << "x data_size: " << x.data_size << std::endl;
    
    std::cout << "Calling layer_norm..." << std::endl;
    TensorOps::layer_norm(x, weight, bias);
    
    std::cout << "Success!" << std::endl;
    return 0;
}

#include <iostream>
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Linear + reshape test..." << std::endl;
    
    size_t d_model = 64;
    Linear W_q(d_model, d_model, false);
    
    std::cout << "W_q weight shape: [" << W_q.weight.shape[0] << ", " << W_q.weight.shape[1] << "]" << std::endl;
    
    Tensor input({1, d_model});
    std::cout << "input shape: [" << input.shape[0] << ", " << input.shape[1] << "]" << std::endl;
    
    float* d = input.as_fp32();
    std::cout << "Writing data..." << std::endl;
    for (size_t i = 0; i < input.numel(); ++i) d[i] = 0.1f * i;
    std::cout << "Data written" << std::endl;
    
    std::cout << "reshaping input..." << std::endl;
    Tensor x_flat = input.reshape({1, d_model});
    std::cout << "x_flat shape: [" << x_flat.shape[0] << ", " << x_flat.shape[1] << "]" << std::endl;
    
    std::cout << "Calling W_q.forward(x_flat)..." << std::endl;
    Tensor q = W_q.forward(x_flat);
    
    std::cout << "q shape: [" << q.shape[0] << ", " << q.shape[1] << "]" << std::endl;
    std::cout << "Success!" << std::endl;
    return 0;
}

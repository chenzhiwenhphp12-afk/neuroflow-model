#include <iostream>
#include <exception>
#include "../include/neuroflow/memory.hpp"

using namespace neuroflow;

int main() {
    try {
        std::cout << "MLA debug test..." << std::endl;
        
        LatentKVCache mla(64, 4, 16, 128);
        
        std::cout << "mla.d_model: " << mla.d_model << std::endl;
        std::cout << "mla.n_heads: " << mla.n_heads << std::endl;
        std::cout << "mla.d_latent: " << mla.d_latent << std::endl;
        std::cout << "mla.head_dim: " << mla.head_dim << std::endl;
        std::cout << "mla.cache_len: " << mla.cache_len << std::endl;
        
        Tensor input({1, 64});
        std::cout << "input shape: [" << input.shape[0] << ", " << input.shape[1] << "]" << std::endl;
        std::cout << "input numel: " << input.numel() << std::endl;
        
        for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
        
        std::cout << "Calling mla.forward(input, true)..." << std::endl;
        Tensor output1 = mla.forward(input, true);
        
        std::cout << "output1 shape size: " << output1.shape.size() << std::endl;
        for (size_t i = 0; i < output1.shape.size(); ++i) std::cout << "  dim " << i << ": " << output1.shape[i] << std::endl;
        std::cout << "mla.cache_len after 1st forward: " << mla.cache_len << std::endl;
        
        std::cout << "Second forward..." << std::endl;
        Tensor input2({1, 64});
        for (size_t i = 0; i < input2.numel(); ++i) input2.as_fp32()[i] = 0.2f * i;
        Tensor output2 = mla.forward(input2, true);
        
        std::cout << "output2 shape size: " << output2.shape.size() << std::endl;
        for (size_t i = 0; i < output2.shape.size(); ++i) std::cout << "  dim " << i << ": " << output2.shape[i] << std::endl;
        std::cout << "mla.cache_len after 2nd forward: " << mla.cache_len << std::endl;
        
        std::cout << "Success!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}

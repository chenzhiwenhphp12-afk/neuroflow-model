#include <iostream>
#include <exception>
#include "../include/neuroflow/model.hpp"

using namespace neuroflow;

int main() {
    try {
        std::cout << "Full forward test..." << std::endl;
        
        NeuroFlowModel::Config cfg;
        cfg.input_dim = 64;
        cfg.hidden_dim = 32;
        cfg.output_dim = 5;
        cfg.memory_slots = 8;
        cfg.memory_dim = 16;
        cfg.num_layers = 1;
        cfg.num_associations = 2;
        cfg.use_mla = false;
        
        NeuroFlowModel model(cfg);
        
        Tensor input({2, cfg.input_dim});
        for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
        
        std::cout << "Calling full forward..." << std::endl;
        auto output = model.forward(input, nullptr, false, false);
        
        std::cout << "Output shape: [" << output.output.shape[0] << ", " << output.output.shape[1] << "]" << std::endl;
        std::cout << "Success!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}

#include <iostream>
#include "../include/neuroflow/model.hpp"

using namespace neuroflow;

int main() {
    std::cout << "Simple model test..." << std::endl;
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 64;
    cfg.hidden_dim = 32;
    cfg.output_dim = 5;
    cfg.memory_slots = 8;
    cfg.memory_dim = 16;
    cfg.num_layers = 1;
    cfg.num_associations = 2;
    cfg.use_mla = false;  // 不使用 MLA
    
    std::cout << "Creating model..." << std::endl;
    NeuroFlowModel model(cfg);
    
    std::cout << "Creating input tensor..." << std::endl;
    Tensor input({1, cfg.input_dim});
    float* data = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = 0.1f * i;
    }
    
    std::cout << "Running forward..." << std::endl;
    auto output = model.forward(input);
    
    std::cout << "Output shape: [" << output.output.shape[0] << ", " << output.output.shape[1] << "]" << std::endl;
    std::cout << "Success!" << std::endl;
    
    return 0;
}

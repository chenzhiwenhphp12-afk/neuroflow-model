#include <iostream>
#include <exception>
#include "../include/neuroflow/model.hpp"
#include "../include/neuroflow/memory.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    try {
        std::cout << "Step by step forward test..." << std::endl;
        
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
        size_t batch = 2;
        
        Tensor x({batch, cfg.input_dim});
        for (size_t i = 0; i < x.numel(); ++i) x.as_fp32()[i] = 0.1f * i;
        
        std::cout << "1. Input projection..." << std::endl;
        Tensor h = model.input_proj_linear->forward(x);
        std::cout << "  h shape: [" << h.shape[0] << ", " << h.shape[1] << "]" << std::endl;
        h = model.input_proj_norm->forward(h);
        h = model.input_proj_gelu->forward(h);
        std::cout << "  after norm/gelu: [" << h.shape[0] << ", " << h.shape[1] << "]" << std::endl;
        
        std::cout << "2. SN forward..." << std::endl;
        auto sn_out = model.sn->forward(h);
        std::cout << "  saliency: [" << sn_out.saliency.shape[0] << ", " << sn_out.saliency.shape[1] << "]" << std::endl;
        std::cout << "  gates: [" << sn_out.gates.shape[0] << ", " << sn_out.gates.shape[1] << "]" << std::endl;
        
        std::cout << "3. ECN forward..." << std::endl;
        auto ecn_out = model.ecn->forward(h);
        std::cout << "  decision: [" << ecn_out.decision.shape[0] << ", " << ecn_out.decision.shape[1] << "]" << std::endl;
        std::cout << "  value: [" << ecn_out.value.shape[0] << ", " << ecn_out.value.shape[1] << "]" << std::endl;
        
        std::cout << "4. Memory encode..." << std::endl;
        Tensor mem_seed = model.memory->encode(h);
        std::cout << "  mem_seed: [" << mem_seed.shape[0] << ", " << mem_seed.shape[1] << "]" << std::endl;
        
        std::cout << "5. DMN forward..." << std::endl;
        auto dmn_out = model.dmn->forward(mem_seed);
        std::cout << "  vision shape size: " << dmn_out.vision.shape.size() << std::endl;
        for (size_t i = 0; i < dmn_out.vision.shape.size(); ++i) 
            std::cout << "    dim " << i << ": " << dmn_out.vision.shape[i] << std::endl;
        
        std::cout << "6. Memory retrieve..." << std::endl;
        auto mem_out = model.memory->forward(h);
        std::cout << "  retrieved: [" << mem_out.retrieved.shape[0] << ", " << mem_out.retrieved.shape[1] << "]" << std::endl;
        
        std::cout << "7. Reshaping dmn_out.vision..." << std::endl;
        std::cout << "  vision numel: " << dmn_out.vision.numel() << std::endl;
        std::cout << "  trying reshape to [" << batch << ", " << dmn_out.vision.shape[1] << "]" << std::endl;
        std::cout << "  expected numel: " << (batch * dmn_out.vision.shape[1]) << std::endl;
        
        if (dmn_out.vision.numel() != batch * dmn_out.vision.shape[1]) {
            std::cout << "  MISMATCH! vision actual shape may be different" << std::endl;
        }
        
        Tensor dmn_weighted = dmn_out.vision.reshape({batch, dmn_out.vision.shape[1]});
        std::cout << "  reshape success" << std::endl;
        
        std::cout << "All steps passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}

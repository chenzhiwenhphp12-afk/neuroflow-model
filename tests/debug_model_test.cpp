#include <iostream>
#include <exception>
#include "../include/neuroflow/model.hpp"
#include "../include/neuroflow/memory.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

void test_ecn() {
    std::cout << "Testing ECN alone..." << std::endl;
    ExecutiveControlNetwork ecn(32, 32, 5, 1);
    Tensor input({2, 32});
    for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
    
    auto output = ecn.forward(input);
    std::cout << "ECN decision shape: [" << output.decision.shape_[0] << ", " << output.decision.shape_[1] << "]" << std::endl;
    std::cout << "ECN passed!" << std::endl;
}

void test_dmn() {
    std::cout << "Testing DMN alone..." << std::endl;
    DefaultModeNetwork dmn(16, 16, 2);
    Tensor input({2, 16});
    for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
    
    auto output = dmn.forward(input);
    std::cout << "DMN vision shape size: " << output.vision.shape_.size() << std::endl;
    for (size_t i = 0; i < output.vision.shape_.size(); ++i) std::cout << "  dim " << i << ": " << output.vision.shape_[i] << std::endl;
    std::cout << "DMN passed!" << std::endl;
}

void test_sn() {
    std::cout << "Testing SN alone..." << std::endl;
    SalienceNetwork sn(32, 16);
    Tensor input({2, 32});
    for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
    
    auto output = sn.forward(input);
    std::cout << "SN saliency shape: [" << output.saliency.shape_[0] << ", " << output.saliency.shape_[1] << "]" << std::endl;
    std::cout << "SN gates shape: [" << output.gates.shape_[0] << ", " << output.gates.shape_[1] << "]" << std::endl;
    std::cout << "SN passed!" << std::endl;
}

void test_memory() {
    std::cout << "Testing Memory alone..." << std::endl;
    MemoryConsolidationModule memory(32, 8, 16);
    Tensor input({2, 32});
    for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
    
    auto output = memory.forward(input);
    std::cout << "Memory retrieved shape: [" << output.retrieved.shape_[0] << ", " << output.retrieved.shape_[1] << "]" << std::endl;
    std::cout << "Memory passed!" << std::endl;
}

int main() {
    try {
        test_ecn();
        test_dmn();
        test_sn();
        test_memory();
        
        std::cout << "\nAll component tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

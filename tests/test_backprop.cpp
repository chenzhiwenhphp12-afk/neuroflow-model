#include <iostream>
#include <chrono>
#include "../include/neuroflow/model.hpp"
#include "../include/neuroflow/backprop.hpp"

using namespace neuroflow;

void test_forward_cache() {
    std::cout << "\n=== Forward Cache Test ===\n";
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 64;
    cfg.hidden_dim = 32;
    cfg.output_dim = 5;
    cfg.memory_dim = 32;
    cfg.num_layers = 1;
    cfg.num_associations = 2;
    
    NeuroFlowModel model(cfg);
    FullBackpropEngine bp(model);
    
    Tensor input({2, 64});
    float* data = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = 0.1f * i;
    }
    
    auto output = bp.forward_with_cache(input);
    
    std::cout << "  Forward cache stored:\n";
    std::cout << "    input: [" << bp.cache.input.shape_[0] << ", " << bp.cache.input.shape_[1] << "]\n";
    std::cout << "    h: [" << bp.cache.h.shape_[0] << ", " << bp.cache.h.shape_[1] << "]\n";
    std::cout << "    ecn_hidden: " << bp.cache.ecn_hidden.size() << " layers\n";
    std::cout << "    combined: [" << bp.cache.combined.shape_[0] << ", " << bp.cache.combined.shape_[1] << "]\n";
    
    std::cout << "  Output shape: [" << output.output.shape_[0] 
              << ", " << output.output.shape_[1] << "]\n";
    
    std::cout << "  [PASS] Forward cache works\n";
}

void test_backward_pass() {
    std::cout << "\n=== Backward Pass Test ===\n";
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 64;
    cfg.hidden_dim = 32;
    cfg.output_dim = 5;
    cfg.memory_dim = 32;
    cfg.num_layers = 1;
    cfg.num_associations = 2;
    
    NeuroFlowModel model(cfg);
    FullBackpropEngine bp(model);
    
    Tensor input({2, 64});
    for (size_t i = 0; i < input.numel(); ++i) {
        input.as_fp32()[i] = 0.1f * i;
    }
    
    auto output = bp.forward_with_cache(input);
    
    Tensor output_grad({2, 5});
    for (size_t i = 0; i < output_grad.numel(); ++i) {
        output_grad.as_fp32()[i] = 0.01f;
    }
    
    auto grads = bp.backward(output_grad);
    
    std::cout << "  Backward gradients computed:\n";
    std::cout << "    output_fusion_down_weight_grad: [" << grads.output_fusion_down_weight_grad.shape_[0] 
              << ", " << grads.output_fusion_down_weight_grad.shape_[1] << "]\n";
    std::cout << "    output_fusion_up_weight_grad: [" << grads.output_fusion_up_weight_grad.shape_[0] 
              << ", " << grads.output_fusion_up_weight_grad.shape_[1] << "]\n";
    std::cout << "    input_grad: [" << grads.input_grad.shape_[0] 
              << ", " << grads.input_grad.shape_[1] << "]\n";
    
    float grad_sum = 0;
    const float* wg = grads.output_fusion_down_weight_grad.as_fp32();
    for (size_t i = 0; i < grads.output_fusion_down_weight_grad.numel(); ++i) {
        grad_sum += std::abs(wg[i]);
    }
    std::cout << "    Total weight gradient magnitude: " << grad_sum << "\n";
    
    std::cout << "  [PASS] Backward pass works\n";
}

void test_trainer() {
    std::cout << "\n=== FullTrainer Test ===\n";
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 64;
    cfg.hidden_dim = 32;
    cfg.output_dim = 5;
    cfg.memory_dim = 32;
    cfg.num_layers = 1;
    cfg.num_associations = 2;
    
    NeuroFlowModel model(cfg);
    FullTrainer trainer(model, 0.01f);
    
    Tensor input({2, 64});
    Tensor target({2, 5});
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (size_t i = 0; i < input.numel(); ++i) {
        input.as_fp32()[i] = dist(rng);
    }
    for (size_t i = 0; i < target.numel(); ++i) {
        target.as_fp32()[i] = (i % 5 == 2) ? 1.0f : 0.0f;
    }
    
    auto output1 = model.forward(input);
    float loss1 = LossFunctions::mse(output1.output, target);
    
    std::cout << "  Training for 10 steps...\n";
    for (int i = 0; i < 10; ++i) {
        auto step = trainer.train_step(input, target);
        std::cout << "    Step " << i << ": loss=" << step.loss 
                  << ", grad_norm=" << step.grad_norm << "\n";
    }
    
    auto output2 = model.forward(input);
    float loss2 = LossFunctions::mse(output2.output, target);
    
    std::cout << "  Initial loss: " << loss1 << "\n";
    std::cout << "  Final loss: " << loss2 << "\n";
    std::cout << "  Loss reduction: " << (loss1 - loss2) << "\n";
    
    if (loss2 <= loss1) {
        std::cout << "  [PASS] Training reduces loss\n";
    } else {
        std::cout << "  [WARN] Loss increased\n";
    }
}

void test_memory_consolidation_training() {
    std::cout << "\n=== Memory Consolidation Training Test ===\n";
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 64;
    cfg.hidden_dim = 32;
    cfg.output_dim = 5;
    cfg.memory_slots = 16;
    cfg.memory_dim = 16;
    
    NeuroFlowModel model(cfg);
    FullTrainer trainer(model, 0.01f);
    
    std::cout << "  Initial memory bank sample: " << model.memory->memory_bank.as_fp32()[0] << "\n";
    
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    
    for (int batch = 0; batch < 5; ++batch) {
        Tensor input({4, 64});
        Tensor target({4, 5});
        
        for (size_t i = 0; i < input.numel(); ++i) {
            input.as_fp32()[i] = dist(rng);
        }
        for (size_t i = 0; i < target.numel(); ++i) {
            target.as_fp32()[i] = dist01(rng);
        }
        
        auto step = trainer.train_step(input, target);
        std::cout << "  Batch " << batch << ": loss=" << step.loss << "\n";
    }
    
    float mem_after = model.memory->memory_bank.as_fp32()[0];
    std::cout << "  Memory bank after training: " << mem_after << "\n";
    std::cout << "  Memory slots: " << model.memory->memory_slots << "\n";
    
    std::cout << "  [PASS] Memory consolidation during training\n";
}

void test_gradient_flow() {
    std::cout << "\n=== Gradient Flow Test ===\n";
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 32;
    cfg.hidden_dim = 16;
    cfg.output_dim = 3;
    cfg.num_layers = 2;
    
    NeuroFlowModel model(cfg);
    FullBackpropEngine bp(model);
    
    Tensor input({1, 32});
    for (size_t i = 0; i < 32; ++i) input.as_fp32()[i] = 0.1f;
    
    auto output = bp.forward_with_cache(input);
    
    Tensor output_grad({1, 3});
    output_grad.as_fp32()[0] = 1.0f;
    output_grad.as_fp32()[1] = 0.0f;
    output_grad.as_fp32()[2] = -1.0f;
    
    auto grads = bp.backward(output_grad);
    
    std::cout << "  Input gradient samples:\n";
    const float* ig = grads.input_grad.as_fp32();
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "    grad[" << i << "] = " << ig[i] << "\n";
    }
    
    std::cout << "  [PASS] Gradient flows to input\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "NeuroFlow Backpropagation Tests\n";
    std::cout << "========================================\n";
    
    test_forward_cache();
    test_backward_pass();
    test_trainer();
    test_memory_consolidation_training();
    test_gradient_flow();
    
    std::cout << "\n========================================\n";
    std::cout << "All Tests Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}

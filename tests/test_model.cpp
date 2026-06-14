/**
 * NeuroFlow Core Tests - Model Tests
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include "../include/neuroflow/model.hpp"
#include "../include/neuroflow/memory.hpp"

using namespace neuroflow;

void test_model_creation() {
    std::cout << "Testing model creation..." << std::endl;
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 512;
    cfg.hidden_dim = 256;
    cfg.output_dim = 10;
    
    NeuroFlowModel model(cfg);
    
    auto stats = model.get_stats();
    std::cout << "  Total params: " << stats.total_params << std::endl;
    std::cout << "  Memory (MB): " << stats.memory_bytes / 1024.0 / 1024.0 << std::endl;
    
    assert(stats.total_params > 0);
    
    std::cout << "  PASSED: model creation" << std::endl;
}

void test_forward_pass() {
    std::cout << "Testing forward pass..." << std::endl;
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 128;
    cfg.hidden_dim = 64;
    cfg.output_dim = 5;
    cfg.memory_slots = 16;
    cfg.memory_dim = 32;
    cfg.num_layers = 1;
    cfg.num_associations = 4;
    
    NeuroFlowModel model(cfg);
    
    // 创建输入
    Tensor input({2, cfg.input_dim});
    float* data = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    
    // 前向传播
    auto output = model.forward(input, nullptr, false, false);
    
    assert(output.output.shape_[0] == 2);
    assert(output.output.shape_[1] == cfg.output_dim);
    assert(output.decision.shape_[1] == cfg.output_dim);
    assert(output.value.shape_[1] == 1);
    
    std::cout << "  Output shape: [" << output.output.shape_[0] << ", " << output.output.shape_[1] << "]" << std::endl;
    std::cout << "  PASSED: forward pass" << std::endl;
}

void test_forward_with_manifold() {
    std::cout << "Testing forward with manifold..." << std::endl;
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 128;
    cfg.hidden_dim = 64;
    cfg.output_dim = 5;
    
    NeuroFlowModel model(cfg);
    
    Tensor input({1, cfg.input_dim});
    auto output = model.forward(input, nullptr, false, true);
    
    assert(output.manifold.shape_[0] == 1);
    assert(output.manifold.shape_[1] == 32);
    
    std::cout << "  Manifold shape: [" << output.manifold.shape_[0] << ", " << output.manifold.shape_[1] << "]" << std::endl;
    std::cout << "  PASSED: forward with manifold" << std::endl;
}

void test_manifold_trajectory() {
    std::cout << "Testing manifold trajectory..." << std::endl;
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 128;
    cfg.hidden_dim = 64;
    cfg.output_dim = 5;
    
    NeuroFlowModel model(cfg);
    
    Tensor input({1, cfg.input_dim});
    auto trajectory = model.get_manifold_trajectory(input, 5);
    
    assert(trajectory.size() == 5);
    for (auto& t : trajectory) {
        assert(t.shape_[0] == 1);
        assert(t.shape_[1] == 32);
    }
    
    std::cout << "  Trajectory length: " << trajectory.size() << std::endl;
    std::cout << "  PASSED: manifold trajectory" << std::endl;
}

void test_memory_module() {
    std::cout << "Testing memory module..." << std::endl;
    
    MemoryConsolidationModule memory(64, 16, 32);
    
    // 编码
    Tensor input({2, 64});
    float* data = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    
    Tensor encoded = memory.encode(input);
    assert(encoded.shape_[1] == 32);
    
    // 检索
    auto result = memory.retrieve(input);
    assert(result.retrieved.shape_[1] == 64);
    assert(result.attention.shape_[1] == 16);
    
    std::cout << "  Memory slots: " << memory.memory_slots << std::endl;
    std::cout << "  PASSED: memory module" << std::endl;
}

void test_memory_consolidation() {
    std::cout << "Testing memory consolidation..." << std::endl;
    
    MemoryConsolidationModule memory(64, 16, 32, 0.1f);
    
    // 多次巩固
    for (int i = 0; i < 5; ++i) {
        Tensor input({1, 64});
        float* data = input.as_fp32();
        for (size_t j = 0; j < input.numel(); ++j) {
            data[j] = static_cast<float>(std::rand()) / RAND_MAX;
        }
        memory.consolidate(input);
    }
    
    std::cout << "  PASSED: memory consolidation" << std::endl;
}

void test_mla_cache() {
    std::cout << "Testing MLA (Latent KV Cache)..." << std::endl;
    
    LatentKVCache mla(64, 4, 16, 128);  // model_dim=64, heads=4, latent=16
    
    // 第一次前向
    Tensor input({1, 64});
    float* data = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    
    Tensor output1 = mla.forward(input, true);
    size_t cache1 = mla.cache_len;
    
    // 第二次前向 (cache应该增长)
    Tensor input2({1, 64});
    Tensor output2 = mla.forward(input2, true);
    size_t cache2 = mla.cache_len;
    
    assert(cache2 >= cache1);
    
    // 检查内存节省比例
    float saving = mla.memory_saving_ratio();
    std::cout << "  MLA memory saving: " << saving * 100 << "%" << std::endl;
    std::cout << "  Cache size: " << mla.cache_size_bytes() << " bytes" << std::endl;
    
    // MLA应该节省至少50%
    assert(saving > 0.5f);
    
    std::cout << "  PASSED: MLA cache" << std::endl;
}

void test_quantized_model() {
    std::cout << "Testing quantized model..." << std::endl;
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 128;
    cfg.hidden_dim = 64;
    cfg.output_dim = 5;
    cfg.use_quantization = true;
    
    NeuroFlowModel model(cfg);
    
    auto stats = model.get_stats();
    std::cout << "  Quantization ratio: " << stats.quantization_ratio * 100 << "%" << std::endl;
    
    // 前向传播应该仍然工作
    Tensor input({2, cfg.input_dim});
    auto output = model.forward(input);
    
    assert(output.output.shape_[1] == cfg.output_dim);
    
    std::cout << "  PASSED: quantized model" << std::endl;
}

void test_performance_comparison() {
    std::cout << "Testing performance comparison..." << std::endl;
    
    NeuroFlowModel::Config orig_cfg;
    orig_cfg.input_dim = 512;
    orig_cfg.hidden_dim = 256;
    orig_cfg.output_dim = 10;
    
    NeuroFlowModel original(orig_cfg);
    
    NeuroFlowModel::Config lite_cfg;
    lite_cfg.input_dim = 512;
    lite_cfg.hidden_dim = 128;
    lite_cfg.output_dim = 10;
    lite_cfg.memory_dim = 64;
    lite_cfg.memory_slots = 32;
    lite_cfg.num_layers = 1;
    lite_cfg.num_associations = 4;
    lite_cfg.use_quantization = true;
    
    NeuroFlowModel lite(lite_cfg);
    
    auto orig_stats = original.get_stats();
    auto lite_stats = lite.get_stats();
    
    std::cout << "  Original params: " << orig_stats.total_params << std::endl;
    std::cout << "  Lite params: " << lite_stats.total_params << std::endl;
    std::cout << "  Size reduction: " << (1.0 - static_cast<double>(lite_stats.total_params) / orig_stats.total_params) * 100 << "%" << std::endl;
    
    // 性能测试
    Tensor input({32, 512});
    float* data = input.as_fp32();
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    
    // 预热
    original.forward(input);
    lite.forward(input);
    
    // 原始模型
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        original.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto orig_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 10;
    
    // Lite模型
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        lite.forward(input);
    }
    end = std::chrono::high_resolution_clock::now();
    auto lite_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 10;
    
    std::cout << "  Original time: " << orig_time << " ms" << std::endl;
    std::cout << "  Lite time: " << lite_time << " ms" << std::endl;
    std::cout << "  Speedup: " << orig_time / lite_time << "x" << std::endl;
    
    std::cout << "  PASSED: performance comparison" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "NeuroFlow Core - Model Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_model_creation();
    test_forward_pass();
    test_forward_with_manifold();
    test_manifold_trajectory();
    test_memory_module();
    test_memory_consolidation();
    test_mla_cache();
    test_quantized_model();
    test_performance_comparison();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
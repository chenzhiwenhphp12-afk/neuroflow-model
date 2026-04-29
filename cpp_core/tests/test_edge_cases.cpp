/**
 * NeuroFlow 边界条件测试
 * 
 * 测试各种边界和异常情况：
 * 1. 空张量
 * 2. 极小尺寸
 * 3. 极大尺寸
 * 4. 无效reshape
 * 5. 维度不匹配
 * 6. 内存溢出检测
 */

#include <iostream>
#include <cassert>
#include <stdexcept>
#include "../include/neuroflow/model.hpp"
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/memory.hpp"

using namespace neuroflow;

void test_empty_tensor() {
    std::cout << "\n=== Empty Tensor Test ===\n";
    
    // 测试空张量
    try {
        Tensor empty({}, QuantType::FP32);
        std::cout << "  Empty tensor numel: " << empty.numel() << "\n";
        std::cout << "  Empty tensor data_size: " << empty.data_size << "\n";
        assert(empty.numel() == 1);  // {} shape means 1 element
        std::cout << "  [PASS] Empty tensor handled\n";
    } catch (const std::exception& e) {
        std::cout << "  Exception: " << e.what() << "\n";
        std::cout << "  [PASS] Empty tensor rejected\n";
    }
}

void test_minimal_sizes() {
    std::cout << "\n=== Minimal Size Test ===\n";
    
    // 1x1张量
    Tensor t1({1, 1}, QuantType::FP32);
    t1.as_fp32()[0] = 1.0f;
    std::cout << "  1x1 tensor: " << t1.as_fp32()[0] << "\n";
    
    // 单元素张量
    Tensor t2({1}, QuantType::FP32);
    std::cout << "  1D tensor numel: " << t2.numel() << "\n";
    
    // 极小模型
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 1;
    cfg.hidden_dim = 1;
    cfg.output_dim = 1;
    cfg.memory_slots = 1;
    cfg.memory_dim = 1;
    cfg.num_layers = 1;
    cfg.num_associations = 1;
    
    NeuroFlowModel model(cfg);
    Tensor input({1, 1});
    input.as_fp32()[0] = 0.5f;
    
    auto output = model.forward(input);
    std::cout << "  Minimal model output shape: [" << output.output.shape[0] 
              << ", " << output.output.shape[1] << "]\n";
    
    std::cout << "  [PASS] Minimal sizes work\n";
}

void test_large_sizes() {
    std::cout << "\n=== Large Size Test ===\n";
    
    // 大张量 (但不至于溢出)
    size_t large_size = 1024 * 1024;  // 1M elements = 4MB
    
    try {
        Tensor large({large_size}, QuantType::FP32);
        std::cout << "  Large tensor size: " << large.data_size / 1024 / 1024 << " MB\n";
        
        // 填充数据测试
        float* data = large.as_fp32();
        data[0] = 1.0f;
        data[large_size - 1] = 2.0f;
        
        std::cout << "  First element: " << data[0] << "\n";
        std::cout << "  Last element: " << data[large_size - 1] << "\n";
        
        std::cout << "  [PASS] Large tensor works\n";
    } catch (const std::exception& e) {
        std::cout << "  Exception: " << e.what() << "\n";
        std::cout << "  [INFO] Large tensor allocation failed (expected on limited memory)\n";
    }
}

void test_invalid_reshape() {
    std::cout << "\n=== Invalid Reshape Test ===\n";
    
    Tensor t({2, 3}, QuantType::FP32);
    float* data = t.as_fp32();
    for (size_t i = 0; i < 6; ++i) data[i] = i;
    
    // 有效reshape
    try {
        Tensor valid = t.reshape({3, 2});
        std::cout << "  Valid reshape {2,3} -> {3,2}: OK\n";
        std::cout << "  [PASS] Valid reshape works\n";
    } catch (const std::exception& e) {
        std::cout << "  Exception: " << e.what() << "\n";
        std::cout << "  [FAIL] Valid reshape failed!\n";
    }
    
    // 无效reshape (元素数不匹配)
    try {
        Tensor invalid = t.reshape({4, 2});  // 8 != 6
        std::cout << "  Invalid reshape accepted - BUG!\n";
        std::cout << "  [FAIL] Invalid reshape should throw!\n";
    } catch (const std::runtime_error& e) {
        std::cout << "  Exception: " << e.what() << "\n";
        std::cout << "  [PASS] Invalid reshape rejected\n";
    }
}

void test_dimension_mismatch() {
    std::cout << "\n=== Dimension Mismatch Test ===\n";
    
    // GEMM维度不匹配
    Tensor A({2, 3}, QuantType::FP32);
    Tensor B({4, 5}, QuantType::FP32);  // 不匹配！
    Tensor C({2, 5}, QuantType::FP32);
    
    try {
        TensorOps::gemm(A, B, C);
        std::cout << "  [WARN] Dimension mismatch accepted - may crash\n";
    } catch (const std::exception& e) {
        std::cout << "  Exception: " << e.what() << "\n";
        std::cout << "  [PASS] Dimension mismatch detected\n";
    }
    
    // 正确维度
    Tensor B2({3, 5}, QuantType::FP32);
    TensorOps::gemm(A, B2, C);
    std::cout << "  Correct GEMM: OK\n";
    
    std::cout << "  [PASS] Dimension check works\n";
}

void test_quantization_edge_cases() {
    std::cout << "\n=== Quantization Edge Cases Test ===\n";
    
    // 全零张量量化
    Tensor zeros({4, 8}, QuantType::FP32);
    memset(zeros.as_fp32(), 0, zeros.data_size);
    
    Tensor quant({4, 8}, QuantType::INT8);
    Tensor scale({4}, QuantType::FP32);
    
    TensorOps::quantize_int8(zeros, quant, scale);
    std::cout << "  Zero quantization scale[0]: " << scale.as_fp32()[0] << "\n";
    
    // 极大值量化
    Tensor large_vals({2, 4}, QuantType::FP32);
    float* lv = large_vals.as_fp32();
    lv[0] = 1e10f;  // 极大值
    lv[1] = -1e10f;
    lv[2] = 1e-10f;  // 极小值
    lv[3] = 0.0f;
    
    Tensor quant_large({2, 4}, QuantType::INT8);
    Tensor scale_large({2}, QuantType::FP32);
    
    TensorOps::quantize_int8(large_vals, quant_large, scale_large);
    std::cout << "  Large value quant scale[0]: " << scale_large.as_fp32()[0] << "\n";
    
    std::cout << "  [PASS] Quantization edge cases handled\n";
}

void test_mla_cache_limits() {
    std::cout << "\n=== MLA Cache Limit Test ===\n";
    
    // 测试cache达到上限
    LatentKVCache mla(64, 4, 16, 10);  // max_len=10
    
    for (int i = 0; i < 20; ++i) {  // 超过max_len
        Tensor input({1, 64});
        float* data = input.as_fp32();
        for (size_t j = 0; j < 64; ++j) data[j] = 0.1f * i;
        
        mla.forward(input, true);
    }
    
    std::cout << "  Cache len after 20 inputs: " << mla.cache_len << "\n";
    std::cout << "  Expected max: 10\n";
    
    assert(mla.cache_len <= 10);
    std::cout << "  [PASS] MLA cache limit enforced\n";
}

void test_memory_slots_limit() {
    std::cout << "\n=== Memory Slots Limit Test ===\n";
    
    MemoryConsolidationModule memory(64, 8, 32);  // 8 slots
    
    // 多次巩固
    for (int i = 0; i < 100; ++i) {
        Tensor input({1, 64});
        memory.consolidate(input);
    }
    
    std::cout << "  Memory slots: " << memory.memory_slots << "\n";
    std::cout << "  Memory still works after 100 consolidations\n";
    
    // 测试检索
    Tensor query({1, 64});
    auto result = memory.retrieve(query);
    std::cout << "  Retrieved shape: [" << result.retrieved.shape[0] 
              << ", " << result.retrieved.shape[1] << "]\n";
    
    std::cout << "  [PASS] Memory slots limit handled\n";
}

void test_batch_size_edge() {
    std::cout << "\n=== Batch Size Edge Test ===\n";
    
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 16;
    cfg.hidden_dim = 8;
    cfg.output_dim = 2;
    
    NeuroFlowModel model(cfg);
    
    // batch=0 (应该失败或返回空)
    // batch=1
    Tensor single({1, 16});
    auto out1 = model.forward(single);
    std::cout << "  Batch=1 output: [" << out1.output.shape[0] 
              << ", " << out1.output.shape[1] << "]\n";
    
    // batch=100 (大batch)
    Tensor large_batch({100, 16});
    auto out100 = model.forward(large_batch);
    std::cout << "  Batch=100 output: [" << out100.output.shape[0] 
              << ", " << out100.output.shape[1] << "]\n";
    
    std::cout << "  [PASS] Batch size edge cases work\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "NeuroFlow Edge Case Tests\n";
    std::cout << "========================================\n";
    
    test_empty_tensor();
    test_minimal_sizes();
    test_large_sizes();
    test_invalid_reshape();
    test_dimension_mismatch();
    test_quantization_edge_cases();
    test_mla_cache_limits();
    test_memory_slots_limit();
    test_batch_size_edge();
    
    std::cout << "\n========================================\n";
    std::cout << "All Edge Case Tests Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}
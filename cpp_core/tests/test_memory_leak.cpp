/**
 * NeuroFlow 内存泄漏检测测试
 * 
 * 使用简单的方法检测内存泄漏：
 * 1. 运行大量迭代测试
 * 2. 检查内存使用变化
 * 3. 验证对象生命周期
 */

#include <iostream>
#include <chrono>
#include <vector>
#include "../include/neuroflow/model.hpp"
#include "../include/neuroflow/memory.hpp"

using namespace neuroflow;

// 内存统计
size_t get_current_memory_mb() {
    // 使用简单方法估算
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    
    char line[256];
    size_t vmrss = 0;
    while (fgets(line, 256, f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &vmrss);
            break;
        }
    }
    fclose(f);
    return vmrss;  // KB
}

void test_tensor_memory_leak() {
    std::cout << "\n=== Tensor Memory Leak Test ===\n";
    
    size_t mem_before = get_current_memory_mb();
    
    // 创建和销毁大量Tensor
    for (int i = 0; i < 10000; ++i) {
        Tensor t({256, 512}, QuantType::FP32);
        Tensor t2 = t.clone();
        Tensor t3 = t.reshape({128, 1024});
    }
    
    size_t mem_after = get_current_memory_mb();
    
    ssize_t mem_change = static_cast<ssize_t>(mem_after) - static_cast<ssize_t>(mem_before);
    
    std::cout << "  Memory before: " << mem_before << " KB\n";
    std::cout << "  Memory after: " << mem_after << " KB\n";
    std::cout << "  Memory change: " << mem_change << " KB\n";
    
    // 内存变化应该很小（< 1MB），因为对象都被正确释放
    if (mem_after - mem_before < 1024) {
        std::cout << "  [PASS] No significant memory leak detected\n";
    } else {
        std::cout << "  [WARN] Possible memory leak\n";
    }
}

void test_model_memory_leak() {
    std::cout << "\n=== Model Memory Leak Test ===\n";
    
    size_t mem_before = get_current_memory_mb();
    
    // 创建和销毁大量模型
    for (int i = 0; i < 100; ++i) {
        NeuroFlowModel::Config cfg;
        cfg.input_dim = 128;
        cfg.hidden_dim = 64;
        cfg.output_dim = 5;
        
        NeuroFlowModel model(cfg);
        
        // 执行forward
        Tensor input({2, 128});
        auto output = model.forward(input);
        
        // 执行manifold trajectory
        auto trajectory = model.get_manifold_trajectory(input, 5);
    }
    
    size_t mem_after = get_current_memory_mb();
    
    ssize_t mem_change = static_cast<ssize_t>(mem_after) - static_cast<ssize_t>(mem_before);
    
    std::cout << "  Memory before: " << mem_before << " KB\n";
    std::cout << "  Memory after: " << mem_after << " KB\n";
    std::cout << "  Memory change: " << mem_change << " KB\n";
    
    if (mem_after - mem_before < 2048) {
        std::cout << "  [PASS] No significant memory leak detected\n";
    } else {
        std::cout << "  [WARN] Possible memory leak\n";
    }
}

void test_mla_cache_memory() {
    std::cout << "\n=== MLA Cache Memory Test ===\n";
    
    size_t mem_before = get_current_memory_mb();
    
    // 测试MLA cache
    LatentKVCache mla(64, 4, 16, 128);
    
    for (int i = 0; i < 1000; ++i) {
        Tensor input({1, 64});
        float* data = input.as_fp32();
        for (size_t j = 0; j < 64; ++j) data[j] = 0.1f * j;
        
        mla.forward(input, true);
        
        if (i % 100 == 0) {
            mla.clear_cache();
        }
    }
    
    size_t mem_after = get_current_memory_mb();
    
    ssize_t mem_change = static_cast<ssize_t>(mem_after) - static_cast<ssize_t>(mem_before);
    
    std::cout << "  Memory before: " << mem_before << " KB\n";
    std::cout << "  Memory after: " << mem_after << " KB\n";
    std::cout << "  Memory change: " << mem_change << " KB\n";
    
    if (mem_after - mem_before < 512) {
        std::cout << "  [PASS] MLA cache memory management OK\n";
    } else {
        std::cout << "  [WARN] MLA cache may have memory issues\n";
    }
}

void test_memory_consolidation() {
    std::cout << "\n=== Memory Consolidation Test ===\n";
    
    size_t mem_before = get_current_memory_mb();
    
    MemoryConsolidationModule memory(64, 16, 32);
    
    for (int i = 0; i < 1000; ++i) {
        Tensor input({1, 64});
        memory.consolidate(input);
        
        auto result = memory.retrieve(input);
    }
    
    size_t mem_after = get_current_memory_mb();
    
    ssize_t mem_change = static_cast<ssize_t>(mem_after) - static_cast<ssize_t>(mem_before);
    
    std::cout << "  Memory before: " << mem_before << " KB\n";
    std::cout << "  Memory after: " << mem_after << " KB\n";
    std::cout << "  Memory change: " << mem_change << " KB\n";
    
    if (mem_after - mem_before < 256) {
        std::cout << "  [PASS] Memory consolidation OK\n";
    } else {
        std::cout << "  [WARN] Memory consolidation may leak\n";
    }
}

void test_shared_ptr_cycle() {
    std::cout << "\n=== Shared Pointer Cycle Test ===\n";
    
    size_t mem_before = get_current_memory_mb();
    
    // 测试shared_ptr是否有循环引用
    for (int i = 0; i < 1000; ++i) {
        NeuroFlowModel::Config cfg;
        NeuroFlowModel model(cfg);
        
        // 内部的shared_ptr应该正确管理
        auto stats = model.get_stats();
    }
    
    size_t mem_after = get_current_memory_mb();
    
    ssize_t mem_change = static_cast<ssize_t>(mem_after) - static_cast<ssize_t>(mem_before);
    
    std::cout << "  Memory before: " << mem_before << " KB\n";
    std::cout << "  Memory after: " << mem_after << " KB\n";
    std::cout << "  Memory change: " << mem_change << " KB\n";
    
    if (std::abs(mem_change) < 512) {
        std::cout << "  [PASS] No shared_ptr cycle detected\n";
    } else if (mem_change > 0) {
        std::cout << "  [WARN] Possible shared_ptr cycle\n";
    } else {
        std::cout << "  [PASS] Memory properly released\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "NeuroFlow Memory Leak Detection Tests\n";
    std::cout << "========================================\n";
    
    test_tensor_memory_leak();
    test_model_memory_leak();
    test_mla_cache_memory();
    test_memory_consolidation();
    test_shared_ptr_cycle();
    
    std::cout << "\n========================================\n";
    std::cout << "Memory Leak Tests Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}
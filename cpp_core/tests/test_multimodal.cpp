/**
 * NeuroFlow MultiModal Tests
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include "../include/neuroflow/multimodal.hpp"
#include "../include/neuroflow/multimodal_model.hpp"

using namespace neuroflow;

void test_patch_embedding() {
    std::cout << "Testing PatchEmbedding..." << std::endl;
    
    PatchEmbedding patch_emb(64, 8, 3, 64);  // 64x64 image, 8x8 patch
    
    std::cout << "  num_patches: " << patch_emb.num_patches << std::endl;
    assert(patch_emb.num_patches == (64/8) * (64/8)); // 64 patches
    
    // 创建模拟图像数据
    Tensor image({2, 3, 64, 64});
    float* img = image.as_fp32();
    for (size_t i = 0; i < image.numel(); ++i) img[i] = 0.1f * i;
    
    Tensor embedded = patch_emb.forward(image);
    std::cout << "  embedded shape: [" << embedded.shape[0] << ", " 
              << embedded.shape[1] << ", " << embedded.shape[2] << "]" << std::endl;
    
    assert(embedded.shape[0] == 2);  // batch
    assert(embedded.shape[1] == 64); // num_patches
    assert(embedded.shape[2] == 64); // embed_dim
    
    std::cout << "  PASSED: PatchEmbedding" << std::endl;
}

void test_vision_encoder() {
    std::cout << "Testing VisionEncoder..." << std::endl;
    
    VisionEncoder encoder(64, 8, 64, 4, 2);  // 小尺寸测试
    
    std::cout << "  embed_dim: " << encoder.embed_dim << std::endl;
    std::cout << "  num_layers: " << encoder.num_layers << std::endl;
    
    // 创建模拟图像
    Tensor image({2, 3, 64, 64});
    float* img = image.as_fp32();
    for (size_t i = 0; i < image.numel(); ++i) img[i] = 0.1f * i;
    
    Tensor vision_feat = encoder.forward(image);
    
    std::cout << "  vision_feat shape: [" << vision_feat.shape[0] 
              << ", " << vision_feat.shape[1] << "]" << std::endl;
    
    assert(vision_feat.shape[0] == 2);  // batch
    assert(vision_feat.shape[1] == 64); // embed_dim
    
    std::cout << "  PASSED: VisionEncoder" << std::endl;
}

void test_cross_modal_fusion() {
    std::cout << "Testing CrossModalFusion..." << std::endl;
    
    CrossModalFusion fusion(64, 64, 64);
    
    Tensor text_feat({2, 64});
    Tensor image_feat({2, 64});
    
    float* t = text_feat.as_fp32();
    float* i = image_feat.as_fp32();
    for (size_t j = 0; j < 64; ++j) {
        t[j] = 0.5f;
        t[64 + j] = 0.3f;
        i[j] = 0.5f;  // 相似
        i[64 + j] = 0.1f;  // 不同
    }
    
    auto output = fusion.forward(text_feat, image_feat);
    
    std::cout << "  fused shape: [" << output.fused.shape[0] 
              << ", " << output.fused.shape[1] << "]" << std::endl;
    std::cout << "  text_feat shape: [" << output.text_feat.shape[0] 
              << ", " << output.text_feat.shape[1] << "]" << std::endl;
    std::cout << "  similarity scores: ";
    float* sim = output.similarity.as_fp32();
    std::cout << sim[0] << ", " << sim[1] << std::endl;
    
    assert(output.fused.shape[0] == 2);
    assert(output.fused.shape[1] == 64);
    
    std::cout << "  PASSED: CrossModalFusion" << std::endl;
}

void test_multimodal_attention() {
    std::cout << "Testing MultiModalAttention..." << std::endl;
    
    MultiModalAttention attn(64, 64, 4);
    
    Tensor text({2, 64});
    Tensor image({2, 64});
    
    for (size_t j = 0; j < 64; ++j) {
        text.as_fp32()[j] = 0.1f * j;
        text.as_fp32()[64 + j] = 0.2f * j;
        image.as_fp32()[j] = 0.3f * j;
        image.as_fp32()[64 + j] = 0.4f * j;
    }
    
    Tensor text_enhanced = attn.text_attend_image(text, image);
    
    std::cout << "  text_enhanced shape: [" << text_enhanced.shape[0] 
              << ", " << text_enhanced.shape[1] << "]" << std::endl;
    
    assert(text_enhanced.shape[0] == 2);
    assert(text_enhanced.shape[1] == 64);
    
    std::cout << "  PASSED: MultiModalAttention" << std::endl;
}

void test_multimodal_model_creation() {
    std::cout << "Testing NeuroFlowMultiModal creation..." << std::endl;
    
    NeuroFlowMultiModal::Config cfg;
    cfg.text_dim = 64;
    cfg.image_size = 64;
    cfg.patch_size = 8;
    cfg.vision_dim = 32;
    cfg.fusion_dim = 32;
    cfg.hidden_dim = 32;
    cfg.output_dim = 5;
    cfg.memory_dim = 16;
    cfg.memory_slots = 8;
    cfg.num_layers = 1;
    cfg.num_associations = 2;
    cfg.vision_layers = 2;
    cfg.vision_heads = 2;
    cfg.use_mla = false;
    
    NeuroFlowMultiModal model(cfg);
    
    auto stats = model.get_stats();
    std::cout << "  Total params: " << stats.total_params << std::endl;
    std::cout << "  Vision params: " << stats.vision_params << std::endl;
    std::cout << "  Fusion params: " << stats.fusion_params << std::endl;
    std::cout << "  Brain params: " << stats.brain_params << std::endl;
    std::cout << "  Memory (KB): " << stats.memory_bytes / 1024.0 << std::endl;
    
    assert(stats.total_params > 0);
    
    std::cout << "  PASSED: NeuroFlowMultiModal creation" << std::endl;
}

void test_multimodal_forward_text() {
    std::cout << "Testing multimodal forward (text only)..." << std::endl;
    
    NeuroFlowMultiModal::Config cfg;
    cfg.text_dim = 64;
    cfg.image_size = 64;
    cfg.patch_size = 8;
    cfg.vision_dim = 32;
    cfg.fusion_dim = 32;
    cfg.hidden_dim = 32;
    cfg.output_dim = 5;
    cfg.memory_dim = 16;
    cfg.memory_slots = 8;
    cfg.num_layers = 1;
    cfg.num_associations = 2;
    cfg.vision_layers = 2;
    cfg.vision_heads = 2;
    
    NeuroFlowMultiModal model(cfg);
    
    Tensor text_input({2, cfg.text_dim});
    for (size_t i = 0; i < text_input.numel(); ++i) 
        text_input.as_fp32()[i] = 0.1f * i;
    
    auto output = model.forward_text(text_input);
    
    std::cout << "  output shape: [" << output.output.shape[0] 
              << ", " << output.output.shape[1] << "]" << std::endl;
    std::cout << "  decision shape: [" << output.decision.shape[0] 
              << ", " << output.decision.shape[1] << "]" << std::endl;
    
    assert(output.output.shape[0] == 2);
    assert(output.output.shape[1] == cfg.output_dim);
    
    std::cout << "  PASSED: multimodal forward (text)" << std::endl;
}

void test_multimodal_forward_with_image() {
    std::cout << "Testing multimodal forward (text + image)..." << std::endl;
    
    NeuroFlowMultiModal::Config cfg;
    cfg.text_dim = 32;
    cfg.image_size = 32;
    cfg.patch_size = 4;
    cfg.vision_dim = 16;
    cfg.fusion_dim = 16;
    cfg.hidden_dim = 16;
    cfg.output_dim = 5;
    cfg.memory_dim = 8;
    cfg.memory_slots = 4;
    cfg.num_layers = 1;
    cfg.num_associations = 2;
    cfg.vision_layers = 1;
    cfg.vision_heads = 2;
    
    NeuroFlowMultiModal model(cfg);
    
    // 文本输入
    Tensor text_input({2, cfg.text_dim});
    for (size_t i = 0; i < text_input.numel(); ++i) 
        text_input.as_fp32()[i] = 0.1f * i;
    
    // 图像输入
    Tensor image_input({2, 3, cfg.image_size, cfg.image_size});
    for (size_t i = 0; i < image_input.numel(); ++i) 
        image_input.as_fp32()[i] = 0.05f * i;
    
    auto output = model.forward_multimodal(text_input, image_input);
    
    std::cout << "  output shape: [" << output.output.shape[0] 
              << ", " << output.output.shape[1] << "]" << std::endl;
    std::cout << "  vision_feat shape: [" << output.vision_feat.shape[0] 
              << ", " << output.vision_feat.shape[1] << "]" << std::endl;
    std::cout << "  fused_feat shape: [" << output.fused_feat.shape[0] 
              << ", " << output.fused_feat.shape[1] << "]" << std::endl;
    std::cout << "  text-image similarity: " << output.text_image_sim.as_fp32()[0] 
              << ", " << output.text_image_sim.as_fp32()[1] << std::endl;
    
    assert(output.output.shape[0] == 2);
    assert(output.output.shape[1] == cfg.output_dim);
    assert(output.vision_feat.shape[1] == cfg.vision_dim);
    
    std::cout << "  PASSED: multimodal forward (text+image)" << std::endl;
}

void test_multimodal_image_only() {
    std::cout << "Testing multimodal forward (image only)..." << std::endl;
    
    NeuroFlowMultiModal::Config cfg;
    cfg.text_dim = 32;
    cfg.image_size = 32;
    cfg.patch_size = 4;
    cfg.vision_dim = 16;
    cfg.fusion_dim = 16;
    cfg.hidden_dim = 16;
    cfg.output_dim = 5;
    cfg.memory_dim = 8;
    cfg.memory_slots = 4;
    cfg.num_layers = 1;
    cfg.num_associations = 2;
    cfg.vision_layers = 1;
    cfg.vision_heads = 2;
    
    NeuroFlowMultiModal model(cfg);
    
    // 图像输入
    Tensor image_input({1, 3, cfg.image_size, cfg.image_size});
    for (size_t i = 0; i < image_input.numel(); ++i) 
        image_input.as_fp32()[i] = 0.05f * i;
    
    auto output = model.forward_image_only(image_input);
    
    std::cout << "  output shape: [" << output.output.shape[0] 
              << ", " << output.output.shape[1] << "]" << std::endl;
    std::cout << "  vision_feat shape: [" << output.vision_feat.shape[0] 
              << ", " << output.vision_feat.shape[1] << "]" << std::endl;
    
    assert(output.output.shape[0] == 1);
    assert(output.output.shape[1] == cfg.output_dim);
    
    std::cout << "  PASSED: multimodal forward (image only)" << std::endl;
}

void test_multimodal_quantization() {
    std::cout << "Testing multimodal quantization..." << std::endl;
    
    NeuroFlowMultiModal::Config cfg;
    cfg.text_dim = 32;
    cfg.image_size = 32;
    cfg.patch_size = 4;
    cfg.vision_dim = 16;
    cfg.fusion_dim = 16;
    cfg.hidden_dim = 16;
    cfg.output_dim = 5;
    cfg.use_quantization = true;
    
    NeuroFlowMultiModal model(cfg);
    model.quantize();
    
    auto stats = model.get_stats();
    std::cout << "  Total params after quant: " << stats.total_params << std::endl;
    
    // 测试量化后仍能运行
    Tensor text_input({1, cfg.text_dim});
    Tensor image_input({1, 3, cfg.image_size, cfg.image_size});
    
    auto output = model.forward_multimodal(text_input, image_input);
    
    std::cout << "  Output after quant: [" << output.output.shape[0] 
              << ", " << output.output.shape[1] << "]" << std::endl;
    
    assert(output.output.shape[1] == cfg.output_dim);
    
    std::cout << "  PASSED: multimodal quantization" << std::endl;
}

void test_multimodal_performance() {
    std::cout << "Testing multimodal performance..." << std::endl;
    
    NeuroFlowMultiModal::Config cfg;
    cfg.text_dim = 128;
    cfg.image_size = 64;
    cfg.patch_size = 8;
    cfg.vision_dim = 64;
    cfg.fusion_dim = 64;
    cfg.hidden_dim = 64;
    cfg.output_dim = 10;
    cfg.memory_dim = 32;
    cfg.memory_slots = 16;
    cfg.num_layers = 1;
    cfg.num_associations = 4;
    cfg.vision_layers = 2;
    cfg.vision_heads = 4;
    
    NeuroFlowMultiModal model(cfg);
    
    auto stats = model.get_stats();
    std::cout << "  Full model params: " << stats.total_params << std::endl;
    
    // Lite版本
    NeuroFlowMultiModal::Config lite_cfg;
    lite_cfg.text_dim = 64;
    lite_cfg.image_size = 32;
    lite_cfg.patch_size = 4;
    lite_cfg.vision_dim = 32;
    lite_cfg.fusion_dim = 32;
    lite_cfg.hidden_dim = 32;
    lite_cfg.output_dim = 10;
    lite_cfg.memory_dim = 16;
    lite_cfg.memory_slots = 8;
    lite_cfg.num_layers = 1;
    lite_cfg.num_associations = 2;
    lite_cfg.vision_layers = 1;
    lite_cfg.vision_heads = 2;
    lite_cfg.use_quantization = true;
    
    NeuroFlowMultiModal lite(lite_cfg);
    
    auto lite_stats = lite.get_stats();
    std::cout << "  Lite model params: " << lite_stats.total_params << std::endl;
    std::cout << "  Size reduction: " << (1.0 - (double)lite_stats.total_params / stats.total_params) * 100 << "%" << std::endl;
    
    // 性能测试
    Tensor text({4, cfg.text_dim});
    Tensor image({4, 3, cfg.image_size, cfg.image_size});
    
    for (size_t i = 0; i < text.numel(); ++i) text.as_fp32()[i] = 0.1f * i;
    for (size_t i = 0; i < image.numel(); ++i) image.as_fp32()[i] = 0.05f * i;
    
    // 预热
    model.forward_multimodal(text, image);
    
    // Full模型
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        model.forward_multimodal(text, image);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto full_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 10;
    
    // Lite模型
    Tensor lite_text({4, lite_cfg.text_dim});
    Tensor lite_image({4, 3, lite_cfg.image_size, lite_cfg.image_size});
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        lite.forward_multimodal(lite_text, lite_image);
    }
    end = std::chrono::high_resolution_clock::now();
    auto lite_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 10;
    
    std::cout << "  Full model time: " << full_time << " ms" << std::endl;
    std::cout << "  Lite model time: " << lite_time << " ms" << std::endl;
    std::cout << "  Speedup: " << (full_time / lite_time) << "x" << std::endl;
    
    std::cout << "  PASSED: multimodal performance" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "NeuroFlow MultiModal Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_patch_embedding();
    test_vision_encoder();
    test_cross_modal_fusion();
    test_multimodal_attention();
    test_multimodal_model_creation();
    test_multimodal_forward_text();
    test_multimodal_forward_with_image();
    test_multimodal_image_only();
    test_multimodal_quantization();
    test_multimodal_performance();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All MultiModal tests PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
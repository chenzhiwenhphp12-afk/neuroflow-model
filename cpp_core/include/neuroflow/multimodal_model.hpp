#ifndef NEUROFLOW_MULTIMODAL_MODEL_HPP
#define NEUROFLOW_MULTIMODAL_MODEL_HPP

/**
 * NeuroFlowMultiModal - 多模态类脑模块化神经网络
 * 
 * 整合：
 * 1. Vision Encoder (图像编码)
 * 2. Cross-Modal Fusion (文本-图像融合)
 * 3. ExecutiveControlNetwork (ECN) - 多模态推理决策
 * 4. DefaultModeNetwork (DMN) - 跨模态联想记忆
 * 5. SalienceNetwork (SN) - 多模态显著性分配
 * 6. Memory Module - 长记忆存储
 */

#include "tensor.hpp"
#include "networks.hpp"
#include "memory.hpp"
#include "multimodal.hpp"
#include <vector>
#include <memory>
#include <unordered_map>

namespace neuroflow {

/**
 * NeuroFlowMultiModal - 多模态类脑神经网络
 */
class NeuroFlowMultiModal {
public:
    // 配置
    struct Config {
        size_t text_dim = 512;        // 文本特征维度
        size_t image_size = 224;      // 图像大小
        size_t patch_size = 16;       // ViT patch大小
        size_t vision_dim = 256;      // 视觉编码维度
        size_t fusion_dim = 256;      // 融合维度
        size_t hidden_dim = 256;      // 隐藏维度
        size_t output_dim = 10;       // 输出维度
        size_t memory_dim = 128;      // 记忆维度
        size_t memory_slots = 64;     // 记忆槽数量
        size_t num_layers = 2;        // ECN层数
        size_t num_associations = 8;  // DMN联想头数量
        size_t vision_layers = 4;     // Vision Encoder层数
        size_t vision_heads = 8;      // Vision注意力头数
        bool use_quantization = false;
        bool use_mla = false;
        size_t mla_latent_dim = 32;
    };
    
    Config config;
    
    // ========== 多模态组件 ==========
    std::unique_ptr<VisionEncoder> vision_encoder;
    std::unique_ptr<CrossModalFusion> cross_modal_fusion;
    std::unique_ptr<MultiModalAttention> multimodal_attention;
    
    // ========== 类脑模块 ==========
    std::unique_ptr<ExecutiveControlNetwork> ecn;
    std::unique_ptr<DefaultModeNetwork> dmn;
    std::unique_ptr<SalienceNetwork> sn;
    std::unique_ptr<MemoryConsolidationModule> memory;
    std::unique_ptr<LatentKVCache> mla_cache;
    
    // ========== 输入投影 ==========
    std::shared_ptr<Linear> text_proj;
    std::shared_ptr<LayerNorm> text_norm;
    
    // ========== 融合后处理 ==========
    std::shared_ptr<Linear> multimodal_proj;
    std::shared_ptr<LayerNorm> multimodal_norm;
    
    // ========== 输出层 ==========
    std::shared_ptr<Linear> output_layer;
    std::shared_ptr<LayerNorm> output_norm;
    
    // ========== 流形投影 ==========
    std::shared_ptr<Linear> manifold_proj1;
    std::shared_ptr<LayerNorm> manifold_norm;
    std::shared_ptr<Linear> manifold_proj2;
    
    bool training_mode;
    
    NeuroFlowMultiModal(const Config& cfg) : config(cfg), training_mode(false) {
        
        // 多模态组件
        vision_encoder = std::make_unique<VisionEncoder>(
            config.image_size, config.patch_size, 
            config.vision_dim, config.vision_heads, config.vision_layers);
        
        cross_modal_fusion = std::make_unique<CrossModalFusion>(
            config.text_dim, config.vision_dim, config.fusion_dim);
        
        multimodal_attention = std::make_unique<MultiModalAttention>(
            config.fusion_dim, config.fusion_dim, 8);
        
        // 文本投影
        text_proj = std::make_shared<Linear>(config.text_dim, config.hidden_dim);
        text_norm = std::make_shared<LayerNorm>(config.hidden_dim);
        
        // 多模态融合投影
        multimodal_proj = std::make_shared<Linear>(config.fusion_dim, config.hidden_dim);
        multimodal_norm = std::make_shared<LayerNorm>(config.hidden_dim);
        
        // 类脑模块 (基于融合后的特征)
        ecn = std::make_unique<ExecutiveControlNetwork>(
            config.hidden_dim, config.hidden_dim, config.output_dim, config.num_layers);
        
        dmn = std::make_unique<DefaultModeNetwork>(
            config.memory_dim, config.hidden_dim / 2, config.num_associations);
        
        sn = std::make_unique<SalienceNetwork>(
            config.hidden_dim, config.hidden_dim / 2);
        
        memory = std::make_unique<MemoryConsolidationModule>(
            config.hidden_dim, config.memory_slots, config.memory_dim);
        
        if (config.use_mla) {
            mla_cache = std::make_unique<LatentKVCache>(
                config.hidden_dim, 8, config.mla_latent_dim, 4096);
        }
        
        // 输出层
        output_layer = std::make_shared<Linear>(config.hidden_dim, config.output_dim);
        output_norm = std::make_shared<LayerNorm>(config.output_dim);
        
        // 流形投影
        manifold_proj1 = std::make_shared<Linear>(config.hidden_dim, config.hidden_dim);
        manifold_norm = std::make_shared<LayerNorm>(config.hidden_dim);
        manifold_proj2 = std::make_shared<Linear>(config.hidden_dim, 32);
        
        if (config.use_quantization) {
            quantize();
        }
    }
    
    NeuroFlowMultiModal() : NeuroFlowMultiModal(Config()) {}
    
    // ========== 输出结构 ==========
    struct Output {
        Tensor output;              // 最终输出
        Tensor decision;            // ECN决策
        Tensor value;               // OFC价值
        Tensor saliency;            // SN显著性
        Tensor text_image_sim;      // 文本-图像相似度
        Tensor gates;               // ECN/DMN门控
        Tensor anomaly;             // 异常评分
        Tensor retrieved_mem;       // 检索记忆
        Tensor manifold;            // 流形表征
        Tensor vision_feat;         // 视觉特征
        Tensor text_feat;           // 文本特征 (对齐后)
        Tensor fused_feat;          // 融合特征
    };
    
    // ========== 纯文本模式 ==========
    Output forward_text(const Tensor& text_input) {
        Output out;
        size_t batch = text_input.shape[0];
        
        // 文本投影
        Tensor h = text_proj->forward(text_input);
        h = text_norm->forward(h);
        
        // 类脑处理 (纯文本模式)
        auto sn_out = sn->forward(h);
        auto ecn_out = ecn->forward(h);
        
        out.saliency = sn_out.saliency;
        out.gates = sn_out.gates;
        out.decision = ecn_out.decision;
        out.value = ecn_out.value;
        
        // 记忆
        auto mem_out = memory->forward(h);
        out.retrieved_mem = mem_out.retrieved;
        
        // 输出
        out.output = output_layer->forward(h);
        out.output = output_norm->forward(out.output);
        
        return out;
    }
    
    // ========== 多模态模式 (文本+图像) ==========
    Output forward_multimodal(const Tensor& text_input, const Tensor& image_input,
                              bool consolidate = false, bool return_manifold = false) {
        Output out;
        size_t batch = text_input.shape[0];
        
        // 1. Vision Encoder: 图像编码
        Tensor vision_feat = vision_encoder->forward(image_input);
        out.vision_feat = vision_feat.clone();
        
        // 2. Cross-Modal Fusion: 文本-图像对齐融合
        auto fusion_out = cross_modal_fusion->forward(text_input, vision_feat);
        out.text_feat = fusion_out.text_feat.clone();
        out.text_image_sim = fusion_out.similarity.clone();
        out.fused_feat = fusion_out.fused.clone();
        
        // 3. 多模态融合投影
        Tensor multimodal_h = multimodal_proj->forward(fusion_out.fused);
        multimodal_h = multimodal_norm->forward(multimodal_h);
        
        // 4. Cross-Modal Attention (可选增强)
        Tensor text_enhanced = multimodal_attention->text_attend_image(
            text_proj->forward(text_input), vision_feat);
        
        // 残差融合
        float* mh = multimodal_h.as_fp32();
        float* te = text_enhanced.as_fp32();
        size_t min_dim = std::min(multimodal_h.shape[1], text_enhanced.shape[1]);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t d = 0; d < min_dim; ++d) {
                mh[b * multimodal_h.shape[1] + d] += 0.3f * te[b * text_enhanced.shape[1] + d];
            }
        }
        
        // 5. Salience Network: 多模态显著性检测
        auto sn_out = sn->forward(multimodal_h);
        out.saliency = sn_out.saliency;
        out.gates = sn_out.gates;
        out.anomaly = sn_out.anomaly;
        
        // 提取门控
        float* gates = out.gates.as_fp32();
        Tensor ecn_gate({batch, 1}, QuantType::FP32);
        Tensor dmn_gate({batch, 1}, QuantType::FP32);
        for (size_t i = 0; i < batch; ++i) {
            ecn_gate.as_fp32()[i] = gates[i * 2];
            dmn_gate.as_fp32()[i] = gates[i * 2 + 1];
        }
        
        // 6. ECN: 多模态推理决策 (模拟前额叶整合视觉信息做决策)
        auto ecn_out = ecn->forward(multimodal_h);
        out.decision = ecn_out.decision;
        out.value = ecn_out.value;
        
        // 7. DMN: 跨模态联想记忆 (模拟后扣带回将视觉与记忆关联)
        Tensor mem_seed = memory->encode(multimodal_h);
        auto dmn_out = dmn->forward(mem_seed);
        
        // 8. Memory Retrieval
        auto mem_out = memory->forward(multimodal_h);
        out.retrieved_mem = mem_out.retrieved;
        
        if (consolidate) {
            memory->consolidate(multimodal_h);
        }
        
        // 9. 门控加权融合
        Tensor ecn_weighted = out.decision.clone();
        Tensor dmn_weighted = dmn_out.vision.reshape({batch, dmn_out.vision.shape[1]});
        
        float* eg = ecn_gate.as_fp32();
        float* dg = dmn_gate.as_fp32();
        float* ew = ecn_weighted.as_fp32();
        float* dw = dmn_weighted.as_fp32();
        
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < config.output_dim && j < ecn_weighted.shape[1]; ++j) {
                ew[i * ecn_weighted.shape[1] + j] *= eg[i];
            }
            for (size_t j = 0; j < dmn_weighted.shape[1]; ++j) {
                dw[i * dmn_weighted.shape[1] + j] *= dg[i];
            }
        }
        
        // 10. 最终融合输出
        // 综合ECN决策 + DMN联想 + Memory + 融合特征
        Tensor combined({batch, config.hidden_dim}, QuantType::FP32);
        float* c = combined.as_fp32();
        
        // 加权组合
        for (size_t b = 0; b < batch; ++b) {
            for (size_t d = 0; d < config.hidden_dim; ++d) {
                float val = 0;
                if (d < out.decision.shape[1]) {
                    val += 0.3f * ew[b * out.decision.shape[1] + d];
                }
                if (d < dmn_weighted.shape[1]) {
                    val += 0.2f * dw[b * dmn_weighted.shape[1] + d];
                }
                if (d < out.retrieved_mem.shape[1]) {
                    val += 0.2f * out.retrieved_mem.as_fp32()[b * out.retrieved_mem.shape[1] + d];
                }
                if (d < multimodal_h.shape[1]) {
                    val += 0.3f * mh[b * multimodal_h.shape[1] + d];
                }
                c[b * config.hidden_dim + d] = val;
            }
        }
        
        out.output = output_layer->forward(combined);
        out.output = output_norm->forward(out.output);
        
        // 11. 流形 (可选)
        if (return_manifold) {
            Tensor manifold_in({batch, config.hidden_dim}, QuantType::FP32);
            float* mi = manifold_in.as_fp32();
            float* eh = ecn_out.hidden_states.back().as_fp32();
            
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < config.hidden_dim && j < ecn_out.hidden_states.back().shape[1]; ++j) {
                    mi[i * config.hidden_dim + j] = eh[i * ecn_out.hidden_states.back().shape[1] + j];
                }
                for (size_t j = ecn_out.hidden_states.back().shape[1]; j < config.hidden_dim; ++j) {
                    mi[i * config.hidden_dim + j] = 0;
                }
            }
            
            Tensor m = manifold_proj1->forward(manifold_in);
            m = manifold_norm->forward(m);
            TensorOps::gelu(m);
            out.manifold = manifold_proj2->forward(m);
        }
        
        return out;
    }
    
    // ========== 通用forward接口 ==========
    Output forward(const Tensor& text_input, const Tensor* image_input = nullptr,
                   bool consolidate = false, bool return_manifold = false) {
        if (image_input) {
            return forward_multimodal(text_input, *image_input, consolidate, return_manifold);
        } else {
            return forward_text(text_input);
        }
    }
    
    // ========== 图像理解模式 (无文本，纯视觉推理) ==========
    Output forward_image_only(const Tensor& image_input) {
        Output out;
        size_t batch = image_input.shape[0];
        
        // Vision Encoder
        Tensor vision_feat = vision_encoder->forward(image_input);
        out.vision_feat = vision_feat.clone();
        
        // 直接投影到隐藏层
        Tensor h = multimodal_proj->forward(vision_feat);
        h = multimodal_norm->forward(h);
        
        // 类脑处理 (纯视觉决策)
        auto sn_out = sn->forward(h);
        auto ecn_out = ecn->forward(h);
        
        out.saliency = sn_out.saliency;
        out.gates = sn_out.gates;
        out.decision = ecn_out.decision;
        out.value = ecn_out.value;
        
        // 记忆
        auto mem_out = memory->forward(h);
        out.retrieved_mem = mem_out.retrieved;
        
        // 输出
        out.output = output_layer->forward(h);
        out.output = output_norm->forward(out.output);
        
        return out;
    }
    
    // ========== 设置训练模式 ==========
    void set_training(bool t) {
        training_mode = t;
        ecn->set_training(t);
    }
    
    // ========== 量化 ==========
    void quantize() {
        vision_encoder->quantize();
        cross_modal_fusion->quantize();
        multimodal_attention->quantize();
        text_proj->quantize();
        multimodal_proj->quantize();
        output_layer->quantize();
        manifold_proj1->quantize();
        manifold_proj2->quantize();
        ecn->quantize();
        dmn->quantize();
        sn->quantize();
        memory->encode_proj->quantize();
        memory->retrieve_proj->quantize();
        memory->query_proj->quantize();
    }
    
    // ========== 获取模型统计 ==========
    struct Stats {
        size_t total_params;
        size_t vision_params;
        size_t fusion_params;
        size_t brain_params;
        size_t memory_bytes;
        float quantization_ratio;
    };
    
    Stats get_stats() {
        Stats s;
        s.total_params = 0;
        s.vision_params = 0;
        s.fusion_params = 0;
        s.brain_params = 0;
        s.memory_bytes = 0;
        s.quantization_ratio = 0.0f;
        
        auto count_linear = [&](std::shared_ptr<Linear>& l, size_t& category) {
            s.total_params += l->weight.numel();
            if (l->bias.data) s.total_params += l->bias.numel();
            category += l->weight.numel();
            s.memory_bytes += l->weight.data_size + l->bias.data_size;
        };
        
        // Vision Encoder
        count_linear(vision_encoder->patch_embed->proj, s.vision_params);
        for (auto& l : vision_encoder->self_attn_qkv) count_linear(l, s.vision_params);
        for (auto& l : vision_encoder->self_attn_proj) count_linear(l, s.vision_params);
        for (auto& l : vision_encoder->mlp_fc1) count_linear(l, s.vision_params);
        for (auto& l : vision_encoder->mlp_fc2) count_linear(l, s.vision_params);
        count_linear(vision_encoder->output_proj, s.vision_params);
        
        // Cross-Modal Fusion
        count_linear(cross_modal_fusion->text_proj, s.fusion_params);
        count_linear(cross_modal_fusion->image_proj, s.fusion_params);
        count_linear(cross_modal_fusion->fusion_layer, s.fusion_params);
        
        // Multi-Modal Attention
        count_linear(multimodal_attention->text_query, s.fusion_params);
        count_linear(multimodal_attention->image_key, s.fusion_params);
        count_linear(multimodal_attention->image_value, s.fusion_params);
        count_linear(multimodal_attention->text_output, s.fusion_params);
        count_linear(multimodal_attention->image_query, s.fusion_params);
        count_linear(multimodal_attention->text_key, s.fusion_params);
        count_linear(multimodal_attention->text_value, s.fusion_params);
        count_linear(multimodal_attention->image_output, s.fusion_params);
        
        // 类脑模块
        count_linear(text_proj, s.brain_params);
        count_linear(multimodal_proj, s.brain_params);
        count_linear(output_layer, s.brain_params);
        count_linear(manifold_proj1, s.brain_params);
        count_linear(manifold_proj2, s.brain_params);
        
        for (auto& l : ecn->dlpfc_linear) count_linear(l, s.brain_params);
        count_linear(ecn->ofc1, s.brain_params);
        count_linear(ecn->ofc2, s.brain_params);
        count_linear(ecn->vmpfc1, s.brain_params);
        count_linear(ecn->vmpfc2, s.brain_params);
        
        count_linear(dmn->mem_encoder1, s.brain_params);
        count_linear(dmn->mem_encoder2, s.brain_params);
        count_linear(dmn->future_proj1, s.brain_params);
        for (auto& [h1, h2] : dmn->association_heads) {
            count_linear(h1, s.brain_params);
            count_linear(h2, s.brain_params);
        }
        
        count_linear(sn->saliency1, s.brain_params);
        count_linear(sn->saliency2, s.brain_params);
        count_linear(sn->saliency3, s.brain_params);
        count_linear(sn->gate1, s.brain_params);
        count_linear(sn->gate2, s.brain_params);
        count_linear(sn->anomaly1, s.brain_params);
        count_linear(sn->anomaly2, s.brain_params);
        
        count_linear(memory->encode_proj, s.brain_params);
        count_linear(memory->retrieve_proj, s.brain_params);
        count_linear(memory->query_proj, s.brain_params);
        
        s.memory_bytes += memory->memory_bank.data_size;
        s.total_params += memory->memory_bank.numel();
        
        return s;
    }
    
    // ========== 序列化 ==========
    void save(const std::string& path) {
        // TODO: 实现完整的序列化
    }
    
    void load(const std::string& path) {
        // TODO: 实现完整的反序列化
    }
};

/**
 * NeuroFlowMultiModalLite - 超轻量多模态版
 */
class NeuroFlowMultiModalLite : public NeuroFlowMultiModal {
public:
    NeuroFlowMultiModalLite(size_t text_dim = 256, size_t image_size = 112) {
        Config cfg;
        cfg.text_dim = text_dim;
        cfg.image_size = image_size;
        cfg.patch_size = 8;
        cfg.vision_dim = 128;
        cfg.fusion_dim = 128;
        cfg.hidden_dim = 128;
        cfg.output_dim = 10;
        cfg.memory_dim = 64;
        cfg.memory_slots = 32;
        cfg.num_layers = 1;
        cfg.num_associations = 4;
        cfg.vision_layers = 2;
        cfg.vision_heads = 4;
        cfg.use_quantization = true;
        cfg.use_mla = true;
        cfg.mla_latent_dim = 32;
        
        // 需要重新初始化...
    }
};

} // namespace neuroflow

#endif // NEUROFLOW_MULTIMODAL_MODEL_HPP
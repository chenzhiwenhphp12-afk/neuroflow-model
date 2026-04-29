#ifndef NEUROFLOW_MODEL_HPP
#define NEUROFLOW_MODEL_HPP

/**
 * NeuroFlowModel - 主模型类
 * 
 * 整合三大网络：
 * 1. ExecutiveControlNetwork (ECN)
 * 2. DefaultModeNetwork (DMN)
 * 3. SalienceNetwork (SN)
 * 
 * + 记忆模块
 * + 神经流形分析
 */

#include "tensor.hpp"
#include "networks.hpp"
#include "memory.hpp"
#include <vector>
#include <memory>
#include <unordered_map>

namespace neuroflow {

/**
 * NeuroFlowModel - 类脑模块化神经网络
 */
class NeuroFlowModel {
public:
    // 配置
    struct Config {
        size_t input_dim = 512;
        size_t hidden_dim = 256;
        size_t output_dim = 10;
        size_t memory_dim = 128;
        size_t memory_slots = 64;
        size_t num_layers = 2;
        size_t num_associations = 8;
        bool use_quantization = false;
        bool use_mla = false;
        size_t mla_latent_dim = 32;  // MLA压缩维度
    };
    
    Config config;
    
    // 输入投影
    std::shared_ptr<Linear> input_proj_linear;
    std::shared_ptr<LayerNorm> input_proj_norm;
    std::shared_ptr<GELU> input_proj_gelu;
    
    // 三大核心网络
    std::unique_ptr<ExecutiveControlNetwork> ecn;
    std::unique_ptr<DefaultModeNetwork> dmn;
    std::unique_ptr<SalienceNetwork> sn;
    
    // 记忆模块
    std::unique_ptr<MemoryConsolidationModule> memory;
    std::unique_ptr<LatentKVCache> mla_cache;  // 可选MLA
    
    // 流形投影
    std::shared_ptr<Linear> manifold_proj1;
    std::shared_ptr<LayerNorm> manifold_norm;
    std::shared_ptr<GELU> manifold_gelu;
    std::shared_ptr<Linear> manifold_proj2;
    
    // 输出融合
    std::shared_ptr<Linear> output_fusion_linear;
    std::shared_ptr<LayerNorm> output_fusion_norm;
    
    // 训练模式
    bool training_mode;
    
    NeuroFlowModel(const Config& cfg) : config(cfg), training_mode(false) {
        // 输入投影
        input_proj_linear = std::make_shared<Linear>(config.input_dim, config.hidden_dim);
        input_proj_norm = std::make_shared<LayerNorm>(config.hidden_dim);
        input_proj_gelu = std::make_shared<GELU>();
        
        // ECN
        ecn = std::make_unique<ExecutiveControlNetwork>(
            config.hidden_dim, config.hidden_dim, config.output_dim, config.num_layers);
        
        // DMN
        dmn = std::make_unique<DefaultModeNetwork>(
            config.memory_dim, config.hidden_dim / 2, config.num_associations);
        
        // SN
        sn = std::make_unique<SalienceNetwork>(
            config.hidden_dim, config.hidden_dim / 2);
        
        // 记忆
        memory = std::make_unique<MemoryConsolidationModule>(
            config.hidden_dim, config.memory_slots, config.memory_dim);
        
        // MLA (可选)
        if (config.use_mla) {
            mla_cache = std::make_unique<LatentKVCache>(
                config.hidden_dim, 8, config.mla_latent_dim, 4096);
        }
        
        // 流形投影
        size_t manifold_in = config.hidden_dim + config.hidden_dim / 2;
        manifold_proj1 = std::make_shared<Linear>(manifold_in, config.hidden_dim);
        manifold_norm = std::make_shared<LayerNorm>(config.hidden_dim);
        manifold_gelu = std::make_shared<GELU>();
        manifold_proj2 = std::make_shared<Linear>(config.hidden_dim, 32);
        
        // 输出融合
        output_fusion_linear = std::make_shared<Linear>(config.output_dim * 3, config.output_dim);
        output_fusion_norm = std::make_shared<LayerNorm>(config.output_dim);
        
        // 量化
        if (config.use_quantization) {
            quantize();
        }
    }
    
    // 默认构造
    NeuroFlowModel() : NeuroFlowModel(Config()) {}
    
    // 输出结构
    struct Output {
        Tensor output;           // 最终输出
        Tensor decision;         // ECN决策
        Tensor value;            // OFC价值
        Tensor saliency;         // SN显著性
        Tensor gates;            // ECN/DMN门控权重 (2-class)
        Tensor ecn_gate;         // ECN门控
        Tensor dmn_gate;         // DMN门控
        Tensor anomaly;          // 异常评分
        Tensor mem_attention;    // 记忆注意力
        Tensor retrieved_mem;    // 检索记忆
        Tensor manifold;         // 流形表征 (可选)
    };
    
    // 前向传播
    Output forward(const Tensor& x, const Tensor* memory_input = nullptr,
                   bool consolidate = false, bool return_manifold = false) {
        Output out;
        size_t batch = x.shape[0];
        
        // 输入投影
        Tensor h = input_proj_linear->forward(x);
        h = input_proj_norm->forward(h);
        h = input_proj_gelu->forward(h);
        
        // SN: 显著性检测 + 门控
        auto sn_out = sn->forward(h);
        out.saliency = sn_out.saliency;
        out.gates = sn_out.gates;
        out.anomaly = sn_out.anomaly;
        
        // 提取门控权重
        float* gates = out.gates.as_fp32();
        Tensor ecn_gate({batch, 1}, QuantType::FP32);
        Tensor dmn_gate({batch, 1}, QuantType::FP32);
        for (size_t i = 0; i < batch; ++i) {
            ecn_gate.as_fp32()[i] = gates[i * 2];
            dmn_gate.as_fp32()[i] = gates[i * 2 + 1];
        }
        out.ecn_gate = ecn_gate;
        out.dmn_gate = dmn_gate;
        
        // ECN: 执行推理
        auto ecn_out = ecn->forward(h);
        out.decision = ecn_out.decision;
        out.value = ecn_out.value;
        
        // DMN: 默认模式网络
        Tensor mem_seed;
        if (memory_input) {
            mem_seed = *memory_input;
        } else {
            mem_seed = memory->encode(h);
        }
        auto dmn_out = dmn->forward(mem_seed);
        
        // 记忆检索
        auto mem_out = memory->forward(h);
        out.retrieved_mem = mem_out.retrieved;
        out.mem_attention = mem_out.attention;
        
        // 记忆巩固 (可选)
        if (consolidate) {
            memory->consolidate(h);
        }
        
        // 门控加权
        Tensor ecn_weighted = out.decision.clone();
        Tensor dmn_weighted = dmn_out.vision.reshape({batch, dmn_out.vision.shape[1]});
        // 只取output_dim部分 (截取，不是reshape)
        if (dmn_weighted.shape[1] > config.output_dim) {
            Tensor truncated({batch, config.output_dim}, QuantType::FP32);
            float* tw = truncated.as_fp32();
            float* dw = dmn_weighted.as_fp32();
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < config.output_dim; ++j) {
                    tw[i * config.output_dim + j] = dw[i * dmn_weighted.shape[1] + j];
                }
            }
            dmn_weighted = truncated;
        }
        
        float* eg = ecn_gate.as_fp32();
        float* dg = dmn_gate.as_fp32();
        float* ew = ecn_weighted.as_fp32();
        float* dw = dmn_weighted.as_fp32();
        
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < config.output_dim; ++j) {
                ew[i * config.output_dim + j] *= eg[i];
                if (j < dmn_weighted.shape[1]) {
                    dw[i * dmn_weighted.shape[1] + j] *= dg[i];
                }
            }
        }
        
        // 融合: ECN + DMN + Memory
        std::vector<Tensor> to_concat;
        to_concat.push_back(ecn_weighted);
        to_concat.push_back(dmn_weighted);
        
        Tensor mem_for_fusion = out.retrieved_mem.clone();
        if (mem_for_fusion.shape[1] > config.output_dim) {
            // 截取 (不是reshape)
            Tensor truncated({batch, config.output_dim}, QuantType::FP32);
            float* tw = truncated.as_fp32();
            float* mw = mem_for_fusion.as_fp32();
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < config.output_dim; ++j) {
                    tw[i * config.output_dim + j] = mw[i * mem_for_fusion.shape[1] + j];
                }
            }
            mem_for_fusion = truncated;
        } else if (mem_for_fusion.shape[1] < config.output_dim) {
            // 补零
            Tensor padded({batch, config.output_dim}, QuantType::FP32);
            float* p = padded.as_fp32();
            float* m = mem_for_fusion.as_fp32();
            memset(p, 0, padded.data_size);
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < mem_for_fusion.shape[1]; ++j) {
                    p[i * config.output_dim + j] = m[i * mem_for_fusion.shape[1] + j];
                }
            }
            mem_for_fusion = padded;
        }
        to_concat.push_back(mem_for_fusion);
        
        Tensor combined = TensorOps::concat(to_concat, 1);
        out.output = output_fusion_linear->forward(combined);
        out.output = output_fusion_norm->forward(out.output);
        
        // 流形 (可选)
        if (return_manifold) {
            Tensor manifold_in({batch, config.hidden_dim + config.hidden_dim / 2}, QuantType::FP32);
            float* mi = manifold_in.as_fp32();
            float* eh = ecn_out.hidden_states.back().as_fp32();
            float* dl = dmn_out.latent.as_fp32();
            
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < config.hidden_dim; ++j) {
                    mi[i * manifold_in.shape[1] + j] = eh[i * config.hidden_dim + j];
                }
                for (size_t j = 0; j < config.hidden_dim / 2; ++j) {
                    mi[i * manifold_in.shape[1] + config.hidden_dim + j] = dl[i * config.hidden_dim / 2 + j];
                }
            }
            
            Tensor m = manifold_proj1->forward(manifold_in);
            m = manifold_norm->forward(m);
            m = manifold_gelu->forward(m);
            out.manifold = manifold_proj2->forward(m);
        }
        
        return out;
    }
    
    // 神经流形轨迹
    std::vector<Tensor> get_manifold_trajectory(const Tensor& x, size_t steps = 10) {
        std::vector<Tensor> trajectory;
        Tensor current = x.clone();
        
        for (size_t s = 0; s < steps; ++s) {
            auto out = forward(current, nullptr, false, true);
            trajectory.push_back(out.manifold.clone());
            
            // 残差更新
            float* c = current.as_fp32();
            float* o = out.output.as_fp32();
            for (size_t i = 0; i < current.shape[0]; ++i) {
                size_t min_dim = std::min(current.shape[1], out.output.shape[1]);
                for (size_t j = 0; j < min_dim; ++j) {
                    c[i * current.shape[1] + j] += 0.1f * o[i * out.output.shape[1] + j];
                }
            }
        }
        
        return trajectory;
    }
    
    // 设置训练模式
    void set_training(bool t) {
        training_mode = t;
        ecn->set_training(t);
    }
    
    // 量化
    void quantize() {
        input_proj_linear->quantize();
        manifold_proj1->quantize();
        manifold_proj2->quantize();
        output_fusion_linear->quantize();
        ecn->quantize();
        dmn->quantize();
        sn->quantize();
        memory->encode_proj->quantize();
        memory->retrieve_proj->quantize();
        memory->query_proj->quantize();
    }
    
    // 获取模型统计
    struct Stats {
        size_t total_params;
        size_t memory_bytes;
        float quantization_ratio;
    };
    
    Stats get_stats() {
        Stats s;
        s.total_params = 0;
        s.memory_bytes = 0;
        s.quantization_ratio = 0.0f;
        
        // 简化统计
        size_t fp32_layers = 0;
        size_t quant_layers = 0;
        
        // 统计各层
        auto count_linear = [&](std::shared_ptr<Linear>& l) {
            s.total_params += l->weight.numel();
            if (l->bias.data) s.total_params += l->bias.numel();
            s.memory_bytes += l->weight.data_size + l->bias.data_size;
            if (l->quantized) quant_layers++;
            else fp32_layers++;
        };
        
        count_linear(input_proj_linear);
        count_linear(manifold_proj1);
        count_linear(manifold_proj2);
        count_linear(output_fusion_linear);
        
        for (auto& l : ecn->dlpfc_linear) count_linear(l);
        count_linear(ecn->ofc1);
        count_linear(ecn->ofc2);
        count_linear(ecn->vmpfc1);
        count_linear(ecn->vmpfc2);
        
        count_linear(dmn->mem_encoder1);
        count_linear(dmn->mem_encoder2);
        count_linear(dmn->future_proj1);
        for (auto& [h1, h2] : dmn->association_heads) {
            count_linear(h1);
            count_linear(h2);
        }
        
        count_linear(sn->saliency1);
        count_linear(sn->saliency2);
        count_linear(sn->saliency3);
        count_linear(sn->gate1);
        count_linear(sn->gate2);
        count_linear(sn->anomaly1);
        count_linear(sn->anomaly2);
        
        count_linear(memory->encode_proj);
        count_linear(memory->retrieve_proj);
        count_linear(memory->query_proj);
        
        s.memory_bytes += memory->memory_bank.data_size;
        s.total_params += memory->memory_bank.numel();
        
        if (fp32_layers + quant_layers > 0) {
            s.quantization_ratio = static_cast<float>(quant_layers) / (fp32_layers + quant_layers);
        }
        
        return s;
    }
    
    // 序列化/反序列化
    void save(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        
        // 写配置
        f.write(reinterpret_cast<char*>(&config), sizeof(Config));
        
        // 写各层权重 (简化版本)
        auto write_tensor = [&](const Tensor& t) {
            f.write(reinterpret_cast<const char*>(t.shape.data()), t.shape.size() * sizeof(size_t));
            size_t dtype_v = static_cast<size_t>(t.dtype);
            f.write(reinterpret_cast<const char*>(&dtype_v), sizeof(size_t));
            f.write(reinterpret_cast<const char*>(t.data.get()), t.data_size);
        };
        
        write_tensor(input_proj_linear->weight);
        write_tensor(input_proj_linear->bias);
        // ... 更多层
        
        f.close();
    }
    
    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        
        // 读配置
        Config loaded_cfg;
        f.read(reinterpret_cast<char*>(&loaded_cfg), sizeof(Config));
        
        // 读权重
        auto read_tensor = [&](Tensor& t) {
            size_t shape_size;
            // 简化实现...
        };
        
        f.close();
    }
};

/**
 * NeuroFlowLite - 超轻量版
 * 适合边缘设备部署
 */
class NeuroFlowLite : public NeuroFlowModel {
public:
    NeuroFlowLite(size_t input_dim = 512) : NeuroFlowModel() {
        Config cfg;
        cfg.input_dim = input_dim;
        cfg.hidden_dim = 128;
        cfg.output_dim = 10;
        cfg.memory_dim = 64;
        cfg.memory_slots = 32;
        cfg.num_layers = 1;
        cfg.num_associations = 4;
        cfg.use_quantization = true;
        cfg.use_mla = true;
        cfg.mla_latent_dim = 32;
        
        // 需要重新初始化...
    }
};

} // namespace neuroflow

#endif // NEUROFLOW_MODEL_HPP
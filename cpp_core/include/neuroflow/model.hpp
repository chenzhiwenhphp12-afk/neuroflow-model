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
        
        // 门控加权 - 创建新的张量，避免修改共享数据
        Tensor ecn_weighted({batch, config.output_dim}, QuantType::FP32);
        Tensor dmn_weighted_full({batch, dmn_out.vision.shape[1]}, QuantType::FP32);
        
        float* ew = ecn_weighted.as_fp32();
        float* dw = dmn_weighted_full.as_fp32();
        float* ed = out.decision.as_fp32();
        float* dv = dmn_out.vision.as_fp32();
        float* eg = ecn_gate.as_fp32();
        float* dg = dmn_gate.as_fp32();
        
        // ECN加权
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < config.output_dim; ++j) {
                ew[i * config.output_dim + j] = ed[i * config.output_dim + j] * eg[i];
            }
        }
        
        // DMN加权
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < dmn_out.vision.shape[1]; ++j) {
                dw[i * dmn_out.vision.shape[1] + j] = dv[i * dmn_out.vision.shape[1] + j] * dg[i];
            }
        }
        
        // 只取output_dim部分
        Tensor dmn_weighted({batch, config.output_dim}, QuantType::FP32);
        float* dwf = dmn_weighted.as_fp32();
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < config.output_dim; ++j) {
                if (j < dmn_out.vision.shape[1]) {
                    dwf[i * config.output_dim + j] = dw[i * dmn_out.vision.shape[1] + j];
                } else {
                    dwf[i * config.output_dim + j] = 0.0f;
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
    
    // ========== 序列化 (NFv1 格式) ==========
    void save(const std::string& path) {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) throw std::runtime_error("Cannot open file for save: " + path);
        
        // Magic header
        ofs.write("NFv1", 4);
        
        // Helper: serialize a named tensor
        auto save_tensor = [&](const std::string& name, const Tensor& t) {
            uint32_t name_len = name.size();
            ofs.write(reinterpret_cast<const char*>(&name_len), 4);
            ofs.write(name.data(), name_len);
            
            uint32_t ndim = t.shape.size();
            ofs.write(reinterpret_cast<const char*>(&ndim), 4);
            for (auto d : t.shape) {
                uint32_t dim = d;
                ofs.write(reinterpret_cast<const char*>(&dim), 4);
            }
            
            uint32_t dsize = t.data_size;
            ofs.write(reinterpret_cast<const char*>(&dsize), 4);
            ofs.write(reinterpret_cast<const char*>(t.data.get()), dsize);
        };
        
        // Helper: save Linear layer (shared_ptr)
        auto save_linear = [&](const std::string& prefix, const std::shared_ptr<Linear>& layer) {
            save_tensor(prefix + ".weight", layer->weight);
            if (layer->bias.data) save_tensor(prefix + ".bias", layer->bias);
        };
        
        // === 输入投影 ===
        save_linear("input_proj", input_proj_linear);
        save_tensor("input_proj_norm.weight", input_proj_norm->weight);
        save_tensor("input_proj_norm.bias", input_proj_norm->bias);
        
        // === ECN ===
        for (size_t i = 0; i < ecn->dlpfc_linear.size(); ++i)
            save_linear("ecn.dlpfc" + std::to_string(i), ecn->dlpfc_linear[i]);
        save_linear("ecn.ofc1", ecn->ofc1);
        save_linear("ecn.ofc2", ecn->ofc2);
        save_linear("ecn.vmpfc1", ecn->vmpfc1);
        save_linear("ecn.vmpfc2", ecn->vmpfc2);
        
        // === DMN ===
        save_linear("dmn.mem_encoder1", dmn->mem_encoder1);
        save_linear("dmn.mem_encoder2", dmn->mem_encoder2);
        save_linear("dmn.future_proj1", dmn->future_proj1);
        int head_idx = 0;
        for (auto& [h1, h2] : dmn->association_heads) {
            save_linear("dmn.head" + std::to_string(head_idx) + ".1", h1);
            save_linear("dmn.head" + std::to_string(head_idx) + ".2", h2);
            head_idx++;
        }
        
        // === SN ===
        save_linear("sn.saliency1", sn->saliency1);
        save_linear("sn.saliency2", sn->saliency2);
        save_linear("sn.saliency3", sn->saliency3);
        save_linear("sn.gate1", sn->gate1);
        save_linear("sn.gate2", sn->gate2);
        save_linear("sn.anomaly1", sn->anomaly1);
        save_linear("sn.anomaly2", sn->anomaly2);
        
        // === 记忆 ===
        save_linear("memory.encode", memory->encode_proj);
        save_linear("memory.retrieve", memory->retrieve_proj);
        save_linear("memory.query_proj", memory->query_proj);
        save_tensor("memory.bank", memory->memory_bank);
        
        // === 流形投影 ===
        save_linear("manifold.proj1", manifold_proj1);
        save_tensor("manifold.norm.weight", manifold_norm->weight);
        save_tensor("manifold.norm.bias", manifold_norm->bias);
        save_linear("manifold.proj2", manifold_proj2);
        
        // === 输出融合 ===
        save_linear("output_fusion", output_fusion_linear);
        save_tensor("output_fusion.norm.weight", output_fusion_norm->weight);
        save_tensor("output_fusion.norm.bias", output_fusion_norm->bias);
        
        // End marker
        uint32_t zero = 0;
        ofs.write(reinterpret_cast<const char*>(&zero), 4);
        ofs.close();
    }
    
    void load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) throw std::runtime_error("Cannot open file for load: " + path);
        
        // Magic header
        char magic[5] = {0};
        ifs.read(magic, 4);
        if (std::string(magic) != "NFv1") throw std::runtime_error("Invalid model file");
        
        // Helper: set tensor data from stream
        auto load_tensor_data = [&](Tensor& t) {
            uint32_t ndim, dsize;
            ifs.read(reinterpret_cast<char*>(&ndim), 4);
            std::vector<size_t> shape(ndim);
            for (uint32_t i = 0; i < ndim; ++i) {
                uint32_t dim;
                ifs.read(reinterpret_cast<char*>(&dim), 4);
                shape[i] = dim;
            }
            ifs.read(reinterpret_cast<char*>(&dsize), 4);
            if (t.data_size != dsize) {
                // Reallocate — data is shared_ptr<uint8_t>
                t.data.reset(reinterpret_cast<uint8_t*>(new float[dsize / sizeof(float)]));
                t.shape = shape;
                t.data_size = dsize;
            }
            ifs.read(reinterpret_cast<char*>(t.data.get()), dsize);
        };
        
        // Helper: load named Linear layer
        auto load_linear = [&](const std::shared_ptr<Linear>& layer, const std::string& suffix) {
            if (suffix == ".weight") load_tensor_data(layer->weight);
            else if (suffix == ".bias") { load_tensor_data(layer->bias); }
        };
        
        while (ifs.good()) {
            uint32_t name_len;
            ifs.read(reinterpret_cast<char*>(&name_len), 4);
            if (name_len == 0) break;  // end marker or EOF
            
            std::string name(name_len, '\0');
            ifs.read(&name[0], name_len);
            
            // === 输入投影 ===
            if (name == "input_proj.weight") load_tensor_data(input_proj_linear->weight);
            else if (name == "input_proj.bias") load_tensor_data(input_proj_linear->bias);
            else if (name == "input_proj_norm.weight") load_tensor_data(input_proj_norm->weight);
            else if (name == "input_proj_norm.bias") load_tensor_data(input_proj_norm->bias);
            
            // === ECN ===
            else if (name.rfind("ecn.dlpfc", 0) == 0) {
                std::string suffix = name.substr(name.find('.', 4));  // e.g. ".weight"
                int idx = std::stoi(name.substr(9, name.find('.', 9) - 9));
                load_linear(ecn->dlpfc_linear[idx], suffix);
            }
            else if (name == "ecn.ofc1.weight") load_tensor_data(ecn->ofc1->weight);
            else if (name == "ecn.ofc1.bias") load_tensor_data(ecn->ofc1->bias);
            else if (name == "ecn.ofc2.weight") load_tensor_data(ecn->ofc2->weight);
            else if (name == "ecn.ofc2.bias") load_tensor_data(ecn->ofc2->bias);
            else if (name == "ecn.vmpfc1.weight") load_tensor_data(ecn->vmpfc1->weight);
            else if (name == "ecn.vmpfc1.bias") load_tensor_data(ecn->vmpfc1->bias);
            else if (name == "ecn.vmpfc2.weight") load_tensor_data(ecn->vmpfc2->weight);
            else if (name == "ecn.vmpfc2.bias") load_tensor_data(ecn->vmpfc2->bias);
            
            // === DMN ===
            else if (name == "dmn.mem_encoder1.weight") load_tensor_data(dmn->mem_encoder1->weight);
            else if (name == "dmn.mem_encoder1.bias") load_tensor_data(dmn->mem_encoder1->bias);
            else if (name == "dmn.mem_encoder2.weight") load_tensor_data(dmn->mem_encoder2->weight);
            else if (name == "dmn.mem_encoder2.bias") load_tensor_data(dmn->mem_encoder2->bias);
            else if (name == "dmn.future_proj1.weight") load_tensor_data(dmn->future_proj1->weight);
            else if (name == "dmn.future_proj1.bias") load_tensor_data(dmn->future_proj1->bias);
            else if (name.rfind("dmn.head", 0) == 0) {
                // dmn.head<N>.<1|2>.<weight|bias>
                int h = std::stoi(name.substr(8, name.find('.', 8) - 8));
                int which = name[10 + (h >= 10 ? 1 : 0)] - '0';
                std::string suffix = name.substr(name.rfind('.'));
                auto& layer = (which == 1) ? dmn->association_heads[h].first : dmn->association_heads[h].second;
                load_linear(layer, suffix);
            }
            
            // === SN ===
            else if (name == "sn.saliency1.weight") load_tensor_data(sn->saliency1->weight);
            else if (name == "sn.saliency1.bias") load_tensor_data(sn->saliency1->bias);
            else if (name == "sn.saliency2.weight") load_tensor_data(sn->saliency2->weight);
            else if (name == "sn.saliency2.bias") load_tensor_data(sn->saliency2->bias);
            else if (name == "sn.saliency3.weight") load_tensor_data(sn->saliency3->weight);
            else if (name == "sn.saliency3.bias") load_tensor_data(sn->saliency3->bias);
            else if (name == "sn.gate1.weight") load_tensor_data(sn->gate1->weight);
            else if (name == "sn.gate1.bias") load_tensor_data(sn->gate1->bias);
            else if (name == "sn.gate2.weight") load_tensor_data(sn->gate2->weight);
            else if (name == "sn.gate2.bias") load_tensor_data(sn->gate2->bias);
            else if (name == "sn.anomaly1.weight") load_tensor_data(sn->anomaly1->weight);
            else if (name == "sn.anomaly1.bias") load_tensor_data(sn->anomaly1->bias);
            else if (name == "sn.anomaly2.weight") load_tensor_data(sn->anomaly2->weight);
            else if (name == "sn.anomaly2.bias") load_tensor_data(sn->anomaly2->bias);
            
            // === 记忆 ===
            else if (name == "memory.encode.weight") load_tensor_data(memory->encode_proj->weight);
            else if (name == "memory.encode.bias") load_tensor_data(memory->encode_proj->bias);
            else if (name == "memory.retrieve.weight") load_tensor_data(memory->retrieve_proj->weight);
            else if (name == "memory.retrieve.bias") load_tensor_data(memory->retrieve_proj->bias);
            else if (name == "memory.query_proj.weight") load_tensor_data(memory->query_proj->weight);
            else if (name == "memory.query_proj.bias") load_tensor_data(memory->query_proj->bias);
            else if (name == "memory.bank") load_tensor_data(memory->memory_bank);
            
            // === 流形投影 ===
            else if (name == "manifold.proj1.weight") load_tensor_data(manifold_proj1->weight);
            else if (name == "manifold.proj1.bias") load_tensor_data(manifold_proj1->bias);
            else if (name == "manifold.norm.weight") load_tensor_data(manifold_norm->weight);
            else if (name == "manifold.norm.bias") load_tensor_data(manifold_norm->bias);
            else if (name == "manifold.proj2.weight") load_tensor_data(manifold_proj2->weight);
            else if (name == "manifold.proj2.bias") load_tensor_data(manifold_proj2->bias);
            
            // === 输出融合 ===
            else if (name == "output_fusion.weight") load_tensor_data(output_fusion_linear->weight);
            else if (name == "output_fusion.bias") load_tensor_data(output_fusion_linear->bias);
            else if (name == "output_fusion.norm.weight") load_tensor_data(output_fusion_norm->weight);
            else if (name == "output_fusion.norm.bias") load_tensor_data(output_fusion_norm->bias);
            
            // Unknown layer — skip
            else {
                uint32_t ndim, dsize;
                ifs.read(reinterpret_cast<char*>(&ndim), 4);
                ifs.seekg(ndim * 4, std::ios::cur);
                ifs.read(reinterpret_cast<char*>(&dsize), 4);
                ifs.seekg(dsize, std::ios::cur);
            }
        }
        ifs.close();
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
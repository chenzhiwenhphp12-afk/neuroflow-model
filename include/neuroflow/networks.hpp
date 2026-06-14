#ifndef NEUROFLOW_NETWORKS_HPP
#define NEUROFLOW_NETWORKS_HPP

/**
 * NeuroFlow 核心网络模块
 * 
 * 1. ExecutiveControlNetwork (ECN) - 执行控制
 * 2. DefaultModeNetwork (DMN) - 默认模式/联想
 * 3. SalienceNetwork (SN) - 显著性检测
 */

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <thread>
#include <variant>

namespace neuroflow {

/**
 * Linear层 - 基础线性变换
 * 支持量化权重
 */
class Linear {
public:
    Tensor weight;
    Tensor bias;
    Tensor weight_scale;  // 量化scale
    bool quantized;
    
    Linear(size_t in_features, size_t out_features, bool use_bias = true, bool quant = false)
        : quantized(quant) {
        if (quant) {
            weight = Tensor({out_features, in_features}, QuantType::INT8);
            weight_scale = Tensor({out_features}, QuantType::FP32);
        } else {
            weight = Tensor({out_features, in_features}, QuantType::FP32);
            // 初始化权重 ( Xavier )
            float* w = weight.as_fp32();
            float scale = std::sqrt(2.0f / (in_features + out_features));
            size_t n = weight.numel();
            std::mt19937 init_rng(std::hash<std::thread::id>{}(std::this_thread::get_id()) + in_features * 31 + out_features);
            std::uniform_real_distribution<float> dist(-scale, scale);
            for (size_t i = 0; i < n; ++i) {
                w[i] = dist(init_rng);
            }
        }
        
        if (use_bias) {
            bias = Tensor({out_features}, QuantType::FP32);
            memset(bias.data_.get(), 0, bias.data_size_);
        }
    }
    
    Tensor forward(const Tensor& input) {
        Tensor output({input.shape_[0], weight.shape_[0]}, QuantType::FP32);
        
        if (quantized) {
            // 量化矩阵乘法 - INT8权重需要不同处理
            // 这里简化为普通gemm
            TensorOps::gemm(input, weight, output, false, true);
        } else {
            // weight形状是 [out_features, in_features]
            // output = input @ weight^T
            TensorOps::gemm(input, weight, output, false, true);
        }
        
        // 加bias (只有当 bias 存在时)
        if (bias.data_) {
            float* out = output.as_fp32();
            float* b = bias.as_fp32();
            for (size_t i = 0; i < output.shape_[0]; ++i) {
                for (size_t j = 0; j < output.shape_[1]; ++j) {
                    out[i * output.shape_[1] + j] += b[j];
                }
            }
        }
        
        return output;
    }
    
    // 量化权重
    void quantize() {
        if (quantized) return;
        
        Tensor new_weight({weight.shape_[0], weight.shape_[1]}, QuantType::INT8);
        Tensor scale({weight.shape_[0]}, QuantType::FP32);
        
        TensorOps::quantize_int8(weight, new_weight, scale);
        
        weight = new_weight;
        weight_scale = scale;
        quantized = true;
    }
};

/**
 * LayerNorm层
 */
class LayerNorm {
public:
    Tensor weight;
    Tensor bias;
    float eps;
    
    LayerNorm(size_t dim, float epsilon = 1e-5f) : eps(epsilon) {
        weight = Tensor({dim}, QuantType::FP32);
        bias = Tensor({dim}, QuantType::FP32);
        
        float* w = weight.as_fp32();
        float* b = bias.as_fp32();
        for (size_t i = 0; i < dim; ++i) {
            w[i] = 1.0f;
            b[i] = 0.0f;
        }
    }
    
    Tensor forward(const Tensor& input) {
        Tensor output = input.clone();
        TensorOps::layer_norm(output, weight, bias, eps);
        return output;
    }
};

/**
 * GELU激活层
 */
class GELU {
public:
    Tensor forward(const Tensor& input) {
        Tensor output = input.clone();
        TensorOps::gelu(output);
        return output;
    }
};

/**
 * Dropout层
 */
class Dropout {
public:
    float rate;
    bool training;
    
    Dropout(float r = 0.1f) : rate(r), training(false) {}
    
    Tensor forward(const Tensor& input) {
        Tensor output = input.clone();
        TensorOps::dropout(output, rate, training);
        return output;
    }
    
    void set_training(bool t) { training = t; }
};

/**
 * Sequential容器
 */
class Sequential {
public:
    using LayerVariant = std::variant<
        std::shared_ptr<Linear>,
        std::shared_ptr<LayerNorm>,
        std::shared_ptr<GELU>,
        std::shared_ptr<Dropout>
    >;
    
    std::vector<LayerVariant> layers;
    
    template<typename T>
    void add(std::shared_ptr<T> layer) {
        layers.push_back(layer);
    }
    
    template<typename T>
    std::shared_ptr<T> get(size_t idx) {
        return std::get<std::shared_ptr<T>>(layers[idx]);
    }
};

/**
 * ExecutiveControlNetwork (ECN)
 * 
 * 模拟背外侧前额叶 (dlPFC)、眶额叶皮层 (OFC)、腹内侧前额叶 (vmPFC)
 * 功能：逻辑推理、价值评估、决策输出
 */
class ExecutiveControlNetwork {
public:
    // dlPFC: 多层处理
    std::vector<std::shared_ptr<Linear>> dlpfc_linear;
    std::vector<std::shared_ptr<LayerNorm>> dlpfc_norm;
    std::vector<std::shared_ptr<GELU>> dlpfc_gelu;
    std::vector<std::shared_ptr<Dropout>> dlpfc_drop;
    
    // OFC: 价值评估
    std::shared_ptr<Linear> ofc1, ofc2;
    
    // vmPFC: 决策输出
    std::shared_ptr<Linear> vmpfc1, vmpfc2;
    
    size_t num_layers;
    size_t hidden_dim;
    
    ExecutiveControlNetwork(size_t input_dim, size_t hidden_dim, size_t output_dim, size_t layers = 2)
        : num_layers(layers), hidden_dim(hidden_dim) {
        
        // dlPFC层
        size_t prev_dim = input_dim;
        for (size_t i = 0; i < layers; ++i) {
            dlpfc_linear.push_back(std::make_shared<Linear>(prev_dim, hidden_dim));
            dlpfc_norm.push_back(std::make_shared<LayerNorm>(hidden_dim));
            dlpfc_gelu.push_back(std::make_shared<GELU>());
            dlpfc_drop.push_back(std::make_shared<Dropout>(0.1f));
            prev_dim = hidden_dim;
        }
        
        // OFC: 价值评估 (hidden -> hidden/2 -> 1)
        size_t half = hidden_dim / 2;
        ofc1 = std::make_shared<Linear>(hidden_dim, half);
        ofc2 = std::make_shared<Linear>(half, 1);
        
        // vmPFC: 决策 (hidden -> hidden/2 -> output)
        vmpfc1 = std::make_shared<Linear>(hidden_dim, half);
        vmpfc2 = std::make_shared<Linear>(half, output_dim);
    }
    
    struct Output {
        Tensor decision;   // 决策输出
        Tensor value;      // 价值评估
        std::vector<Tensor> hidden_states;  // 中间层激活 (用于流形分析)
    };
    
    Output forward(const Tensor& x) {
        Output out;
        Tensor h = x;
        
        // dlPFC处理
        for (size_t i = 0; i < num_layers; ++i) {
            h = dlpfc_linear[i]->forward(h);
            h = dlpfc_norm[i]->forward(h);
            h = dlpfc_gelu[i]->forward(h);
            h = dlpfc_drop[i]->forward(h);
            out.hidden_states.push_back(h.clone());
        }
        
        // OFC: 价值评估
        Tensor v = ofc1->forward(h);
        TensorOps::gelu(v);
        out.value = ofc2->forward(v);
        
        // vmPFC: 决策
        Tensor d = vmpfc1->forward(h);
        TensorOps::gelu(d);
        out.decision = vmpfc2->forward(d);
        
        return out;
    }
    
    void set_training(bool t) {
        for (auto& drop : dlpfc_drop) drop->set_training(t);
    }
    
    // 量化所有线性层
    void quantize() {
        for (auto& l : dlpfc_linear) l->quantize();
        ofc1->quantize();
        ofc2->quantize();
        vmpfc1->quantize();
        vmpfc2->quantize();
    }
};

/**
 * DefaultModeNetwork (DMN)
 * 
 * 模拟后扣带回 (PCC)、内侧前额叶 (mPFC)
 * 功能：记忆检索、未来规划、创造性联想
 */
class DefaultModeNetwork {
public:
    size_t memory_dim;
    size_t latent_dim;
    size_t num_associations;
    
    // 记忆编码
    std::shared_ptr<Linear> mem_encoder1, mem_encoder2;
    
    // 联想头
    std::vector<std::pair<std::shared_ptr<Linear>, std::shared_ptr<Linear>>> association_heads;
    
    // 未来投影
    std::shared_ptr<Linear> future_proj1;
    std::shared_ptr<LayerNorm> future_norm;
    std::shared_ptr<GELU> future_gelu;
    
    DefaultModeNetwork(size_t memory_dim, size_t latent_dim, size_t num_assoc = 8)
        : memory_dim(memory_dim), latent_dim(latent_dim), num_associations(num_assoc) {
        
        // 记忆编码器
        mem_encoder1 = std::make_shared<Linear>(memory_dim, latent_dim * 2);
        mem_encoder2 = std::make_shared<Linear>(latent_dim * 2, latent_dim);
        
        // 联想头
        for (size_t i = 0; i < num_assoc; ++i) {
            auto head1 = std::make_shared<Linear>(latent_dim, latent_dim);
            auto head2 = std::make_shared<Linear>(latent_dim, latent_dim);
            association_heads.push_back({head1, head2});
        }
        
        // 未来投影
        future_proj1 = std::make_shared<Linear>(latent_dim * num_assoc, latent_dim * 2);
        future_norm = std::make_shared<LayerNorm>(latent_dim * 2);
        future_gelu = std::make_shared<GELU>();
    }
    
    struct Output {
        Tensor vision;        // 未来愿景
        std::vector<Tensor> associations;  // 各联想头输出
        Tensor latent;        // 潜在记忆表征
    };
    
    Output forward(const Tensor& memory_input) {
        Output out;
        
        // 编码记忆
        Tensor h = mem_encoder1->forward(memory_input);
        TensorOps::gelu(h);
        out.latent = mem_encoder2->forward(h);
        
        // 各联想头处理
        for (auto& head : association_heads) {
            Tensor assoc = head.first->forward(out.latent);
            TensorOps::gelu(assoc);
            assoc = head.second->forward(assoc);
            out.associations.push_back(assoc);
        }
        
        // 合并联想
        out.vision = TensorOps::concat(out.associations, 1);
        out.vision = future_proj1->forward(out.vision);
        out.vision = future_norm->forward(out.vision);
        out.vision = future_gelu->forward(out.vision);
        
        return out;
    }
    
    void quantize() {
        mem_encoder1->quantize();
        mem_encoder2->quantize();
        future_proj1->quantize();
        for (auto& [h1, h2] : association_heads) {
            h1->quantize();
            h2->quantize();
        }
    }
};

/**
 * SalienceNetwork (SN)
 * 
 * 模拟前岛叶 (AI)、前扣带回 (ACC)
 * 功能：显著性检测、ECN/DMN门控、异常检测
 */
class SalienceNetwork {
public:
    // 显著性评分
    std::shared_ptr<Linear> saliency1, saliency2, saliency3;
    
    // 门控生成
    std::shared_ptr<Linear> gate1, gate2;
    
    // 异常检测
    std::shared_ptr<Linear> anomaly1, anomaly2;
    
    SalienceNetwork(size_t input_dim, size_t hidden_dim) {
        // 显著性评分 (sigmoid输出)
        saliency1 = std::make_shared<Linear>(input_dim, hidden_dim);
        saliency2 = std::make_shared<Linear>(hidden_dim, hidden_dim / 2);
        saliency3 = std::make_shared<Linear>(hidden_dim / 2, 1);
        
        // 门控 (softmax 2-class)
        gate1 = std::make_shared<Linear>(input_dim, hidden_dim);
        gate2 = std::make_shared<Linear>(hidden_dim, 2);
        
        // 异常检测
        anomaly1 = std::make_shared<Linear>(input_dim, hidden_dim);
        anomaly2 = std::make_shared<Linear>(hidden_dim, 1);
    }
    
    struct Output {
        Tensor saliency;   // 显著性评分 [0,1]
        Tensor gates;      // ECN/DMN门控权重
        Tensor anomaly;    // 异常评分
    };
    
    Output forward(const Tensor& x, const Tensor* baseline = nullptr) {
        Output out;
        
        // 显著性
        Tensor h = saliency1->forward(x);
        TensorOps::gelu(h);
        h = saliency2->forward(h);
        TensorOps::gelu(h);
        out.saliency = saliency3->forward(h);
        // sigmoid
        float* s = out.saliency.as_fp32();
        for (size_t i = 0; i < out.saliency.numel(); ++i) {
            s[i] = 1.0f / (1.0f + std::exp(-s[i]));
        }
        
        // 门控
        h = gate1->forward(x);
        TensorOps::gelu(h);
        out.gates = gate2->forward(h);
        TensorOps::softmax(out.gates);
        
        // 异常
        if (baseline) {
            Tensor diff = x.clone();

            float* d = diff.as_fp32();
            const float* b = baseline->as_fp32();
            for (size_t i = 0; i < diff.numel(); ++i) d[i] -= b[i];
            
            h = anomaly1->forward(diff);
            TensorOps::gelu(h);
            out.anomaly = anomaly2->forward(h);
        } else {
            out.anomaly = Tensor({x.shape_[0], 1}, QuantType::FP32);
        }
        
        return out;
    }
    
    void quantize() {
        saliency1->quantize();
        saliency2->quantize();
        saliency3->quantize();
        gate1->quantize();
        gate2->quantize();
        anomaly1->quantize();
        anomaly2->quantize();
    }
};

} // namespace neuroflow

#endif // NEUROFLOW_NETWORKS_HPP
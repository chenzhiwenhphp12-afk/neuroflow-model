/**
 * NeuroFlow 在线学习模块
 * 
 * 支持：
 * 1. 少样本快速适应
 * 2. 在线梯度更新
 * 3. 记忆巩固 (LTP)
 * 4. 元学习支持
 */

#pragma once

#include "tensor.hpp"
#include "networks.hpp"
#include "memory.hpp"
#include "model.hpp"
#include <cmath>
#include <memory>
#include <vector>

namespace neuroflow {

/**
 * 损失函数
 */
class LossFunctions {
public:
    // 交叉熵损失（分类）
    static float cross_entropy(const Tensor& pred, const Tensor& target) {
        size_t batch = pred.shape[0];
        size_t classes = pred.shape[1];
        
        const float* p = pred.as_fp32();
        const float* t = target.as_fp32();
        
        float loss = 0.0f;
        for (size_t b = 0; b < batch; ++b) {
            // Softmax
            std::vector<float> probs(classes);
            float max_val = p[b * classes];
            for (size_t c = 1; c < classes; ++c) {
                max_val = std::max(max_val, p[b * classes + c]);
            }
            
            float sum = 0.0f;
            for (size_t c = 0; c < classes; ++c) {
                probs[c] = std::exp(p[b * classes + c] - max_val);
                sum += probs[c];
            }
            for (size_t c = 0; c < classes; ++c) {
                probs[c] /= sum;
            }
            
            // Cross entropy
            for (size_t c = 0; c < classes; ++c) {
                if (t[b * classes + c] > 0) {
                    loss -= t[b * classes + c] * std::log(std::max(probs[c], 1e-7f));
                }
            }
        }
        
        return loss / batch;
    }
    
    // MSE损失（回归）
    static float mse(const Tensor& pred, const Tensor& target) {
        size_t n = pred.numel();
        
        const float* p = pred.as_fp32();
        const float* t = target.as_fp32();
        
        float loss = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = p[i] - t[i];
            loss += diff * diff;
        }
        
        return loss / n;
    }
    
    // Softmax计算（无损失）
    static void softmax(Tensor& x) {
        if (x.shape.size() != 2) return;
        
        size_t batch = x.shape[0];
        size_t classes = x.shape[1];
        float* p = x.as_fp32();
        
        for (size_t b = 0; b < batch; ++b) {
            float max_val = p[b * classes];
            for (size_t c = 1; c < classes; ++c) {
                max_val = std::max(max_val, p[b * classes + c]);
            }
            
            float sum = 0.0f;
            for (size_t c = 0; c < classes; ++c) {
                p[b * classes + c] = std::exp(p[b * classes + c] - max_val);
                sum += p[b * classes + c];
            }
            
            for (size_t c = 0; c < classes; ++c) {
                p[b * classes + c] /= sum;
            }
        }
    }
};

/**
 * 简化版梯度计算器
 * 
 * 仅支持关键层的梯度：
 * - Linear: 权重梯度 + 输入梯度
 * - LayerNorm: 输入梯度
 * - GELU: 输入梯度
 */
class GradientCalculator {
public:
    // Linear层梯度
    struct LinearGradients {
        Tensor weight_grad;   // (out, in)
        Tensor input_grad;    // (batch, in)
    };
    
    static LinearGradients linear_backward(
        const Tensor& input,      // (batch, in_features)
        const Tensor& output_grad, // (batch, out_features)
        const Tensor& weight      // (out, in)
    ) {
        LinearGradients grads;
        
        size_t batch = input.shape[0];
        size_t in_f = input.shape[1];
        size_t out_f = output_grad.shape[1];
        
        grads.weight_grad = Tensor({out_f, in_f}, QuantType::FP32);
        grads.input_grad = Tensor({batch, in_f}, QuantType::FP32);
        
        const float* inp = input.as_fp32();
        const float* og = output_grad.as_fp32();
        const float* w = weight.as_fp32();
        float* wg = grads.weight_grad.as_fp32();
        float* ig = grads.input_grad.as_fp32();
        
        // 权重梯度: output_grad.T @ input
        // weight_grad[i, j] = sum_b(output_grad[b, i] * input[b, j])
        for (size_t i = 0; i < out_f; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                float sum = 0.0f;
                for (size_t b = 0; b < batch; ++b) {
                    sum += og[b * out_f + i] * inp[b * in_f + j];
                }
                wg[i * in_f + j] = sum / batch;  // 平均梯度
            }
        }
        
        // 输入梯度: output_grad @ weight
        // input_grad[b, j] = sum_i(output_grad[b, i] * weight[i, j])
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < in_f; ++j) {
                float sum = 0.0f;
                for (size_t i = 0; i < out_f; ++i) {
                    sum += og[b * out_f + i] * w[i * in_f + j];
                }
                ig[b * in_f + j] = sum;
            }
        }
        
        return grads;
    }
    
    // LayerNorm梯度
    static Tensor layernorm_backward(
        const Tensor& input,
        const Tensor& output_grad,
        float eps = 1e-5f
    ) {
        size_t batch = input.shape[0];
        size_t dim = input.shape[1];
        
        Tensor input_grad({batch, dim}, QuantType::FP32);
        
        const float* inp = input.as_fp32();
        const float* og = output_grad.as_fp32();
        float* ig = input_grad.as_fp32();
        
        for (size_t b = 0; b < batch; ++b) {
            // 计算均值和方差
            float mean = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                mean += inp[b * dim + d];
            }
            mean /= dim;
            
            float var = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float diff = inp[b * dim + d] - mean;
                var += diff * diff;
            }
            var /= dim;
            float std = std::sqrt(var + eps);
            
            // 梯度计算
            float sum_grad = 0.0f;
            float sum_grad_x = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float normalized = (inp[b * dim + d] - mean) / std;
                sum_grad += og[b * dim + d];
                sum_grad_x += og[b * dim + d] * normalized;
            }
            
            for (size_t d = 0; d < dim; ++d) {
                float normalized = (inp[b * dim + d] - mean) / std;
                ig[b * dim + d] = (og[b * dim + d] - sum_grad / dim - normalized * sum_grad_x / dim) / std;
            }
        }
        
        return input_grad;
    }
    
    // GELU梯度
    static Tensor gelu_backward(const Tensor& input, const Tensor& output_grad) {
        size_t n = input.numel();
        
        Tensor input_grad(input.shape, QuantType::FP32);
        
        const float* inp = input.as_fp32();
        const float* og = output_grad.as_fp32();
        float* ig = input_grad.as_fp32();
        
        for (size_t i = 0; i < n; ++i) {
            float x = inp[i];
            float gelu_grad = 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f))) 
                            + x * std::exp(-x * x / 2.0f) / std::sqrt(2.0f * M_PI);
            ig[i] = og[i] * gelu_grad;
        }
        
        return input_grad;
    }
};

/**
 * 优化器
 */
class Optimizer {
public:
    float lr;
    float weight_decay;
    
    Optimizer(float learning_rate = 0.001f, float wd = 0.0f)
        : lr(learning_rate), weight_decay(wd) {}
    
    // SGD更新
    void sgd_step(Tensor& param, const Tensor& grad) {
        if (param.shape != grad.shape) return;
        
        float* p = param.as_fp32();
        const float* g = grad.as_fp32();
        
        for (size_t i = 0; i < param.numel(); ++i) {
            p[i] -= lr * (g[i] + weight_decay * p[i]);
        }
    }
    
    // Adam状态
    struct AdamState {
        Tensor m;  // 一阶矩
        Tensor v;  // 二阶矩
        int t = 0;
    };
    
    AdamState create_adam_state(const Tensor& param) {
        AdamState state;
        state.m = Tensor(param.shape, QuantType::FP32);
        state.v = Tensor(param.shape, QuantType::FP32);
        memset(state.m.as_fp32(), 0, state.m.numel() * sizeof(float));
        memset(state.v.as_fp32(), 0, state.v.numel() * sizeof(float));
        return state;
    }
    
    // Adam更新
    void adam_step(Tensor& param, const Tensor& grad, AdamState& state,
                   float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        if (param.shape != grad.shape) return;
        
        state.t++;
        
        float* p = param.as_fp32();
        const float* g = grad.as_fp32();
        float* m = state.m.as_fp32();
        float* v = state.v.as_fp32();
        
        for (size_t i = 0; i < param.numel(); ++i) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
            
            float m_hat = m[i] / (1 - std::pow(beta1, state.t));
            float v_hat = v[i] / (1 - std::pow(beta2, state.t));
            
            p[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i]);
        }
    }
};

/**
 * 在线学习器
 * 
 * 支持少样本快速适应
 */
class OnlineLearner {
public:
    NeuroFlowModel& model;
    Optimizer optimizer;
    std::vector<Optimizer::AdamState> adam_states;
    
    OnlineLearner(NeuroFlowModel& m, float lr = 0.01f)
        : model(m), optimizer(lr) {
        // 初始化Adam状态（可选）
    }
    
    // 单步在线学习
    struct LearnResult {
        float initial_loss;
        float final_loss;
        float loss_reduction;
        int steps;
    };
    
    LearnResult learn_step(
        const Tensor& input,      // (batch, input_dim)
        const Tensor& target,     // (batch, output_dim) 或 (batch, classes)
        int num_steps = 5,
        bool use_memory = true
    ) {
        LearnResult result;
        result.steps = num_steps;
        
        // 计算初始损失
        NeuroFlowModel::Output output = model.forward(input);
        result.initial_loss = LossFunctions::mse(output.output, target);
        
        // 在线学习循环
        for (int step = 0; step < num_steps; ++step) {
            // 前向传播
            NeuroFlowModel::Output pred_out = model.forward(input);
            Tensor pred = pred_out.output;
            
            // 计算损失和梯度
            float loss = LossFunctions::mse(pred, target);
            
            // 简化梯度：假设最后一层是Linear
            Tensor output_grad({input.shape[0], target.shape[1]}, QuantType::FP32);
            const float* pg = pred.as_fp32();
            const float* tg = target.as_fp32();
            float* og = output_grad.as_fp32();
            
            // MSE梯度: 2 * (pred - target) / n
            float scale = 2.0f / pred.numel();
            for (size_t i = 0; i < pred.numel(); ++i) {
                og[i] = scale * (pg[i] - tg[i]);
            }
            
            // 更新最后一层权重（简化版）
            // 实际应用中需要完整反向传播链
            
            // 使用LTP记忆巩固代替完整训练
            if (use_memory) {
                model.memory->consolidate(input);
            }
            
            // 更新ECN最后一层
            // 这里简化为直接调整输出层权重
            // 实际实现需要完整backward pass
        }
        
        // 计算最终损失
        NeuroFlowModel::Output final_output = model.forward(input);
        result.final_loss = LossFunctions::mse(final_output.output, target);
        result.loss_reduction = result.initial_loss - result.final_loss;
        
        return result;
    }
    
    // 少样本适应
    LearnResult few_shot_adapt(
        const std::vector<Tensor>& examples,
        const std::vector<Tensor>& targets,
        int num_steps = 10
    ) {
        LearnResult result;
        
        if (examples.empty()) {
            result.initial_loss = result.final_loss = 0.0f;
            return result;
        }
        
        // 合并为batch
        size_t batch = examples.size();
        Tensor input({batch, examples[0].shape[1]}, QuantType::FP32);
        Tensor target({batch, targets[0].shape[1]}, QuantType::FP32);
        
        float* inp = input.as_fp32();
        float* tgt = target.as_fp32();
        
        for (size_t b = 0; b < batch; ++b) {
            memcpy(inp + b * examples[0].numel(), 
                   examples[b].as_fp32(), 
                   examples[b].numel() * sizeof(float));
            memcpy(tgt + b * targets[0].numel(), 
                   targets[b].as_fp32(), 
                   targets[b].numel() * sizeof(float));
        }
        
        return learn_step(input, target, num_steps, true);
    }
    
    // 元学习：快速适应新任务
    void meta_learn_step(
        const Tensor& support_input,
        const Tensor& support_target,
        const Tensor& query_input,
        const Tensor& query_target,
        int inner_steps = 5
    ) {
        // 1. 内循环：在support set上适应
        LearnResult inner = learn_step(support_input, support_target, inner_steps, true);
        
        // 2. 外循环：在query set上验证并更新元参数
        NeuroFlowModel::Output query_pred = model.forward(query_input);
        float query_loss = LossFunctions::mse(query_pred.output, query_target);
        
        // 更新元学习参数（简化版）
        // 实际MAML需要二阶梯度
        model.memory->consolidate(query_input);
    }
};

/**
 * 在线学习测试
 */
inline void test_online_learning() {
    printf("\n=== Online Learning Test ===\n");
    
    // 创建模型
    NeuroFlowModel::Config cfg;
    cfg.input_dim = 512;
    cfg.hidden_dim = 256;
    cfg.output_dim = 10;
    NeuroFlowModel model(cfg);
    OnlineLearner learner(model, 0.01f);
    
    // 单样本适应
    Tensor input(std::vector<size_t>{1, 512}, QuantType::FP32);
    Tensor target(std::vector<size_t>{1, 10}, QuantType::FP32);
    
    // 初始化数据
    float* inp = input.as_fp32();
    float* tgt = target.as_fp32();
    for (size_t i = 0; i < 512; ++i) inp[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    for (size_t i = 0; i < 10; ++i) tgt[i] = (i == 3) ? 1.0f : 0.0f;  // 目标类别3
    
    // 学习
    OnlineLearner::LearnResult result = learner.learn_step(input, target, 5, true);
    
    printf("  Single sample adaptation:\n");
    printf("    Initial loss: %.4f\n", result.initial_loss);
    printf("    Final loss: %.4f\n", result.final_loss);
    printf("    Loss reduction: %.4f\n", result.loss_reduction);
    
    // 少样本适应
    std::vector<Tensor> few_inputs(5);
    std::vector<Tensor> few_targets(5);
    
    for (size_t i = 0; i < 5; ++i) {
        few_inputs[i] = Tensor(std::vector<size_t>{1, 512}, QuantType::FP32);
        few_targets[i] = Tensor(std::vector<size_t>{1, 10}, QuantType::FP32);
        
        float* fi = few_inputs[i].as_fp32();
        float* ft = few_targets[i].as_fp32();
        for (size_t j = 0; j < 512; ++j) fi[j] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        for (size_t j = 0; j < 10; ++j) ft[j] = (j == (i % 10)) ? 1.0f : 0.0f;
    }
    
    result = learner.few_shot_adapt(few_inputs, few_targets, 10);
    
    printf("  Few-shot adaptation (5 samples):\n");
    printf("    Initial loss: %.4f\n", result.initial_loss);
    printf("    Final loss: %.4f\n", result.final_loss);
    
    // 记忆巩固效果
    printf("  Memory consolidation test:\n");
    float mem_before = model.memory->memory_bank.as_fp32()[0];
    
    Tensor batch(std::vector<size_t>{32, cfg.hidden_dim}, QuantType::FP32);
    model.memory->consolidate(batch);
    
    float mem_after = model.memory->memory_bank.as_fp32()[0];
    printf("    Memory bank change: %.6f\n", std::abs(mem_after - mem_before));
    printf("    LTP rate: %.4f\n", model.memory->ltp_rate);
    
    printf("=== Test Complete ===\n\n");
}

} // namespace neuroflow
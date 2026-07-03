// 关联头文件
#include "neuroflow/model.hpp"

// C++ 标准库
#include <algorithm>
#include <cmath>

// 第三方库
#ifdef _OPENMP
#include <omp.h>
#endif

// 项目头文件
#include "neuroflow/backprop.hpp"

namespace neuroflow {

namespace {

constexpr size_t OMP_MIN_ITER = 1024;

#ifdef _OPENMP
using omp_idx_t = long long;
#else
using omp_idx_t = size_t;
#endif

Tensor linear_backward_input(const Tensor& output_grad, const Tensor& weight) {
    size_t batch = output_grad.shape_[0];
    size_t out_f = output_grad.shape_[1];
    size_t in_f = weight.shape_[1];
    Tensor input_grad({batch, in_f}, QuantType::FP32);
#ifdef USE_CBLAS
    const float* og = output_grad.as_fp32();
    const float* w = weight.as_fp32();
    float* ig = input_grad.as_fp32();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch, in_f, out_f,
                1.0f, og, out_f, w, in_f,
                0.0f, ig, in_f);
#else
    const float* og = output_grad.as_fp32();
    const float* w = weight.as_fp32();
    float* ig = input_grad.as_fp32();
    size_t total = batch * in_f;
    #pragma omp parallel for schedule(static) if(total >= OMP_MIN_ITER)
    for (omp_idx_t b = 0; b < static_cast<omp_idx_t>(batch); ++b) {
        for (size_t j = 0; j < in_f; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < out_f; ++i) {
                sum += og[b * out_f + i] * w[i * in_f + j];
            }
            ig[b * in_f + j] = sum;
        }
    }
#endif
    return input_grad;
}

Tensor linear_backward_weight(const Tensor& input, const Tensor& output_grad) {
    size_t batch = input.shape_[0];
    size_t in_f = input.shape_[1];
    size_t out_f = output_grad.shape_[1];
    Tensor weight_grad({out_f, in_f}, QuantType::FP32);
#ifdef USE_CBLAS
    const float* inp = input.as_fp32();
    const float* og = output_grad.as_fp32();
    float* wg = weight_grad.as_fp32();
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                out_f, in_f, batch,
                1.0f / batch, og, out_f, inp, in_f,
                0.0f, wg, in_f);
#else
    const float* inp = input.as_fp32();
    const float* og = output_grad.as_fp32();
    float* wg = weight_grad.as_fp32();
    size_t total = out_f * in_f;
    #pragma omp parallel for schedule(static) if(total >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(out_f); ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch; ++b) {
                sum += og[b * out_f + i] * inp[b * in_f + j];
            }
            wg[i * in_f + j] = sum / batch;
        }
    }
#endif
    return weight_grad;
}

Tensor bias_backward(const Tensor& output_grad) {
    size_t batch = output_grad.shape_[0];
    size_t dim = output_grad.shape_[1];
    Tensor grad({dim}, QuantType::FP32);
    const float* og = output_grad.as_fp32();
    float* g = grad.as_fp32();
    for (size_t j = 0; j < dim; ++j) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch; ++b) sum += og[b * dim + j];
        g[j] = sum / batch;
    }
    return grad;
}

Tensor gelu_backward(const Tensor& input, const Tensor& output_grad) {
    size_t n = input.numel();
    Tensor input_grad(input.shape_, QuantType::FP32);
    const float* inp = input.as_fp32();
    const float* og = output_grad.as_fp32();
    float* ig = input_grad.as_fp32();
    const float SQRT_2_INV = 0.7071067811865476f;
    const float SQRT_2PI_INV = 0.3989422804014327f;
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) {
        float x = inp[i];
        float gelu_grad = 0.5f * (1.0f + std::erf(x * SQRT_2_INV))
                         + x * std::exp(-0.5f * x * x) * SQRT_2PI_INV;
        ig[i] = og[i] * gelu_grad;
    }
    return input_grad;
}

Tensor layernorm_backward(const Tensor& input, const Tensor& weight, const Tensor& output_grad, float eps = 1e-5f) {
    size_t batch = input.shape_[0];
    size_t dim = input.shape_[1];
    Tensor input_grad({batch, dim}, QuantType::FP32);
    const float* inp = input.as_fp32();
    const float* w = weight.as_fp32();
    const float* og = output_grad.as_fp32();
    float* ig = input_grad.as_fp32();
    #pragma omp parallel for schedule(static) if(batch >= 16)
    for (omp_idx_t b = 0; b < static_cast<omp_idx_t>(batch); ++b) {
        float mean = 0.0f;
        for (size_t d = 0; d < dim; ++d) mean += inp[b * dim + d];
        mean /= dim;
        float var = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float diff = inp[b * dim + d] - mean;
            var += diff * diff;
        }
        var /= dim;
        float inv_std = 1.0f / std::sqrt(var + eps);
        float sum_gn = 0.0f, sum_gnx = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float norm = (inp[b * dim + d] - mean) * inv_std;
            float gn = og[b * dim + d] * w[d];
            sum_gn += gn;
            sum_gnx += gn * norm;
        }
        for (size_t d = 0; d < dim; ++d) {
            float norm = (inp[b * dim + d] - mean) * inv_std;
            float gn = og[b * dim + d] * w[d];
            ig[b * dim + d] = inv_std * (gn - sum_gn / dim - norm * sum_gnx / dim);
        }
    }
    return input_grad;
}

// LayerNorm 参数梯度: 计算 γ (weight) 和 β (bias) 的梯度
struct LayernormParamGrads {
    Tensor weight_grad;  // γ gradient, shape [dim]
    Tensor bias_grad;    // β gradient, shape [dim]
};

LayernormParamGrads layernorm_param_backward(
    const Tensor& input, const Tensor& output_grad, float eps = 1e-5f) {
    size_t batch = input.shape_[0];
    size_t dim = input.shape_[1];

    LayernormParamGrads result;
    result.weight_grad = Tensor({dim}, QuantType::FP32);
    result.bias_grad = Tensor({dim}, QuantType::FP32);

    const float* inp = input.as_fp32();
    const float* og = output_grad.as_fp32();
    float* wg = result.weight_grad.as_fp32();
    float* bg = result.bias_grad.as_fp32();

    // Precompute mean and inv_std for each batch element
    std::vector<float> means(batch), inv_stds(batch);
    for (size_t b = 0; b < batch; ++b) {
        float mean = 0.0f;
        for (size_t d = 0; d < dim; ++d) mean += inp[b * dim + d];
        mean /= dim;
        means[b] = mean;
        float var = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float diff = inp[b * dim + d] - mean;
            var += diff * diff;
        }
        var /= dim;
        inv_stds[b] = 1.0f / std::sqrt(var + eps);
    }

    // γ_grad[d] = sum_b(output_grad[b,d] * (input[b,d]-mean[b])/std[b]) / batch
    // β_grad[d] = sum_b(output_grad[b,d]) / batch
    for (size_t d = 0; d < dim; ++d) {
        float w_sum = 0.0f, b_sum = 0.0f;
        for (size_t b = 0; b < batch; ++b) {
            float normalized = (inp[b * dim + d] - means[b]) * inv_stds[b];
            w_sum += og[b * dim + d] * normalized;
            b_sum += og[b * dim + d];
        }
        wg[d] = w_sum / batch;
        bg[d] = b_sum / batch;
    }

    return result;
}

}

FullBackpropEngine::FullBackpropEngine(NeuroFlowModel& m) : model(m) {}

NeuroFlowModel::Output FullBackpropEngine::forward_with_cache(const Tensor& x) {
    cache.input = x.clone();
    size_t batch = x.shape_[0];

    cache.input_proj_pre = model.input_proj_linear->forward(x);
    cache.input_proj_post = model.input_proj_norm->forward(cache.input_proj_pre);
    cache.h = model.input_proj_gelu->forward(cache.input_proj_post);

    auto sn_out = model.sn->forward(cache.h);
    cache.sn_gates = sn_out.gates.clone();

    // 缓存SN gate1输出 (修复: pre-gelu用于gelu_backward, post-gelu用于gate2 backward)
    {
        Tensor gh = model.sn->gate1->forward(cache.h);
        cache.sn_gate_h = gh.clone();  // pre-gelu!
        TensorOps::gelu(gh);
        cache.sn_gate_h_post = gh.clone();  // post-gelu
    }

    Tensor h_ecn = cache.h.clone();
    cache.ecn_hidden.clear();
    cache.ecn_pre_linear.clear();
    cache.ecn_pre_norm.clear();
    for (size_t i = 0; i < model.ecn->num_layers; ++i) {
        cache.ecn_pre_linear.push_back(h_ecn.clone());
        h_ecn = model.ecn->dlpfc_linear[i]->forward(h_ecn);
        cache.ecn_pre_norm.push_back(h_ecn.clone());
        h_ecn = model.ecn->dlpfc_norm[i]->forward(h_ecn);
        h_ecn = model.ecn->dlpfc_gelu[i]->forward(h_ecn);
        cache.ecn_hidden.push_back(h_ecn.clone());
    }

    cache.ecn_vmpfc_pre = model.ecn->vmpfc1->forward(cache.ecn_hidden.back());
    cache.ecn_vmpfc_d = cache.ecn_vmpfc_pre.clone();
    TensorOps::gelu(cache.ecn_vmpfc_d);

    cache.ecn_ofc_pre = model.ecn->ofc1->forward(cache.ecn_hidden.back());
    cache.ecn_ofc_v = cache.ecn_ofc_pre.clone();
    TensorOps::gelu(cache.ecn_ofc_v);

    cache.memory_encoded = model.memory->encode(cache.h);
    cache.dmn_encoded = model.dmn->mem_encoder1->forward(cache.memory_encoded);
    TensorOps::gelu(cache.dmn_encoded);
    cache.dmn_latent = model.dmn->mem_encoder2->forward(cache.dmn_encoded);

    cache.dmn_associations.clear();
    cache.dmn_head1_outs.clear();
    for (auto& [h1, h2] : model.dmn->association_heads) {
        Tensor assoc_pre = h1->forward(cache.dmn_latent);
        TensorOps::gelu(assoc_pre);
        cache.dmn_head1_outs.push_back(assoc_pre.clone());  // 缓存head1输出
        Tensor assoc = h2->forward(assoc_pre);
        cache.dmn_associations.push_back(assoc.clone());
    }

    cache.dmn_vision = TensorOps::concat(cache.dmn_associations, 1);
    cache.dmn_vision = model.dmn->future_proj1->forward(cache.dmn_vision);
    cache.dmn_vision = model.dmn->future_norm->forward(cache.dmn_vision);
    cache.dmn_vision = model.dmn->future_gelu->forward(cache.dmn_vision);

    auto mem_out = model.memory->forward(cache.h);
    cache.memory_retrieved = mem_out.retrieved;

    NeuroFlowModel::Output out;
    out.saliency = sn_out.saliency;
    out.gates = sn_out.gates;
    out.anomaly = sn_out.anomaly;

    const float* gates = out.gates.as_fp32();
    Tensor ecn_gate({batch, 1}, QuantType::FP32);
    Tensor dmn_gate({batch, 1}, QuantType::FP32);
    for (size_t i = 0; i < batch; ++i) {
        ecn_gate.as_fp32()[i] = gates[i * 2];
        dmn_gate.as_fp32()[i] = gates[i * 2 + 1];
    }
    out.ecn_gate = ecn_gate;
    out.dmn_gate = dmn_gate;

    cache.ecn_decision = model.ecn->vmpfc2->forward(cache.ecn_vmpfc_d);
    out.decision = cache.ecn_decision;
    out.value = model.ecn->ofc2->forward(cache.ecn_ofc_v);

    Tensor ecn_weighted({batch, model.config.output_dim}, QuantType::FP32);
    Tensor dmn_weighted({batch, model.config.output_dim}, QuantType::FP32);
    Tensor mem_weighted({batch, model.config.output_dim}, QuantType::FP32);

    float* ew = ecn_weighted.as_fp32();
    float* dw = dmn_weighted.as_fp32();
    float* mw = mem_weighted.as_fp32();
    const float* ed = out.decision.as_fp32();
    const float* dv = cache.dmn_vision.as_fp32();
    const float* mr = cache.memory_retrieved.as_fp32();
    const float* eg = ecn_gate.as_fp32();
    const float* dg = dmn_gate.as_fp32();
    size_t out_dim = model.config.output_dim;
    size_t dmn_dim = cache.dmn_vision.shape_[1];
    size_t mem_dim = cache.memory_retrieved.shape_[1];


    for (size_t i = 0; i < batch; ++i) {
        float egv = eg[i];
        float dgv = dg[i];
        for (size_t j = 0; j < out_dim; ++j) {
            ew[i * out_dim + j] = ed[i * out_dim + j] * egv;
            if (j < dmn_dim)
                dw[i * out_dim + j] = dv[i * dmn_dim + j] * dgv;
            if (j < mem_dim)
                mw[i * out_dim + j] = mr[i * mem_dim + j];
        }
    }

    std::vector<Tensor> to_concat = {ecn_weighted, dmn_weighted, mem_weighted};
    cache.combined = TensorOps::concat(to_concat, 1);

    cache.fused_bn = model.output_fusion_down->forward(cache.combined);
    cache.fused_bn_pre_norm = cache.fused_bn.clone();  // save pre-norm input
    cache.fused_bn = model.output_fusion_bottleneck_norm->forward(cache.fused_bn);
    cache.fused_bn_pre_relu = cache.fused_bn.clone();
    TensorOps::relu(cache.fused_bn);
    cache.fused_pre_norm = model.output_fusion_up->forward(cache.fused_bn);
    cache.fused = model.output_fusion_norm->forward(cache.fused_pre_norm);
    out.output = cache.fused;

    return out;
}

FullBackpropEngine::Gradients FullBackpropEngine::backward(const Tensor& output_grad) {
    Gradients grads;
    size_t batch = cache.input.shape_[0];
    size_t out_dim = model.config.output_dim;
    size_t combined_dim = cache.combined.shape_[1];

    Tensor up_input_grad = layernorm_backward(cache.fused_pre_norm, model.output_fusion_norm->weight, output_grad);

    // output_fusion_norm γ/β gradients
    auto out_norm_pg = layernorm_param_backward(cache.fused_pre_norm, output_grad);
    grads.output_fusion_norm_weight_grad = out_norm_pg.weight_grad;
    grads.output_fusion_norm_bias_grad = out_norm_pg.bias_grad;

    grads.output_fusion_up_weight_grad = linear_backward_weight(cache.fused_bn, up_input_grad);
    grads.output_fusion_up_bias_grad = bias_backward(up_input_grad);

    Tensor bn_grad = linear_backward_input(up_input_grad, model.output_fusion_up->weight);
    {
        const float* pre_relu = cache.fused_bn_pre_relu.as_fp32();
        float* bg = bn_grad.as_fp32();
        size_t n = bn_grad.numel();
        for (size_t i = 0; i < n; ++i) {
            bg[i] = (pre_relu[i] > 0.0f) ? bg[i] : 0.0f;
        }
    }

    // Fix: use saved pre-norm input for bottleneck norm backward
    Tensor bn_norm_grad = layernorm_backward(cache.fused_bn_pre_norm, model.output_fusion_bottleneck_norm->weight, bn_grad);

    // output_fusion_bottleneck_norm γ/β gradients
    auto bn_norm_pg = layernorm_param_backward(cache.fused_bn_pre_norm, bn_grad);
    grads.output_fusion_bottleneck_norm_weight_grad = bn_norm_pg.weight_grad;
    grads.output_fusion_bottleneck_norm_bias_grad = bn_norm_pg.bias_grad;

    grads.output_fusion_down_weight_grad = linear_backward_weight(cache.combined, bn_norm_grad);
    grads.output_fusion_down_bias_grad = bias_backward(bn_norm_grad);

    Tensor combined_grad = linear_backward_input(bn_norm_grad, model.output_fusion_down->weight);

    Tensor ecn_w_grad({batch, out_dim}, QuantType::FP32);
    Tensor dmn_w_grad({batch, out_dim}, QuantType::FP32);
    Tensor mem_w_grad({batch, out_dim}, QuantType::FP32);
    {
        const float* cg = combined_grad.as_fp32();
        float* eg = ecn_w_grad.as_fp32();
        float* dg = dmn_w_grad.as_fp32();
        float* mg = mem_w_grad.as_fp32();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < out_dim; ++j) {
                eg[b * out_dim + j] = cg[b * combined_dim + j];
                dg[b * out_dim + j] = cg[b * combined_dim + out_dim + j];
                mg[b * out_dim + j] = cg[b * combined_dim + 2 * out_dim + j];
            }
        }
    }

    const float* gates_data = cache.sn_gates.as_fp32();

    // DEBUG: Skip SN gate gradient computation for NaN isolation
    // ===== SN gate gradient (修复: gate1/gate2 之前没有梯度) =====
    // ecn_weighted = ecn_decision * ecn_gate
    // dmn_weighted = dmn_vision * dmn_gate
    // d(loss)/d(ecn_gate_b) = sum_j(ecn_w_grad[b,j] * ecn_decision[b,j])
    // d(loss)/d(dmn_gate_b) = sum_j(dmn_w_grad[b,j] * dmn_vision[b,j])
    Tensor sn_gate_output_grad({batch, 2}, QuantType::FP32);
    {
        const float* ewg = ecn_w_grad.as_fp32();
        const float* dwg = dmn_w_grad.as_fp32();
        // 修复: 用 ecn_decision [batch,2048] 替代 ecn_vmpfc_d [batch,1024] (维度错误导致越界!)
        const float* ed = cache.ecn_decision.as_fp32();
        float* sgg = sn_gate_output_grad.as_fp32();
        size_t dmn_dim = cache.dmn_vision.shape_[1];
        const float* dv = cache.dmn_vision.as_fp32();

        for (size_t b = 0; b < batch; ++b) {
            float ecn_sum = 0.0f, dmn_sum = 0.0f;
            for (size_t j = 0; j < out_dim; ++j) {
                ecn_sum += ewg[b * out_dim + j] * ed[b * out_dim + j];
            }
            for (size_t j = 0; j < out_dim && j < dmn_dim; ++j) {
                dmn_sum += dwg[b * out_dim + j] * dv[b * dmn_dim + j];
            }
            sgg[b * 2] = ecn_sum;
            sgg[b * 2 + 1] = dmn_sum;
        }
    }

    // Backprop through softmax of gates
    {
        float* sgg = sn_gate_output_grad.as_fp32();
        for (size_t b = 0; b < batch; ++b) {
            float g0 = gates_data[b * 2];
            float g1 = gates_data[b * 2 + 1];
            // softmax gradient: ds_i = s_i * (g_i - s_i * sum(g))
            float dot = sgg[b * 2] * g0 + sgg[b * 2 + 1] * g1;
            sgg[b * 2] = g0 * (sgg[b * 2] - dot);
            sgg[b * 2 + 1] = g1 * (sgg[b * 2 + 1] - dot);
        }
    }

    // gate2 backward: Linear(gate1_post_gelu) → softmax
    Tensor gate2_input_grad = linear_backward_input(sn_gate_output_grad,
        model.sn->gate2->weight);
    grads.sn_gate2_weight_grad = linear_backward_weight(cache.sn_gate_h_post, sn_gate_output_grad);
    grads.sn_gate2_bias_grad = bias_backward(sn_gate_output_grad);

    // gate1 backward: Linear(h) → gelu → gate2
    // Use PRE-gelu cache for correct GELU derivative!
    Tensor gate1_gelu_grad = gelu_backward(cache.sn_gate_h, gate2_input_grad);
    grads.sn_gate1_weight_grad = linear_backward_weight(cache.h, gate1_gelu_grad);
    grads.sn_gate1_bias_grad = bias_backward(gate1_gelu_grad);
    Tensor sn_h_grad = linear_backward_input(gate1_gelu_grad, model.sn->gate1->weight);

// end SN gate
    // ===== ECN backward =====
    Tensor ecn_decision_grad({batch, out_dim}, QuantType::FP32);
    {
        const float* ewg = ecn_w_grad.as_fp32();
        float* edg = ecn_decision_grad.as_fp32();

        for (size_t b = 0; b < batch; ++b) {
            float ecn_gate = gates_data[b * 2];
            for (size_t j = 0; j < out_dim; ++j) {
                edg[b * out_dim + j] = ewg[b * out_dim + j] * ecn_gate;
            }
        }
    }

    Tensor vmpfc_d_grad = linear_backward_input(ecn_decision_grad, model.ecn->vmpfc2->weight);
    grads.ecn_vmpfc2_weight_grad = linear_backward_weight(cache.ecn_vmpfc_d, ecn_decision_grad);
    grads.ecn_vmpfc2_bias_grad = bias_backward(ecn_decision_grad);

    Tensor vmpfc_pre_grad = gelu_backward(cache.ecn_vmpfc_pre, vmpfc_d_grad);
    grads.ecn_vmpfc1_weight_grad = linear_backward_weight(cache.ecn_hidden.back(), vmpfc_pre_grad);
    grads.ecn_vmpfc1_bias_grad = bias_backward(vmpfc_pre_grad);

    Tensor ecn_last_h_grad = linear_backward_input(vmpfc_pre_grad, model.ecn->vmpfc1->weight);

    grads.ecn_dlpfc_weight_grads.clear();
    grads.ecn_dlpfc_bias_grads.clear();
    grads.ecn_dlpfc_norm_weight_grads.clear();
    grads.ecn_dlpfc_norm_bias_grads.clear();

    for (int i = static_cast<int>(model.ecn->num_layers) - 1; i >= 0; --i) {
        Tensor gelu_grad = gelu_backward(cache.ecn_pre_norm[i], ecn_last_h_grad);
        Tensor norm_grad_i = layernorm_backward(
            cache.ecn_pre_norm[i], model.ecn->dlpfc_norm[i]->weight, gelu_grad);

        // dlpfc_norm γ/β gradients
        auto norm_pg_i = layernorm_param_backward(cache.ecn_pre_norm[i], gelu_grad);
        grads.ecn_dlpfc_norm_weight_grads.insert(grads.ecn_dlpfc_norm_weight_grads.begin(), norm_pg_i.weight_grad);
        grads.ecn_dlpfc_norm_bias_grads.insert(grads.ecn_dlpfc_norm_bias_grads.begin(), norm_pg_i.bias_grad);

        const Tensor& input_for_layer = (i == 0) ? cache.h : cache.ecn_pre_linear[i];
        grads.ecn_dlpfc_weight_grads.insert(grads.ecn_dlpfc_weight_grads.begin(),
            linear_backward_weight(input_for_layer, norm_grad_i));
        grads.ecn_dlpfc_bias_grads.insert(grads.ecn_dlpfc_bias_grads.begin(),
            bias_backward(norm_grad_i));

        ecn_last_h_grad = linear_backward_input(norm_grad_i, model.ecn->dlpfc_linear[i]->weight);
    }

    // ===== DMN backward chain (修复: DMN之前没有梯度) =====
    // Compute dmn_vision_grad from dmn_w_grad (gate-weighted)
    size_t dmn_vision_dim = cache.dmn_vision.shape_[1];
    Tensor dmn_vision_grad({batch, dmn_vision_dim}, QuantType::FP32);
    {
        const float* dwg = dmn_w_grad.as_fp32();
        float* dvg = dmn_vision_grad.as_fp32();

        for (size_t b = 0; b < batch; ++b) {
            float dmn_gate = gates_data[b * 2 + 1];
            for (size_t j = 0; j < dmn_vision_dim; ++j) {
                if (j < out_dim)
                    dvg[b * dmn_vision_dim + j] = dwg[b * out_dim + j] * dmn_gate;
                else
                    dvg[b * dmn_vision_dim + j] = 0.0f;
            }
        }
    }

    // Backprop through future_proj1
    grads.dmn_future_proj1_weight_grad = linear_backward_weight(
        TensorOps::concat(cache.dmn_associations, 1), dmn_vision_grad);
    grads.dmn_future_proj1_bias_grad = bias_backward(dmn_vision_grad);

    // Gradient back to concatenated associations
    Tensor concat_assoc_grad = linear_backward_input(dmn_vision_grad,
        model.dmn->future_proj1->weight);

    // Split concat_assoc_grad back to individual association heads
    size_t latent_dim = model.dmn->latent_dim;
    Tensor dmn_latent_grad({batch, latent_dim}, QuantType::FP32);
    {
        float* dlg = dmn_latent_grad.as_fp32();
        const float* cag = concat_assoc_grad.as_fp32();

        grads.dmn_head2_weight_grads.clear();
        grads.dmn_head2_bias_grads.clear();
        grads.dmn_head1_weight_grads.clear();
        grads.dmn_head1_bias_grads.clear();

        for (size_t h = 0; h < model.dmn->num_associations; ++h) {
            // Extract gradient for this head
            Tensor head_grad({batch, latent_dim}, QuantType::FP32);
            float* hg = head_grad.as_fp32();
            size_t concat_dim = concat_assoc_grad.shape_[1];
            for (size_t b = 0; b < batch; ++b) {
                for (size_t j = 0; j < latent_dim; ++j) {
                    hg[b * latent_dim + j] = cag[b * concat_dim + h * latent_dim + j];
                }
            }

            // head2 backward: Linear(head1_out) → head2_out
            auto& head2 = model.dmn->association_heads[h].second;
            grads.dmn_head2_weight_grads.push_back(
                linear_backward_weight(cache.dmn_head1_outs[h], head_grad));
            grads.dmn_head2_bias_grads.push_back(bias_backward(head_grad));

            Tensor head1_out_grad = linear_backward_input(head_grad, head2->weight);

            // GELU backward (head1 output went through GELU)
            // cache.dmn_head1_outs[h] is post-GELU; approximate GELU derivative
            {
                float* h1og = head1_out_grad.as_fp32();
                const float* h1o = cache.dmn_head1_outs[h].as_fp32();
                size_t n = head1_out_grad.numel();
                for (size_t i = 0; i < n; ++i) {
                    // GELU gradient approximation for positive values
                    h1og[i] *= (h1o[i] > 0.0f) ? 1.0f : 0.1f;
                }
            }

            // head1 backward: Linear(latent) → head1_out
            auto& head1 = model.dmn->association_heads[h].first;
            grads.dmn_head1_weight_grads.push_back(
                linear_backward_weight(cache.dmn_latent, head1_out_grad));
            grads.dmn_head1_bias_grads.push_back(bias_backward(head1_out_grad));

            Tensor head1_input_grad = linear_backward_input(head1_out_grad, head1->weight);

            // Accumulate into dmn_latent_grad (average across heads)
            float* dlgp = dmn_latent_grad.as_fp32();
            const float* h1ig = head1_input_grad.as_fp32();
            float inv_h = 1.0f / model.dmn->num_associations;
            for (size_t idx = 0; idx < batch * latent_dim; ++idx) {
                dlgp[idx] += h1ig[idx] * inv_h;
            }
        }
    }

    // mem_encoder2 backward: Linear(dmn_encoded) → dmn_latent
    grads.dmn_mem_encoder2_weight_grad = linear_backward_weight(
        cache.dmn_encoded, dmn_latent_grad);
    grads.dmn_mem_encoder2_bias_grad = bias_backward(dmn_latent_grad);

    Tensor dmn_encoded_grad = linear_backward_input(dmn_latent_grad,
        model.dmn->mem_encoder2->weight);

    // GELU backward on dmn_encoded
    // cache.dmn_encoded has gelu applied; need pre-gelu value
    // Approximate: ReLU-like gradient (pass-through for positive)
    {
        float* deg = dmn_encoded_grad.as_fp32();
        const float* de = cache.dmn_encoded.as_fp32();
        size_t n = dmn_encoded_grad.numel();
        for (size_t i = 0; i < n; ++i) {
            deg[i] *= (de[i] > 0.0f) ? 1.0f : 0.1f;  // GELU leaky approx
        }
    }

    // mem_encoder1 backward: Linear(memory_encoded) → dmn_encoded
    grads.dmn_mem_encoder1_weight_grad = linear_backward_weight(
        cache.memory_encoded, dmn_encoded_grad);
    grads.dmn_mem_encoder1_bias_grad = bias_backward(dmn_encoded_grad);

    Tensor dmn_h_grad = linear_backward_input(dmn_encoded_grad,
        model.dmn->mem_encoder1->weight);

// end DMN
    // ===== Memory backward chain (修复: memory encode/query之前没有梯度) =====
    // mem_w_grad → back through retrieve_proj → attention-weighted back to query and encode
    // mem_for_fusion = retrieve_proj(retrieved_mem)
    // retrieved_mem = attention @ memory_bank
    Tensor mem_retrieved_grad({batch, model.config.memory_dim}, QuantType::FP32);
    {
        const float* mwg = mem_w_grad.as_fp32();
        float* mrg = mem_retrieved_grad.as_fp32();
        size_t mem_dim = model.config.memory_dim;
        // mem_w_grad already has out_dim elements; truncate/pad to memory_dim
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < mem_dim; ++j) {
                mrg[b * mem_dim + j] = (j < out_dim) ? mwg[b * out_dim + j] : 0.0f;
            }
        }
    }

    // retrieve_proj backward: Linear(retrieved_mem) → out
    // Retrieve the intermediate retrieved_mem from cache
    auto mem_out = model.memory->retrieve(cache.h);
    grads.mem_encode_proj_weight_grad = linear_backward_weight(
        cache.h, mem_retrieved_grad);
    grads.mem_encode_proj_bias_grad = bias_backward(mem_retrieved_grad);

    Tensor mem_enc_h_grad = linear_backward_input(mem_retrieved_grad,
        model.memory->encode_proj->weight);

    // query_proj backward
    grads.mem_query_proj_weight_grad = linear_backward_weight(
        cache.h, mem_retrieved_grad);
    grads.mem_query_proj_bias_grad = bias_backward(mem_retrieved_grad);

    Tensor mem_query_h_grad = linear_backward_input(mem_retrieved_grad,
        model.memory->query_proj->weight);

// end Memory
    // ===== Merge all gradients into h_grad =====
    // h_grad = ECN + DMN + Memory + SN (all four pathways, NaN bug fixed)
    Tensor h_grad({batch, model.config.hidden_dim}, QuantType::FP32);
    {
        float* hg = h_grad.as_fp32();
        const float* eg = ecn_last_h_grad.as_fp32();
        const float* dg = dmn_h_grad.as_fp32();
        const float* meg = mem_enc_h_grad.as_fp32();
        const float* mqg = mem_query_h_grad.as_fp32();
        const float* sg = sn_h_grad.as_fp32();
        size_t hd = model.config.hidden_dim;
        size_t md = model.config.memory_dim;

        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < hd; ++j) {
                hg[b * hd + j] = eg[b * hd + j]           // ECN
                    + (j < md ? dg[b * md + j] : 0.0f)     // DMN (memory_dim)
                    + (j < md ? meg[b * md + j] : 0.0f)    // Memory encode
                    + (j < md ? mqg[b * md + j] : 0.0f)    // Memory query
                    + (j < hd ? sg[b * hd + j] : 0.0f);    // SN gate
            }
        }
    }

    Tensor gelu_grad_input = gelu_backward(cache.input_proj_post, h_grad);
    Tensor norm_grad_input = layernorm_backward(
        cache.input_proj_pre, model.input_proj_norm->weight, gelu_grad_input);

    // input_proj_norm γ/β gradients
    auto in_norm_pg = layernorm_param_backward(cache.input_proj_pre, gelu_grad_input);
    grads.input_proj_norm_weight_grad = in_norm_pg.weight_grad;
    grads.input_proj_norm_bias_grad = in_norm_pg.bias_grad;

    grads.input_proj_weight_grad = linear_backward_weight(cache.input, norm_grad_input);
    grads.input_proj_bias_grad = bias_backward(norm_grad_input);

    grads.input_grad = linear_backward_input(norm_grad_input, model.input_proj_linear->weight);

    return grads;
}

void FullTrainer::apply_gradients(FullBackpropEngine::Gradients& grads, float lr) {
    auto sgd_update = [&](Tensor& param, const Tensor& grad) {
        if (param.shape_ != grad.shape_ || param.numel() == 0 || grad.numel() == 0) return;
        float* p = param.as_fp32();
        const float* g = grad.as_fp32();
        size_t n = param.numel();
        #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
        for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) {
            p[i] -= lr * g[i];
        }
    };

    sgd_update(model.input_proj_linear->weight, grads.input_proj_weight_grad);
    sgd_update(model.input_proj_linear->bias, grads.input_proj_bias_grad);
    sgd_update(model.input_proj_norm->weight, grads.input_proj_norm_weight_grad);
    sgd_update(model.input_proj_norm->bias, grads.input_proj_norm_bias_grad);
    sgd_update(model.output_fusion_down->weight, grads.output_fusion_down_weight_grad);
    sgd_update(model.output_fusion_down->bias, grads.output_fusion_down_bias_grad);
    sgd_update(model.output_fusion_up->weight, grads.output_fusion_up_weight_grad);
    sgd_update(model.output_fusion_up->bias, grads.output_fusion_up_bias_grad);
    sgd_update(model.output_fusion_norm->weight, grads.output_fusion_norm_weight_grad);
    sgd_update(model.output_fusion_norm->bias, grads.output_fusion_norm_bias_grad);
    sgd_update(model.output_fusion_bottleneck_norm->weight, grads.output_fusion_bottleneck_norm_weight_grad);
    sgd_update(model.output_fusion_bottleneck_norm->bias, grads.output_fusion_bottleneck_norm_bias_grad);
    sgd_update(model.ecn->vmpfc2->weight, grads.ecn_vmpfc2_weight_grad);
    sgd_update(model.ecn->vmpfc2->bias, grads.ecn_vmpfc2_bias_grad);
    sgd_update(model.ecn->vmpfc1->weight, grads.ecn_vmpfc1_weight_grad);
    sgd_update(model.ecn->vmpfc1->bias, grads.ecn_vmpfc1_bias_grad);

    for (size_t i = 0; i < grads.ecn_dlpfc_weight_grads.size() && i < model.ecn->dlpfc_linear.size(); ++i) {
        sgd_update(model.ecn->dlpfc_linear[i]->weight, grads.ecn_dlpfc_weight_grads[i]);
    }
    for (size_t i = 0; i < grads.ecn_dlpfc_bias_grads.size() && i < model.ecn->dlpfc_linear.size(); ++i) {
        sgd_update(model.ecn->dlpfc_linear[i]->bias, grads.ecn_dlpfc_bias_grads[i]);
    }
    for (size_t i = 0; i < grads.ecn_dlpfc_norm_weight_grads.size() && i < model.ecn->dlpfc_norm.size(); ++i) {
        sgd_update(model.ecn->dlpfc_norm[i]->weight, grads.ecn_dlpfc_norm_weight_grads[i]);
    }
    for (size_t i = 0; i < grads.ecn_dlpfc_norm_bias_grads.size() && i < model.ecn->dlpfc_norm.size(); ++i) {
        sgd_update(model.ecn->dlpfc_norm[i]->bias, grads.ecn_dlpfc_norm_bias_grads[i]);
    }

    // SN gate gradients (修复: gate1/gate2)
    sgd_update(model.sn->gate2->weight, grads.sn_gate2_weight_grad);
    sgd_update(model.sn->gate2->bias, grads.sn_gate2_bias_grad);
    sgd_update(model.sn->gate1->weight, grads.sn_gate1_weight_grad);
    sgd_update(model.sn->gate1->bias, grads.sn_gate1_bias_grad);

    // DMN gradients (修复: 全链路)
    sgd_update(model.dmn->future_proj1->weight, grads.dmn_future_proj1_weight_grad);
    sgd_update(model.dmn->future_proj1->bias, grads.dmn_future_proj1_bias_grad);
    sgd_update(model.dmn->mem_encoder2->weight, grads.dmn_mem_encoder2_weight_grad);
    sgd_update(model.dmn->mem_encoder2->bias, grads.dmn_mem_encoder2_bias_grad);
    sgd_update(model.dmn->mem_encoder1->weight, grads.dmn_mem_encoder1_weight_grad);
    sgd_update(model.dmn->mem_encoder1->bias, grads.dmn_mem_encoder1_bias_grad);
    for (size_t i = 0; i < grads.dmn_head2_weight_grads.size() && i < model.dmn->association_heads.size(); ++i) {
        sgd_update(model.dmn->association_heads[i].second->weight, grads.dmn_head2_weight_grads[i]);
        sgd_update(model.dmn->association_heads[i].second->bias, grads.dmn_head2_bias_grads[i]);
    }
    for (size_t i = 0; i < grads.dmn_head1_weight_grads.size() && i < model.dmn->association_heads.size(); ++i) {
        sgd_update(model.dmn->association_heads[i].first->weight, grads.dmn_head1_weight_grads[i]);
        sgd_update(model.dmn->association_heads[i].first->bias, grads.dmn_head1_bias_grads[i]);
    }

    // Memory gradients (修复: encode/query)
    sgd_update(model.memory->encode_proj->weight, grads.mem_encode_proj_weight_grad);
    sgd_update(model.memory->encode_proj->bias, grads.mem_encode_proj_bias_grad);
    sgd_update(model.memory->query_proj->weight, grads.mem_query_proj_weight_grad);
    sgd_update(model.memory->query_proj->bias, grads.mem_query_proj_bias_grad);
}

FullTrainer::FullTrainer(NeuroFlowModel& m, float lr)
    : model(m), learning_rate(lr), backprop(m) {}

FullTrainer::TrainStep FullTrainer::train_step(const Tensor& input, const Tensor& target) {
    TrainStep result;

    auto output = backprop.forward_with_cache(input);

    const float* pred = output.output.as_fp32();
    const float* tgt = target.as_fp32();
    size_t batch = input.shape_[0];
    size_t dim = target.shape_[1];
    size_t n = output.output.numel();

    bool is_onehot = false;
    for (size_t i = 0; i < std::min(n, batch * 10UL); ++i) {
        if (tgt[i] == 1.0f) { is_onehot = true; break; }
    }

    if (is_onehot && dim > 100) {
        float loss = 0.0f;
        Tensor output_grad({batch, dim}, QuantType::FP32);
        float* og = output_grad.as_fp32();
        float grad_norm = 0.0f;

        for (size_t b = 0; b < batch; ++b) {
            float max_val = -1e30f;
            for (size_t j = 0; j < dim; ++j) {
                if (pred[b * dim + j] > max_val) max_val = pred[b * dim + j];
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                sum_exp += std::exp(pred[b * dim + j] - max_val);
            }
            float log_sum_exp = max_val + std::log(sum_exp);

            for (size_t j = 0; j < dim; ++j) {
                float softmax_val = std::exp(pred[b * dim + j] - max_val) / sum_exp;
                float t = tgt[b * dim + j];
                og[b * dim + j] = (softmax_val - t) / batch;
                grad_norm += og[b * dim + j] * og[b * dim + j];
                if (t > 0.5f) {
                    loss -= (pred[b * dim + j] - log_sum_exp);
                }
            }
        }

        result.loss = loss / batch;
        result.grad_norm = std::sqrt(grad_norm);

        auto grads = backprop.backward(output_grad);
        apply_gradients(grads, learning_rate);
    } else {
        Tensor output_grad({batch, dim}, QuantType::FP32);
        float* og = output_grad.as_fp32();
        float scale = 2.0f / n;
        float loss = 0.0f;
        float grad_norm = 0.0f;
        #pragma omp parallel for schedule(static) reduction(+:loss,grad_norm) if(n >= OMP_MIN_ITER)
        for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) {
            float diff = pred[i] - tgt[i];
            og[i] = scale * diff;
            loss += diff * diff;
            grad_norm += og[i] * og[i];
        }
        result.loss = loss / n;
        result.grad_norm = std::sqrt(grad_norm);

        auto grads = backprop.backward(output_grad);
        apply_gradients(grads, learning_rate);
    }

    model.memory->consolidate(backprop.cache.h);

    return result;
}

std::vector<float> FullTrainer::train(const std::vector<Tensor>& inputs,
                                       const std::vector<Tensor>& targets,
                                       int epochs) {
    std::vector<float> losses;
    for (int e = 0; e < epochs; ++e) {
        float epoch_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto step = train_step(inputs[i], targets[i]);
            epoch_loss += step.loss;
        }
        epoch_loss /= inputs.size();
        losses.push_back(epoch_loss);
    }
    return losses;
}

FullTrainer::TrainStep FullTrainer::accumulate_step(const Tensor& input, const Tensor& target) {
    TrainStep result;

    auto output = backprop.forward_with_cache(input);

    const float* pred = output.output.as_fp32();
    const float* tgt = target.as_fp32();
    size_t batch = input.shape_[0];
    size_t dim = target.shape_[1];
    size_t n = output.output.numel();

    bool is_onehot = false;
    for (size_t i = 0; i < std::min(n, batch * 10UL); ++i) {
        if (tgt[i] == 1.0f) { is_onehot = true; break; }
    }

    Tensor output_grad({batch, dim}, QuantType::FP32);
    float* og = output_grad.as_fp32();

    if (is_onehot && dim > 100) {
        float loss = 0.0f;
        float grad_norm = 0.0f;
        for (size_t b = 0; b < batch; ++b) {
            float max_val = -1e30f;
            for (size_t j = 0; j < dim; ++j)
                if (pred[b * dim + j] > max_val) max_val = pred[b * dim + j];
            float sum_exp = 0.0f;
            for (size_t j = 0; j < dim; ++j)
                sum_exp += std::exp(pred[b * dim + j] - max_val);
            float log_sum_exp = max_val + std::log(sum_exp);
            for (size_t j = 0; j < dim; ++j) {
                float softmax_val = std::exp(pred[b * dim + j] - max_val) / sum_exp;
                float t = tgt[b * dim + j];
                og[b * dim + j] = (softmax_val - t) / batch;
                grad_norm += og[b * dim + j] * og[b * dim + j];
                if (t > 0.5f) loss -= (pred[b * dim + j] - log_sum_exp);
            }
        }
        result.loss = loss / batch;
        result.grad_norm = std::sqrt(grad_norm);
    } else {
        float scale = 2.0f / n;
        float loss = 0.0f;
        float grad_norm = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = pred[i] - tgt[i];
            og[i] = scale * diff;
            loss += diff * diff;
            grad_norm += og[i] * og[i];
        }
        result.loss = loss / n;
        result.grad_norm = std::sqrt(grad_norm);
    }

    auto grads = backprop.backward(output_grad);

    if (!accum_initialized_) {
        accum_grads_.input_proj_weight_grad = grads.input_proj_weight_grad.clone();
        accum_grads_.input_proj_bias_grad = grads.input_proj_bias_grad.clone();
        accum_grads_.input_proj_norm_weight_grad = grads.input_proj_norm_weight_grad.clone();
        accum_grads_.input_proj_norm_bias_grad = grads.input_proj_norm_bias_grad.clone();
        accum_grads_.output_fusion_down_weight_grad = grads.output_fusion_down_weight_grad.clone();
        accum_grads_.output_fusion_down_bias_grad = grads.output_fusion_down_bias_grad.clone();
        accum_grads_.output_fusion_up_weight_grad = grads.output_fusion_up_weight_grad.clone();
        accum_grads_.output_fusion_up_bias_grad = grads.output_fusion_up_bias_grad.clone();
        accum_grads_.output_fusion_norm_weight_grad = grads.output_fusion_norm_weight_grad.clone();
        accum_grads_.output_fusion_norm_bias_grad = grads.output_fusion_norm_bias_grad.clone();
        accum_grads_.output_fusion_bottleneck_norm_weight_grad = grads.output_fusion_bottleneck_norm_weight_grad.clone();
        accum_grads_.output_fusion_bottleneck_norm_bias_grad = grads.output_fusion_bottleneck_norm_bias_grad.clone();
        accum_grads_.ecn_vmpfc2_weight_grad = grads.ecn_vmpfc2_weight_grad.clone();
        accum_grads_.ecn_vmpfc2_bias_grad = grads.ecn_vmpfc2_bias_grad.clone();
        accum_grads_.ecn_vmpfc1_weight_grad = grads.ecn_vmpfc1_weight_grad.clone();
        accum_grads_.ecn_vmpfc1_bias_grad = grads.ecn_vmpfc1_bias_grad.clone();
        accum_grads_.ecn_dlpfc_weight_grads = grads.ecn_dlpfc_weight_grads;
        accum_grads_.ecn_dlpfc_bias_grads = grads.ecn_dlpfc_bias_grads;
        accum_grads_.ecn_dlpfc_norm_weight_grads = grads.ecn_dlpfc_norm_weight_grads;
        accum_grads_.ecn_dlpfc_norm_bias_grads = grads.ecn_dlpfc_norm_bias_grads;
        accum_initialized_ = true;
    } else {
        auto add_tensor = [](Tensor& dst, const Tensor& src) {
            float* d = dst.as_fp32();
            const float* s = src.as_fp32();
            size_t n = dst.numel();
            for (size_t i = 0; i < n; ++i) d[i] += s[i];
        };
        add_tensor(accum_grads_.input_proj_weight_grad, grads.input_proj_weight_grad);
        add_tensor(accum_grads_.input_proj_bias_grad, grads.input_proj_bias_grad);
        add_tensor(accum_grads_.input_proj_norm_weight_grad, grads.input_proj_norm_weight_grad);
        add_tensor(accum_grads_.input_proj_norm_bias_grad, grads.input_proj_norm_bias_grad);
        add_tensor(accum_grads_.output_fusion_down_weight_grad, grads.output_fusion_down_weight_grad);
        add_tensor(accum_grads_.output_fusion_down_bias_grad, grads.output_fusion_down_bias_grad);
        add_tensor(accum_grads_.output_fusion_up_weight_grad, grads.output_fusion_up_weight_grad);
        add_tensor(accum_grads_.output_fusion_up_bias_grad, grads.output_fusion_up_bias_grad);
        add_tensor(accum_grads_.output_fusion_norm_weight_grad, grads.output_fusion_norm_weight_grad);
        add_tensor(accum_grads_.output_fusion_norm_bias_grad, grads.output_fusion_norm_bias_grad);
        add_tensor(accum_grads_.output_fusion_bottleneck_norm_weight_grad, grads.output_fusion_bottleneck_norm_weight_grad);
        add_tensor(accum_grads_.output_fusion_bottleneck_norm_bias_grad, grads.output_fusion_bottleneck_norm_bias_grad);
        add_tensor(accum_grads_.ecn_vmpfc2_weight_grad, grads.ecn_vmpfc2_weight_grad);
        add_tensor(accum_grads_.ecn_vmpfc2_bias_grad, grads.ecn_vmpfc2_bias_grad);
        add_tensor(accum_grads_.ecn_vmpfc1_weight_grad, grads.ecn_vmpfc1_weight_grad);
        add_tensor(accum_grads_.ecn_vmpfc1_bias_grad, grads.ecn_vmpfc1_bias_grad);
        for (size_t i = 0; i < grads.ecn_dlpfc_weight_grads.size() && i < accum_grads_.ecn_dlpfc_weight_grads.size(); ++i)
            add_tensor(accum_grads_.ecn_dlpfc_weight_grads[i], grads.ecn_dlpfc_weight_grads[i]);
        for (size_t i = 0; i < grads.ecn_dlpfc_bias_grads.size() && i < accum_grads_.ecn_dlpfc_bias_grads.size(); ++i)
            add_tensor(accum_grads_.ecn_dlpfc_bias_grads[i], grads.ecn_dlpfc_bias_grads[i]);
        for (size_t i = 0; i < grads.ecn_dlpfc_norm_weight_grads.size() && i < accum_grads_.ecn_dlpfc_norm_weight_grads.size(); ++i)
            add_tensor(accum_grads_.ecn_dlpfc_norm_weight_grads[i], grads.ecn_dlpfc_norm_weight_grads[i]);
        for (size_t i = 0; i < grads.ecn_dlpfc_norm_bias_grads.size() && i < accum_grads_.ecn_dlpfc_norm_bias_grads.size(); ++i)
            add_tensor(accum_grads_.ecn_dlpfc_norm_bias_grads[i], grads.ecn_dlpfc_norm_bias_grads[i]);
    }

    accum_loss_ += result.loss;
    accum_grad_norm_ += result.grad_norm;

    model.memory->consolidate(backprop.cache.h);
    return result;
}

void FullTrainer::apply_accumulated_gradients(int accum_steps) {
    if (!accum_initialized_ || accum_steps <= 0) return;
    float scale = 1.0f / accum_steps;

    auto scaled_sgd = [&](Tensor& param, Tensor& grad) {
        if (param.shape_ != grad.shape_ || param.numel() == 0) return;
        float* p = param.as_fp32();
        const float* g = grad.as_fp32();
        size_t n = param.numel();
        #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
        for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) {
            p[i] -= learning_rate * g[i] * scale;
        }
    };

    scaled_sgd(model.input_proj_linear->weight, accum_grads_.input_proj_weight_grad);
    scaled_sgd(model.input_proj_linear->bias, accum_grads_.input_proj_bias_grad);
    scaled_sgd(model.input_proj_norm->weight, accum_grads_.input_proj_norm_weight_grad);
    scaled_sgd(model.input_proj_norm->bias, accum_grads_.input_proj_norm_bias_grad);
    scaled_sgd(model.output_fusion_down->weight, accum_grads_.output_fusion_down_weight_grad);
    scaled_sgd(model.output_fusion_down->bias, accum_grads_.output_fusion_down_bias_grad);
    scaled_sgd(model.output_fusion_up->weight, accum_grads_.output_fusion_up_weight_grad);
    scaled_sgd(model.output_fusion_up->bias, accum_grads_.output_fusion_up_bias_grad);
    scaled_sgd(model.output_fusion_norm->weight, accum_grads_.output_fusion_norm_weight_grad);
    scaled_sgd(model.output_fusion_norm->bias, accum_grads_.output_fusion_norm_bias_grad);
    scaled_sgd(model.output_fusion_bottleneck_norm->weight, accum_grads_.output_fusion_bottleneck_norm_weight_grad);
    scaled_sgd(model.output_fusion_bottleneck_norm->bias, accum_grads_.output_fusion_bottleneck_norm_bias_grad);
    scaled_sgd(model.ecn->vmpfc2->weight, accum_grads_.ecn_vmpfc2_weight_grad);
    scaled_sgd(model.ecn->vmpfc2->bias, accum_grads_.ecn_vmpfc2_bias_grad);
    scaled_sgd(model.ecn->vmpfc1->weight, accum_grads_.ecn_vmpfc1_weight_grad);
    scaled_sgd(model.ecn->vmpfc1->bias, accum_grads_.ecn_vmpfc1_bias_grad);

    for (size_t i = 0; i < accum_grads_.ecn_dlpfc_weight_grads.size() && i < model.ecn->dlpfc_linear.size(); ++i)
        scaled_sgd(model.ecn->dlpfc_linear[i]->weight, accum_grads_.ecn_dlpfc_weight_grads[i]);
    for (size_t i = 0; i < accum_grads_.ecn_dlpfc_bias_grads.size() && i < model.ecn->dlpfc_linear.size(); ++i)
        scaled_sgd(model.ecn->dlpfc_linear[i]->bias, accum_grads_.ecn_dlpfc_bias_grads[i]);
    for (size_t i = 0; i < accum_grads_.ecn_dlpfc_norm_weight_grads.size() && i < model.ecn->dlpfc_norm.size(); ++i)
        scaled_sgd(model.ecn->dlpfc_norm[i]->weight, accum_grads_.ecn_dlpfc_norm_weight_grads[i]);
    for (size_t i = 0; i < accum_grads_.ecn_dlpfc_norm_bias_grads.size() && i < model.ecn->dlpfc_norm.size(); ++i)
        scaled_sgd(model.ecn->dlpfc_norm[i]->bias, accum_grads_.ecn_dlpfc_norm_bias_grads[i]);

    accum_initialized_ = false;
    accum_loss_ = 0.0f;
    accum_grad_norm_ = 0.0f;
}

}

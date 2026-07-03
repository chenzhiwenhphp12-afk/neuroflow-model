#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <iostream>
#include <vector>
#include "neuroflow/generative.hpp"
#include "neuroflow/model.hpp"
#include "neuroflow/tensor.hpp"

using namespace neuroflow;

int main() {
    std::cerr << "=== CausalLMHead 通路测试 ===" << std::endl;

    CausalLMConfig cfg;
    cfg.vocab_size = 100;
    cfg.d_model = 32;
    cfg.max_seq_len = 16;
    cfg.causal_window_size = 4;
    cfg.sae_k = 8;
    cfg.ntm_memory_slots = 4;
    cfg.use_mla = false;
    cfg.weight_tying = true;
    cfg.num_attn_layers = 1;
    cfg.num_attn_heads = 2;
    cfg.pooling = "mean";

    std::cerr << "1. 构造 CausalLMHead..." << std::endl;
    CausalLMHead lm_head(cfg);
    lm_head.tie_weights();
    std::cerr << "   OK" << std::endl;

    std::cerr << "2. forward (推理)..." << std::endl;
    std::vector<size_t> token_ids = {1, 5, 10, 20, 30};
    Tensor logits = lm_head.forward(token_ids);
    std::cerr << "   logits shape: [" << logits.shape_[0] << "," << logits.shape_[1] << "]" << std::endl;
    std::cerr << "   OK" << std::endl;

    std::cerr << "3. forward_for_training..." << std::endl;
    std::vector<size_t> train_ids = {1, 5, 10, 20};
    Tensor train_logits = lm_head.forward_for_training(train_ids);
    std::cerr << "   train_logits shape: [" << train_logits.shape_[0] << "," << train_logits.shape_[1] << "]" << std::endl;
    std::cerr << "   OK" << std::endl;

    std::cerr << "4. backward_from_logits..." << std::endl;
    size_t target_id = 30;
    const float* pred = train_logits.as_fp32();
    float max_val = -1e30f;
    for (size_t j = 0; j < cfg.vocab_size; ++j) {
        if (pred[j] > max_val) max_val = pred[j];
    }
    float sum_exp = 0.0f;
    for (size_t j = 0; j < cfg.vocab_size; ++j) {
        sum_exp += std::exp(pred[j] - max_val);
    }
    float loss = -(pred[target_id] - max_val - std::log(sum_exp));
    std::cerr << "   loss = " << loss << std::endl;

    Tensor logits_grad({1, cfg.vocab_size}, QuantType::FP32);
    float* lg = logits_grad.as_fp32();
    for (size_t j = 0; j < cfg.vocab_size; ++j) {
        float softmax_val = std::exp(pred[j] - max_val) / sum_exp;
        lg[j] = softmax_val;
        if (j == target_id) lg[j] -= 1.0f;
    }

    auto grads = lm_head.backward_from_logits(logits_grad);
    std::cerr << "   attn_grads.size() = " << grads.attn_grads.size() << std::endl;
    std::cerr << "   w_proj_weight_grad shape: [" << grads.w_proj_weight_grad.shape_[0] << "," << grads.w_proj_weight_grad.shape_[1] << "]" << std::endl;
    std::cerr << "   embed_grad shape: [" << grads.embed_grad.shape_[0] << "," << grads.embed_grad.shape_[1] << "]" << std::endl;
    std::cerr << "   dw_kernel_grad shape: [" << grads.dw_kernel_grad.shape_[0] << "," << grads.dw_kernel_grad.shape_[1] << "]" << std::endl;

    // NaN diagnostics
    auto check_nan = [](const std::string& name, const Tensor& t) {
        if (t.numel() == 0 || t.data_size_ == 0) return;
        const float* d = t.as_fp32();
        size_t nan_count = 0, inf_count = 0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < t.numel(); ++i) {
            if (std::isnan(d[i])) nan_count++;
            else if (std::isinf(d[i])) inf_count++;
            else max_abs = std::max(max_abs, std::abs(d[i]));
        }
        std::cerr << "   [DIAG] " << name << ": nan=" << nan_count << " inf=" << inf_count << " max_abs=" << max_abs << std::endl;
    };
    check_nan("w_proj_weight_grad", grads.w_proj_weight_grad);
    std::cerr << "   [DEBUG] w_out_weight_grad numel=" << grads.w_out_weight_grad.numel() << std::endl;
    check_nan("w_out_weight_grad", grads.w_out_weight_grad);
    check_nan("ln_weight_grad", grads.ln_weight_grad);
    check_nan("sae_encode_weight_grad", grads.sae_encode_weight_grad);
    check_nan("sae_decode_weight_grad", grads.sae_decode_weight_grad);
    check_nan("ntm_read_weight_grad", grads.ntm_read_weight_grad);
    check_nan("dw_kernel_grad", grads.dw_kernel_grad);
    check_nan("pw_conv_weight_grad", grads.pw_conv_weight_grad);
    check_nan("embed_grad", grads.embed_grad);
    if (!grads.attn_grads.empty()) {
        check_nan("attn0.w_qkv_weight_grad", grads.attn_grads[0].w_qkv_weight_grad);
        check_nan("attn0.w_out_weight_grad", grads.attn_grads[0].w_out_weight_grad);
        check_nan("attn0.input_grad", grads.attn_grads[0].input_grad);
    }
    std::cerr << "   OK" << std::endl;

    std::cerr << "5. apply_lm_gradients..." << std::endl;
    lm_head.apply_lm_gradients(grads, 1e-5f);
    std::cerr << "   OK" << std::endl;

    std::cerr << "6. 第二次 forward_for_training (验证梯度更新有效)..." << std::endl;
    Tensor train_logits2 = lm_head.forward_for_training(train_ids);
    const float* pred2 = train_logits2.as_fp32();
    float max_val2 = -1e30f;
    for (size_t j = 0; j < cfg.vocab_size; ++j) {
        if (pred2[j] > max_val2) max_val2 = pred2[j];
    }
    float sum_exp2 = 0.0f;
    for (size_t j = 0; j < cfg.vocab_size; ++j) {
        sum_exp2 += std::exp(pred2[j] - max_val2);
    }
    float loss2 = -(pred2[target_id] - max_val2 - std::log(sum_exp2));
    std::cerr << "   loss2 = " << loss2 << std::endl;
    if (loss2 != loss) {
        std::cerr << "   梯度更新有效 (loss变化)" << std::endl;
    } else {
        std::cerr << "   警告: loss未变化" << std::endl;
    }

    std::cerr << "7. 保存 LM Head..." << std::endl;
    {
        auto sl = [](std::ofstream& o, const std::string& n, const Tensor& t) {
            uint32_t nl = n.size(); o.write((char*)&nl, 4); o.write(n.data(), nl);
            uint32_t nd = t.shape_.size(); o.write((char*)&nd, 4);
            for (auto d : t.shape_) { uint32_t dd = d; o.write((char*)&dd, 4); }
            uint32_t ds = t.data_size_; o.write((char*)&ds, 4);
            o.write((char*)t.data_.get(), ds);
        };
        std::ofstream o("D:/neuroflow-C++/test_run/lm_head_test.nfv1", std::ios::binary);
        o.write("LMH2", 4);
        sl(o, "w_embed", lm_head.w_embed_);
        sl(o, "w_proj.weight", lm_head.w_proj_->weight);
        sl(o, "w_out.weight", lm_head.w_out_->weight);
        sl(o, "dw_kernel", lm_head.dw_kernel_);
        sl(o, "sae_encode.weight", lm_head.sae_w_encode_->weight);
        sl(o, "sae_decode.weight", lm_head.sae_w_decode_->weight);
        sl(o, "ntm_read.weight", lm_head.ntm_w_read_->weight);
        sl(o, "ntm_write.weight", lm_head.ntm_w_write_->weight);
        sl(o, "ntm_erase.weight", lm_head.ntm_w_erase_->weight);
        sl(o, "ntm_memory", lm_head.ntm_memory_);
        sl(o, "ln.weight", lm_head.ln_->weight);
        sl(o, "ln.bias", lm_head.ln_->bias);
        for (size_t i = 0; i < lm_head.attn_layers_.size(); ++i) {
            std::string p = "attn" + std::to_string(i) + ".";
            sl(o, p + "w_qkv.weight", lm_head.attn_layers_[i]->w_qkv->weight);
            sl(o, p + "w_qkv.bias", lm_head.attn_layers_[i]->w_qkv->bias);
            sl(o, p + "w_out.weight", lm_head.attn_layers_[i]->w_out->weight);
            sl(o, p + "w_out.bias", lm_head.attn_layers_[i]->w_out->bias);
            sl(o, p + "norm.weight", lm_head.attn_layers_[i]->norm->weight);
            sl(o, p + "norm.bias", lm_head.attn_layers_[i]->norm->bias);
        }
        uint32_t z = 0; o.write((char*)&z, 4); o.close();
    }
    std::cerr << "   OK" << std::endl;

    std::cerr << "=== 所有通路测试通过! ===" << std::endl;
    return 0;
}
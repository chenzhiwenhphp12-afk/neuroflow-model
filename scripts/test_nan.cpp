/**
 * 最小 NaN 诊断测试 — 绕过 DataLoader，直接测试 forward + CE loss
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "neuroflow/backprop.hpp"
#include "neuroflow/generative.hpp"
#include "neuroflow/model.hpp"

using namespace neuroflow;

int main() {
    // Load model config
    NeuroFlowModel::Config cfg;
    // Use defaults: d_model=512, hidden_dim=2048, output_dim=2048, vocab_size=128000

    NeuroFlowModel model(cfg);
    FullBackpropEngine backprop(model);

    // Load tokenizer
    BPETokenizer tok("configs/tokenizer_128k.json");
    printf("Tokenizer loaded: vocab=%zu\n", tok.vocab_size());

    // Create CausalLMHead
    CausalLMConfig lm_cfg;
    lm_cfg.vocab_size = cfg.vocab_size;
    lm_cfg.d_model = cfg.hidden_dim;  // 2048
    lm_cfg.max_seq_len = cfg.max_seq_len;
    lm_cfg.causal_window_size = cfg.causal_window_size;
    lm_cfg.sae_k = cfg.sae_k;
    lm_cfg.ntm_memory_slots = cfg.ntm_memory_slots;
    lm_cfg.weight_tying = true;
    lm_cfg.num_attn_layers = cfg.lm_num_attn_layers;
    lm_cfg.pooling = cfg.lm_pooling;
    CausalLMHead lm_head(lm_cfg);
    lm_head.tie_weights();

    printf("CausalLMHead: d_model=%zu vocab=%zu\n", lm_cfg.d_model, lm_cfg.vocab_size);

    // Create bridge projection (matching train_v2.cpp)
    size_t d_model = lm_cfg.d_model;
    size_t hidden_dim = cfg.hidden_dim;
    Tensor lm_bridge_weight({d_model, hidden_dim}, QuantType::FP32);
    Tensor lm_bridge_bias({d_model}, QuantType::FP32);
    {
        float* bw = lm_bridge_weight.as_fp32();
        float scale = std::sqrt(2.0f / (hidden_dim + d_model));
        std::mt19937 br_rng(hidden_dim * 31 + d_model);
        std::uniform_real_distribution<float> br_dist(-scale, scale);
        for (size_t i = 0; i < lm_bridge_weight.numel(); ++i) bw[i] = br_dist(br_rng);
        memset(lm_bridge_bias.as_fp32(), 0, lm_bridge_bias.data_size_);
    }

    // Create random test data
    size_t batch_sz = 4;
    size_t seq_len = 16;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> id_dist(4, tok.vocab_size() - 1);

    printf("\n=== Testing %zu batches ===\n", batch_sz);

    for (int step = 0; step < 200; ++step) {
        // Create random token sequence
        std::vector<size_t> all_ids;
        for (size_t b = 0; b < batch_sz; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                all_ids.push_back(id_dist(rng));
            }
        }

        // Create input tensor [batch_sz, input_dim=512]
        Tensor input({batch_sz, cfg.input_dim}, QuantType::FP32);
        float* inp = input.as_fp32();
        for (size_t b = 0; b < batch_sz; ++b) {
            for (size_t j = 0; j < cfg.input_dim; ++j) {
                inp[b * cfg.input_dim + j] = static_cast<float>(all_ids[b * seq_len + (j % seq_len)]) / 128000.0f;
            }
        }

        // Create target tensor [batch_sz, vocab_size]
        size_t vocab_sz = lm_cfg.vocab_size;
        Tensor target({batch_sz, vocab_sz}, QuantType::FP32);
        float* tgt = target.as_fp32();
        memset(tgt, 0, target.data_size_);
        for (size_t b = 0; b < batch_sz; ++b) {
            size_t next_id = all_ids[b * seq_len + 1]; // next token as target
            if (next_id < vocab_sz) tgt[b * vocab_sz + next_id] = 1.0f;
        }

        // Forward pass
        auto nf_output = backprop.forward_with_cache(input);
        const Tensor& nf_hidden = nf_output.output;

        // Bridge projection
        Tensor hidden_proj({batch_sz, d_model}, QuantType::FP32);
#ifdef USE_CBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    batch_sz, d_model, hidden_dim,
                    1.0f, nf_hidden.as_fp32(), hidden_dim,
                    lm_bridge_weight.as_fp32(), hidden_dim,
                    0.0f, hidden_proj.as_fp32(), d_model);
#else
        {
            float* hp = hidden_proj.as_fp32();
            const float* nh = nf_hidden.as_fp32();
            const float* bw = lm_bridge_weight.as_fp32();
            for (size_t b = 0; b < batch_sz; ++b) {
                for (size_t j = 0; j < d_model; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < hidden_dim; ++k) {
                        sum += nh[b * hidden_dim + k] * bw[j * hidden_dim + k];
                    }
                    hp[b * d_model + j] = sum;
                }
            }
        }
#endif
        // Add bias
        {
            float* hp = hidden_proj.as_fp32();
            const float* bb = lm_bridge_bias.as_fp32();
            for (size_t b = 0; b < batch_sz; ++b)
                for (size_t j = 0; j < d_model; ++j)
                    hp[b * d_model + j] += bb[j];
        }

        Tensor projected = lm_head.w_proj_->forward(hidden_proj);
        Tensor logits = lm_head.w_out_->forward(projected);

        // Check logits for NaN
        const float* pred = logits.as_fp32();
        size_t n = logits.numel();
        bool pred_nan = false;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(pred[i])) { pred_nan = true; break; }
        }
        if (pred_nan) {
            printf("Step %d: logits NaN BEFORE loss!\n", step);
            return 1;
        }

        // Compute CE loss
        size_t dim = vocab_sz;
        float loss = 0.0f;
        for (size_t b = 0; b < batch_sz; ++b) {
            float max_val = -1e30f;
            for (size_t j = 0; j < dim; ++j) {
                if (pred[b * dim + j] > max_val) max_val = pred[b * dim + j];
            }
            float sum_exp = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                sum_exp += std::exp(pred[b * dim + j] - max_val);
            }
            if (!std::isfinite(sum_exp) || sum_exp <= 0) {
                printf("Step %d batch %zu: sum_exp=%f max_val=%f\n", step, b, sum_exp, max_val);
                return 1;
            }
            float log_sum_exp = max_val + std::log(sum_exp);
            if (!std::isfinite(log_sum_exp)) {
                printf("Step %d batch %zu: log_sum_exp=%f\n", step, b, log_sum_exp);
                return 1;
            }

            for (size_t j = 0; j < dim; ++j) {
                float softmax_val = std::exp(pred[b * dim + j] - max_val) / sum_exp;
                float t = tgt[b * dim + j];
                if (t > 0.5f) {
                    float contrib = pred[b * dim + j] - log_sum_exp;
                    loss -= contrib;
                    if (!std::isfinite(loss)) {
                        printf("Step %d batch %zu j=%zu: LOSS NaN! pred=%f log_sum_exp=%f contrib=%f\n",
                               step, b, j, pred[b*dim+j], log_sum_exp, contrib);
                        return 1;
                    }
                }
            }
        }
        loss /= batch_sz;

        // Backward
        Tensor output_grad({batch_sz, dim}, QuantType::FP32);
        float* og = output_grad.as_fp32();
        for (size_t b = 0; b < batch_sz; ++b) {
            float max_val = -1e30f;
            for (size_t j = 0; j < dim; ++j)
                if (pred[b*dim+j] > max_val) max_val = pred[b*dim+j];
            float sum_exp = 0.0f;
            for (size_t j = 0; j < dim; ++j)
                sum_exp += std::exp(pred[b*dim+j] - max_val);
            for (size_t j = 0; j < dim; ++j) {
                float softmax_val = std::exp(pred[b*dim+j] - max_val) / sum_exp;
                float t = tgt[b*dim+j];
                og[b*dim+j] = (softmax_val - t) / batch_sz;
            }
        }

        // Check output_grad
        for (size_t i = 0; i < output_grad.numel(); ++i) {
            if (!std::isfinite(og[i])) {
                printf("Step %d: output_grad NaN at i=%zu\n", step, i);
                return 1;
            }
        }

        // Backward through LM head
        // (simplified for test — just check that backward doesn't crash)
        auto nf_grads = backprop.backward(output_grad);

        // Check nf_grads for NaN
        auto check_nan = [](const Tensor& t, const char* name) {
            if (t.numel() == 0) return;
            const float* d = t.as_fp32();
            for (size_t i = 0; i < t.numel(); ++i) {
                if (!std::isfinite(d[i])) {
                    printf("  GRAD NaN in %s at i=%zu\n", name, i);
                    return;
                }
            }
        };
        check_nan(nf_grads.input_proj_weight_grad, "input_proj_weight");
        check_nan(nf_grads.output_fusion_up_weight_grad, "output_fusion_up_weight");
        check_nan(nf_grads.ecn_vmpfc2_weight_grad, "ecn_vmpfc2_weight");
        check_nan(nf_grads.sn_gate1_weight_grad, "sn_gate1_weight");
        check_nan(nf_grads.dmn_mem_encoder1_weight_grad, "dmn_mem_encoder1_weight");
        check_nan(nf_grads.mem_encode_proj_weight_grad, "mem_encode_proj_weight");

        if (step % 50 == 0) {
            float lmin = 1e30f, lmax = -1e30f;
            for (size_t i = 0; i < std::min(n, size_t(1000)); ++i) {
                if (pred[i] < lmin) lmin = pred[i];
                if (pred[i] > lmax) lmax = pred[i];
            }
            printf("Step %d: loss=%.4f logits=[%.4f, %.4f] OK\n", step, loss, lmin, lmax);
        }
    }

    printf("\nAll %d steps passed! No NaN detected.\n", 200);
    return 0;
}

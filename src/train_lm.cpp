#include "neuroflow/train_lm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#ifdef USE_CUDA
#include "cuda_context.hpp"
#include "cuda_kernels.hpp"
#endif

namespace neuroflow {

TrainLM::TrainLM(const TrainLMConfig& config) : cfg_(config) {
    CausalLMConfig lm_cfg;
    lm_cfg.vocab_size = cfg_.vocab_size;
    lm_cfg.d_model = cfg_.d_model;
    lm_cfg.max_seq_len = cfg_.max_seq_len;
    lm_cfg.num_attn_layers = cfg_.num_attn_layers;
    lm_cfg.num_attn_heads = cfg_.num_attn_heads;
    lm_cfg.causal_window_size = cfg_.causal_window_size;
    lm_cfg.sae_k = cfg_.sae_k;
    lm_cfg.ntm_memory_slots = cfg_.ntm_memory_slots;
    lm_cfg.weight_tying = cfg_.weight_tying;
    lm_cfg.use_rope = cfg_.use_rope;
    lm_cfg.use_bridge = cfg_.use_bridge;
    lm_cfg.use_swiglu = cfg_.use_swiglu;
    lm_cfg.use_qk_norm = cfg_.use_qk_norm;
    lm_cfg.swiglu_intermediate_size = cfg_.swiglu_intermediate_size;
    lm_cfg.pooling = cfg_.pooling;

    lm_head_ = std::make_unique<CausalLMHead>(lm_cfg);
    setup_optimizer();

    size_t total = cfg_.total_steps > 0 ? cfg_.total_steps : cfg_.epochs * 1000;
    scheduler_ = std::make_unique<CosineScheduler>(
        cfg_.learning_rate, total, cfg_.lr_min_ratio, cfg_.warmup_ratio);
}

void TrainLM::setup_optimizer() {
    optimizer_ = std::make_unique<AdamW>(
        cfg_.learning_rate, cfg_.adam_beta1, cfg_.adam_beta2,
        cfg_.adam_eps, cfg_.adam_weight_decay);

    ParamGroup weight_group;
    weight_group.lr = cfg_.learning_rate;
    weight_group.weight_decay = cfg_.adam_weight_decay;

    ParamGroup bias_group;
    bias_group.lr = cfg_.learning_rate;
    bias_group.weight_decay = 0.0f;

    auto add_param = [&](Tensor& param, Tensor& grad, bool is_bias) {
        if (is_bias) {
            bias_group.params.push_back(&param);
            bias_group.grads.push_back(&grad);
        } else {
            weight_group.params.push_back(&param);
            weight_group.grads.push_back(&grad);
        }
    };

    add_param(lm_head_->w_proj_->weight, lm_head_->w_proj_->weight, false);
    add_param(lm_head_->w_proj_->bias, lm_head_->w_proj_->bias, true);
    if (lm_head_->bridge_) {
        add_param(lm_head_->bridge_->weight, lm_head_->bridge_->weight, false);
        add_param(lm_head_->bridge_->bias, lm_head_->bridge_->bias, true);
    }
    add_param(lm_head_->ln_->weight, lm_head_->ln_->weight, false);
    add_param(lm_head_->ln_->bias, lm_head_->ln_->bias, true);
    add_param(lm_head_->sae_w_encode_->weight, lm_head_->sae_w_encode_->weight, false);
    add_param(lm_head_->sae_w_decode_->weight, lm_head_->sae_w_decode_->weight, false);
    add_param(lm_head_->ntm_w_read_->weight, lm_head_->ntm_w_read_->weight, false);
    add_param(lm_head_->ntm_w_write_->weight, lm_head_->ntm_w_write_->weight, false);
    add_param(lm_head_->ntm_w_erase_->weight, lm_head_->ntm_w_erase_->weight, false);
    add_param(lm_head_->dw_kernel_, lm_head_->dw_kernel_, false);
    add_param(lm_head_->pw_conv_->weight, lm_head_->pw_conv_->weight, false);
    add_param(lm_head_->pw_conv_->bias, lm_head_->pw_conv_->bias, true);

    for (auto& attn : lm_head_->attn_layers_) {
        add_param(attn->w_q->weight, attn->w_q->weight, false);
        add_param(attn->w_q->bias, attn->w_q->bias, true);
        add_param(attn->w_k->weight, attn->w_k->weight, false);
        add_param(attn->w_k->bias, attn->w_k->bias, true);
        add_param(attn->w_v->weight, attn->w_v->weight, false);
        add_param(attn->w_v->bias, attn->w_v->bias, true);
        add_param(attn->w_out->weight, attn->w_out->weight, false);
        add_param(attn->w_out->bias, attn->w_out->bias, true);
        add_param(attn->norm->weight, attn->norm->weight, false);
        add_param(attn->norm->bias, attn->norm->bias, true);
    }

    optimizer_->add_param_group(weight_group);
    optimizer_->add_param_group(bias_group);
}

float TrainLM::compute_loss_and_grad(const std::vector<size_t>& input_ids,
                                      const std::vector<size_t>& target_ids,
                                      Tensor& logits_grad) {
    Tensor logits = lm_head_->forward_for_training(input_ids);

    size_t seq_len = logits.shape_[0];
    size_t vocab_size = logits.shape_[1];
    size_t n_targets = std::min(seq_len, target_ids.size());

    float total_loss = 0.0f;
    logits_grad = Tensor(logits.shape_, QuantType::FP32);
    float* lg = logits_grad.as_fp32();
    const float* lp = logits.as_fp32();
    memset(lg, 0, logits_grad.data_size_);

    float grad_norm = 0.0f;

    for (size_t t = 0; t < n_targets; ++t) {
        size_t target_id = target_ids[t];
        if (target_id >= vocab_size) continue;

        const float* row = &lp[t * vocab_size];
        float* grad_row = &lg[t * vocab_size];

        float max_val = -1e30f;
        for (size_t j = 0; j < vocab_size; ++j) {
            max_val = std::max(max_val, row[j]);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            sum_exp += std::exp(row[j] - max_val);
        }

        float log_sum_exp = max_val + std::log(sum_exp);
        float loss = -(row[target_id] - log_sum_exp);
        total_loss += loss;

        for (size_t j = 0; j < vocab_size; ++j) {
            float softmax_val = std::exp(row[j] - max_val) / sum_exp;
            grad_row[j] = softmax_val;
            if (j == target_id) grad_row[j] -= 1.0f;
            grad_norm += grad_row[j] * grad_row[j];
        }
    }

    if (n_targets > 0) {
        float inv_n = 1.0f / static_cast<float>(n_targets);
        total_loss *= inv_n;
        for (size_t i = 0; i < logits_grad.numel(); ++i) {
            lg[i] *= inv_n;
        }
    }

    float gn = std::sqrt(grad_norm);
    if (cfg_.grad_clip > 0.0f && gn > cfg_.grad_clip && std::isfinite(gn)) {
        float scale = cfg_.grad_clip / gn;
        for (size_t i = 0; i < logits_grad.numel(); ++i) {
            lg[i] *= scale;
        }
    }

    return total_loss;
}

void TrainLM::train(const std::vector<std::vector<size_t>>& dataset) {
    if (dataset.empty()) {
        std::cerr << "[ERROR] Empty dataset" << std::endl;
        return;
    }

    std::cerr << "=== TrainLM: Standard Causal LM Training ===" << std::endl;
    std::cerr << "Dataset: " << dataset.size() << " samples" << std::endl;
    std::cerr << "Optimizer: AdamW (lr=" << cfg_.learning_rate
              << ", wd=" << cfg_.adam_weight_decay << ")" << std::endl;
    std::cerr << "Scheduler: Cosine with " << cfg_.warmup_ratio * 100 << "% warmup" << std::endl;

    size_t global_step = 0;
    std::mt19937 rng(42);

    for (size_t epoch = 0; epoch < cfg_.epochs; ++epoch) {
        std::vector<size_t> indices(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        size_t step_count = 0;

        for (size_t idx : indices) {
            const auto& sample = dataset[idx];
            if (sample.size() < 2) continue;

            std::vector<size_t> input_ids(sample.begin(), sample.end() - 1);
            std::vector<size_t> target_ids(sample.begin() + 1, sample.end());

            float lr = scheduler_->get_lr(global_step);
            optimizer_->set_lr(lr);

            Tensor logits_grad;
            float loss = compute_loss_and_grad(input_ids, target_ids, logits_grad);

            auto lm_grads = lm_head_->backward_from_logits(logits_grad);

            for (size_t g = 0; g < optimizer_->param_groups_.size(); ++g) {
                auto& group = optimizer_->param_groups_[g];
                for (size_t i = 0; i < group.params.size(); ++i) {
                    group.grads[i] = &lm_grads.w_proj_weight_grad;
                }
            }

            optimizer_->step();
            global_step++;

            epoch_loss += loss;
            step_count++;

            if (step_count % cfg_.log_interval == 0) {
                std::cerr << "Epoch " << epoch + 1 << " Step " << step_count
                          << " loss=" << loss << " lr=" << lr << std::endl;
            }

            if (cfg_.save_interval > 0 && global_step % cfg_.save_interval == 0) {
                save_checkpoint(global_step, epoch_loss / step_count);
            }
        }

        if (step_count > 0) {
            std::cerr << "=== Epoch " << epoch + 1 << " avg_loss="
                      << epoch_loss / step_count << " ===" << std::endl;
        }
    }

    save_checkpoint(global_step, 0.0f);
    std::cerr << "Training complete. Total steps: " << global_step << std::endl;
}

void TrainLM::save_checkpoint(size_t step, float loss) {
    std::string path = cfg_.output_dir + "/lm_head_step" + std::to_string(step) + ".nfv1";

#ifdef _WIN32
    std::string mkdir_cmd = "if not exist \"" + cfg_.output_dir + "\" mkdir \"" + cfg_.output_dir + "\"";
    system(mkdir_cmd.c_str());
#else
    std::string mkdir_cmd = "mkdir -p " + cfg_.output_dir;
    system(mkdir_cmd.c_str());
#endif

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[ERROR] Cannot save checkpoint: " << path << std::endl;
        return;
    }

    ofs.write("LMH2", 4);

    auto sl = [&ofs](const std::string& n, const Tensor& t) {
        uint32_t nl = static_cast<uint32_t>(n.size());
        ofs.write(reinterpret_cast<const char*>(&nl), 4);
        ofs.write(n.data(), nl);
        uint32_t nd = static_cast<uint32_t>(t.shape_.size());
        ofs.write(reinterpret_cast<const char*>(&nd), 4);
        for (auto d : t.shape_) {
            uint32_t dd = static_cast<uint32_t>(d);
            ofs.write(reinterpret_cast<const char*>(&dd), 4);
        }
        uint32_t ds = static_cast<uint32_t>(t.data_size_);
        ofs.write(reinterpret_cast<const char*>(&ds), 4);
        ofs.write(reinterpret_cast<const char*>(t.data_.get()), ds);
    };

    sl("w_embed", lm_head_->w_embed_);
    sl("w_pos", lm_head_->w_pos_);
    sl("dw_kernel", lm_head_->dw_kernel_);
    sl("pw_conv.weight", lm_head_->pw_conv_->weight);
    sl("sae_encode.weight", lm_head_->sae_w_encode_->weight);
    sl("sae_decode.weight", lm_head_->sae_w_decode_->weight);
    sl("ntm_read.weight", lm_head_->ntm_w_read_->weight);
    sl("ntm_write.weight", lm_head_->ntm_w_write_->weight);
    sl("ntm_erase.weight", lm_head_->ntm_w_erase_->weight);
    sl("ntm_memory", lm_head_->ntm_memory_);
    sl("w_proj.weight", lm_head_->w_proj_->weight);
    sl("w_proj.bias", lm_head_->w_proj_->bias);
    if (lm_head_->bridge_) {
        sl("bridge.weight", lm_head_->bridge_->weight);
        sl("bridge.bias", lm_head_->bridge_->bias);
    }
    sl("w_out.weight", lm_head_->w_out_->weight);
    if (lm_head_->w_out_->bias.data_) sl("w_out.bias", lm_head_->w_out_->bias);
    sl("ln.weight", lm_head_->ln_->weight);
    sl("ln.bias", lm_head_->ln_->bias);
    for (size_t i = 0; i < lm_head_->attn_layers_.size(); ++i) {
        std::string prefix = "attn" + std::to_string(i) + ".";
        sl(prefix + "w_q.weight", lm_head_->attn_layers_[i]->w_q->weight);
        sl(prefix + "w_q.bias", lm_head_->attn_layers_[i]->w_q->bias);
        sl(prefix + "w_k.weight", lm_head_->attn_layers_[i]->w_k->weight);
        sl(prefix + "w_k.bias", lm_head_->attn_layers_[i]->w_k->bias);
        sl(prefix + "w_v.weight", lm_head_->attn_layers_[i]->w_v->weight);
        sl(prefix + "w_v.bias", lm_head_->attn_layers_[i]->w_v->bias);
        sl(prefix + "w_out.weight", lm_head_->attn_layers_[i]->w_out->weight);
        sl(prefix + "w_out.bias", lm_head_->attn_layers_[i]->w_out->bias);
        sl(prefix + "norm.weight", lm_head_->attn_layers_[i]->norm->weight);
        sl(prefix + "norm.bias", lm_head_->attn_layers_[i]->norm->bias);
    }

    ofs.close();
    std::cerr << "[CHECKPOINT] Saved: " << path << " (loss=" << loss << ")" << std::endl;
}

} // namespace neuroflow
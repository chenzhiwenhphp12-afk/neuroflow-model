#ifndef NEUROFLOW_TRAIN_LM_HPP
#define NEUROFLOW_TRAIN_LM_HPP

#include <cstddef>
#include <string>
#include <vector>
#include "tensor.hpp"
#include "causal_lm.hpp"
#include "adamw.hpp"
#include "scheduler.hpp"

namespace neuroflow {

struct TrainLMConfig {
    size_t vocab_size = 128000;
    size_t d_model = 256;
    size_t max_seq_len = 128;
    size_t num_attn_layers = 2;
    size_t num_attn_heads = 4;
    size_t causal_window_size = 32;
    size_t sae_k = 64;
    size_t ntm_memory_slots = 16;
    bool weight_tying = true;
    bool use_rope = true;
    bool use_bridge = true;
    bool use_swiglu = true;
    bool use_qk_norm = true;
    size_t swiglu_intermediate_size = 0;
    std::string pooling = "mean";

    float learning_rate = 5e-4f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;
    float adam_weight_decay = 0.01f;
    float grad_clip = 1.0f;
    float lr_min_ratio = 0.1f;
    float warmup_ratio = 0.01f;

    size_t epochs = 10;
    size_t total_steps = 0;
    size_t log_interval = 10;
    size_t save_interval = 1000;
    std::string output_dir = "./checkpoints";
    std::string data_path;
};

class TrainLM {
public:
    TrainLM(const TrainLMConfig& config);
    void train(const std::vector<std::vector<size_t>>& dataset);

private:
    TrainLMConfig cfg_;
    std::unique_ptr<CausalLMHead> lm_head_;
    std::unique_ptr<AdamW> optimizer_;
    std::unique_ptr<CosineScheduler> scheduler_;

    void setup_optimizer();
    float compute_loss_and_grad(const std::vector<size_t>& input_ids,
                                const std::vector<size_t>& target_ids,
                                Tensor& logits_grad);
    void save_checkpoint(size_t step, float loss);
};

} // namespace neuroflow

#endif // NEUROFLOW_TRAIN_LM_HPP
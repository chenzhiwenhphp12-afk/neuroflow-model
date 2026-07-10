#ifndef NEUROFLOW_CAUSAL_LM_HPP
#define NEUROFLOW_CAUSAL_LM_HPP

#include <memory>
#include <string>
#include <vector>
#include "memory.hpp"
#include "model.hpp"
#include "networks.hpp"
#include "tensor.hpp"
#include "rope.hpp"
#include "swiglu.hpp"
#include "rms_norm.hpp"

#ifdef USE_CUDA
#include "cuda_kernels.hpp"
#endif

namespace neuroflow {

enum class FinishReason : uint8_t {
    EOS_TOKEN = 0,
    MAX_LENGTH = 1,
    GEN_ERROR = 2
};

struct CausalLMConfig {
    size_t vocab_size = 128000;
    size_t d_model = 512;
    size_t max_seq_len = 512;
    size_t causal_window_size = 32;
    size_t sae_k = 128;
    size_t ntm_memory_slots = 32;
    bool use_mla = true;
    size_t mla_latent_dim = 64;
    size_t mla_n_heads = 8;
    size_t mla_max_cache_len = 4096;
    bool use_quantization = false;
    bool weight_tying = true;
    bool use_rope = true;
    bool use_bridge = true;
    bool use_swiglu = true;
    size_t swiglu_intermediate_size = 0;
    bool use_qk_norm = true;
    size_t num_attn_layers = 4;
    size_t num_attn_heads = 8;
    size_t n_kv_heads = 2;
    int padding_id = -1;
    std::string pooling = "last";
};

struct CacheStats {
    size_t cache_len = 0;
    size_t memory_bytes = 0;
    float saving_ratio = 0.0f;
    size_t sliding_window_drops = 0;
};

struct GenerateOutput {
    std::string text;
    std::vector<size_t> token_ids;
    std::vector<Tensor> logits_history;
    FinishReason finish_reason = FinishReason::MAX_LENGTH;
    CacheStats cache_stats;
};

class CausalSelfAttention {
public:
    size_t n_q_heads_;
    size_t n_kv_heads_;
    size_t n_rep_;
    size_t d_model_;
    size_t head_dim_;

    std::shared_ptr<Linear> w_q;
    std::shared_ptr<Linear> w_k;
    std::shared_ptr<Linear> w_v;
    std::shared_ptr<Linear> w_out;
    std::shared_ptr<LayerNorm> norm;
    std::unique_ptr<RoPE> rope_;
    bool use_rope_;
    size_t max_seq_len_;
    bool use_qk_norm_;
    std::unique_ptr<RMSNorm> q_norm_;

    std::unique_ptr<RMSNorm> k_norm_;
    bool training_mode_ = false;
    float yarn_temp_scale_ = 1.0f;

    struct Cache {
        Tensor input;
        Tensor q_proj;
        Tensor k_proj;
        Tensor v_proj;
        Tensor attn_weights;
        Tensor attn_output;
        Tensor w_out_input;
        Tensor residual;
    };
    Cache cache_;

    CausalSelfAttention(size_t d_model, size_t n_q_heads, size_t n_kv_heads,
                        bool use_rope = true, size_t max_seq_len = 128, bool use_qk_norm = true);
    Tensor forward(const Tensor& x, const Tensor* padding_mask = nullptr);

    void train() { training_mode_ = true; }
    void eval() { training_mode_ = false; }

    struct Gradients {
        Tensor w_q_weight_grad;
        Tensor w_q_bias_grad;
        Tensor w_k_weight_grad;
        Tensor w_k_bias_grad;
        Tensor w_v_weight_grad;
        Tensor w_v_bias_grad;
        Tensor w_out_weight_grad;
        Tensor w_out_bias_grad;
        Tensor input_grad;
    };

    Gradients backward(const Tensor& output_grad);

private:
    Tensor layernorm_backward_impl(const Tensor& input, const Tensor& weight, const Tensor& output_grad, float eps = 1e-5f);
    Tensor linear_backward_weight_impl(const Tensor& input, const Tensor& output_grad);
    Tensor linear_backward_input_impl(const Tensor& output_grad, const Tensor& weight);
    Tensor bias_backward_impl(const Tensor& output_grad);
};

class CausalLMHead {
public:
    CausalLMConfig config_;

    Tensor w_embed_;
    Tensor w_pos_;
    Tensor dw_kernel_;
    std::shared_ptr<Linear> pw_conv_;
    std::shared_ptr<Linear> sae_w_encode_;
    std::shared_ptr<Linear> sae_w_decode_;
    std::shared_ptr<Linear> ntm_w_read_;
    std::shared_ptr<Linear> ntm_w_write_;
    std::shared_ptr<Linear> ntm_w_erase_;
    Tensor ntm_memory_;
    Tensor shadow_memory_;
    bool training_mode_ = false;
    bool mode_set_ = false;
    std::shared_ptr<Linear> w_proj_;
    std::shared_ptr<Linear> bridge_;
    std::shared_ptr<Linear> w_out_;
    std::shared_ptr<LayerNorm> ln_;
    std::shared_ptr<LatentKVCache> kv_cache_;
    Tensor last_hidden_;
    Tensor last_projected_;
    std::vector<std::unique_ptr<CausalSelfAttention>> attn_layers_;
    std::unique_ptr<SwiGLUFFN> swiglu_;

    struct TrainingCache {
        std::vector<size_t> input_ids;
        Tensor x_embed;
        Tensor x_pos;
        std::vector<Tensor> attn_inputs;
        std::vector<Tensor> attn_outputs;
        Tensor x_gate_in;
        Tensor x_gate_pre_sigmoid;
        Tensor x_after_gate;
        Tensor x_after_swiglu;
        Tensor x_sae_encoded;
        Tensor x_after_sae;
        Tensor x_ntm_read_weights;
        Tensor x_ntm_read_content;
        Tensor x_ntm_h;
        Tensor x_ntm_erase;
        Tensor x_ntm_write;
        Tensor x_after_ntm;
        Tensor x_after_ln;
        Tensor x_pooled;
        Tensor x_bridge;
        Tensor x_projected;
    };
    TrainingCache train_cache_;

    size_t sliding_window_drops_;

    void tie_weights();

    CausalLMHead(const CausalLMConfig& config);

    void train();
    void eval();
    bool is_training() const { return training_mode_; }
    void set_yarn_scale(float scale_factor);

    Tensor embed_lookup(const std::vector<size_t>& ids);
    Tensor positional_encode(const Tensor& x, size_t offset = 0);
    Tensor causal_window_gate(const Tensor& x);
    Tensor sae_sparse(const Tensor& x);
    Tensor ntm_memory_access(const Tensor& x);
    Tensor last_token_pool(const Tensor& x);
    Tensor mean_pool(const Tensor& x);
    Tensor pool(const Tensor& x);

    Tensor make_padding_mask(const std::vector<size_t>& token_ids) const;

    Tensor forward(const std::vector<size_t>& token_ids);
    Tensor forward_step(size_t token_id, size_t pos);
    Tensor forward_for_training(const std::vector<size_t>& token_ids);

    struct LMGradients {
        std::vector<CausalSelfAttention::Gradients> attn_grads;
        Tensor w_proj_weight_grad;
        Tensor w_proj_bias_grad;
        Tensor bridge_weight_grad;
        Tensor bridge_bias_grad;
        Tensor w_out_weight_grad;
        Tensor w_out_bias_grad;
        Tensor embed_grad;
        std::vector<size_t> used_token_ids;
        Tensor ln_weight_grad;
        Tensor ln_bias_grad;
        Tensor ntm_read_weight_grad;
        Tensor ntm_write_weight_grad;
        Tensor ntm_erase_weight_grad;
        Tensor sae_encode_weight_grad;
        Tensor sae_decode_weight_grad;
        Tensor dw_kernel_grad;
        Tensor pw_conv_weight_grad;
        Tensor pw_conv_bias_grad;
        SwiGLUFFN::Gradients swiglu_grads;
    };

    LMGradients backward_from_logits(const Tensor& logits_grad);
    void apply_lm_gradients(LMGradients& grads, float lr);

    void clear_cache();
    CacheStats cache_stats() const;

private:
    Tensor ln_backward_impl(const Tensor& input, const Tensor& weight, const Tensor& output_grad, float eps = 1e-5f);
    Tensor lm_head_linear_backward_input(const Tensor& output_grad, const Tensor& weight);
    Tensor lm_head_linear_backward_weight(const Tensor& input, const Tensor& output_grad);
    Tensor lm_head_bias_backward(const Tensor& output_grad);
};

} // namespace neuroflow

#endif // NEUROFLOW_CAUSAL_LM_HPP
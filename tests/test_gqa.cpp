#include "test_framework.hpp"
#include "neuroflow/causal_lm.hpp"
#include <cmath>

using namespace neuroflow;

TEST(GQA, MHAWhenQEqualsKV) {
    CausalLMConfig config;
    config.d_model = 64;
    config.num_attn_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 100;
    config.max_seq_len = 32;
    config.use_rope = false;
    config.use_qk_norm = false;

    CausalLMHead lm(config);
    lm.eval();

    std::vector<size_t> ids = {1, 2, 3, 4};
    Tensor logits = lm.forward(ids);
    EXPECT_EQ(logits.shape_[0], 1u);
    EXPECT_EQ(logits.shape_[1], 100u);
}

TEST(GQA, GQAReducedKVHeads) {
    CausalLMConfig config;
    config.d_model = 64;
    config.num_attn_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 32;
    config.use_rope = false;
    config.use_qk_norm = false;

    CausalLMHead lm(config);
    lm.eval();

    std::vector<size_t> ids = {1, 2, 3, 4};
    Tensor logits = lm.forward(ids);
    EXPECT_EQ(logits.shape_[0], 1u);
    EXPECT_EQ(logits.shape_[1], 100u);

    EXPECT_FALSE(std::isnan(logits.as_fp32()[0]));
}

TEST(GQA, InvalidRatioThrows) {
    EXPECT_THROW({
        CausalSelfAttention attn(64, 5, 2, false, 32, false);
    }, std::invalid_argument);
}

TEST(GQA, TrainingBackwardWithGQA) {
    CausalLMConfig config;
    config.d_model = 64;
    config.num_attn_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 32;
    config.use_rope = false;
    config.use_qk_norm = false;

    CausalLMHead lm(config);
    lm.train();

    std::vector<size_t> ids = {1, 2, 3, 4};
    Tensor logits = lm.forward_for_training(ids);

    Tensor grad({1, 100}, QuantType::FP32);
    float* gp = grad.as_fp32();
    for (size_t i = 0; i < 100; ++i) gp[i] = 0.01f;

    auto grads = lm.backward_from_logits(grad);
    EXPECT_GT(grads.attn_grads.size(), 0u);
    EXPECT_GT(grads.attn_grads[0].w_q_weight_grad.numel(), 0u);
    EXPECT_GT(grads.attn_grads[0].w_k_weight_grad.numel(), 0u);
    EXPECT_GT(grads.attn_grads[0].w_v_weight_grad.numel(), 0u);
}

TEST(GQA, KVParamsSmallerWithGQA) {
    CausalLMConfig config_mha;
    config_mha.d_model = 64;
    config_mha.num_attn_heads = 4;
    config_mha.n_kv_heads = 4;
    config_mha.vocab_size = 100;
    config_mha.max_seq_len = 32;
    config_mha.use_rope = false;
    config_mha.use_qk_norm = false;

    CausalLMConfig config_gqa;
    config_gqa.d_model = 64;
    config_gqa.num_attn_heads = 4;
    config_gqa.n_kv_heads = 2;
    config_gqa.vocab_size = 100;
    config_gqa.max_seq_len = 32;
    config_gqa.use_rope = false;
    config_gqa.use_qk_norm = false;

    CausalLMHead lm_mha(config_mha);
    CausalLMHead lm_gqa(config_gqa);

    size_t mha_kv_params = 0;
    size_t gqa_kv_params = 0;
    for (auto& attn : lm_mha.attn_layers_) {
        mha_kv_params += attn->w_k->weight.numel() + attn->w_v->weight.numel();
    }
    for (auto& attn : lm_gqa.attn_layers_) {
        gqa_kv_params += attn->w_k->weight.numel() + attn->w_v->weight.numel();
    }

    EXPECT_LT(gqa_kv_params, mha_kv_params);
}

int main() { RUN_ALL_TESTS(); }

#include "test_framework.hpp"
#include "neuroflow/rope.hpp"
#include <cmath>

using namespace neuroflow;

TEST(RoPE, ConstructionAndFreqShape) {
    RoPE rope(64, 128);
    EXPECT_EQ(rope.head_dim_, 64u);
    EXPECT_EQ(rope.max_seq_len_, 128u);
    EXPECT_EQ(rope.freqs_cos_.shape_.size(), 2u);
    EXPECT_EQ(rope.freqs_cos_.shape_[0], 128u);
    EXPECT_EQ(rope.freqs_cos_.shape_[1], 32u);
}

TEST(RoPE, FreqValuesInRange) {
    RoPE rope(64, 128);
    const float* cos_data = rope.freqs_cos_.as_fp32();
    const float* sin_data = rope.freqs_sin_.as_fp32();
    size_t n = rope.freqs_cos_.numel();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_TRUE(cos_data[i] >= -1.0f - 1e-6f && cos_data[i] <= 1.0f + 1e-6f);
        EXPECT_TRUE(sin_data[i] >= -1.0f - 1e-6f && sin_data[i] <= 1.0f + 1e-6f);
    }
}

TEST(RoPE, ApplySingleModifiesQK) {
    size_t head_dim = 64;
    size_t n_heads = 4;
    size_t seq_len = 8;
    size_t total = seq_len * n_heads * head_dim;

    RoPE rope(head_dim, 128);

    Tensor q({seq_len, n_heads * head_dim}, QuantType::FP32);
    float* qp = q.as_fp32();
    for (size_t i = 0; i < total; ++i) qp[i] = 1.0f;

    Tensor q_copy({seq_len, n_heads * head_dim}, QuantType::FP32);
    memcpy(q_copy.as_fp32(), qp, total * sizeof(float));

    rope.apply_single(q, seq_len, n_heads, 0);

    bool changed = false;
    const float* qp_after = q.as_fp32();
    const float* qcp = q_copy.as_fp32();
    for (size_t i = 0; i < total; ++i) {
        if (std::abs(qp_after[i] - qcp[i]) > 1e-6f) { changed = true; break; }
    }
    EXPECT_TRUE(changed);
}

TEST(RoPE, PositionZeroCosOneSinZero) {
    RoPE rope(64, 128);
    const float* cos_data = rope.freqs_cos_.as_fp32();
    const float* sin_data = rope.freqs_sin_.as_fp32();
    for (size_t d = 0; d < rope.freqs_cos_.shape_[1]; ++d) {
        EXPECT_NEAR(cos_data[0 * rope.freqs_cos_.shape_[1] + d], 1.0f, 1e-5f);
        EXPECT_NEAR(sin_data[0 * rope.freqs_sin_.shape_[1] + d], 0.0f, 1e-5f);
    }
}

TEST(RoPE, YarnScaleUpdatesFreqs) {
    RoPE rope(64, 128);
    float orig_cos_1 = rope.freqs_cos_.as_fp32()[rope.freqs_cos_.shape_[1]];
    rope.set_yarn_scale(2.0f);
    float new_cos_1 = rope.freqs_cos_.as_fp32()[rope.freqs_cos_.shape_[1]];
    EXPECT_NE(orig_cos_1, new_cos_1);
}

int main() { RUN_ALL_TESTS(); }

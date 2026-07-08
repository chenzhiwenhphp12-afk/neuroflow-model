#include "test_framework.hpp"
#include "neuroflow/rms_norm.hpp"
#include <cmath>

using namespace neuroflow;

TEST(RMSNorm, Construction) {
    RMSNorm norm(64);
    EXPECT_EQ(norm.dim_, 64u);
    EXPECT_EQ(norm.weight_.shape_[0], 64u);
    const float* w = norm.weight_.as_fp32();
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_NEAR(w[i], 1.0f, 1e-6f);
    }
}

TEST(RMSNorm, ZeroInputGivesZeroOutput) {
    RMSNorm norm(32);
    Tensor x({1, 32}, QuantType::FP32);
    memset(x.as_fp32(), 0, 32 * sizeof(float));

    Tensor out = norm.forward(x);
    const float* op = out.as_fp32();
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_NEAR(op[i], 0.0f, 1e-6f);
    }
}

TEST(RMSNorm, NormalizedL2ApproxOne) {
    RMSNorm norm(64, 1e-5f);
    Tensor x({1, 64}, QuantType::FP32);
    float* xp = x.as_fp32();
    for (size_t i = 0; i < 64; ++i) xp[i] = static_cast<float>(i) * 0.1f + 0.5f;

    Tensor out = norm.forward(x);
    const float* op = out.as_fp32();
    float l2 = 0.0f;
    for (size_t i = 0; i < 64; ++i) l2 += op[i] * op[i];
    float rms = std::sqrt(l2 / 64.0f);
    EXPECT_NEAR(rms, 1.0f, 0.05f);
}

TEST(RMSNorm, BackwardGradientsShape) {
    RMSNorm norm(32);
    Tensor x({2, 32}, QuantType::FP32);
    float* xp = x.as_fp32();
    for (size_t i = 0; i < x.numel(); ++i) xp[i] = 0.5f;

    Tensor out = norm.forward(x);

    Tensor grad({2, 32}, QuantType::FP32);
    float* gp = grad.as_fp32();
    for (size_t i = 0; i < grad.numel(); ++i) gp[i] = 1.0f;

    auto grads = norm.backward(grad);
    EXPECT_EQ(grads.input_grad.shape_[0], 2u);
    EXPECT_EQ(grads.input_grad.shape_[1], 32u);
    EXPECT_EQ(grads.weight_grad.shape_[0], 32u);
}

TEST(RMSNorm, MultipleRows) {
    RMSNorm norm(16);
    Tensor x({4, 16}, QuantType::FP32);
    float* xp = x.as_fp32();
    for (size_t i = 0; i < x.numel(); ++i) xp[i] = 1.0f;

    Tensor out = norm.forward(x);
    EXPECT_EQ(out.shape_[0], 4u);
    EXPECT_EQ(out.shape_[1], 16u);
}

int main() { RUN_ALL_TESTS(); }

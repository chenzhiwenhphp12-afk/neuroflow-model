#include "test_framework.hpp"
#include "neuroflow/swiglu.hpp"
#include <cmath>

using namespace neuroflow;

TEST(SwiGLU, ConstructionDefaultFF) {
    SwiGLUFFN ffn(256);
    EXPECT_EQ(ffn.d_model_, 256u);
    EXPECT_GT(ffn.d_ff_, 0u);
    EXPECT_NE(ffn.d_ff_, 256u);
}

TEST(SwiGLU, IntermediateSizeComputation) {
    SwiGLUFFN ffn_256(256);
    EXPECT_EQ(ffn_256.d_ff_, 256u * 4);

    SwiGLUFFN ffn_512(512);
    EXPECT_EQ(ffn_512.d_ff_, 512u * 4);

    SwiGLUFFN ffn_custom(64, 128);
    EXPECT_EQ(ffn_custom.d_ff_, 128u);
}

TEST(SwiGLU, ForwardOutputShape) {
    SwiGLUFFN ffn(64, 128);
    Tensor x({4, 64}, QuantType::FP32);
    float* xp = x.as_fp32();
    for (size_t i = 0; i < x.numel(); ++i) xp[i] = 0.1f;

    Tensor out = ffn.forward(x);
    EXPECT_EQ(out.shape_.size(), 2u);
    EXPECT_EQ(out.shape_[0], 4u);
    EXPECT_EQ(out.shape_[1], 64u);
}

TEST(SwiGLU, ForwardNoNaN) {
    SwiGLUFFN ffn(64, 128);
    Tensor x({2, 64}, QuantType::FP32);
    float* xp = x.as_fp32();
    for (size_t i = 0; i < x.numel(); ++i) xp[i] = 0.5f;

    Tensor out = ffn.forward(x);
    const float* op = out.as_fp32();
    for (size_t i = 0; i < out.numel(); ++i) {
        EXPECT_FALSE(std::isnan(op[i]));
        EXPECT_FALSE(std::isinf(op[i]));
    }
}

TEST(SwiGLU, BackwardGradientsExist) {
    SwiGLUFFN ffn(64, 128);
    ffn.training_mode_ = true;
    Tensor x({2, 64}, QuantType::FP32);
    float* xp = x.as_fp32();
    for (size_t i = 0; i < x.numel(); ++i) xp[i] = 0.5f;

    Tensor out = ffn.forward(x);

    Tensor grad({2, 64}, QuantType::FP32);
    float* gp = grad.as_fp32();
    for (size_t i = 0; i < grad.numel(); ++i) gp[i] = 1.0f;

    auto grads = ffn.backward(grad);
    EXPECT_GT(grads.w_gate_weight_grad.numel(), 0u);
    EXPECT_GT(grads.w_down_weight_grad.numel(), 0u);
    EXPECT_EQ(grads.input_grad.shape_[0], 2u);
    EXPECT_EQ(grads.input_grad.shape_[1], 64u);
}

int main() { RUN_ALL_TESTS(); }

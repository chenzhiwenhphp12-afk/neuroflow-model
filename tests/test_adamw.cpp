#include "test_framework.hpp"
#include "neuroflow/adamw.hpp"
#include <cmath>

using namespace neuroflow;

TEST(AdamW, Construction) {
    AdamW opt(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    EXPECT_NEAR(opt.lr_, 0.001f, 1e-8f);
    EXPECT_NEAR(opt.beta1_, 0.9f, 1e-8f);
    EXPECT_EQ(opt.step_, 0u);
}

TEST(AdamW, SingleStepUpdate) {
    AdamW opt(0.01f);
    Tensor param({4}, QuantType::FP32);
    float* pp = param.as_fp32();
    pp[0] = 1.0f; pp[1] = 2.0f; pp[2] = 3.0f; pp[3] = 4.0f;

    Tensor grad({4}, QuantType::FP32);
    float* gp = grad.as_fp32();
    gp[0] = 0.1f; gp[1] = 0.2f; gp[2] = 0.3f; gp[3] = 0.4f;

    ParamGroup pg;
    pg.params = {&param};
    pg.grads = {&grad};
    pg.lr = 0.01f;
    pg.weight_decay = 0.01f;
    opt.add_param_group(pg);

    float orig_p0 = pp[0];
    opt.step();

    EXPECT_NE(pp[0], orig_p0);
    EXPECT_EQ(opt.step_, 1u);
}

TEST(AdamW, BiasCorrectionStep1) {
    AdamW opt(0.01f);
    Tensor param({2}, QuantType::FP32);
    float* pp = param.as_fp32();
    pp[0] = 1.0f; pp[1] = 1.0f;

    Tensor grad({2}, QuantType::FP32);
    float* gp = grad.as_fp32();
    gp[0] = 1.0f; gp[1] = 1.0f;

    ParamGroup pg;
    pg.params = {&param};
    pg.grads = {&grad};
    pg.lr = 0.01f;
    pg.weight_decay = 0.0f;
    opt.add_param_group(pg);

    opt.step();

    float m_hat = 0.1f / (1.0f - 0.9f);
    float v_hat = 0.01f / (1.0f - 0.999f);
    float expected = 1.0f - 0.01f * m_hat / (std::sqrt(v_hat) + 1e-8f);
    EXPECT_NEAR(pp[0], expected, 0.01f);
}

TEST(AdamW, WeightDecayApplied) {
    AdamW opt_no_wd(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
    AdamW opt_wd(0.01f, 0.9f, 0.999f, 1e-8f, 0.1f);

    Tensor p1({2}, QuantType::FP32);
    Tensor p2({2}, QuantType::FP32);
    float* p1p = p1.as_fp32();
    float* p2p = p2.as_fp32();
    p1p[0] = 5.0f; p1p[1] = 5.0f;
    p2p[0] = 5.0f; p2p[1] = 5.0f;

    Tensor g({2}, QuantType::FP32);
    float* gp = g.as_fp32();
    gp[0] = 0.0f; gp[1] = 0.0f;

    ParamGroup pg1;
    pg1.params = {&p1}; pg1.grads = {&g}; pg1.lr = 0.01f; pg1.weight_decay = 0.0f;
    opt_no_wd.add_param_group(pg1);

    ParamGroup pg2;
    pg2.params = {&p2}; pg2.grads = {&g}; pg2.lr = 0.01f; pg2.weight_decay = 0.1f;
    opt_wd.add_param_group(pg2);

    opt_no_wd.step();
    opt_wd.step();

    EXPECT_GT(std::abs(p1p[0] - p2p[0]), 1e-6f);
}

TEST(AdamW, SetLr) {
    AdamW opt(0.001f);
    opt.set_lr(0.01f);
    EXPECT_NEAR(opt.get_lr(), 0.01f, 1e-8f);
}

TEST(AdamW, NaNGradientSkipped) {
    AdamW opt(0.01f);
    Tensor param({2}, QuantType::FP32);
    float* pp = param.as_fp32();
    pp[0] = 1.0f; pp[1] = 2.0f;

    Tensor grad({2}, QuantType::FP32);
    float* gp = grad.as_fp32();
    gp[0] = std::nanf(""); gp[1] = 0.1f;

    ParamGroup pg;
    pg.params = {&param}; pg.grads = {&grad}; pg.lr = 0.01f; pg.weight_decay = 0.0f;
    opt.add_param_group(pg);

    opt.step();

    EXPECT_NEAR(pp[0], 1.0f, 1e-6f);
}

int main() { RUN_ALL_TESTS(); }

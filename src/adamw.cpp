#include "neuroflow/adamw.hpp"

#include <cmath>
#include <iostream>

namespace neuroflow {

AdamW::AdamW(float lr, float beta1, float beta2, float eps, float weight_decay)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
      weight_decay_(weight_decay), step_(0) {}

void AdamW::add_param_group(const ParamGroup& group) {
    param_groups_.push_back(group);
    std::vector<ParamState> group_states;
    for (size_t i = 0; i < group.params.size(); ++i) {
        ParamState ps;
        ps.m = Tensor(group.params[i]->shape_, QuantType::FP32);
        ps.v = Tensor(group.params[i]->shape_, QuantType::FP32);
        memset(ps.m.as_fp32(), 0, ps.m.data_size_);
        memset(ps.v.as_fp32(), 0, ps.v.data_size_);
        group_states.push_back(std::move(ps));
    }
    states_.push_back(std::move(group_states));
}

void AdamW::step() {
    step_++;
    double bias_corr1 = 1.0 - std::pow(static_cast<double>(beta1_), static_cast<double>(step_));
    double bias_corr2 = 1.0 - std::pow(static_cast<double>(beta2_), static_cast<double>(step_));

    for (size_t g = 0; g < param_groups_.size(); ++g) {
        auto& group = param_groups_[g];
        auto& gstates = states_[g];
        float group_lr = group.lr > 0 ? group.lr : lr_;
        float group_wd = group.weight_decay;

        for (size_t i = 0; i < group.params.size(); ++i) {
            Tensor* param = group.params[i];
            Tensor* grad = group.grads[i];
            if (!param || !grad || param->numel() == 0 || grad->numel() == 0) continue;

            float* p = param->as_fp32();
            const float* g = grad->as_fp32();
            float* m = gstates[i].m.as_fp32();
            float* v = gstates[i].v.as_fp32();
            size_t n = param->numel();

            for (size_t j = 0; j < n; ++j) {
                if (!std::isfinite(g[j])) continue;

                p[j] -= group_lr * group_wd * p[j];

                m[j] = beta1_ * m[j] + (1.0f - beta1_) * g[j];
                v[j] = beta2_ * v[j] + (1.0f - beta2_) * g[j] * g[j];

                float m_hat = static_cast<float>(static_cast<double>(m[j]) / bias_corr1);
                float v_hat = static_cast<float>(static_cast<double>(v[j]) / bias_corr2);

                p[j] -= group_lr * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
}

void AdamW::set_lr(float lr) { lr_ = lr; }

} // namespace neuroflow
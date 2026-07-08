#include "neuroflow/grad_scaler.hpp"

#include <cmath>
#include <iostream>

namespace neuroflow {

GradScaler::GradScaler(float init_scale, float growth_factor,
                       float backoff_factor, size_t growth_interval)
    : scale_(init_scale), growth_factor_(growth_factor),
      backoff_factor_(backoff_factor), growth_interval_(growth_interval),
      growth_tracker_(0) {}

bool GradScaler::has_inf_or_nan(const std::vector<Tensor*>& grads) const {
    for (const auto* grad : grads) {
        if (!grad || grad->numel() == 0) continue;
        const float* data = grad->as_fp32();
        for (size_t i = 0; i < grad->numel(); ++i) {
            if (!std::isfinite(data[i])) return true;
        }
    }
    return false;
}

void GradScaler::unscale(std::vector<Tensor*>& grads) {
    float inv_scale = 1.0f / scale_;
    for (auto* grad : grads) {
        if (!grad || grad->numel() == 0) continue;
        float* data = grad->as_fp32();
        for (size_t i = 0; i < grad->numel(); ++i) {
            data[i] *= inv_scale;
        }
    }
}

void GradScaler::scale_loss(Tensor& loss) {
    float* d = loss.as_fp32();
    d[0] *= scale_;
}

void GradScaler::update(bool found_inf) {
    if (found_inf) {
        scale_ *= backoff_factor_;
        growth_tracker_ = 0;
    } else {
        growth_tracker_++;
        if (growth_tracker_ >= growth_interval_) {
            scale_ *= growth_factor_;
            growth_tracker_ = 0;
        }
    }
}

} // namespace neuroflow
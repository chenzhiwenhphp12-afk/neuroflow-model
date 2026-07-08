#include "neuroflow/scheduler.hpp"

#include <algorithm>
#include <cmath>

namespace neuroflow {

CosineScheduler::CosineScheduler(float lr_max, size_t total_steps,
                                  float lr_min_ratio, float warmup_ratio)
    : lr_max_(lr_max),
      lr_min_(lr_max * lr_min_ratio),
      warmup_steps_(std::max(size_t(1), static_cast<size_t>(static_cast<float>(total_steps) * warmup_ratio))),
      total_steps_(total_steps) {}

float CosineScheduler::get_lr(size_t step) const {
    if (total_steps_ == 0) return lr_max_;
    if (step >= total_steps_) return lr_min_;
    if (step < warmup_steps_) {
        return lr_max_ * static_cast<float>(step + 1) / static_cast<float>(warmup_steps_);
    }
    float progress = static_cast<float>(step - warmup_steps_)
                   / static_cast<float>(total_steps_ - warmup_steps_);
    return lr_min_ + 0.5f * (lr_max_ - lr_min_)
                   * (1.0f + std::cos(3.14159265358979f * progress));
}

CosineScheduler::Phase CosineScheduler::get_phase(size_t step) const {
    if (step < warmup_steps_) return Phase::WARMUP;
    if (step < total_steps_) return Phase::COSINE;
    return Phase::DONE;
}

} // namespace neuroflow
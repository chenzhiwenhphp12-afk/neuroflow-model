#ifndef NEUROFLOW_SCHEDULER_HPP
#define NEUROFLOW_SCHEDULER_HPP

#include <cstddef>

namespace neuroflow {

class CosineScheduler {
public:
    enum class Phase { WARMUP, COSINE, DONE };

    float lr_max_;
    float lr_min_;
    size_t warmup_steps_;
    size_t total_steps_;

    CosineScheduler(float lr_max, size_t total_steps,
                    float lr_min_ratio = 0.1f, float warmup_ratio = 0.01f);

    float get_lr(size_t step) const;
    Phase get_phase(size_t step) const;
};

} // namespace neuroflow

#endif // NEUROFLOW_SCHEDULER_HPP
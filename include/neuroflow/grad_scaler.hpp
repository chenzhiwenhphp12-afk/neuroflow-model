#ifndef NEUROFLOW_GRAD_SCALER_HPP
#define NEUROFLOW_GRAD_SCALER_HPP

#include <cstddef>
#include <vector>
#include "tensor.hpp"

namespace neuroflow {

class GradScaler {
public:
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    size_t growth_interval_;
    size_t growth_tracker_;

    GradScaler(float init_scale = 65536.0f,
               float growth_factor = 2.0f,
               float backoff_factor = 0.5f,
               size_t growth_interval = 2000);

    float get_scale() const { return scale_; }

    bool has_inf_or_nan(const std::vector<Tensor*>& grads) const;

    void unscale(std::vector<Tensor*>& grads);

    void scale_loss(Tensor& loss);

    void update(bool found_inf);
};

} // namespace neuroflow

#endif // NEUROFLOW_GRAD_SCALER_HPP
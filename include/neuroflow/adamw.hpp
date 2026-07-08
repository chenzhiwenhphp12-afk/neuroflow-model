#ifndef NEUROFLOW_ADAMW_HPP
#define NEUROFLOW_ADAMW_HPP

#include <cstddef>
#include <vector>
#include "tensor.hpp"

namespace neuroflow {

struct ParamState {
    Tensor m;
    Tensor v;
};

struct ParamGroup {
    std::vector<Tensor*> params;
    std::vector<Tensor*> grads;
    float lr;
    float weight_decay;
};

class AdamW {
public:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    size_t step_;

    std::vector<ParamGroup> param_groups_;
    std::vector<std::vector<ParamState>> states_;

    AdamW(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.01f);

    void add_param_group(const ParamGroup& group);
    void step();
    void set_lr(float lr);
    float get_lr() const { return lr_; }
    size_t get_step() const { return step_; }
};

} // namespace neuroflow

#endif // NEUROFLOW_ADAMW_HPP
#ifndef NEUROFLOW_SWIGLU_HPP
#define NEUROFLOW_SWIGLU_HPP

#include <cstddef>
#include "tensor.hpp"
#include "model.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace neuroflow {

class SwiGLUFFN {
public:
    size_t d_model_;
    size_t d_ff_;
    std::shared_ptr<Linear> w_gate_;
    std::shared_ptr<Linear> w_up_;
    std::shared_ptr<Linear> w_down_;

    struct Cache {
        Tensor input;
        Tensor gate_out;
        Tensor up_out;
        Tensor gate_activated;
        Tensor multiplied;
    };
    Cache cache_;
    bool training_mode_ = false;

    SwiGLUFFN(size_t d_model, size_t d_ff = 0);
    Tensor forward(const Tensor& x);

    struct Gradients {
        Tensor w_gate_weight_grad;
        Tensor w_gate_bias_grad;
        Tensor w_up_weight_grad;
        Tensor w_up_bias_grad;
        Tensor w_down_weight_grad;
        Tensor w_down_bias_grad;
        Tensor input_grad;
    };

    Gradients backward(const Tensor& output_grad);
};

#ifdef USE_CUDA
void launch_silu(float* data, size_t n, cudaStream_t stream);
void launch_elementwise_mul(float* out, const float* a, const float* b, size_t n, cudaStream_t stream);
#endif

} // namespace neuroflow

#endif // NEUROFLOW_SWIGLU_HPP
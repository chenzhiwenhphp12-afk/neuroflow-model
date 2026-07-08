#ifndef NEUROFLOW_RMS_NORM_HPP
#define NEUROFLOW_RMS_NORM_HPP

#include <cstddef>
#include "tensor.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace neuroflow {

class RMSNorm {
public:
    size_t dim_;
    float eps_;
    Tensor weight_;

    struct Cache {
        Tensor input;
        Tensor rms;
        Tensor normalized;
    };
    Cache cache_;

    RMSNorm(size_t dim, float eps = 1e-5f);
    Tensor forward(const Tensor& x);

    struct Gradients {
        Tensor weight_grad;
        Tensor input_grad;
    };

    Gradients backward(const Tensor& output_grad);
};

#ifdef USE_CUDA
void launch_rms_norm_forward(float* out, const float* input, const float* weight,
                              float* rms, float* normalized, size_t batch, size_t dim,
                              float eps, cudaStream_t stream);
void launch_rms_norm_backward(float* input_grad, float* weight_grad,
                               const float* output_grad, const float* input,
                               const float* rms, const float* normalized,
                               const float* weight, size_t batch, size_t dim,
                               cudaStream_t stream);
#endif

} // namespace neuroflow

#endif // NEUROFLOW_RMS_NORM_HPP
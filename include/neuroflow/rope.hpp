#ifndef NEUROFLOW_ROPE_HPP
#define NEUROFLOW_ROPE_HPP

#include <cstddef>
#include <vector>
#include "tensor.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace neuroflow {

class RoPE {
public:
    size_t head_dim_;
    size_t max_seq_len_;
    float yarn_scale_factor_ = 1.0f;
    Tensor freqs_cos_;
    Tensor freqs_sin_;

    RoPE(size_t head_dim, size_t max_seq_len);

    void apply(Tensor& qkv, size_t seq_len, size_t n_heads, size_t d_model, size_t offset = 0) const;
    void apply_single(Tensor& x, size_t seq_len, size_t n_heads, size_t offset = 0) const;
    void set_yarn_scale(float scale_factor);
};

#ifdef USE_CUDA
void launch_rope(float* qkv, const float* cos_table, const float* sin_table,
                 size_t seq_len, size_t n_heads, size_t d_model, size_t head_dim,
                 size_t offset, cudaStream_t stream);
void launch_rope_single(float* data, const float* cos_table, const float* sin_table,
                         size_t seq_len, size_t n_heads, size_t stride, size_t head_dim,
                         size_t offset, cudaStream_t stream);
#endif

} // namespace neuroflow

#endif // NEUROFLOW_ROPE_HPP
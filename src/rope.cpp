#include "neuroflow/rope.hpp"

#include <cmath>
#include <iostream>

#ifdef USE_CUDA
#include "cuda_context.hpp"
#endif

namespace neuroflow {

RoPE::RoPE(size_t head_dim, size_t max_seq_len)
    : head_dim_(head_dim), max_seq_len_(max_seq_len),
      freqs_cos_({max_seq_len, head_dim / 2}, QuantType::FP32),
      freqs_sin_({max_seq_len, head_dim / 2}, QuantType::FP32) {
    float* cos_p = freqs_cos_.as_fp32();
    float* sin_p = freqs_sin_.as_fp32();
    float base = 10000.0f;
    size_t half_dim = head_dim / 2;

    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        for (size_t d = 0; d < half_dim; ++d) {
            float freq = 1.0f / std::pow(base, static_cast<float>(2 * d) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;
            cos_p[pos * half_dim + d] = std::cos(angle);
            sin_p[pos * half_dim + d] = std::sin(angle);
        }
    }

#ifdef USE_CUDA
    if (CudaContext::instance().is_available()) {
        freqs_cos_.to_gpu();
        freqs_sin_.to_gpu();
    }
#endif
}

void RoPE::apply(Tensor& qkv, size_t seq_len, size_t n_heads, size_t d_model, size_t offset) const {
    size_t half_dim = head_dim_ / 2;

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && qkv.is_on_gpu()) {
        launch_rope(qkv.as_gpu_fp32(), freqs_cos_.as_gpu_fp32(), freqs_sin_.as_gpu_fp32(),
                    seq_len, n_heads, d_model, head_dim_, offset,
                    CudaContext::instance().stream());
        qkv.gpu_dirty_ = true;
        return;
    }
#endif

    float* qkvp = qkv.as_fp32();
    const float* cos_p = freqs_cos_.as_fp32();
    const float* sin_p = freqs_sin_.as_fp32();

    for (size_t h = 0; h < n_heads; ++h) {
        size_t q_off = h * head_dim_;
        size_t k_off = d_model + h * head_dim_;

        for (size_t i = 0; i < seq_len; ++i) {
            size_t pos = offset + i;
            if (pos >= max_seq_len_) {
                if (yarn_scale_factor_ > 1.0f) {
                    pos = pos % max_seq_len_;
                } else {
                    pos = max_seq_len_ - 1;
                }
            }

            for (size_t d = 0; d < half_dim; ++d) {
                float c = cos_p[pos * half_dim + d];
                float s = sin_p[pos * half_dim + d];

                size_t qi = i * 3 * d_model + q_off + d;
                size_t qi2 = i * 3 * d_model + q_off + half_dim + d;
                float q0 = qkvp[qi];
                float q1 = qkvp[qi2];
                qkvp[qi] = q0 * c - q1 * s;
                qkvp[qi2] = q0 * s + q1 * c;

                size_t ki = i * 3 * d_model + k_off + d;
                size_t ki2 = i * 3 * d_model + k_off + half_dim + d;
                float k0 = qkvp[ki];
                float k1 = qkvp[ki2];
                qkvp[ki] = k0 * c - k1 * s;
                qkvp[ki2] = k0 * s + k1 * c;
            }
        }
    }
}

void RoPE::apply_single(Tensor& x, size_t seq_len, size_t n_heads, size_t offset) const {
    size_t half_dim = head_dim_ / 2;
    size_t stride = n_heads * head_dim_;

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && x.is_on_gpu()) {
        launch_rope_single(x.as_gpu_fp32(), freqs_cos_.as_gpu_fp32(), freqs_sin_.as_gpu_fp32(),
                           seq_len, n_heads, stride, head_dim_, offset,
                           CudaContext::instance().stream());
        x.gpu_dirty_ = true;
        return;
    }
#endif

    float* xp = x.as_fp32();
    const float* cos_p = freqs_cos_.as_fp32();
    const float* sin_p = freqs_sin_.as_fp32();

    for (size_t h = 0; h < n_heads; ++h) {
        size_t h_off = h * head_dim_;
        for (size_t i = 0; i < seq_len; ++i) {
            size_t pos = offset + i;
            if (pos >= max_seq_len_) {
                pos = yarn_scale_factor_ > 1.0f ? pos % max_seq_len_ : max_seq_len_ - 1;
            }
            for (size_t d = 0; d < half_dim; ++d) {
                float c = cos_p[pos * half_dim + d];
                float s = sin_p[pos * half_dim + d];
                size_t idx0 = i * stride + h_off + d;
                size_t idx1 = i * stride + h_off + half_dim + d;
                float x0 = xp[idx0];
                float x1 = xp[idx1];
                xp[idx0] = x0 * c - x1 * s;
                xp[idx1] = x0 * s + x1 * c;
            }
        }
    }
}

void RoPE::set_yarn_scale(float scale_factor) {
    if (scale_factor <= 1.0f) return;
    if (scale_factor > 100.0f) {
        std::cerr << "[YaRN WARNING] Extreme scale_factor=" << scale_factor
                  << ", results may be unreliable" << std::endl;
    }
    yarn_scale_factor_ = scale_factor;

    float* cos_p = freqs_cos_.as_fp32();
    float* sin_p = freqs_sin_.as_fp32();
    float base = 10000.0f;
    size_t half_dim = head_dim_ / 2;

    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        for (size_t d = 0; d < half_dim; ++d) {
            float freq = 1.0f / std::pow(base, static_cast<float>(2 * d) / static_cast<float>(head_dim_));
            float scaled_freq = freq / scale_factor;
            float angle = static_cast<float>(pos) * scaled_freq;
            cos_p[pos * half_dim + d] = std::cos(angle);
            sin_p[pos * half_dim + d] = std::sin(angle);
        }
    }

#ifdef USE_CUDA
    if (CudaContext::instance().is_available()) {
        freqs_cos_.to_gpu();
        freqs_sin_.to_gpu();
    }
#endif

    std::cerr << "[YaRN] Applied scale_factor=" << scale_factor
              << ", temp_scale=" << (std::sqrt(std::log(scale_factor)) + 1.0f) << std::endl;
}

#ifdef USE_CUDA
__global__ void kernel_rope_impl(float* qkv, const float* cos_table, const float* sin_table,
                                  size_t seq_len, size_t n_heads, size_t d_model,
                                  size_t head_dim, size_t half_dim, size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * n_heads * half_dim;
    if (idx >= total) return;

    size_t d = idx % half_dim;
    size_t h = (idx / half_dim) % n_heads;
    size_t i = idx / (n_heads * half_dim);

    size_t pos = offset + i;

    float c = cos_table[pos * half_dim + d];
    float s = sin_table[pos * half_dim + d];

    size_t q_off = h * head_dim;
    size_t k_off = d_model + h * head_dim;

    size_t qi = i * 3 * d_model + q_off + d;
    size_t qi2 = i * 3 * d_model + q_off + half_dim + d;
    float q0 = qkv[qi];
    float q1 = qkv[qi2];
    qkv[qi] = q0 * c - q1 * s;
    qkv[qi2] = q0 * s + q1 * c;

    size_t ki = i * 3 * d_model + k_off + d;
    size_t ki2 = i * 3 * d_model + k_off + half_dim + d;
    float k0 = qkv[ki];
    float k1 = qkv[ki2];
    qkv[ki] = k0 * c - k1 * s;
    qkv[ki2] = k0 * s + k1 * c;
}

void launch_rope(float* qkv, const float* cos_table, const float* sin_table,
                 size_t seq_len, size_t n_heads, size_t d_model, size_t head_dim,
                 size_t offset, cudaStream_t stream) {
    size_t half_dim = head_dim / 2;
    size_t total = seq_len * n_heads * half_dim;
    int block = 256;
    int grid = (static_cast<int>(total) + block - 1) / block;
    kernel_rope_impl<<<grid, block, 0, stream>>>(qkv, cos_table, sin_table,
        seq_len, n_heads, d_model, head_dim, half_dim, offset);
}
__global__ void kernel_rope_single_impl(float* data, const float* cos_table, const float* sin_table,
                                          size_t seq_len, size_t n_heads, size_t stride,
                                          size_t head_dim, size_t half_dim, size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * n_heads * half_dim;
    if (idx >= total) return;

    size_t d = idx % half_dim;
    size_t h = (idx / half_dim) % n_heads;
    size_t i = idx / (n_heads * half_dim);

    size_t pos = offset + i;
    float c = cos_table[pos * half_dim + d];
    float s = sin_table[pos * half_dim + d];

    size_t h_off = h * head_dim;
    size_t idx0 = i * stride + h_off + d;
    size_t idx1 = i * stride + h_off + half_dim + d;
    float x0 = data[idx0];
    float x1 = data[idx1];
    data[idx0] = x0 * c - x1 * s;
    data[idx1] = x0 * s + x1 * c;
}

void launch_rope_single(float* data, const float* cos_table, const float* sin_table,
                         size_t seq_len, size_t n_heads, size_t stride, size_t head_dim,
                         size_t offset, cudaStream_t stream) {
    size_t half_dim = head_dim / 2;
    size_t total = seq_len * n_heads * half_dim;
    int block = 256;
    int grid = (static_cast<int>(total) + block - 1) / block;
    kernel_rope_single_impl<<<grid, block, 0, stream>>>(data, cos_table, sin_table,
        seq_len, n_heads, stride, head_dim, half_dim, offset);
}
#endif

} // namespace neuroflow
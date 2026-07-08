#include "neuroflow/rms_norm.hpp"

#include <cmath>
#include <cstring>

#ifdef USE_CUDA
#include "cuda_context.hpp"
#endif

namespace neuroflow {

RMSNorm::RMSNorm(size_t dim, float eps)
    : dim_(dim), eps_(eps), weight_({dim}, QuantType::FP32) {
    float* wp = weight_.as_fp32();
    for (size_t i = 0; i < dim; ++i) wp[i] = 1.0f;
}

Tensor RMSNorm::forward(const Tensor& x) {
    cache_.input = x.clone();
    size_t batch = x.numel() / dim_;

    cache_.rms = Tensor({batch}, QuantType::FP32);
    cache_.normalized = Tensor(x.shape_, QuantType::FP32);
    Tensor output(x.shape_, QuantType::FP32);

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && x.is_on_gpu()) {
        output.to_gpu();
        cache_.rms.to_gpu();
        cache_.normalized.to_gpu();
        launch_rms_norm_forward(output.as_gpu_fp32(), x.as_gpu_fp32(), weight_.as_gpu_fp32(),
                                 cache_.rms.as_gpu_fp32(), cache_.normalized.as_gpu_fp32(),
                                 batch, dim_, eps_, CudaContext::instance().stream());
        output.gpu_dirty_ = true;
        cache_.rms.gpu_dirty_ = true;
        cache_.normalized.gpu_dirty_ = true;
        return output;
    }
#endif

    const float* xp = x.as_fp32();
    const float* wp = weight_.as_fp32();
    float* rms_p = cache_.rms.as_fp32();
    float* norm_p = cache_.normalized.as_fp32();
    float* op = output.as_fp32();

    for (size_t b = 0; b < batch; ++b) {
        float ss = 0.0f;
        for (size_t d = 0; d < dim_; ++d) {
            float v = xp[b * dim_ + d];
            ss += v * v;
        }
        float rms = std::sqrt(ss / static_cast<float>(dim_) + eps_);
        rms_p[b] = rms;

        for (size_t d = 0; d < dim_; ++d) {
            float n = xp[b * dim_ + d] / rms;
            norm_p[b * dim_ + d] = n;
            op[b * dim_ + d] = n * wp[d];
        }
    }
    return output;
}

RMSNorm::Gradients RMSNorm::backward(const Tensor& output_grad) {
    Gradients grads;
    size_t batch = cache_.input.numel() / dim_;

    grads.weight_grad = Tensor({dim_}, QuantType::FP32);
    grads.input_grad = Tensor(cache_.input.shape_, QuantType::FP32);

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && output_grad.is_on_gpu()) {
        grads.weight_grad.to_gpu();
        grads.input_grad.to_gpu();
        launch_rms_norm_backward(grads.input_grad.as_gpu_fp32(), grads.weight_grad.as_gpu_fp32(),
                                  output_grad.as_gpu_fp32(), cache_.input.as_gpu_fp32(),
                                  cache_.rms.as_gpu_fp32(), cache_.normalized.as_gpu_fp32(),
                                  weight_.as_gpu_fp32(), batch, dim_,
                                  CudaContext::instance().stream());
        grads.weight_grad.gpu_dirty_ = true;
        grads.input_grad.gpu_dirty_ = true;
        return grads;
    }
#endif

    const float* og = output_grad.as_fp32();
    const float* xp = cache_.input.as_fp32();
    const float* rms_p = cache_.rms.as_fp32();
    const float* norm_p = cache_.normalized.as_fp32();
    const float* wp = weight_.as_fp32();
    float* wg = grads.weight_grad.as_fp32();
    float* ig = grads.input_grad.as_fp32();

    memset(wg, 0, dim_ * sizeof(float));

    for (size_t b = 0; b < batch; ++b) {
        float rms = rms_p[b];
        float inv_rms = 1.0f / rms;

        for (size_t d = 0; d < dim_; ++d) {
            wg[d] += og[b * dim_ + d] * norm_p[b * dim_ + d];
        }

        float dot_og_n = 0.0f;
        for (size_t d = 0; d < dim_; ++d) {
            dot_og_n += og[b * dim_ + d] * wp[d] * xp[b * dim_ + d];
        }

        for (size_t d = 0; d < dim_; ++d) {
            float d_norm = og[b * dim_ + d] * wp[d];
            float d_rms = -dot_og_n / (rms * static_cast<float>(dim_));
            float d_ss = d_rms * 0.5f / rms;
            ig[b * dim_ + d] = d_norm * inv_rms + 2.0f * xp[b * dim_ + d] * d_ss;
        }
    }

    return grads;
}

#ifdef USE_CUDA
__global__ void kernel_rms_norm_forward_impl(float* out, const float* input, const float* weight,
                                              float* rms, float* normalized,
                                              size_t batch, size_t dim, float eps) {
    size_t b = blockIdx.x;
    if (b >= batch) return;

    float ss = 0.0f;
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = input[b * dim + d];
        ss += v * v;
    }
    __shared__ float s_ss;
    if (threadIdx.x == 0) s_ss = 0.0f;
    __syncthreads();
    atomicAdd(&s_ss, ss);
    __syncthreads();

    float rms_val = sqrtf(s_ss / static_cast<float>(dim) + eps);
    if (threadIdx.x == 0) rms[b] = rms_val;

    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float n = input[b * dim + d] / rms_val;
        normalized[b * dim + d] = n;
        out[b * dim + d] = n * weight[d];
    }
}

void launch_rms_norm_forward(float* out, const float* input, const float* weight,
                              float* rms, float* normalized, size_t batch, size_t dim,
                              float eps, cudaStream_t stream) {
    int block = min(static_cast<int>(dim), 512);
    kernel_rms_norm_forward_impl<<<static_cast<int>(batch), block, 0, stream>>>(
        out, input, weight, rms, normalized, batch, dim, eps);
}

__global__ void kernel_rms_norm_backward_impl(float* input_grad, float* weight_grad,
                                               const float* output_grad, const float* input,
                                               const float* rms, const float* normalized,
                                               const float* weight,
                                               size_t batch, size_t dim) {
    size_t b = blockIdx.x;
    if (b >= batch) return;

    float rms_val = rms[b];
    float inv_rms = 1.0f / rms_val;

    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        atomicAdd(&weight_grad[d], output_grad[b * dim + d] * normalized[b * dim + d]);
    }

    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = 0.0f;
    __syncthreads();

    float local_dot = 0.0f;
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        local_dot += output_grad[b * dim + d] * weight[d] * input[b * dim + d];
    }
    atomicAdd(&s_dot, local_dot);
    __syncthreads();

    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float d_norm = output_grad[b * dim + d] * weight[d];
        float d_rms = -s_dot / (rms_val * static_cast<float>(dim));
        float d_ss = d_rms * 0.5f / rms_val;
        input_grad[b * dim + d] = d_norm * inv_rms + 2.0f * input[b * dim + d] * d_ss;
    }
}

void launch_rms_norm_backward(float* input_grad, float* weight_grad,
                               const float* output_grad, const float* input,
                               const float* rms, const float* normalized,
                               const float* weight, size_t batch, size_t dim,
                               cudaStream_t stream) {
    int block = min(static_cast<int>(dim), 512);
    kernel_rms_norm_backward_impl<<<static_cast<int>(batch), block, 0, stream>>>(
        input_grad, weight_grad, output_grad, input, rms, normalized, weight, batch, dim);
}
#endif

} // namespace neuroflow
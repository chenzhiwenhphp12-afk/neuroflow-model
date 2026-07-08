#include "neuroflow/swiglu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

#ifdef USE_CUDA
#include "cuda_context.hpp"
#endif

namespace neuroflow {

static Tensor linear_backward_weight(const Tensor& input, const Tensor& output_grad) {
    size_t in_dim = input.shape_.back();
    size_t out_dim = output_grad.shape_.back();
    size_t batch = input.numel() / in_dim;

    Tensor grad({out_dim, in_dim}, QuantType::FP32);
    float* gp = grad.as_fp32();
    memset(gp, 0, grad.data_size_);

    const float* ip = input.as_fp32();
    const float* op = output_grad.as_fp32();

#ifdef USE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<int>(out_dim), static_cast<int>(in_dim), static_cast<int>(batch),
        1.0f, op, static_cast<int>(out_dim), ip, static_cast<int>(in_dim),
        0.0f, gp, static_cast<int>(in_dim));
#else
    for (size_t b = 0; b < batch; ++b) {
        for (size_t o = 0; o < out_dim; ++o) {
            for (size_t i = 0; i < in_dim; ++i) {
                gp[o * in_dim + i] += op[b * out_dim + o] * ip[b * in_dim + i];
            }
        }
    }
#endif
    return grad;
}

static Tensor linear_backward_input(const Tensor& output_grad, const Tensor& weight) {
    size_t out_dim = output_grad.shape_.back();
    size_t batch = output_grad.numel() / out_dim;
    size_t in_dim = weight.shape_[1];

    Tensor grad({batch, in_dim}, QuantType::FP32);
    float* gp = grad.as_fp32();
    const float* op = output_grad.as_fp32();
    const float* wp = weight.as_fp32();

#ifdef USE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        static_cast<int>(batch), static_cast<int>(in_dim), static_cast<int>(out_dim),
        1.0f, op, static_cast<int>(out_dim), wp, static_cast<int>(in_dim),
        0.0f, gp, static_cast<int>(in_dim));
#else
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < in_dim; ++i) {
            float val = 0.0f;
            for (size_t o = 0; o < out_dim; ++o) {
                val += op[b * out_dim + o] * wp[o * in_dim + i];
            }
            gp[b * in_dim + i] = val;
        }
    }
#endif
    return grad;
}

static Tensor bias_backward(const Tensor& output_grad) {
    size_t out_dim = output_grad.shape_.back();
    size_t batch = output_grad.numel() / out_dim;

    Tensor grad({out_dim}, QuantType::FP32);
    float* gp = grad.as_fp32();
    const float* op = output_grad.as_fp32();

    memset(gp, 0, grad.data_size_);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t o = 0; o < out_dim; ++o) {
            gp[o] += op[b * out_dim + o];
        }
    }
    return grad;
}

SwiGLUFFN::SwiGLUFFN(size_t d_model, size_t d_ff)
    : d_model_(d_model), d_ff_(d_ff > 0 ? d_ff : d_model * 4) {
    w_gate_ = std::make_shared<Linear>(d_model_, d_ff_, true);
    w_up_ = std::make_shared<Linear>(d_model_, d_ff_, true);
    w_down_ = std::make_shared<Linear>(d_ff_, d_model_, true);
}

Tensor SwiGLUFFN::forward(const Tensor& x) {
    cache_.input = x.clone();

    cache_.gate_out = w_gate_->forward(x);
    cache_.up_out = w_up_->forward(x);

    size_t n = cache_.gate_out.numel();
    cache_.gate_activated = cache_.gate_out.clone();

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && cache_.gate_activated.is_on_gpu()) {
        launch_silu(cache_.gate_activated.as_gpu_fp32(), n, CudaContext::instance().stream());
        cache_.gate_activated.gpu_dirty_ = true;
    } else {
#endif
    float* ga = cache_.gate_activated.as_fp32();
    for (size_t i = 0; i < n; ++i) {
        float v = ga[i];
        ga[i] = v / (1.0f + expf(-v));
    }
#ifdef USE_CUDA
    }
#endif

    cache_.multiplied = Tensor(cache_.gate_activated.shape_, QuantType::FP32);

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && cache_.gate_activated.is_on_gpu()) {
        cache_.multiplied.to_gpu();
        launch_elementwise_mul(cache_.multiplied.as_gpu_fp32(),
                    cache_.gate_activated.as_gpu_fp32(),
                    cache_.up_out.as_gpu_fp32(), n,
                    CudaContext::instance().stream());
        cache_.multiplied.gpu_dirty_ = true;
    } else {
#endif
    const float* ga_p = cache_.gate_activated.as_fp32();
    const float* up_p = cache_.up_out.as_fp32();
    float* mp = cache_.multiplied.as_fp32();
    for (size_t i = 0; i < n; ++i) {
        mp[i] = ga_p[i] * up_p[i];
    }
#ifdef USE_CUDA
    }
#endif

    return w_down_->forward(cache_.multiplied);
}

SwiGLUFFN::Gradients SwiGLUFFN::backward(const Tensor& output_grad) {
    Gradients grads;

    grads.w_down_weight_grad = linear_backward_weight(cache_.multiplied, output_grad);
    grads.w_down_bias_grad = bias_backward(output_grad);
    Tensor d_multiplied = linear_backward_input(output_grad, w_down_->weight);

    Tensor d_gate_activated(d_multiplied.shape_, QuantType::FP32);
    Tensor d_up(d_multiplied.shape_, QuantType::FP32);
    const float* dm = d_multiplied.as_fp32();
    const float* ga_p = cache_.gate_activated.as_fp32();
    const float* up_p = cache_.up_out.as_fp32();
    float* dga = d_gate_activated.as_fp32();
    float* dup = d_up.as_fp32();
    size_t n = d_multiplied.numel();
    for (size_t i = 0; i < n; ++i) {
        dga[i] = dm[i] * up_p[i];
        dup[i] = dm[i] * ga_p[i];
    }

    const float* go_p = cache_.gate_out.as_fp32();
    Tensor d_gate_out(d_gate_activated.shape_, QuantType::FP32);
    float* dgo = d_gate_out.as_fp32();
    for (size_t i = 0; i < n; ++i) {
        float sig = ga_p[i];
        dgo[i] = dga[i] * sig * (1.0f + go_p[i] * (1.0f - sig));
    }

    grads.w_gate_weight_grad = linear_backward_weight(cache_.input, d_gate_out);
    grads.w_gate_bias_grad = bias_backward(d_gate_out);
    Tensor d_input_gate = linear_backward_input(d_gate_out, w_gate_->weight);

    grads.w_up_weight_grad = linear_backward_weight(cache_.input, d_up);
    grads.w_up_bias_grad = bias_backward(d_up);
    Tensor d_input_up = linear_backward_input(d_up, w_up_->weight);

    grads.input_grad = Tensor(cache_.input.shape_, QuantType::FP32);
    const float* dig = d_input_gate.as_fp32();
    const float* diu = d_input_up.as_fp32();
    float* ig = grads.input_grad.as_fp32();
    for (size_t i = 0; i < grads.input_grad.numel(); ++i) {
        ig[i] = dig[i] + diu[i];
    }

    return grads;
}

#ifdef USE_CUDA
__global__ void kernel_silu_impl(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = data[idx];
    data[idx] = v / (1.0f + expf(-v));
}

void launch_silu(float* data, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_silu_impl<<<grid, block, 0, stream>>>(data, n);
}

__global__ void kernel_elementwise_mul_impl(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

void launch_elementwise_mul(float* out, const float* a, const float* b, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_elementwise_mul_impl<<<grid, block, 0, stream>>>(out, a, b, n);
}
#endif

} // namespace neuroflow

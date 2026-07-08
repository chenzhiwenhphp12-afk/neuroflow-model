#ifndef NEUROFLOW_CUDA_KERNELS_HPP
#define NEUROFLOW_CUDA_KERNELS_HPP

#ifdef USE_CUDA

#include <cstddef>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_context.hpp"

namespace neuroflow {

__device__ inline void safe_atomic_max_float(float* addr, float val) {
    unsigned int old = atomicCAS(reinterpret_cast<unsigned int*>(addr),
                                  __float_as_uint(val),
                                  __float_as_uint(val));
    while (val > __uint_as_float(old)) {
        unsigned int assumed = old;
        old = atomicCAS(reinterpret_cast<unsigned int*>(addr),
                         assumed,
                         __float_as_uint(val));
        if (old == assumed) break;
    }
}

// ═══════════════════════════════════════════════════════
// CUDA Kernel 实现
// ═══════════════════════════════════════════════════════

// --- GELU ---
__global__ void kernel_gelu_impl(float* data, size_t n) {
    const float SQRT_2_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = data[idx];
    float inner = SQRT_2_PI * (v + COEFF * v * v * v);
    data[idx] = 0.5f * v * (1.0f + tanhf(inner));
}

inline void launch_gelu(float* d_data, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_gelu_impl<<<grid, block, 0, stream>>>(d_data, n);
}

// --- Sigmoid ---
__global__ void kernel_sigmoid_impl(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

inline void launch_sigmoid(float* d_data, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_sigmoid_impl<<<grid, block, 0, stream>>>(d_data, n);
}

// --- Element-wise Add (residual) ---
__global__ void kernel_add_impl(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

inline void launch_add(float* d_out, const float* d_a, const float* d_b, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_add_impl<<<grid, block, 0, stream>>>(d_out, d_a, d_b, n);
}

// --- Scale + Add ---
__global__ void kernel_scale_add_impl(float* out, const float* a, float scale, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * scale + b[idx];
}

inline void launch_scale_add(float* d_out, const float* d_a, float scale, const float* d_b, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_scale_add_impl<<<grid, block, 0, stream>>>(d_out, d_a, scale, d_b, n);
}

// --- Bias Add ---
__global__ void kernel_bias_add_impl(float* out, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int j = idx % cols;
    out[idx] += bias[j];
}

inline void launch_bias_add(float* d_out, const float* d_bias, int rows, int cols, cudaStream_t stream) {
    int block = 256;
    int total = rows * cols;
    int grid = (total + block - 1) / block;
    kernel_bias_add_impl<<<grid, block, 0, stream>>>(d_out, d_bias, rows, cols);
}

// --- Softmax (online, per-row) ---
__global__ void kernel_softmax_impl(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* row_data = data + row * cols;

    float max_val = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        max_val = fmaxf(max_val, row_data[j]);
    }
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -1e30f;
    __syncthreads();
    safe_atomic_max_float(&s_max, max_val);
    __syncthreads();
    max_val = s_max;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        row_data[j] = expf(row_data[j] - max_val);
        sum += row_data[j];
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        row_data[j] /= s_sum;
    }
}

inline void launch_softmax(float* d_data, int rows, int cols, cudaStream_t stream) {
    kernel_softmax_impl<<<rows, 256, 0, stream>>>(d_data, rows, cols);
}

// --- Causal Softmax (per-row, zero out j > i) ---
__global__ void kernel_causal_softmax_impl(float* data, int seq_len) {
    int i = blockIdx.x;
    if (i >= seq_len) return;
    float* row = data + i * seq_len;

    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        // keep
    }
    // Zero future positions
    for (int j = i + 1 + threadIdx.x; j < seq_len; j += blockDim.x) {
        row[j] = 0.0f;
    }
    __syncthreads();

    float max_val = -1e30f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        max_val = fmaxf(max_val, row[j]);
    }
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -1e30f;
    __syncthreads();
    safe_atomic_max_float(&s_max, max_val);
    __syncthreads();
    max_val = s_max;

    float sum = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        row[j] = expf(row[j] - max_val);
        sum += row[j];
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();

    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        row[j] /= s_sum;
    }
}

inline void launch_causal_softmax(float* d_data, int seq_len, cudaStream_t stream) {
    kernel_causal_softmax_impl<<<seq_len, 256, 0, stream>>>(d_data, seq_len);
}

// --- LayerNorm Forward ---
__global__ void kernel_layer_norm_impl(float* out, const float* inp, const float* w, const float* b, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* row_in = inp + row * cols;
    float* row_out = out + row * cols;

    float mean = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        mean += row_in[j];
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = 0.0f;
    __syncthreads();
    atomicAdd(&s_mean, mean);
    __syncthreads();
    mean = s_mean / cols;

    float var = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float diff = row_in[j] - mean;
        var += diff * diff;
    }
    __shared__ float s_var;
    if (threadIdx.x == 0) s_var = 0.0f;
    __syncthreads();
    atomicAdd(&s_var, var);
    __syncthreads();
    var = s_var / cols;

    float inv_std = 1.0f / sqrtf(var + eps);
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float norm = (row_in[j] - mean) * inv_std;
        row_out[j] = w[j] * norm + b[j];
    }
}

inline void launch_layer_norm(float* d_out, const float* d_inp, const float* d_w, const float* d_b, int rows, int cols, float eps, cudaStream_t stream) {
    kernel_layer_norm_impl<<<rows, 256, 0, stream>>>(d_out, d_inp, d_w, d_b, rows, cols, eps);
}

// --- Embed Lookup ---
__global__ void kernel_embed_lookup_impl(float* out, const float* embed, const int* token_ids, int seq_len, int d_model, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;
    int i = idx / d_model;
    int d = idx % d_model;
    int tid = token_ids[i];
    out[idx] = embed[tid * d_model + d] * scale;
}

inline void launch_embed_lookup(float* d_out, const float* d_embed, const int* d_token_ids, int seq_len, int d_model, float scale, cudaStream_t stream) {
    int total = seq_len * d_model;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_embed_lookup_impl<<<grid, block, 0, stream>>>(d_out, d_embed, d_token_ids, seq_len, d_model, scale);
}

// --- Positional Encode (add) ---
__global__ void kernel_positional_encode_impl(float* out, const float* pos_enc, int seq_len, int d_model, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;
    int i = idx / d_model;
    int d = idx % d_model;
    int p = offset + i;
    out[idx] += pos_enc[p * d_model + d];
}

inline void launch_positional_encode(float* d_out, const float* d_pos_enc, int seq_len, int d_model, int offset, cudaStream_t stream) {
    int total = seq_len * d_model;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_positional_encode_impl<<<grid, block, 0, stream>>>(d_out, d_pos_enc, seq_len, d_model, offset);
}

// --- SGD Update ---
__global__ void kernel_sgd_update_impl(float* param, const float* grad, size_t n, float lr) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    if (isfinite(g)) {
        param[idx] -= lr * g;
    }
}

inline void launch_sgd_update(float* d_param, const float* d_grad, size_t n, float lr, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_sgd_update_impl<<<grid, block, 0, stream>>>(d_param, d_grad, n, lr);
}

// --- Sparse Embed Update ---
__global__ void kernel_sparse_embed_update_impl(float* embed, const float* grad, const int* token_ids, int seq_len, int d_model, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;
    int i = idx / d_model;
    int d = idx % d_model;
    int tid = token_ids[i];
    float g = grad[i * d_model + d];
    if (isfinite(g)) {
        atomicAdd(&embed[tid * d_model + d], -lr * g);
    }
}

inline void launch_sparse_embed_update(float* d_embed, const float* d_grad, const int* d_token_ids, int seq_len, int d_model, float lr, cudaStream_t stream) {
    int total = seq_len * d_model;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_sparse_embed_update_impl<<<grid, block, 0, stream>>>(d_embed, d_grad, d_token_ids, seq_len, d_model, lr);
}

// --- Cross Entropy Forward (returns loss on GPU) ---
__global__ void kernel_cross_entropy_impl(float* loss, const float* logits, int target_id, int vocab_size) {
    __shared__ float s_max;
    __shared__ float s_sum;

    float local_max = -1e30f;
    for (int j = threadIdx.x; j < vocab_size; j += blockDim.x) {
        local_max = fmaxf(local_max, logits[j]);
    }
    if (threadIdx.x == 0) s_max = -1e30f;
    __syncthreads();
    safe_atomic_max_float(&s_max, local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < vocab_size; j += blockDim.x) {
        local_sum += expf(logits[j] - s_max);
    }
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        float log_sum_exp = s_max + logf(s_sum);
        *loss = -(logits[target_id] - log_sum_exp);
    }
}

inline void launch_cross_entropy(float* d_loss, const float* d_logits, int target_id, int vocab_size, cudaStream_t stream) {
    kernel_cross_entropy_impl<<<1, 256, 0, stream>>>(d_loss, d_logits, target_id, vocab_size);
}

// --- Cross Entropy Backward (softmax - one_hot) ---
__global__ void kernel_cross_entropy_backward_impl(float* grad, const float* logits, int target_id, int vocab_size) {
    __shared__ float s_max;
    __shared__ float s_sum;

    float local_max = -1e30f;
    for (int j = threadIdx.x; j < vocab_size; j += blockDim.x) {
        local_max = fmaxf(local_max, logits[j]);
    }
    if (threadIdx.x == 0) s_max = -1e30f;
    __syncthreads();
    safe_atomic_max_float(&s_max, local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < vocab_size; j += blockDim.x) {
        float val = expf(logits[j] - s_max);
        local_sum += val;
    }
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    for (int j = threadIdx.x; j < vocab_size; j += blockDim.x) {
        float softmax_val = expf(logits[j] - s_max) / s_sum;
        grad[j] = softmax_val;
        if (j == target_id) grad[j] -= 1.0f;
    }
}

inline void launch_cross_entropy_backward(float* d_grad, const float* d_logits, int target_id, int vocab_size, cudaStream_t stream) {
    kernel_cross_entropy_backward_impl<<<1, 256, 0, stream>>>(d_grad, d_logits, target_id, vocab_size);
}

// --- Mean Pool ---
__global__ void kernel_mean_pool_impl(float* out, const float* inp, int seq_len, int d_model) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_model) return;
    float sum = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        sum += inp[i * d_model + d];
    }
    out[d] = sum / seq_len;
}

inline void launch_mean_pool(float* d_out, const float* d_inp, int seq_len, int d_model, cudaStream_t stream) {
    int block = 256;
    int grid = (d_model + block - 1) / block;
    kernel_mean_pool_impl<<<grid, block, 0, stream>>>(d_out, d_inp, seq_len, d_model);
}

// --- Last Token Pool ---
__global__ void kernel_last_token_pool_impl(float* out, const float* inp, int seq_len, int d_model) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_model) return;
    out[d] = inp[(seq_len - 1) * d_model + d];
}

inline void launch_last_token_pool(float* d_out, const float* d_inp, int seq_len, int d_model, cudaStream_t stream) {
    int block = 256;
    int grid = (d_model + block - 1) / block;
    kernel_last_token_pool_impl<<<grid, block, 0, stream>>>(d_out, d_inp, seq_len, d_model);
}

// --- Fill Zero ---
__global__ void kernel_fill_zero_impl(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = 0.0f;
}

inline void launch_fill_zero(float* d_data, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_fill_zero_impl<<<grid, block, 0, stream>>>(d_data, n);
}

// --- Causal Mask Zero (zero out j > i in [seq_len, seq_len]) ---
__global__ void kernel_causal_mask_zero_impl(float* data, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    if (idx >= total) return;
    int i = idx / seq_len;
    int j = idx % seq_len;
    if (j > i) data[idx] = 0.0f;
}

inline void launch_causal_mask_zero(float* d_data, int seq_len, cudaStream_t stream) {
    int total = seq_len * seq_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_causal_mask_zero_impl<<<grid, block, 0, stream>>>(d_data, seq_len);
}

// --- Scatter QKV gradients ---
__global__ void kernel_scatter_qkv_impl(float* d_qkv, const float* d_q, const float* d_k, const float* d_v,
                                         int seq_len, int d_model, int head_dim, int n_heads, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    if (idx >= total) return;
    int i = idx / head_dim;
    int d = idx % head_dim;
    size_t q_off = h * head_dim;
    size_t k_off = d_model + h * head_dim;
    size_t v_off = 2 * d_model + h * head_dim;
    d_qkv[i * 3 * d_model + q_off + d] += d_q[i * head_dim + d];
    d_qkv[i * 3 * d_model + k_off + d] += d_k[i * head_dim + d];
    d_qkv[i * 3 * d_model + v_off + d] += d_v[i * head_dim + d];
}

inline void launch_scatter_qkv(float* d_qkv, const float* d_q, const float* d_k, const float* d_v,
                                int seq_len, int d_model, int head_dim, int n_heads, int h, cudaStream_t stream) {
    int total = seq_len * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_scatter_qkv_impl<<<grid, block, 0, stream>>>(d_qkv, d_q, d_k, d_v, seq_len, d_model, head_dim, n_heads, h);
}

// --- Softmax Backward (for attention) ---
__global__ void kernel_softmax_backward_impl(float* d_scores, const float* attn_weights, const float* d_attn_weights,
                                              int seq_len, float inv_scale) {
    int i = blockIdx.x;
    if (i >= seq_len) return;
    const float* aw_row = attn_weights + i * seq_len;
    const float* daw_row = d_attn_weights + i * seq_len;
    float* ds_row = d_scores + i * seq_len;

    float dot = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        dot += aw_row[j] * daw_row[j];
    }
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = 0.0f;
    __syncthreads();
    atomicAdd(&s_dot, dot);
    __syncthreads();

    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        ds_row[j] = aw_row[j] * (daw_row[j] - s_dot) * inv_scale;
    }
    for (int j = i + 1 + threadIdx.x; j < seq_len; j += blockDim.x) {
        ds_row[j] = 0.0f;
    }
}

inline void launch_softmax_backward(float* d_scores, const float* d_attn_weights, const float* d_daw,
                                     int seq_len, float inv_scale, cudaStream_t stream) {
    kernel_softmax_backward_impl<<<seq_len, 256, 0, stream>>>(d_scores, d_attn_weights, d_daw, seq_len, inv_scale);
}

// --- LayerNorm Backward ---
__global__ void kernel_layer_norm_backward_impl(float* input_grad, const float* input, const float* weight,
                                                  const float* output_grad, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* inp_row = input + row * cols;
    const float* og_row = output_grad + row * cols;
    float* ig_row = input_grad + row * cols;

    float mean = 0.0f;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) mean += inp_row[d];
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = 0.0f;
    __syncthreads();
    atomicAdd(&s_mean, mean);
    __syncthreads();
    mean = s_mean / cols;

    float var = 0.0f;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) {
        float diff = inp_row[d] - mean;
        var += diff * diff;
    }
    __shared__ float s_var;
    if (threadIdx.x == 0) s_var = 0.0f;
    __syncthreads();
    atomicAdd(&s_var, var);
    __syncthreads();
    var = s_var / cols;

    float inv_std = 1.0f / sqrtf(var + eps);

    float sum_gn = 0.0f, sum_gnx = 0.0f;
    for (int d = threadIdx.x; d < cols; d += blockDim.x) {
        float norm = (inp_row[d] - mean) * inv_std;
        float gn = og_row[d] * weight[d];
        sum_gn += gn;
        sum_gnx += gn * norm;
    }
    __shared__ float s_sum_gn;
    __shared__ float s_sum_gnx;
    if (threadIdx.x == 0) { s_sum_gn = 0.0f; s_sum_gnx = 0.0f; }
    __syncthreads();
    atomicAdd(&s_sum_gn, sum_gn);
    atomicAdd(&s_sum_gnx, sum_gnx);
    __syncthreads();

    for (int d = threadIdx.x; d < cols; d += blockDim.x) {
        float norm = (inp_row[d] - mean) * inv_std;
        float gn = og_row[d] * weight[d];
        ig_row[d] = inv_std * (gn - s_sum_gn / cols - norm * s_sum_gnx / cols);
    }
}

inline void launch_layer_norm_backward(float* d_input_grad, const float* d_input, const float* d_weight,
                                         const float* d_output_grad, int rows, int cols, float eps, cudaStream_t stream) {
    kernel_layer_norm_backward_impl<<<rows, 256, 0, stream>>>(d_input_grad, d_input, d_weight, d_output_grad, rows, cols, eps);
}

// --- Pool Backward (mean) ---
__global__ void kernel_mean_pool_backward_impl(float* x_grad, const float* pooled_grad, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;
    int d = idx % d_model;
    float inv_n = 1.0f / static_cast<float>(seq_len);
    x_grad[idx] = pooled_grad[d] * inv_n;
}

inline void launch_mean_pool_backward(float* d_x_grad, const float* d_pooled_grad, int seq_len, int d_model, cudaStream_t stream) {
    int total = seq_len * d_model;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_mean_pool_backward_impl<<<grid, block, 0, stream>>>(d_x_grad, d_pooled_grad, seq_len, d_model);
}

// --- Bias Backward ---
__global__ void kernel_bias_backward_impl(float* bias_grad, const float* output_grad, int batch, int dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dim) return;
    float sum = 0.0f;
    for (int b = 0; b < batch; ++b) {
        sum += output_grad[b * dim + j];
    }
    bias_grad[j] = sum / batch;
}

inline void launch_bias_backward(float* d_bias_grad, const float* d_output_grad, int batch, int dim, cudaStream_t stream) {
    int block = 256;
    int grid = (dim + block - 1) / block;
    kernel_bias_backward_impl<<<grid, block, 0, stream>>>(d_bias_grad, d_output_grad, batch, dim);
}

// --- Sigmoid Backward (for causal gate) ---
__global__ void kernel_sigmoid_backward_impl(float* gate_grad, const float* gate_output, const float* input, const float* x_grad, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = gate_output[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    gate_grad[idx] = x_grad[idx] * sig * (1.0f - sig) * input[idx] + x_grad[idx] * sig;
}

inline void launch_sigmoid_backward(float* d_gate_grad, const float* d_gate_output, const float* d_input, const float* d_x_grad, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_sigmoid_backward_impl<<<grid, block, 0, stream>>>(d_gate_grad, d_gate_output, d_input, d_x_grad, n);
}

// --- Extract per-head Q, K, V from QKV ---
__global__ void kernel_extract_qkv_impl(float* Q_h, float* K_h, float* V_h,
                                          const float* qkv, int seq_len, int d_model, int head_dim, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    if (idx >= total) return;
    int i = idx / head_dim;
    int d = idx % head_dim;
    size_t q_off = h * head_dim;
    size_t k_off = d_model + h * head_dim;
    size_t v_off = 2 * d_model + h * head_dim;
    Q_h[i * head_dim + d] = qkv[i * 3 * d_model + q_off + d];
    K_h[i * head_dim + d] = qkv[i * 3 * d_model + k_off + d];
    V_h[i * head_dim + d] = qkv[i * 3 * d_model + v_off + d];
}

inline void launch_extract_qkv(float* d_Q_h, float* d_K_h, float* d_V_h,
                                const float* d_qkv, int seq_len, int d_model, int head_dim, int h, cudaStream_t stream) {
    int total = seq_len * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_extract_qkv_impl<<<grid, block, 0, stream>>>(d_Q_h, d_K_h, d_V_h, d_qkv, seq_len, d_model, head_dim, h);
}

__global__ void kernel_extract_head_impl(float* out, const float* src, int seq_len, int n_heads, int head_dim, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    if (idx >= total) return;
    int i = idx / head_dim;
    int d = idx % head_dim;
    out[i * head_dim + d] = src[i * n_heads * head_dim + h * head_dim + d];
}

inline void launch_extract_head(float* out, const float* src, int seq_len, int n_heads, int head_dim, int h, cudaStream_t stream) {
    int total = seq_len * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_extract_head_impl<<<grid, block, 0, stream>>>(out, src, seq_len, n_heads, head_dim, h);
}

// --- Extract per-head attn_out_grad ---
__global__ void kernel_extract_attn_out_grad_impl(float* aog_h, const float* aog, int seq_len, int d_model, int head_dim, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    if (idx >= total) return;
    int i = idx / head_dim;
    int d = idx % head_dim;
    aog_h[i * head_dim + d] = aog[i * d_model + h * head_dim + d];
}

inline void launch_extract_attn_out_grad(float* d_aog_h, const float* d_aog, int seq_len, int d_model, int head_dim, int h, cudaStream_t stream) {
    int total = seq_len * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_extract_attn_out_grad_impl<<<grid, block, 0, stream>>>(d_aog_h, d_aog, seq_len, d_model, head_dim, h);
}

// --- NTM softmax (per-batch) ---
__global__ void kernel_ntm_softmax_impl(float* data, int batch, int slots) {
    int b = blockIdx.x;
    if (b >= batch) return;
    float* row = data + b * slots;
    float max_val = -1e30f;
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        max_val = fmaxf(max_val, row[s]);
    }
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -1e30f;
    __syncthreads();
    safe_atomic_max_float(&s_max, max_val);
    __syncthreads();

    float sum = 0.0f;
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        row[s] = expf(row[s] - s_max);
        sum += row[s];
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();

    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        row[s] /= s_sum;
    }
}

inline void launch_ntm_softmax(float* d_data, int batch, int slots, cudaStream_t stream) {
    kernel_ntm_softmax_impl<<<batch, 256, 0, stream>>>(d_data, batch, slots);
}

// --- NTM read content (read_weights @ memory) ---
__global__ void kernel_ntm_read_impl(float* read_content, const float* read_weights, const float* memory, int batch, int slots, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_model;
    if (idx >= total) return;
    int b = idx / d_model;
    int d = idx % d_model;
    float val = 0.0f;
    for (int s = 0; s < slots; ++s) {
        val += read_weights[b * slots + s] * memory[s * d_model + d];
    }
    read_content[idx] = val;
}

inline void launch_ntm_read(float* d_read_content, const float* d_read_weights, const float* d_memory, int batch, int slots, int d_model, cudaStream_t stream) {
    int total = batch * d_model;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_ntm_read_impl<<<grid, block, 0, stream>>>(d_read_content, d_read_weights, d_memory, batch, slots, d_model);
}

// --- NTM write (memory update) ---
__global__ void kernel_ntm_write_impl(float* memory, const float* read_weights, const float* erase, const float* write_val, int batch, int slots, int d_model) {
    int s = blockIdx.x;
    int d = threadIdx.x;
    if (s >= slots || d >= d_model) return;
    for (int b = 0; b < batch; ++b) {
        float rw = read_weights[b * slots + s];
        float e = 1.0f / (1.0f + expf(-erase[b * d_model + d]));
        float w = tanhf(write_val[b * d_model + d]);
        memory[s * d_model + d] = memory[s * d_model + d] * (1.0f - rw * e) + rw * w;
    }
}

inline void launch_ntm_write(float* d_memory, const float* d_read_weights, const float* d_erase, const float* d_write_val, int batch, int slots, int d_model, cudaStream_t stream) {
    kernel_ntm_write_impl<<<slots, d_model, 0, stream>>>(d_memory, d_read_weights, d_erase, d_write_val, batch, slots, d_model);
}

// --- SAE Top-K mask (bitonic sort approach for small n) ---
// Step 1: compute abs values into a temp buffer
// Step 2: bitonic sort to find the k-th largest absolute value (threshold)
// Step 3: zero out elements whose abs value < threshold

__global__ void kernel_sae_topk_mask_impl(float* data, size_t n, size_t k) {
    // For small n (typical d_model=128), use single-block bitonic sort
    // Each thread handles multiple elements if n > blockDim
    extern __shared__ float s_abs[];

    // Load abs values into shared memory
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        s_abs[i] = fabsf(data[i]);
    }
    __syncthreads();

    // Bitonic sort descending in shared memory
    for (size_t stage = 1; stage < n; stage *= 2) {
        for (size_t step = stage; step >= 1; step /= 2) {
            for (size_t i = threadIdx.x; i < n / 2; i += blockDim.x) {
                size_t dir = ((i / stage) % 2) == 0 ? 1 : 0;
                size_t pair = i ^ step;
                if (pair > i && pair < n) {
                    float a = s_abs[i];
                    float b = s_abs[pair];
                    if ((dir && a < b) || (!dir && a > b)) {
                        s_abs[i] = b;
                        s_abs[pair] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // k-th largest absolute value is at index k-1 (0-indexed)
    float threshold = 0.0f;
    if (k > 0 && k <= n) {
        threshold = s_abs[k - 1];
    }
    __syncthreads();

    // Zero out elements below threshold
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (fabsf(data[i]) < threshold) {
            data[i] = 0.0f;
        }
    }
}

inline void launch_sae_topk_mask(float* d_data, size_t n, size_t k, cudaStream_t stream) {
    int block = 256;
    size_t shared_mem = n * sizeof(float);
    kernel_sae_topk_mask_impl<<<1, block, shared_mem, stream>>>(d_data, n, k);
}

// --- SAE Top-K mask for backward (same logic but on grad, using cached encoded values) ---
__global__ void kernel_sae_topk_mask_backward_impl(float* grad, const float* encoded, size_t n, size_t k) {
    extern __shared__ float s_abs[];

    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        s_abs[i] = fabsf(encoded[i]);
    }
    __syncthreads();

    for (size_t stage = 1; stage < n; stage *= 2) {
        for (size_t step = stage; step >= 1; step /= 2) {
            for (size_t i = threadIdx.x; i < n / 2; i += blockDim.x) {
                size_t dir = ((i / stage) % 2) == 0 ? 1 : 0;
                size_t pair = i ^ step;
                if (pair > i && pair < n) {
                    float a = s_abs[i];
                    float b = s_abs[pair];
                    if ((dir && a < b) || (!dir && a > b)) {
                        s_abs[i] = b;
                        s_abs[pair] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    float threshold = 0.0f;
    if (k > 0 && k <= n) {
        threshold = s_abs[k - 1];
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (fabsf(encoded[i]) < threshold) {
            grad[i] = 0.0f;
        }
    }
}

inline void launch_sae_topk_mask_backward(float* d_grad, const float* d_encoded, size_t n, size_t k, cudaStream_t stream) {
    int block = 256;
    size_t shared_mem = n * sizeof(float);
    kernel_sae_topk_mask_backward_impl<<<1, block, shared_mem, stream>>>(d_grad, d_encoded, n, k);
}

// --- Gradient clipping (compute norm + scale) ---
// Returns total squared norm on GPU (single float)
__global__ void kernel_grad_norm_sq_impl(float* norm_sq, const float* const* grads, const size_t* sizes, int n_tensors) {
    float local_sum = 0.0f;
    for (int t = 0; t < n_tensors; ++t) {
        const float* g = grads[t];
        size_t sz = sizes[t];
        for (size_t i = threadIdx.x; i < sz; i += blockDim.x) {
            local_sum += g[i] * g[i];
        }
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    if (threadIdx.x == 0) *norm_sq = s_sum;
}

// --- Scale multiple gradients by a single factor ---
__global__ void kernel_scale_grads_impl(float** grads, const size_t* sizes, int n_tensors, float scale) {
    int t = blockIdx.y;
    if (t >= n_tensors) return;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizes[t]) return;
    grads[t][idx] *= scale;
}

// --- SGD update with gradient clipping (fused: scale + update) ---
__global__ void kernel_sgd_update_clipped_impl(float* param, const float* grad, size_t n, float lr, float clip_scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx] * clip_scale;
    if (isfinite(g)) {
        param[idx] -= lr * g;
    }
}

inline void launch_sgd_update_clipped(float* d_param, const float* d_grad, size_t n, float lr, float clip_scale, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_sgd_update_clipped_impl<<<grid, block, 0, stream>>>(d_param, d_grad, n, lr, clip_scale);
}

// --- Padding Mask Generation ---
__global__ void kernel_padding_mask_impl(float* mask, const int* token_ids, int seq_len, int padding_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;
    mask[idx] = (token_ids[idx] == padding_id) ? 0.0f : 1.0f;
}

inline void launch_padding_mask(float* d_mask, const int* d_token_ids, int seq_len, int padding_id, cudaStream_t stream) {
    int block = 256;
    int grid = (seq_len + block - 1) / block;
    kernel_padding_mask_impl<<<grid, block, 0, stream>>>(d_mask, d_token_ids, seq_len, padding_id);
}

// --- Fused Causal + Padding Mask (applied to attention scores before softmax) ---
// Sets attn_scores[i][j] = -inf if (j > i) OR (padding_mask[j] == 0)
__global__ void kernel_fused_causal_padding_mask_impl(float* attn_scores, const float* padding_mask,
                                                       int seq_len, int n_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * seq_len * seq_len;
    if (idx >= total) return;
    int h = idx / (seq_len * seq_len);
    int rem = idx % (seq_len * seq_len);
    int i = rem / seq_len;
    int j = rem % seq_len;
    if (j > i || padding_mask[j] == 0.0f) {
        attn_scores[idx] = -1e30f;
    }
}

inline void launch_fused_causal_padding_mask(float* d_attn_scores, const float* d_padding_mask,
                                              int seq_len, int n_heads, cudaStream_t stream) {
    int total = n_heads * seq_len * seq_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_fused_causal_padding_mask_impl<<<grid, block, 0, stream>>>(d_attn_scores, d_padding_mask, seq_len, n_heads);
}


// --- GPU Top-K/Top-P Sampling ---
// Performs temperature scaling, top-k filtering, top-p filtering, softmax, and sampling
// entirely on GPU. Only the sampled token ID is copied back to CPU.
// Uses a simple deterministic pseudo-random based on block/thread index + seed.

__global__ void kernel_topk_topp_sampling_impl(
    const float* d_logits, int* d_output_token,
    int vocab_size, int top_k, float top_p, float temperature,
    unsigned int seed, int num_candidates) {

    extern __shared__ float smem[];

    float* sm_scores = smem;
    int* sm_indices = (int*)(sm_scores + num_candidates);

    int tid = threadIdx.x;

    for (int i = tid; i < num_candidates; ++i) {
        sm_scores[i] = -1e30f;
        sm_indices[i] = 0;
    }
    __syncthreads();

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = d_logits[i] / temperature;
        if (val > sm_scores[num_candidates - 1]) {
            int pos = num_candidates - 1;
            while (pos > 0 && val > sm_scores[pos - 1]) {
                sm_scores[pos] = sm_scores[pos - 1];
                sm_indices[pos] = sm_indices[pos - 1];
                pos--;
            }
            sm_scores[pos] = val;
            sm_indices[pos] = i;
        }
    }
    __syncthreads();

    if (tid == 0) {
        int k = min(top_k, vocab_size);
        k = min(k, num_candidates);

        float max_val = sm_scores[0];
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sm_scores[i] = expf(sm_scores[i] - max_val);
            sum += sm_scores[i];
        }
        for (int i = 0; i < k; ++i) {
            sm_scores[i] /= sum;
        }

        float cumulative = 0.0f;
        int top_p_cutoff = k;
        for (int i = 0; i < k; ++i) {
            cumulative += sm_scores[i];
            if (cumulative >= top_p) {
                top_p_cutoff = i + 1;
                break;
            }
        }

        float renorm_sum = 0.0f;
        for (int i = 0; i < top_p_cutoff; ++i) {
            renorm_sum += sm_scores[i];
        }

        unsigned int rng_state = seed + blockIdx.x * 12345;
        rng_state = rng_state * 1103515245 + 12345;
        float r = ((rng_state >> 16) & 0x7fff) / 32768.0f;

        float threshold = r * renorm_sum;
        cumulative = 0.0f;
        int chosen = 0;
        for (int i = 0; i < top_p_cutoff; ++i) {
            cumulative += sm_scores[i];
            if (cumulative >= threshold) {
                chosen = sm_indices[i];
                break;
            }
        }

        *d_output_token = chosen;
    }
}

inline bool launch_topk_topp_sampling(const float* d_logits, int* d_output_token,
                                       int vocab_size, int top_k, float top_p,
                                       float temperature, unsigned int seed,
                                       cudaStream_t stream) {
    int num_candidates = min(top_k, vocab_size);
    num_candidates = max(num_candidates, 1);

    int block = 256;
    size_t smem_size = sizeof(float) * num_candidates + sizeof(int) * num_candidates;

    if (smem_size > 48 * 1024) {
        return false;
    }

    kernel_topk_topp_sampling_impl<<<1, block, smem_size, stream>>>(
        d_logits, d_output_token, vocab_size, top_k, top_p, temperature, seed, num_candidates);

    return true;
}

// --- FP16/BF16 Conversion Kernels ---

__global__ void kernel_fp32_to_fp16(uint16_t* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(in[idx]);
}

__global__ void kernel_fp16_to_fp32(float* out, const uint16_t* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __half2float(in[idx]);
}

__global__ void kernel_fp32_to_bf16(uint16_t* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t bits;
    memcpy(&bits, &in[idx], 4);
    out[idx] = static_cast<uint16_t>(bits >> 16);
}

__global__ void kernel_bf16_to_fp32(float* out, const uint16_t* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t bits = static_cast<uint32_t>(in[idx]) << 16;
    memcpy(&out[idx], &bits, 4);
}

inline void launch_fp32_to_fp16(uint16_t* d_out, const float* d_in, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_fp32_to_fp16<<<grid, block, 0, stream>>>(d_out, d_in, n);
}

inline void launch_fp16_to_fp32(float* d_out, const uint16_t* d_in, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_fp16_to_fp32<<<grid, block, 0, stream>>>(d_out, d_in, n);
}

inline void launch_fp32_to_bf16(uint16_t* d_out, const float* d_in, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_fp32_to_bf16<<<grid, block, 0, stream>>>(d_out, d_in, n);
}

inline void launch_bf16_to_fp32(float* d_out, const uint16_t* d_in, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (static_cast<int>(n) + block - 1) / block;
    kernel_bf16_to_fp32<<<grid, block, 0, stream>>>(d_out, d_in, n);
}

// --- Flash Attention (Tiled Online Softmax) ---

template<int BLOCK_SIZE = 64, int HEAD_DIM = 64>
__global__ void kernel_flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, int seq_len, float scale) {

    int h = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= seq_len) return;

    extern __shared__ char smem[];
    float* s_K = reinterpret_cast<float*>(smem);
    float* s_V = reinterpret_cast<float*>(smem + BLOCK_SIZE * HEAD_DIM * sizeof(float));

    float m = -1e30f;
    float l = 0.0f;
    float o[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) o[d] = 0.0f;

    const float* q_row = Q + (h * seq_len + i) * HEAD_DIM;

    for (int jb = 0; jb < seq_len; jb += BLOCK_SIZE) {
        int j_end = min(jb + BLOCK_SIZE, seq_len);

        for (int j = threadIdx.x; j < (j_end - jb) * HEAD_DIM; j += blockDim.x) {
            int jj = j / HEAD_DIM;
            int dd = j % HEAD_DIM;
            s_K[jj * HEAD_DIM + dd] = K[(h * seq_len + jb + jj) * HEAD_DIM + dd];
            s_V[jj * HEAD_DIM + dd] = V[(h * seq_len + jb + jj) * HEAD_DIM + dd];
        }
        __syncthreads();

        for (int jj = 0; jj < (j_end - jb); ++jj) {
            int j = jb + jj;
            if (j > i) break;

            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += q_row[d] * s_K[jj * HEAD_DIM + d];
            }
            dot *= scale;

            float m_new = max(m, dot);
            float l_new = expf(m - m_new) * l + expf(dot - m_new);

            for (int d = 0; d < HEAD_DIM; ++d) {
                o[d] = expf(m - m_new) * o[d] + expf(dot - m_new) * s_V[jj * HEAD_DIM + d];
            }
            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }

    if (l > 0.0f) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            O[(h * seq_len + i) * HEAD_DIM + d] = o[d] / l;
        }
    } else {
        for (int d = 0; d < HEAD_DIM; ++d) {
            O[(h * seq_len + i) * HEAD_DIM + d] = 0.0f;
        }
    }
}

inline void launch_flash_attention(const float* d_Q, const float* d_K, const float* d_V,
                                   float* d_O, int n_heads, int seq_len,
                                   int head_dim, float scale, cudaStream_t stream) {
    dim3 grid(n_heads, (seq_len + 63) / 64);
    int block = 64;
    size_t smem = 64 * head_dim * 2 * sizeof(float);
    if (head_dim <= 64) {
        kernel_flash_attention_forward<64, 64><<<grid, block, smem, stream>>>(
            d_Q, d_K, d_V, d_O, seq_len, scale);
    } else if (head_dim <= 128) {
        kernel_flash_attention_forward<64, 128><<<grid, block, smem, stream>>>(
            d_Q, d_K, d_V, d_O, seq_len, scale);
    } else {
        kernel_flash_attention_forward<64, 256><<<grid, block, smem, stream>>>(
            d_Q, d_K, d_V, d_O, seq_len, scale);
    }
}

} // namespace neuroflow

#endif // USE_CUDA
#endif // NEUROFLOW_CUDA_KERNELS_HPP
/*
 * NeuroFlow GPU Kernels (HIP/ROCm)
 * 编译: hipcc -shared -fPIC -O3 -o libneuroflow_gpu.so neuroflow_gpu.cpp --offload-arch=gfx90a
 * 用法: Python ctypes加载
 */

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK_HIP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d '%s'(%d)\n", __FILE__, __LINE__, hipGetErrorString(e), e); \
        exit(1); \
    } \
} while(0)

// ═══════════════════════════════════════════
// GPU Kernels
// ═══════════════════════════════════════════

// ReLU: y = max(x, 0)
__global__ void relu_kernel(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(x[i], 0.0f);
}

// Sigmoid: y = 1/(1+exp(-x))
__global__ void sigmoid_kernel(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

// L2 normalize rows (in-place)
__global__ void l2_normalize_rows_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float sum_sq = 0.0f;
    for (int j = 0; j < cols; j++) {
        float v = x[row * cols + j];
        sum_sq += v * v;
    }
    float norm = sqrtf(sum_sq) + 1e-8f;
    for (int j = 0; j < cols; j++) {
        x[row * cols + j] /= norm;
    }
}

// Masked noise: y = x * mask + noise
__global__ void mask_noise_kernel(float* y, const float* x, const float* mask, int n, float noise_std, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Simple LCG random
    unsigned long s = seed + i * 2654435761UL;
    float r = (float)(s & 0xFFFFFF) / 0xFFFFFF - 0.5f;
    y[i] = x[i] * mask[i] + r * noise_std;
}

// MSE loss gradient: grad = 2*(pred - target)/N
__global__ void mse_grad_kernel(float* grad, const float* pred, const float* target, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 2.0f * (pred[i] - target[i]) / (float)n;
}

// SGD update: w -= lr * (grad + wd * w)
__global__ void sgd_update_kernel(float* w, const float* grad, int n, float lr, float wd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * (grad[i] + wd * w[i]);
}

// Masked top-K selection (SAE): keep only top K
__global__ void topk_mask_kernel(float* y, int n, int k, int offset) {
    // Each block processes one row
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    // Load row into shared memory
    if (tid < k * 2) {  // double buffer
        int idx = row * n + offset + tid;
        shared[tid] = (tid < n - offset) ? fabsf(y[idx]) : -1.0f;
    }
    __syncthreads();
    
    // Simple bitonic sort for top-k (k is small, e.g., 65)
    // ... simplified: just threshold
    if (tid == 0) {
        // Copy to local, sort, find threshold
        float local[256];
        int len = min(n - offset, 256);
        for (int j = 0; j < len; j++) {
            int idx = row * n + offset + j;
            local[j] = fabsf(y[idx]);
        }
        // Simple bubble sort for top-k
        for (int j = 0; j < k && j < len; j++) {
            int max_idx = j;
            for (int m = j+1; m < len; m++) {
                if (local[m] > local[max_idx]) max_idx = m;
            }
            float tmp = local[j];
            local[j] = local[max_idx];
            local[max_idx] = tmp;
        }
        float threshold = (k <= len) ? local[k-1] : 0.0f;
        // Apply mask
        for (int j = 0; j < len; j++) {
            int idx = row * n + offset + j;
            if (fabsf(y[idx]) < threshold && fabsf(y[idx]) > 0) {
                y[idx] = 0.0f;
            }
        }
    }
}

// ═══════════════════════════════════════════
// Python-callable C functions
// ═══════════════════════════════════════════

extern "C" {

// GPU info
int gpu_get_count() {
    int count;
    CHECK_HIP(hipGetDeviceCount(&count));
    return count;
}

void gpu_get_name(int id, char* name, int max_len) {
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, id));
    strncpy(name, prop.name, max_len);
}

// Init
void gpu_init(int device_id) {
    CHECK_HIP(hipSetDevice(device_id));
    printf("[GPU] Initialized device %d\n", device_id);
}

// Allocate / Free
float* gpu_alloc(int n) {
    float* ptr;
    CHECK_HIP(hipMalloc(&ptr, n * sizeof(float)));
    return ptr;
}

void gpu_free(float* ptr) {
    CHECK_HIP(hipFree(ptr));
}

// Copy H→D, D→H
void gpu_memcpy_htod(float* d_ptr, const float* h_ptr, int n) {
    CHECK_HIP(hipMemcpy(d_ptr, h_ptr, n * sizeof(float), hipMemcpyHostToDevice));
}

void gpu_memcpy_dtoh(float* h_ptr, const float* d_ptr, int n) {
    CHECK_HIP(hipMemcpy(h_ptr, d_ptr, n * sizeof(float), hipMemcpyDeviceToHost));
}

// ReLU
void gpu_relu(float* d_y, const float* d_x, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_kernel<<<grid, block>>>(d_y, d_x, n);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

// Sigmoid
void gpu_sigmoid(float* d_y, const float* d_x, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sigmoid_kernel<<<grid, block>>>(d_y, d_x, n);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

// L2 normalize rows
void gpu_l2_norm_rows(float* d_x, int rows, int cols) {
    l2_normalize_rows_kernel<<<rows, 1>>>(d_x, rows, cols);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

// Mask + Noise
void gpu_mask_noise(float* d_y, const float* d_x, const float* d_mask, int n, float noise_std, unsigned long seed) {
    int block = 256;
    int grid = (n + block - 1) / block;
    mask_noise_kernel<<<grid, block>>>(d_y, d_x, d_mask, n, noise_std, seed);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

// Matrix multiply: C = A @ B  (using hipBLAS)
// A: [M, K], B: [K, N], C: [M, N]
void gpu_matmul(float* d_C, const float* d_A, const float* d_B, int M, int N, int K) {
    static hipblasHandle_t handle = nullptr;
    if (!handle) hipblasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                 N, M, K,
                 &alpha, d_B, N, d_A, K,
                 &beta, d_C, N);
    CHECK_HIP(hipDeviceSynchronize());
}

// SGD update
void gpu_sgd_update(float* d_w, const float* d_grad, int n, float lr, float wd) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sgd_update_kernel<<<grid, block>>>(d_w, d_grad, n, lr, wd);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

// Fill with zeros
void gpu_fill_zero(float* d_ptr, int n) {
    CHECK_HIP(hipMemset(d_ptr, 0, n * sizeof(float)));
}

// Print GPU memory info
void gpu_mem_info() {
    size_t free, total;
    CHECK_HIP(hipMemGetInfo(&free, &total));
    printf("[GPU] Memory: %.1f GB free / %.1f GB total\n", 
           free / 1e9, total / 1e9);
}

} // extern "C"

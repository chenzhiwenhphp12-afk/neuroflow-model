#ifndef NEUROFLOW_CUDA_CONTEXT_HPP
#define NEUROFLOW_CUDA_CONTEXT_HPP

#ifdef USE_CUDA

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("[CUDA ERROR] ") + cudaGetErrorString(err)          \
                + " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            throw std::runtime_error(                                           \
                std::string("[CUBLAS ERROR] code=") + std::to_string(status)    \
                + " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                       \
    } while (0)

namespace neuroflow {

class CudaContext {
public:
    static CudaContext& instance() {
        static CudaContext ctx;
        return ctx;
    }

    bool initialize(int device_id = 0) {
        if (initialized_) return true;

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            std::cerr << "[CUDA WARNING] No CUDA-capable GPU detected" << std::endl;
            return false;
        }

        if (device_id >= device_count) {
            std::cerr << "[CUDA WARNING] Device " << device_id
                      << " not available (count=" << device_count << ")" << std::endl;
            return false;
        }

        CUDA_CHECK(cudaSetDevice(device_id));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        if (prop.major < 8) {
            std::cerr << "[CUDA WARNING] GPU Compute Capability " << prop.major
                      << "." << prop.minor << " < 8.0 (Ampere required)" << std::endl;
            return false;
        }

        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));

        device_id_ = device_id;
        initialized_ = true;

        std::cerr << "[CUDA] Initialized on " << prop.name
                  << " (CC " << prop.major << "." << prop.minor
                  << ", " << prop.totalGlobalMem / (1024*1024) << " MB)" << std::endl;
        return true;
    }

    void finalize() {
        if (!initialized_) return;
        CUBLAS_CHECK(cublasDestroy(cublas_handle_));
        CUDA_CHECK(cudaStreamDestroy(stream_));
        initialized_ = false;
        device_id_ = -1;
    }

    void sgemm(bool transA, bool transB,
               int M, int N, int K,
               float alpha, const float* d_A, int lda,
               const float* d_B, int ldb,
               float beta, float* d_C, int ldc) {
        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_CHECK(cublasSgemm(cublas_handle_, opA, opB,
                                 M, N, K, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
    }

    void sgemm_rowmajor(bool transA, bool transB,
                        int M, int N, int K,
                        float alpha, const float* d_A, int lda,
                        const float* d_B, int ldb,
                        float beta, float* d_C, int ldc) {
        cublasOperation_t opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_CHECK(cublasSgemm(cublas_handle_, opA, opB,
                                 N, M, K, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
    }

    void* alloc(size_t bytes) {
        void* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
        return d_ptr;
    }

    void free(void* d_ptr) {
        if (d_ptr) CUDA_CHECK(cudaFree(d_ptr));
    }

    void copy_h2d(void* dst, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_));
    }

    void copy_d2h(void* dst, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_));
    }

    void copy_d2d(void* dst, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream_));
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    bool is_available() const { return initialized_; }

    size_t free_memory() const {
        if (!initialized_) return 0;
        size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return free;
    }

    size_t total_memory() const {
        if (!initialized_) return 0;
        size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return total;
    }

    cudaStream_t stream() const { return stream_; }
    cublasHandle_t cublas_handle() const { return cublas_handle_; }

private:
    CudaContext() = default;
    ~CudaContext() { finalize(); }
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    cublasHandle_t cublas_handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool initialized_ = false;
    int device_id_ = -1;
};

} // namespace neuroflow

#endif // USE_CUDA
#endif // NEUROFLOW_CUDA_CONTEXT_HPP
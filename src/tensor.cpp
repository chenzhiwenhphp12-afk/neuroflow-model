#include "neuroflow/tensor.hpp"
#include <iostream>
#include <numeric>
#include <future>
#include <functional>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace neuroflow {

namespace {

constexpr size_t OMP_MIN_ITER = 1024;

#ifdef _OPENMP
using omp_idx_t = long long;
#else
using omp_idx_t = size_t;
#endif

}

void TensorOps::parallel_gemm(const Tensor& A, const Tensor& B, Tensor& C,
                               bool transA, bool transB,
                               float alpha, float beta,
                               size_t num_threads) {
    if (A.dtype_ != QuantType::FP32 || B.dtype_ != QuantType::FP32) {
        quantized_gemm(A, B, C);
        return;
    }

    size_t M = transA ? A.shape_[1] : A.shape_[0];
    size_t K = transA ? A.shape_[0] : A.shape_[1];
    size_t N = transB ? B.shape_[0] : B.shape_[1];

    auto* a = reinterpret_cast<const float*>(A.data_.get());
    auto* b = reinterpret_cast<const float*>(B.data_.get());
    auto* c = reinterpret_cast<float*>(C.data_.get());

#ifdef USE_CBLAS
    CBLAS_TRANSPOSE ctransA = transA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE ctransB = transB ? CblasTrans : CblasNoTrans;
    size_t lda = transA ? M : K;
    size_t ldb = transB ? K : N;
    cblas_sgemm(CblasRowMajor, ctransA, ctransB, M, N, K, alpha, a, lda, b, ldb, beta, c, N);
#else
    if (beta != 0.0f) {
        for (size_t i = 0; i < M * N; ++i) c[i] *= beta;
    } else {
        memset(c, 0, M * N * sizeof(float));
    }

    if (num_threads <= 1 || M < 8) {
        gemm_scalar(a, b, c, M, K, N, transA, transB, alpha, beta);
        return;
    }

  #ifdef _OPENMP
    size_t use_threads = std::min(static_cast<size_t>(omp_get_max_threads()), num_threads);
    #pragma omp parallel for schedule(static) num_threads(use_threads) if(M >= 64)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(M); ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float av = transA ? a[k * M + i] : a[i * K + k];
                float bv = transB ? b[j * K + k] : b[k * N + j];
                sum += av * bv;
            }
            c[i * N + j] += alpha * sum;
        }
    }
  #else
    ThreadPool pool;
    size_t rows_per_thread = (M + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t row_start = t * rows_per_thread;
        size_t row_end = std::min(row_start + rows_per_thread, M);
        if (row_start >= M) break;

        pool.submit([=]() {
            for (size_t i = row_start; i < row_end; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        float av = transA ? a[k * M + i] : a[i * K + k];
                        float bv = transB ? b[j * K + k] : b[k * N + j];
                        sum += av * bv;
                    }
                    c[i * N + j] += alpha * sum;
                }
            }
        });
    }
    pool.wait_all();
  #endif
#endif
}

Tensor TensorOps::matmul(const Tensor& A, const Tensor& B) {
    if (A.shape_.size() != 2 || B.shape_.size() != 2)
        throw std::runtime_error("matmul requires 2D tensors");
    if (A.shape_[1] != B.shape_[0])
        throw std::runtime_error("matmul dimension mismatch");

    Tensor C({A.shape_[0], B.shape_[1]}, A.dtype_);
    gemm(A, B, C, false, false);
    return C;
}

Tensor TensorOps::elementwise_add(const Tensor& A, const Tensor& B) {
    if (A.shape_ != B.shape_) throw std::runtime_error("shape mismatch in add");
    Tensor C = A.clone();
    add(C, B);
    return C;
}

Tensor TensorOps::elementwise_mul(const Tensor& A, const Tensor& B) {
    if (A.shape_ != B.shape_) throw std::runtime_error("shape mismatch in mul");
    Tensor C(A.shape_, A.dtype_);
    const float* a = A.as_fp32();
    const float* b = B.as_fp32();
    float* c = C.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) c[i] = a[i] * b[i];
    return C;
}

Tensor TensorOps::scalar_mul(const Tensor& A, float s) {
    Tensor C = A.clone();
    mul(C, s);
    return C;
}

Tensor TensorOps::elementwise_sub(const Tensor& A, const Tensor& B) {
    if (A.shape_ != B.shape_) throw std::runtime_error("shape mismatch in sub");
    Tensor C(A.shape_, A.dtype_);
    const float* a = A.as_fp32();
    const float* b = B.as_fp32();
    float* c = C.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) c[i] = a[i] - b[i];
    return C;
}

Tensor TensorOps::transpose2d(const Tensor& A) {
    if (A.shape_.size() != 2) throw std::runtime_error("transpose2d requires 2D tensor");
    size_t rows = A.shape_[0], cols = A.shape_[1];
    Tensor C({cols, rows}, A.dtype_);
    const float* a = A.as_fp32();
    float* c = C.as_fp32();
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            c[j * rows + i] = a[i * cols + j];
    return C;
}

Tensor TensorOps::broadcast_add(const Tensor& A, const Tensor& B) {
    Tensor C = A.clone();
    float* c = C.as_fp32();
    const float* b = B.as_fp32();

    if (B.shape_.size() == 1 && A.shape_.size() == 2 && B.shape_[0] == A.shape_[1]) {
        size_t batch = A.shape_[0], dim = A.shape_[1];
        size_t n = batch * dim;
        #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
        for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(batch); ++i)
            for (size_t j = 0; j < dim; ++j)
                c[i * dim + j] += b[j];
    } else if (B.shape_ == A.shape_) {
        size_t n = A.numel();
        #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
        for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) c[i] += b[i];
    } else {
        throw std::runtime_error("broadcast_add: incompatible shapes");
    }
    return C;
}

void TensorOps::fill(Tensor& A, float value) {
    float* a = A.as_fp32();
    size_t n = A.numel();
    if (n >= 1024) {
        memset(a, 0, n * sizeof(float));
        if (value != 0.0f) {
            #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
            for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) a[i] = value;
        }
    } else {
        for (size_t i = 0; i < n; ++i) a[i] = value;
    }
}

void TensorOps::copy_data(Tensor& dst, const Tensor& src) {
    if (dst.numel() != src.numel()) throw std::runtime_error("copy_data: size mismatch");
    memcpy(dst.as_fp32(), src.as_fp32(), dst.numel() * sizeof(float));
}

float TensorOps::dot(const Tensor& A, const Tensor& B) {
    if (A.shape_ != B.shape_) throw std::runtime_error("dot: shape mismatch");
    const float* a = A.as_fp32();
    const float* b = B.as_fp32();
    size_t n = A.numel();
    float sum = 0.0f;
    #pragma omp parallel for schedule(static) reduction(+:sum) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) sum += a[i] * b[i];
    return sum;
}

float TensorOps::norm2(const Tensor& A) {
    return std::sqrt(dot(A, A));
}

Tensor TensorOps::reduce_sum(const Tensor& A, int axis) {
    if (axis == -1) axis = static_cast<int>(A.shape_.size()) - 1;
    if (A.shape_.size() != 2 || axis != 1)
        throw std::runtime_error("reduce_sum: only 2D axis=1 supported");

    size_t batch = A.shape_[0], dim = A.shape_[1];
    Tensor C({batch, 1}, QuantType::FP32);
    const float* a = A.as_fp32();
    float* c = C.as_fp32();
    for (size_t i = 0; i < batch; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < dim; ++j) sum += a[i * dim + j];
        c[i] = sum;
    }
    return C;
}

Tensor TensorOps::slice(const Tensor& A, size_t dim, size_t start, size_t end) {
    if (dim >= A.shape_.size()) throw std::runtime_error("slice: dim out of range");
    if (end > A.shape_[dim]) end = A.shape_[dim];

    auto new_shape = A.shape_;
    new_shape[dim] = end - start;
    Tensor C(new_shape, A.dtype_);

    if (A.shape_.size() == 2 && dim == 1) {
        size_t rows = A.shape_[0];
        const float* a = A.as_fp32();
        float* c = C.as_fp32();
        for (size_t i = 0; i < rows; ++i)
            memcpy(c + i * (end - start), a + i * A.shape_[1] + start, (end - start) * sizeof(float));
    } else if (A.shape_.size() == 2 && dim == 0) {
        const float* a = A.as_fp32();
        float* c = C.as_fp32();
        memcpy(c, a + start * A.shape_[1], (end - start) * A.shape_[1] * sizeof(float));
    } else {
        throw std::runtime_error("slice: only 2D supported");
    }
    return C;
}

Tensor TensorOps::pad1d(const Tensor& A, size_t left, size_t right, float value) {
    if (A.shape_.size() != 2) throw std::runtime_error("pad1d requires 2D tensor");
    size_t batch = A.shape_[0], dim = A.shape_[1];
    size_t new_dim = left + dim + right;
    Tensor C({batch, new_dim}, QuantType::FP32);
    float* c = C.as_fp32();
    const float* a = A.as_fp32();
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < left; ++j) c[i * new_dim + j] = value;
        memcpy(c + i * new_dim + left, a + i * dim, dim * sizeof(float));
        for (size_t j = left + dim; j < new_dim; ++j) c[i * new_dim + j] = value;
    }
    return C;
}

void TensorOps::apply_inplace(Tensor& A, std::function<float(float)> fn) {
    float* a = A.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) a[i] = fn(a[i]);
}

Tensor TensorOps::apply(const Tensor& A, std::function<float(float)> fn) {
    Tensor C = A.clone();
    apply_inplace(C, fn);
    return C;
}

Tensor TensorOps::relu(const Tensor& A) {
    Tensor C = A.clone();
    float* c = C.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) c[i] = std::max(0.0f, c[i]);
    return C;
}

Tensor TensorOps::sigmoid(const Tensor& A) {
    Tensor C = A.clone();
    float* c = C.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) c[i] = 1.0f / (1.0f + std::exp(-c[i]));
    return C;
}

Tensor TensorOps::tanh_act(const Tensor& A) {
    Tensor C = A.clone();
    float* c = C.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) c[i] = std::tanh(c[i]);
    return C;
}

Tensor TensorOps::log(const Tensor& A) {
    Tensor C = A.clone();
    float* c = C.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) c[i] = std::log(std::max(c[i], 1e-7f));
    return C;
}

Tensor TensorOps::exp(const Tensor& A) {
    Tensor C = A.clone();
    float* c = C.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i) c[i] = std::exp(c[i]);
    return C;
}

void TensorOps::clip_inplace(Tensor& A, float min_val, float max_val) {
    float* a = A.as_fp32();
    size_t n = A.numel();
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_ITER)
    for (omp_idx_t i = 0; i < static_cast<omp_idx_t>(n); ++i)
        a[i] = std::clamp(a[i], min_val, max_val);
}

}

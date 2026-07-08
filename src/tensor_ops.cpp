#include "neuroflow/tensor.hpp"
#include "neuroflow/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#ifdef USE_CBLAS
extern "C" {
#include <cblas.h>
}
#define cblas_sgemm scipy_cblas_sgemm
#endif

#ifdef USE_CUDA
#include "cuda_context.hpp"
#endif

#ifdef __AVX2__
#include <immintrin.h>
#define HAS_AVX2 1
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#define HAS_NEON 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <condition_variable>
#include <future>
#include <mutex>
#include <numeric>
#include <queue>

namespace neuroflow {

namespace {

constexpr size_t OMP_MIN_ITER = 1024;

#ifdef _OPENMP
using omp_idx_t = long long;
#else
using omp_idx_t = size_t;
#endif

#ifndef _OPENMP
class ThreadPool {
public:
    explicit ThreadPool(size_t n = std::thread::hardware_concurrency()) {
        n = std::max(n, size_t(1));
        for (size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

    void wait_all() {
        while (true) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.empty()) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};
#endif

}

void TensorOps::gemm(const Tensor& A, const Tensor& B, Tensor& C,
                     bool transA, bool transB,
                     float alpha, float beta) {
    size_t K_A = transA ? A.shape_[0] : A.shape_[1];
    size_t K_B = transB ? B.shape_[1] : B.shape_[0];
    if (K_A != K_B) {
        throw std::runtime_error("GEMM dimension mismatch: K dimensions don't match");
    }

    if (A.dtype_ != QuantType::FP32 || B.dtype_ != QuantType::FP32) {
        quantized_gemm(A, B, C);
        return;
    }

    size_t M = transA ? A.shape_[1] : A.shape_[0];
    size_t K = transA ? A.shape_[0] : A.shape_[1];
    size_t N = transB ? B.shape_[0] : B.shape_[1];

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && A.is_on_gpu() && B.is_on_gpu()) {
        const float* d_a = A.as_gpu_fp32();
        const float* d_b = B.as_gpu_fp32();
        float* d_c = C.as_gpu_fp32();
        int lda = static_cast<int>(transA ? M : K);
        int ldb = static_cast<int>(transB ? K : N);
        int ldc = static_cast<int>(N);
        CudaContext::instance().sgemm_rowmajor(transA, transB,
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            alpha, d_a, lda, d_b, ldb, beta, d_c, ldc);
        C.gpu_dirty_ = true;
        return;
    }
#endif

    auto a = reinterpret_cast<const float*>(A.data_.get());
    auto b = reinterpret_cast<const float*>(B.data_.get());
    auto c = reinterpret_cast<float*>(C.data_.get());

#ifdef USE_CBLAS
    CBLAS_TRANSPOSE ctransA = transA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE ctransB = transB ? CblasTrans : CblasNoTrans;
    CBLAS_ORDER order = CblasRowMajor;
    size_t lda_cblas = transA ? M : K;
    size_t ldb_cblas = transB ? K : N;
    cblas_sgemm(order, ctransA, ctransB, M, N, K, alpha, a, lda_cblas, b, ldb_cblas, beta, c, N);
#elif HAS_AVX2
    gemm_avx2(a, b, c, M, K, N, transA, transB, alpha, beta);
#elif HAS_NEON
    gemm_neon(a, b, c, M, K, N, transA, transB, alpha, beta);
#else
    gemm_scalar(a, b, c, M, K, N, transA, transB, alpha, beta);
#endif
}

#ifdef HAS_AVX2
void TensorOps::gemm_avx2(const float* a, const float* b, float* c,
                           size_t M, size_t K, size_t N,
                           bool transA, bool transB,
                           float alpha, float beta) {
    if (beta != 0.0f) {
        for (size_t i = 0; i < M * N; ++i) c[i] *= beta;
    } else {
        memset(c, 0, M * N * sizeof(float));
    }

    const size_t BLOCK = 8;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; j += BLOCK) {
            __m256 sum = _mm256_setzero_ps();

            for (size_t k = 0; k < K; ++k) {
                float av = transA ? a[k * M + i] : a[i * K + k];
                __m256 bv;

                if (transB) {
                    float bvals[BLOCK];
                    for (size_t jj = 0; jj < BLOCK && j + jj < N; ++jj) {
                        bvals[jj] = b[(j + jj) * K + k];
                    }
                    bv = _mm256_loadu_ps(bvals);
                } else {
                    bv = _mm256_loadu_ps(&b[k * N + j]);
                }

                sum = _mm256_fmadd_ps(_mm256_set1_ps(av), bv, sum);
            }

            sum = _mm256_mul_ps(sum, _mm256_set1_ps(alpha));

            if (j + BLOCK <= N) {
                __m256 cv = _mm256_loadu_ps(&c[i * N + j]);
                cv = _mm256_add_ps(cv, sum);
                _mm256_storeu_ps(&c[i * N + j], cv);
            } else {
                float tmp[BLOCK];
                _mm256_storeu_ps(tmp, sum);
                for (size_t jj = 0; j + jj < N; ++jj) {
                    c[i * N + j + jj] += tmp[jj];
                }
            }
        }
    }
}
#endif

#ifdef HAS_NEON
void TensorOps::gemm_neon(const float* a, const float* b, float* c,
                           size_t M, size_t K, size_t N,
                           bool transA, bool transB,
                           float alpha, float beta) {
    if (beta != 0.0f) {
        for (size_t i = 0; i < M * N; ++i) c[i] *= beta;
    } else {
        memset(c, 0, M * N * sizeof(float));
    }

    const size_t BLOCK = 4;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; j += BLOCK) {
            float32x4_t sum = vdupq_n_f32(0.0f);

            for (size_t k = 0; k < K; ++k) {
                float av = transA ? a[k * M + i] : a[i * K + k];
                float32x4_t bv;

                if (transB) {
                    float bvals[BLOCK];
                    for (size_t jj = 0; jj < BLOCK && j + jj < N; ++jj) {
                        bvals[jj] = b[(j + jj) * K + k];
                    }
                    bv = vld1q_f32(bvals);
                } else {
                    bv = vld1q_f32(&b[k * N + j]);
                }

                sum = vmlaq_n_f32(sum, bv, av);
            }

            sum = vmulq_n_f32(sum, alpha);

            float32x4_t cv = vld1q_f32(&c[i * N + j]);
            cv = vaddq_f32(cv, sum);
            vst1q_f32(&c[i * N + j], cv);
        }
    }
}
#endif

void TensorOps::gemm_scalar(const float* a, const float* b, float* c,
                            size_t M, size_t K, size_t N,
                            bool transA, bool transB,
                            float alpha, float beta) {
    if (beta != 0.0f) {
        for (size_t i = 0; i < M * N; ++i) c[i] *= beta;
    } else {
        memset(c, 0, M * N * sizeof(float));
    }

    for (size_t i = 0; i < M; ++i) {
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
}

void TensorOps::quantized_gemm(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.dtype_ == QuantType::INT8 && B.dtype_ == QuantType::FP32) {
        const int8_t* a = reinterpret_cast<const int8_t*>(A.data_.get());
        const float* b = reinterpret_cast<const float*>(B.data_.get());
        float* c = reinterpret_cast<float*>(C.data_.get());

        size_t M = A.shape_[0];
        size_t K = A.shape_[1];
        size_t N = B.shape_[1];

        memset(c, 0, M * N * sizeof(float));

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                int32_t sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[i * K + k] * static_cast<int32_t>(b[k * N + j]);
                }
                c[i * N + j] = static_cast<float>(sum);
            }
        }
    }
}

void TensorOps::layer_norm(Tensor& x, Tensor& weight, Tensor& bias, float eps) {
    float* data = x.as_fp32();
    float* w = weight.as_fp32();
    float* b = bias.as_fp32();

    size_t batch = x.shape_[0];
    size_t dim = x.shape_[1];

    for (size_t i = 0; i < batch; ++i) {
        float mean = 0.0f;
        float var = 0.0f;

        for (size_t j = 0; j < dim; ++j) {
            mean += data[i * dim + j];
        }
        mean /= dim;

        for (size_t j = 0; j < dim; ++j) {
            float diff = data[i * dim + j] - mean;
            var += diff * diff;
        }
        var /= dim;

        float std = std::sqrt(var + eps);
        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = w[j] * (data[i * dim + j] - mean) / std + b[j];
        }
    }
}

void TensorOps::gelu(Tensor& x) {
    float* data = x.as_fp32();
    size_t n = x.numel();

    const float SQRT_2_PI = 0.7978845608f;
    const float COEFF = 0.044715f;

    for (size_t i = 0; i < n; ++i) {
        float v = data[i];
        float inner = SQRT_2_PI * (v + COEFF * v * v * v);
        data[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

void TensorOps::softmax(Tensor& x, int axis) {
    float* data = x.as_fp32();

    if (axis == -1) axis = static_cast<int>(x.shape_.size()) - 1;

    size_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= x.shape_[i];

    size_t inner_dim = x.shape_[axis];
    size_t inner = 1;
    for (size_t i = axis + 1; i < x.shape_.size(); ++i) inner *= x.shape_[i];

    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            float* row = data + o * inner_dim * inner + i;

            float max_val = row[0];
            for (size_t d = 1; d < inner_dim; ++d) {
                max_val = std::max(max_val, row[d * inner]);
            }

            float sum = 0.0f;
            for (size_t d = 0; d < inner_dim; ++d) {
                row[d * inner] = std::exp(row[d * inner] - max_val);
                sum += row[d * inner];
            }

            for (size_t d = 0; d < inner_dim; ++d) {
                row[d * inner] /= sum;
            }
        }
    }
}

void TensorOps::dropout(Tensor& x, float rate, bool training) {
    if (!training || rate == 0.0f) return;

    float* data = x.as_fp32();
    size_t n = x.numel();

    float scale = 1.0f / (1.0f - rate);

    size_t addr_bits = reinterpret_cast<uintptr_t>(data);
    std::mt19937 rng(static_cast<unsigned>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ addr_bits ^ n));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n; ++i) {
        if (dist(rng) < rate) {
            data[i] = 0.0f;
        } else {
            data[i] *= scale;
        }
    }
}

void TensorOps::quantize_int8(const Tensor& src, Tensor& dst, Tensor& scale) {
    if (src.dtype_ != QuantType::FP32) throw std::runtime_error("Source must be FP32");

    const float* sdata = reinterpret_cast<const float*>(src.data_.get());
    int8_t* ddata = reinterpret_cast<int8_t*>(dst.data_.get());
    float* scales = reinterpret_cast<float*>(scale.data_.get());

    size_t rows = src.shape_[0];
    size_t cols = src.shape_[1];

    for (size_t i = 0; i < rows; ++i) {
        float max_abs = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            max_abs = std::max(max_abs, std::abs(sdata[i * cols + j]));
        }

        scales[i] = max_abs / 127.0f;
        if (scales[i] < 1e-8f) scales[i] = 1e-8f;

        for (size_t j = 0; j < cols; ++j) {
            float v = sdata[i * cols + j] / scales[i];
            ddata[i * cols + j] = static_cast<int8_t>(std::round(std::clamp(v, -128.0f, 127.0f)));
        }
    }
}

void TensorOps::dequantize_int8(const Tensor& src, Tensor& dst, const Tensor& scale) {
    const int8_t* sdata = reinterpret_cast<const int8_t*>(src.data_.get());
    float* ddata = reinterpret_cast<float*>(dst.data_.get());
    const float* scales = reinterpret_cast<const float*>(scale.data_.get());

    size_t rows = src.shape_[0];
    size_t cols = src.shape_[1];

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ddata[i * cols + j] = static_cast<float>(sdata[i * cols + j]) * scales[i];
        }
    }
}

void TensorOps::add(Tensor& a, const Tensor& b) {
    if (a.dtype_ != QuantType::FP32 || b.dtype_ != QuantType::FP32)
        throw std::runtime_error("add: only FP32 supported");
    float* da = a.as_fp32();
    const float* db = reinterpret_cast<const float*>(b.data_.get());
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) da[i] += db[i];
}

void TensorOps::mul(Tensor& a, float scalar) {
    float* data = a.as_fp32();
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) data[i] *= scalar;
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

Tensor TensorOps::concat(const std::vector<Tensor>& tensors, size_t axis) {
    if (tensors.empty()) throw std::runtime_error("Empty tensors");

    auto new_shape = tensors[0].shape_;
    size_t total = 0;
    for (auto& t : tensors) {
        total += t.shape_[axis];
    }
    new_shape[axis] = total;

    Tensor result(new_shape);
    float* out = result.as_fp32();

    size_t offset = 0;
    size_t outer = 1;
    for (size_t i = 0; i < axis; ++i) outer *= tensors[0].shape_[i];

    size_t inner = 1;
    for (size_t i = axis + 1; i < tensors[0].shape_.size(); ++i) inner *= tensors[0].shape_[i];

    for (auto& t : tensors) {
        const float* in = reinterpret_cast<const float*>(t.data_.get());
        size_t dim = t.shape_[axis];

        for (size_t o = 0; o < outer; ++o) {
            memcpy(out + o * total * inner + offset * inner,
                   in + o * dim * inner,
                   dim * inner * sizeof(float));
        }
        offset += dim;
    }

    return result;
}

} // namespace neuroflow
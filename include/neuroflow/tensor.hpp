#ifndef NEUROFLOW_TENSOR_HPP
#define NEUROFLOW_TENSOR_HPP

/**
 * NeuroFlow Core Tensor Engine
 * 
 * 轻量化高性能张量计算库
 * 特点：
 * - SIMD优化 (AVX2/ARM NEON)
 * - 零拷贝内存管理
 * - 内存映射支持
 * - INT8/FP8量化支持
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
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

namespace neuroflow {

// 量化类型枚举
enum class QuantType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    FP8_E4M3 = 4,  // DeepSeek FP8
    FP8_E5M2 = 5,
};

// 内存布局
enum class MemoryLayout : uint8_t {
    ROW_MAJOR = 0,
    COL_MAJOR = 1,
};

/**
 * Tensor类 - 核心数据结构
 * 支持多种量化格式和内存布局
 */
class Tensor {
public:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    QuantType dtype_;
    MemoryLayout layout_;
    std::shared_ptr<uint8_t> data_;
    size_t data_size_;
    bool owns_data_;

#ifdef USE_CUDA
    void* gpu_data_ = nullptr;
    bool on_gpu_ = false;
    bool gpu_dirty_ = false;

    void to_gpu() {
        if (data_size_ == 0 || !data_) return;
        auto& ctx = CudaContext::instance();
        if (!ctx.is_available()) return;
        if (!gpu_data_) {
            gpu_data_ = ctx.alloc(data_size_);
        }
        ctx.copy_h2d(gpu_data_, data_.get(), data_size_);
        on_gpu_ = true;
        gpu_dirty_ = false;
    }

    void to_cpu() {
        if (!on_gpu_ || !gpu_data_ || data_size_ == 0) return;
        auto& ctx = CudaContext::instance();
        if (!ctx.is_available()) return;
        if (!data_) {
            data_ = std::shared_ptr<uint8_t>(new uint8_t[data_size_], std::default_delete<uint8_t[]>());
        }
        ctx.copy_d2h(data_.get(), gpu_data_, data_size_);
        ctx.synchronize();
        gpu_dirty_ = false;
    }

    float* as_gpu_fp32() {
        if (dtype_ != QuantType::FP32) throw std::runtime_error("Not FP32 tensor");
        return reinterpret_cast<float*>(gpu_data_);
    }

    const float* as_gpu_fp32() const {
        if (dtype_ != QuantType::FP32) throw std::runtime_error("Not FP32 tensor");
        return reinterpret_cast<const float*>(gpu_data_);
    }

    bool is_on_gpu() const { return on_gpu_; }

    void gpu_free() {
        if (gpu_data_) {
            auto& ctx = CudaContext::instance();
            if (ctx.is_available()) ctx.free(gpu_data_);
            gpu_data_ = nullptr;
            on_gpu_ = false;
            gpu_dirty_ = false;
        }
    }
#endif
    
    Tensor() : dtype_(QuantType::FP32), layout_(MemoryLayout::ROW_MAJOR), 
               data_size_(0), owns_data_(true) {}

    ~Tensor() {
#ifdef USE_CUDA
        gpu_free();
#endif
    }

    Tensor(const Tensor& other)
        : shape_(other.shape_), strides_(other.strides_), dtype_(other.dtype_),
          layout_(other.layout_), data_(other.data_), data_size_(other.data_size_),
          owns_data_(false)
#ifdef USE_CUDA
          , gpu_data_(other.gpu_data_), on_gpu_(other.on_gpu_), gpu_dirty_(other.gpu_dirty_)
#endif
    {}

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
#ifdef USE_CUDA
            gpu_free();
#endif
            shape_ = other.shape_;
            strides_ = other.strides_;
            dtype_ = other.dtype_;
            layout_ = other.layout_;
            data_ = other.data_;
            data_size_ = other.data_size_;
            owns_data_ = false;
#ifdef USE_CUDA
            gpu_data_ = other.gpu_data_;
            on_gpu_ = other.on_gpu_;
            gpu_dirty_ = other.gpu_dirty_;
#endif
        }
        return *this;
    }

    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
          dtype_(other.dtype_), layout_(other.layout_), data_(std::move(other.data_)),
          data_size_(other.data_size_), owns_data_(other.owns_data_)
#ifdef USE_CUDA
          , gpu_data_(other.gpu_data_), on_gpu_(other.on_gpu_), gpu_dirty_(other.gpu_dirty_)
#endif
    {
        other.data_size_ = 0;
        other.owns_data_ = false;
#ifdef USE_CUDA
        other.gpu_data_ = nullptr;
        other.on_gpu_ = false;
        other.gpu_dirty_ = false;
#endif
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
#ifdef USE_CUDA
            gpu_free();
#endif
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            dtype_ = other.dtype_;
            layout_ = other.layout_;
            data_ = std::move(other.data_);
            data_size_ = other.data_size_;
            owns_data_ = other.owns_data_;
#ifdef USE_CUDA
            gpu_data_ = other.gpu_data_;
            on_gpu_ = other.on_gpu_;
            gpu_dirty_ = other.gpu_dirty_;
            other.gpu_data_ = nullptr;
            other.on_gpu_ = false;
            other.gpu_dirty_ = false;
#endif
            other.data_size_ = 0;
            other.owns_data_ = false;
        }
        return *this;
    }
    
    Tensor(const std::vector<size_t>& dims, QuantType type = QuantType::FP32)
        : shape_(dims), dtype_(type), layout_(MemoryLayout::ROW_MAJOR), owns_data_(true) {
        strides_.resize(dims.size());
        size_t total = 1;
        for (size_t i = dims.size(); i > 0; --i) {
            strides_[i-1] = total;
            total *= dims[i-1];
        }
        data_size_ = total * get_type_size();
        data_ = std::shared_ptr<uint8_t>(new uint8_t[data_size_], std::default_delete<uint8_t[]>());
        memset(data_.get(), 0, data_size_);
    }
    
    // 类型大小
    size_t get_type_size() const {
        switch (dtype_) {
            case QuantType::FP32: return 4;
            case QuantType::FP16: return 2;
            case QuantType::INT8: return 1;
            case QuantType::INT4: return 1; // packed
            case QuantType::FP8_E4M3: return 1;
            case QuantType::FP8_E5M2: return 1;
            default: return 4;
        }
    }
    
    // 总元素数
    size_t numel() const {
        size_t n = 1;
        for (auto d : shape_) n *= d;
        return n;
    }
    
    // FP32数据访问
    float* as_fp32() {
        if (dtype_ != QuantType::FP32) throw std::runtime_error("Not FP32 tensor");
        return reinterpret_cast<float*>(data_.get());
    }
    
    const float* as_fp32() const {
        if (dtype_ != QuantType::FP32) throw std::runtime_error("Not FP32 tensor");
        return reinterpret_cast<const float*>(data_.get());
    }
    
    // 别名，兼容性
    const float* as_fp32_const() const {
        return as_fp32();
    }
    
    // INT8数据访问
    int8_t* as_int8() {
        if (dtype_ != QuantType::INT8) throw std::runtime_error("Not INT8 tensor");
        return reinterpret_cast<int8_t*>(data_.get());
    }
    
    const int8_t* as_int8() const {
        if (dtype_ != QuantType::INT8) throw std::runtime_error("Not INT8 tensor");
        return reinterpret_cast<const int8_t*>(data_.get());
    }
    
    // reshape (零拷贝)
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        Tensor t;
        t.shape_ = new_shape;
        t.dtype_ = dtype_;
        t.layout_ = layout_;
        t.data_ = data_;
        t.data_size_ = data_size_;
        t.owns_data_ = false;
#ifdef USE_CUDA
        t.gpu_data_ = gpu_data_;
        t.on_gpu_ = on_gpu_;
        t.gpu_dirty_ = gpu_dirty_;
#endif
        
        size_t total = 1;
        for (auto d : new_shape) total *= d;
        if (total != numel()) throw std::runtime_error("Invalid reshape");
        
        t.strides_.resize(new_shape.size());
        size_t acc = 1;
        for (size_t i = new_shape.size(); i > 0; --i) {
            t.strides_[i-1] = acc;
            acc *= new_shape[i-1];
        }
        return t;
    }
    
    Tensor clone() const {
        Tensor t(shape_, dtype_);
        memcpy(t.data_.get(), data_.get(), data_size_);
        return t;
    }
};

/**
 * TensorOps - 张量运算 (SIMD优化)
 */
class TensorOps {
public:
    // 矩阵乘法 GEMM (SIMD优化)
    static void gemm(const Tensor& A, const Tensor& B, Tensor& C,
                     bool transA = false, bool transB = false,
                     float alpha = 1.0f, float beta = 0.0f) {
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
    
private:
    // AVX2优化的GEMM
    #ifdef HAS_AVX2
    static void gemm_avx2(const float* a, const float* b, float* c,
                          size_t M, size_t K, size_t N,
                          bool transA, bool transB,
                          float alpha, float beta) {
        // 初始化C
        if (beta != 0.0f) {
            for (size_t i = 0; i < M * N; ++i) c[i] *= beta;
        } else {
            memset(c, 0, M * N * sizeof(float));
        }
        
        // 8-wide SIMD
        const size_t BLOCK = 8;
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; j += BLOCK) {
                __m256 sum = _mm256_setzero_ps();
                
                for (size_t k = 0; k < K; ++k) {
                    float av = transA ? a[k * M + i] : a[i * K + k];
                    __m256 bv;
                    
                    if (transB) {
                        // 手动加载8个元素
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
    
    // ARM NEON优化的GEMM
    #ifdef HAS_NEON
    static void gemm_neon(const float* a, const float* b, float* c,
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
    
    // 普通GEMM
    static void gemm_scalar(const float* a, const float* b, float* c,
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
    
public:
    // 量化GEMM
    static void quantized_gemm(const Tensor& A, const Tensor& B, Tensor& C) {
        // INT8量化矩阵乘法
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
    
public:
    // LayerNorm
    static void layer_norm(Tensor& x, Tensor& weight, Tensor& bias, float eps = 1e-5f) {
        float* data = x.as_fp32();
        float* w = weight.as_fp32();
        float* b = bias.as_fp32();
        
        size_t batch = x.shape_[0];
        size_t dim = x.shape_[1];
        
        for (size_t i = 0; i < batch; ++i) {
            float mean = 0.0f;
            float var = 0.0f;
            
            // 计算均值
            for (size_t j = 0; j < dim; ++j) {
                mean += data[i * dim + j];
            }
            mean /= dim;
            
            // 计算方差
            for (size_t j = 0; j < dim; ++j) {
                float diff = data[i * dim + j] - mean;
                var += diff * diff;
            }
            var /= dim;
            
            // 归一化
            float std = std::sqrt(var + eps);
            for (size_t j = 0; j < dim; ++j) {
                data[i * dim + j] = w[j] * (data[i * dim + j] - mean) / std + b[j];
            }
        }
    }
    
    // GELU激活 (近似)
    static void gelu(Tensor& x) {
        float* data = x.as_fp32();
        size_t n = x.numel();
        
        // GELU近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float SQRT_2_PI = 0.7978845608f;
        const float COEFF = 0.044715f;
        
        for (size_t i = 0; i < n; ++i) {
            float v = data[i];
            float inner = SQRT_2_PI * (v + COEFF * v * v * v);
            data[i] = 0.5f * v * (1.0f + std::tanh(inner));
        }
    }
    
    // Softmax
    static void softmax(Tensor& x, int axis = -1) {
        float* data = x.as_fp32();
        
        if (axis == -1) axis = x.shape_.size() - 1;
        
        size_t outer = 1;
        for (int i = 0; i < axis; ++i) outer *= x.shape_[i];
        
        size_t inner_dim = x.shape_[axis];
        size_t inner = 1;
        for (size_t i = axis + 1; i < x.shape_.size(); ++i) inner *= x.shape_[i];
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                float* row = data + o * inner_dim * inner + i;
                
                // 找最大值
                float max_val = row[0];
                for (size_t d = 1; d < inner_dim; ++d) {
                    max_val = std::max(max_val, row[d * inner]);
                }
                
                // exp和求和
                float sum = 0.0f;
                for (size_t d = 0; d < inner_dim; ++d) {
                    row[d * inner] = std::exp(row[d * inner] - max_val);
                    sum += row[d * inner];
                }
                
                // 归一化
                for (size_t d = 0; d < inner_dim; ++d) {
                    row[d * inner] /= sum;
                }
            }
        }
    }
    
    // Dropout (训练用)
    static void dropout(Tensor& x, float rate, bool training = true) {
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
    
    // 量化: FP32 -> INT8
    static void quantize_int8(const Tensor& src, Tensor& dst, Tensor& scale) {
        if (src.dtype_ != QuantType::FP32) throw std::runtime_error("Source must be FP32");
        
        const float* sdata = reinterpret_cast<const float*>(src.data_.get());
        int8_t* ddata = reinterpret_cast<int8_t*>(dst.data_.get());
        float* scales = reinterpret_cast<float*>(scale.data_.get());
        
        size_t rows = src.shape_[0];
        size_t cols = src.shape_[1];
        
        // 每行单独量化
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
    
    // 反量化: INT8 -> FP32
    static void dequantize_int8(const Tensor& src, Tensor& dst, const Tensor& scale) {
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
    
    // 元素加法
    static void add(Tensor& a, const Tensor& b) {
        if (a.dtype_ != QuantType::FP32 || b.dtype_ != QuantType::FP32)
            throw std::runtime_error("add: only FP32 supported");
        float* da = a.as_fp32();
        const float* db = reinterpret_cast<const float*>(b.data_.get());
        size_t n = a.numel();
        for (size_t i = 0; i < n; ++i) da[i] += db[i];
    }
    
    // 元素乘法
    static void mul(Tensor& a, float scalar) {
        float* data = a.as_fp32();
        size_t n = a.numel();
        for (size_t i = 0; i < n; ++i) data[i] *= scalar;
    }
    
    // 并行GEMM
    static void parallel_gemm(const Tensor& A, const Tensor& B, Tensor& C,
                              bool transA = false, bool transB = false,
                              float alpha = 1.0f, float beta = 0.0f,
                              size_t num_threads = 4);

    // 便捷矩阵乘法
    static Tensor matmul(const Tensor& A, const Tensor& B);

    // 逐元素运算（返回新张量）
    static Tensor elementwise_add(const Tensor& A, const Tensor& B);
    static Tensor elementwise_sub(const Tensor& A, const Tensor& B);
    static Tensor elementwise_mul(const Tensor& A, const Tensor& B);
    static Tensor scalar_mul(const Tensor& A, float s);
    static Tensor broadcast_add(const Tensor& A, const Tensor& B);

    // 转置
    static Tensor transpose2d(const Tensor& A);

    // 填充/拷贝
    static void fill(Tensor& A, float value);
    static void copy_data(Tensor& dst, const Tensor& src);

    // 内积/范数
    static float dot(const Tensor& A, const Tensor& B);
    static float norm2(const Tensor& A);

    // 归约
    static Tensor reduce_sum(const Tensor& A, int axis = -1);

    // 切片/填充
    static Tensor slice(const Tensor& A, size_t dim, size_t start, size_t end);
    static Tensor pad1d(const Tensor& A, size_t left, size_t right, float value = 0.0f);

    // 通用函数应用
    static void apply_inplace(Tensor& A, std::function<float(float)> fn);
    static Tensor apply(const Tensor& A, std::function<float(float)> fn);

    // 常用激活函数
    static Tensor relu(const Tensor& A);
    static Tensor sigmoid(const Tensor& A);
    static Tensor tanh_act(const Tensor& A);
    static Tensor log(const Tensor& A);
    static Tensor exp(const Tensor& A);
    static void clip_inplace(Tensor& A, float min_val, float max_val);

    // concat
    static Tensor concat(const std::vector<Tensor>& tensors, size_t axis = 0) {
        if (tensors.empty()) throw std::runtime_error("Empty tensors");
        
        // 计算新形状
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
};

} // namespace neuroflow

#endif // NEUROFLOW_TENSOR_HPP
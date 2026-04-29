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

#include <vector>
#include <memory>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>

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
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    QuantType dtype;
    MemoryLayout layout;
    std::shared_ptr<uint8_t> data;
    size_t data_size;
    bool owns_data;
    
    // 构造函数
    Tensor() : dtype(QuantType::FP32), layout(MemoryLayout::ROW_MAJOR), 
               data_size(0), owns_data(true) {}
    
    Tensor(const std::vector<size_t>& dims, QuantType type = QuantType::FP32)
        : shape(dims), dtype(type), layout(MemoryLayout::ROW_MAJOR), owns_data(true) {
        strides.resize(dims.size());
        size_t total = 1;
        for (size_t i = dims.size(); i > 0; --i) {
            strides[i-1] = total;
            total *= dims[i-1];
        }
        data_size = total * get_type_size();
        data = std::shared_ptr<uint8_t>(new uint8_t[data_size], std::default_delete<uint8_t[]>());
        memset(data.get(), 0, data_size);
    }
    
    // 类型大小
    size_t get_type_size() const {
        switch (dtype) {
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
        for (auto d : shape) n *= d;
        return n;
    }
    
    // FP32数据访问
    float* as_fp32() {
        if (dtype != QuantType::FP32) throw std::runtime_error("Not FP32 tensor");
        return reinterpret_cast<float*>(data.get());
    }
    
    const float* as_fp32() const {
        if (dtype != QuantType::FP32) throw std::runtime_error("Not FP32 tensor");
        return reinterpret_cast<const float*>(data.get());
    }
    
    // 别名，兼容性
    const float* as_fp32_const() const {
        return as_fp32();
    }
    
    // INT8数据访问
    int8_t* as_int8() {
        if (dtype != QuantType::INT8) throw std::runtime_error("Not INT8 tensor");
        return reinterpret_cast<int8_t*>(data.get());
    }
    
    const int8_t* as_int8() const {
        if (dtype != QuantType::INT8) throw std::runtime_error("Not INT8 tensor");
        return reinterpret_cast<const int8_t*>(data.get());
    }
    
    // reshape (零拷贝)
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        Tensor t;
        t.shape = new_shape;
        t.dtype = dtype;
        t.layout = layout;
        t.data = data;
        t.data_size = data_size;
        t.owns_data = false;
        
        size_t total = 1;
        for (auto d : new_shape) total *= d;
        if (total != numel()) throw std::runtime_error("Invalid reshape");
        
        t.strides.resize(new_shape.size());
        size_t acc = 1;
        for (size_t i = new_shape.size(); i > 0; --i) {
            t.strides[i-1] = acc;
            acc *= new_shape[i-1];
        }
        return t;
    }
    
    // 拷贝
    Tensor clone() const {
        Tensor t(shape, dtype);
        memcpy(t.data.get(), data.get(), data_size);
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
        // 维度检查
        size_t K_A = transA ? A.shape[0] : A.shape[1];
        size_t K_B = transB ? B.shape[1] : B.shape[0];
        if (K_A != K_B) {
            throw std::runtime_error("GEMM dimension mismatch: K dimensions don't match");
        }
        
        if (A.dtype != QuantType::FP32 || B.dtype != QuantType::FP32) {
            quantized_gemm(A, B, C);
            return;
        }
        
        auto a = reinterpret_cast<float*>(A.data.get());
        auto b = reinterpret_cast<float*>(B.data.get());
        auto c = reinterpret_cast<float*>(C.data.get());
        
        size_t M = transA ? A.shape[1] : A.shape[0];
        size_t K = transA ? A.shape[0] : A.shape[1];
        size_t N = transB ? B.shape[0] : B.shape[1];
        
        // SIMD优化的GEMM
        #ifdef HAS_AVX2
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
                    
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_set1_ps(av), bv));
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
        if (A.dtype == QuantType::INT8 && B.dtype == QuantType::FP32) {
            int8_t* a = reinterpret_cast<int8_t*>(A.data.get());
            float* b = reinterpret_cast<float*>(B.data.get());
            float* c = reinterpret_cast<float*>(C.data.get());
            
            size_t M = A.shape[0];
            size_t K = A.shape[1];
            size_t N = B.shape[1];
            
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
        
        size_t batch = x.shape[0];
        size_t dim = x.shape[1];
        
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
        
        if (axis == -1) axis = x.shape.size() - 1;
        
        size_t outer = 1;
        for (int i = 0; i < axis; ++i) outer *= x.shape[i];
        
        size_t inner_dim = x.shape[axis];
        size_t inner = 1;
        for (size_t i = axis + 1; i < x.shape.size(); ++i) inner *= x.shape[i];
        
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
        
        // 简单随机mask (实际应用应使用更好的随机数生成器)
        for (size_t i = 0; i < n; ++i) {
            float rand_val = static_cast<float>(std::rand()) / RAND_MAX;
            if (rand_val < rate) {
                data[i] = 0.0f;
            } else {
                data[i] *= scale;
            }
        }
    }
    
    // 量化: FP32 -> INT8
    static void quantize_int8(const Tensor& src, Tensor& dst, Tensor& scale) {
        if (src.dtype != QuantType::FP32) throw std::runtime_error("Source must be FP32");
        
        float* sdata = reinterpret_cast<float*>(src.data.get());
        int8_t* ddata = reinterpret_cast<int8_t*>(dst.data.get());
        float* scales = reinterpret_cast<float*>(scale.data.get());
        
        size_t rows = src.shape[0];
        size_t cols = src.shape[1];
        
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
        int8_t* sdata = reinterpret_cast<int8_t*>(src.data.get());
        float* ddata = reinterpret_cast<float*>(dst.data.get());
        float* scales = reinterpret_cast<float*>(scale.data.get());
        
        size_t rows = src.shape[0];
        size_t cols = src.shape[1];
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                ddata[i * cols + j] = static_cast<float>(sdata[i * cols + j]) * scales[i];
            }
        }
    }
    
    // 元素加法
    static void add(Tensor& a, const Tensor& b) {
        float* da = a.as_fp32();
        float* db = reinterpret_cast<float*>(b.data.get());
        size_t n = a.numel();
        for (size_t i = 0; i < n; ++i) da[i] += db[i];
    }
    
    // 元素乘法
    static void mul(Tensor& a, float scalar) {
        float* data = a.as_fp32();
        size_t n = a.numel();
        for (size_t i = 0; i < n; ++i) data[i] *= scalar;
    }
    
    // concat
    static Tensor concat(const std::vector<Tensor>& tensors, size_t axis = 0) {
        if (tensors.empty()) throw std::runtime_error("Empty tensors");
        
        // 计算新形状
        auto new_shape = tensors[0].shape;
        size_t total = 0;
        for (auto& t : tensors) {
            total += t.shape[axis];
        }
        new_shape[axis] = total;
        
        Tensor result(new_shape);
        float* out = result.as_fp32();
        
        size_t offset = 0;
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) outer *= tensors[0].shape[i];
        
        size_t inner = 1;
        for (size_t i = axis + 1; i < tensors[0].shape.size(); ++i) inner *= tensors[0].shape[i];
        
        for (auto& t : tensors) {
            float* in = reinterpret_cast<float*>(t.data.get());
            size_t dim = t.shape[axis];
            
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
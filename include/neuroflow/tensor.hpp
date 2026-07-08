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

} // namespace neuroflow

#include "tensor_ops.hpp"

#endif // NEUROFLOW_TENSOR_HPP
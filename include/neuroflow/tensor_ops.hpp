#ifndef NEUROFLOW_TENSOR_OPS_HPP
#define NEUROFLOW_TENSOR_OPS_HPP


namespace neuroflow {

class TensorOps {
public:
    static void gemm(const Tensor& A, const Tensor& B, Tensor& C,
                     bool transA = false, bool transB = false,
                     float alpha = 1.0f, float beta = 0.0f);

    static void quantized_gemm(const Tensor& A, const Tensor& B, Tensor& C);

    static void layer_norm(Tensor& x, Tensor& weight, Tensor& bias, float eps = 1e-5f);

    static void gelu(Tensor& x);

    static void softmax(Tensor& x, int axis = -1);

    static void dropout(Tensor& x, float rate, bool training = true);

    static void quantize_int8(const Tensor& src, Tensor& dst, Tensor& scale);

    static void dequantize_int8(const Tensor& src, Tensor& dst, const Tensor& scale);

    static void add(Tensor& a, const Tensor& b);

    static void mul(Tensor& a, float scalar);

    static void parallel_gemm(const Tensor& A, const Tensor& B, Tensor& C,
                              bool transA = false, bool transB = false,
                              float alpha = 1.0f, float beta = 0.0f,
                              size_t num_threads = 4);

    static Tensor matmul(const Tensor& A, const Tensor& B);

    static Tensor elementwise_add(const Tensor& A, const Tensor& B);
    static Tensor elementwise_sub(const Tensor& A, const Tensor& B);
    static Tensor elementwise_mul(const Tensor& A, const Tensor& B);
    static Tensor scalar_mul(const Tensor& A, float s);
    static Tensor broadcast_add(const Tensor& A, const Tensor& B);

    static Tensor transpose2d(const Tensor& A);

    static void fill(Tensor& A, float value);
    static void copy_data(Tensor& dst, const Tensor& src);

    static float dot(const Tensor& A, const Tensor& B);
    static float norm2(const Tensor& A);

    static Tensor reduce_sum(const Tensor& A, int axis = -1);

    static Tensor slice(const Tensor& A, size_t dim, size_t start, size_t end);
    static Tensor pad1d(const Tensor& A, size_t left, size_t right, float value = 0.0f);

    static void apply_inplace(Tensor& A, std::function<float(float)> fn);
    static Tensor apply(const Tensor& A, std::function<float(float)> fn);

    static Tensor relu(const Tensor& A);
    static Tensor sigmoid(const Tensor& A);
    static Tensor tanh_act(const Tensor& A);
    static Tensor log(const Tensor& A);
    static Tensor exp(const Tensor& A);
    static void clip_inplace(Tensor& A, float min_val, float max_val);

    static Tensor concat(const std::vector<Tensor>& tensors, size_t axis = 0);

private:
    static void gemm_scalar(const float* a, const float* b, float* c,
                            size_t M, size_t K, size_t N,
                            bool transA, bool transB,
                            float alpha, float beta);

#ifdef HAS_AVX2
    static void gemm_avx2(const float* a, const float* b, float* c,
                          size_t M, size_t K, size_t N,
                          bool transA, bool transB,
                          float alpha, float beta);
#endif

#ifdef HAS_NEON
    static void gemm_neon(const float* a, const float* b, float* c,
                          size_t M, size_t K, size_t N,
                          bool transA, bool transB,
                          float alpha, float beta);
#endif
};

} // namespace neuroflow

#endif // NEUROFLOW_TENSOR_OPS_HPP
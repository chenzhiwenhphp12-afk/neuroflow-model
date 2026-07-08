#include "neuroflow/causal_lm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

namespace neuroflow {

CausalSelfAttention::CausalSelfAttention(size_t d_model, size_t n_q_heads, size_t n_kv_heads,
                                           bool use_rope, size_t max_seq_len, bool use_qk_norm)
    : n_q_heads_(n_q_heads), n_kv_heads_(n_kv_heads),
      n_rep_(n_q_heads / n_kv_heads), d_model_(d_model),
      head_dim_(d_model / n_q_heads),
      use_rope_(use_rope), max_seq_len_(max_seq_len), use_qk_norm_(use_qk_norm) {
    if (n_q_heads % n_kv_heads != 0) {
        throw std::invalid_argument("n_q_heads must be divisible by n_kv_heads");
    }
    size_t q_dim = n_q_heads * head_dim_;
    size_t kv_dim = n_kv_heads * head_dim_;
    w_q = std::make_shared<Linear>(d_model, q_dim, true);
    w_k = std::make_shared<Linear>(d_model, kv_dim, true);
    w_v = std::make_shared<Linear>(d_model, kv_dim, true);
    w_out = std::make_shared<Linear>(q_dim, d_model, true);
    norm = std::make_shared<LayerNorm>(d_model);
    if (use_rope_) {
        rope_ = std::make_unique<RoPE>(head_dim_, max_seq_len);
    }
    if (use_qk_norm_) {
        q_norm_ = std::make_unique<RMSNorm>(head_dim_);
        k_norm_ = std::make_unique<RMSNorm>(head_dim_);
    }
}

Tensor CausalSelfAttention::forward(const Tensor& x, const Tensor* padding_mask) {
    if (training_mode_) {
        cache_.input = x.clone();
    }
    size_t seq_len = x.shape_[0];

    cache_.q_proj = w_q->forward(x);
    cache_.k_proj = w_k->forward(x);
    cache_.v_proj = w_v->forward(x);

    if (use_rope_ && rope_) {
        rope_->apply_single(cache_.q_proj, seq_len, n_q_heads_, 0);
        rope_->apply_single(cache_.k_proj, seq_len, n_kv_heads_, 0);
    }

    if (use_qk_norm_ && q_norm_ && k_norm_) {
        float* qp = cache_.q_proj.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t h = 0; h < n_q_heads_; ++h) {
                Tensor q_vec({1, head_dim_}, QuantType::FP32);
                float* qvp = q_vec.as_fp32();
                for (size_t d = 0; d < head_dim_; ++d) {
                    qvp[d] = qp[i * n_q_heads_ * head_dim_ + h * head_dim_ + d];
                }
                q_vec = q_norm_->forward(q_vec);
                const float* nqp = q_vec.as_fp32();
                for (size_t d = 0; d < head_dim_; ++d) {
                    qp[i * n_q_heads_ * head_dim_ + h * head_dim_ + d] = nqp[d];
                }
            }
        }
        float* kp = cache_.k_proj.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t h = 0; h < n_kv_heads_; ++h) {
                Tensor k_vec({1, head_dim_}, QuantType::FP32);
                float* kvp = k_vec.as_fp32();
                for (size_t d = 0; d < head_dim_; ++d) {
                    kvp[d] = kp[i * n_kv_heads_ * head_dim_ + h * head_dim_ + d];
                }
                k_vec = k_norm_->forward(k_vec);
                const float* nkp = k_vec.as_fp32();
                for (size_t d = 0; d < head_dim_; ++d) {
                    kp[i * n_kv_heads_ * head_dim_ + h * head_dim_ + d] = nkp[d];
                }
            }
        }
    }

    cache_.attn_weights = Tensor({n_q_heads_, seq_len, seq_len}, QuantType::FP32);
    cache_.attn_output = Tensor({seq_len, n_q_heads_ * head_dim_}, QuantType::FP32);

    float scale = 1.0f / (std::sqrt(static_cast<float>(head_dim_)) * yarn_temp_scale_);

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && x.is_on_gpu()) {
        cache_.attn_weights.to_gpu();
        cache_.attn_output.to_gpu();
        const float* qp = cache_.q_proj.as_gpu_fp32();
        const float* kp = cache_.k_proj.as_gpu_fp32();
        const float* vp = cache_.v_proj.as_gpu_fp32();
        float* aw = cache_.attn_weights.as_gpu_fp32();
        float* ao = cache_.attn_output.as_gpu_fp32();
        auto stream = CudaContext::instance().stream();

        launch_fill_zero(ao, seq_len * n_q_heads_ * head_dim_, stream);

        for (size_t h_q = 0; h_q < n_q_heads_; ++h_q) {
            size_t kv_h = h_q / n_rep_;

            Tensor Q_h({seq_len, head_dim_}, QuantType::FP32);
            Tensor K_h({seq_len, head_dim_}, QuantType::FP32);
            Tensor V_h({seq_len, head_dim_}, QuantType::FP32);
            Q_h.to_gpu(); K_h.to_gpu(); V_h.to_gpu();

            launch_extract_head(Q_h.as_gpu_fp32(), qp, seq_len, n_q_heads_, head_dim_, h_q, stream);
            launch_extract_head(K_h.as_gpu_fp32(), kp, seq_len, n_kv_heads_, head_dim_, kv_h, stream);
            launch_extract_head(V_h.as_gpu_fp32(), vp, seq_len, n_kv_heads_, head_dim_, kv_h, stream);

            float* aw_h = aw + h_q * seq_len * seq_len;
            CudaContext::instance().sgemm_rowmajor(false, true,
                static_cast<int>(seq_len), static_cast<int>(seq_len), static_cast<int>(head_dim_),
                scale, Q_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                K_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                0.0f, aw_h, static_cast<int>(seq_len));

            if (padding_mask && padding_mask->is_on_gpu()) {
                launch_fused_causal_padding_mask(aw_h, padding_mask->as_gpu_fp32(),
                    static_cast<int>(seq_len), 1, stream);
                launch_softmax(aw_h, static_cast<int>(seq_len), static_cast<int>(seq_len), stream);
            } else {
                launch_causal_softmax(aw_h, seq_len, stream);
            }

            float* ao_ptr = ao + h_q * head_dim_;
            for (size_t i = 0; i < seq_len; ++i) {
                CudaContext::instance().sgemm_rowmajor(false, false,
                    1, static_cast<int>(head_dim_), static_cast<int>(seq_len),
                    1.0f, aw_h + i * seq_len, static_cast<int>(seq_len),
                    V_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                    0.0f, ao_ptr + i * n_q_heads_ * head_dim_, static_cast<int>(n_q_heads_ * head_dim_));
            }

            cache_.attn_weights.gpu_dirty_ = true;
            cache_.attn_output.gpu_dirty_ = true;
        }

        if (padding_mask && padding_mask->is_on_gpu()) {
            const float* pm = padding_mask->as_gpu_fp32();
            float* ao = cache_.attn_output.as_gpu_fp32();
            for (size_t i = 0; i < seq_len; ++i) {
                float pm_val = 0.0f;
                cudaMemcpyAsync(&pm_val, pm + i, sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                if (pm_val == 0.0f) {
                    launch_fill_zero(ao + i * n_q_heads_ * head_dim_, n_q_heads_ * head_dim_, stream);
                }
            }
        }

        if (training_mode_) {
            cache_.w_out_input = cache_.attn_output.clone();
        }
        Tensor projected = w_out->forward(cache_.attn_output);

        cache_.residual = Tensor({seq_len, d_model_}, QuantType::FP32);
        cache_.residual.to_gpu();
        launch_add(cache_.residual.as_gpu_fp32(), projected.as_gpu_fp32(), cache_.input.as_gpu_fp32(), seq_len * d_model_, stream);
        cache_.residual.gpu_dirty_ = true;

        return norm->forward(cache_.residual);
    }
#endif

    const float* qp = cache_.q_proj.as_fp32();
    const float* kp = cache_.k_proj.as_fp32();
    const float* vp = cache_.v_proj.as_fp32();
    float* aw = cache_.attn_weights.as_fp32();
    float* ao = cache_.attn_output.as_fp32();
    memset(ao, 0, cache_.attn_output.data_size_);
    const float* xp = x.as_fp32();
    const float* pm = (padding_mask && padding_mask->numel() > 0) ? padding_mask->as_fp32() : nullptr;

    for (size_t h_q = 0; h_q < n_q_heads_; ++h_q) {
        size_t kv_h = h_q / n_rep_;
        size_t q_stride = n_q_heads_ * head_dim_;
        size_t kv_stride = n_kv_heads_ * head_dim_;

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                if (pm && pm[j] == 0.0f) {
                    aw[h_q * seq_len * seq_len + i * seq_len + j] = -1e30f;
                } else {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim_; ++d) {
                        dot += qp[i * q_stride + h_q * head_dim_ + d]
                             * kp[j * kv_stride + kv_h * head_dim_ + d];
                    }
                    aw[h_q * seq_len * seq_len + i * seq_len + j] = dot * scale;
                }
            }
            for (size_t j = i + 1; j < seq_len; ++j) {
                aw[h_q * seq_len * seq_len + i * seq_len + j] = -1e30f;
            }
        }

        for (size_t i = 0; i < seq_len; ++i) {
            if (pm && pm[i] == 0.0f) {
                for (size_t j = 0; j < seq_len; ++j) {
                    aw[h_q * seq_len * seq_len + i * seq_len + j] = 0.0f;
                }
                continue;
            }
            float max_val = -1e30f;
            for (size_t j = 0; j <= i; ++j) {
                float v = aw[h_q * seq_len * seq_len + i * seq_len + j];
                if (v > max_val) max_val = v;
            }
            float sum = 0.0f;
            for (size_t j = 0; j <= i; ++j) {
                float v = aw[h_q * seq_len * seq_len + i * seq_len + j];
                if (v > -1e29f) {
                    aw[h_q * seq_len * seq_len + i * seq_len + j] = std::exp(v - max_val);
                    sum += aw[h_q * seq_len * seq_len + i * seq_len + j];
                } else {
                    aw[h_q * seq_len * seq_len + i * seq_len + j] = 0.0f;
                }
            }
            if (sum > 0.0f) {
                for (size_t j = 0; j <= i; ++j) {
                    aw[h_q * seq_len * seq_len + i * seq_len + j] /= sum;
                }
            }
            for (size_t j = i + 1; j < seq_len; ++j) {
                aw[h_q * seq_len * seq_len + i * seq_len + j] = 0.0f;
            }
        }

        for (size_t i = 0; i < seq_len; ++i) {
            if (pm && pm[i] == 0.0f) continue;
            for (size_t d = 0; d < head_dim_; ++d) {
                float val = 0.0f;
                for (size_t j = 0; j <= i; ++j) {
                    val += aw[h_q * seq_len * seq_len + i * seq_len + j] * vp[j * kv_stride + kv_h * head_dim_ + d];
                }
                ao[i * q_stride + h_q * head_dim_ + d] = val;
            }
        }
    }

    if (training_mode_) {
        cache_.w_out_input = cache_.attn_output.clone();
    }
    Tensor projected = w_out->forward(cache_.attn_output);

    cache_.residual = Tensor({seq_len, d_model_}, QuantType::FP32);
    float* rp = cache_.residual.as_fp32();
    const float* pp = projected.as_fp32();
    const float* xpp = xp;
    for (size_t i = 0; i < seq_len * d_model_; ++i) {
        rp[i] = pp[i] + xpp[i];
    }

    return norm->forward(cache_.residual);
}


CausalSelfAttention::Gradients CausalSelfAttention::backward(const Tensor& output_grad) {
    Gradients grads;
    size_t seq_len = cache_.input.shape_[0];
    const float* og = output_grad.as_fp32();
    const float* aw = cache_.attn_weights.as_fp32();
    const float* qp = cache_.q_proj.as_fp32();
    const float* kp = cache_.k_proj.as_fp32();
    const float* vp = cache_.v_proj.as_fp32();

    Tensor residual_grad = layernorm_backward_impl(cache_.residual, norm->weight, output_grad);

    const float* rg = residual_grad.as_fp32();

    Tensor proj_grad({seq_len, n_q_heads_ * head_dim_}, QuantType::FP32);
    float* pg = proj_grad.as_fp32();
    memcpy(pg, rg, proj_grad.data_size_);

    grads.w_out_weight_grad = linear_backward_weight_impl(cache_.w_out_input, proj_grad);
    grads.w_out_bias_grad = bias_backward_impl(proj_grad);

    Tensor attn_out_grad = linear_backward_input_impl(proj_grad, w_out->weight);

    const float* aog = attn_out_grad.as_fp32();

    size_t q_dim = n_q_heads_ * head_dim_;
    size_t kv_dim = n_kv_heads_ * head_dim_;

    Tensor d_Q({seq_len, q_dim}, QuantType::FP32);
    Tensor d_K({seq_len, kv_dim}, QuantType::FP32);
    Tensor d_V({seq_len, kv_dim}, QuantType::FP32);
    float* dqp = d_Q.as_fp32();
    float* dkp = d_K.as_fp32();
    float* dvp = d_V.as_fp32();
    memset(dqp, 0, d_Q.data_size_);
    memset(dkp, 0, d_K.data_size_);
    memset(dvp, 0, d_V.data_size_);

    float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (size_t h_q = 0; h_q < n_q_heads_; ++h_q) {
        size_t kv_h = h_q / n_rep_;

        Tensor Q_h({seq_len, head_dim_}, QuantType::FP32);
        Tensor K_h({seq_len, head_dim_}, QuantType::FP32);
        Tensor V_h({seq_len, head_dim_}, QuantType::FP32);
        float* qh = Q_h.as_fp32();
        float* kh = K_h.as_fp32();
        float* vh = V_h.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            memcpy(qh + i * head_dim_, qp + i * q_dim + h_q * head_dim_, head_dim_ * sizeof(float));
            memcpy(kh + i * head_dim_, kp + i * kv_dim + kv_h * head_dim_, head_dim_ * sizeof(float));
            memcpy(vh + i * head_dim_, vp + i * kv_dim + kv_h * head_dim_, head_dim_ * sizeof(float));
        }

        Tensor aog_h({seq_len, head_dim_}, QuantType::FP32);
        float* aogh = aog_h.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            memcpy(aogh + i * head_dim_, aog + i * q_dim + h_q * head_dim_, head_dim_ * sizeof(float));
        }

        Tensor d_attn_weights({seq_len, seq_len}, QuantType::FP32);
        float* daw = d_attn_weights.as_fp32();
        memset(daw, 0, d_attn_weights.data_size_);

#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && cache_.q_proj.is_on_gpu()) {
            d_attn_weights.to_gpu();
            CudaContext::instance().sgemm_rowmajor(false, true,
                static_cast<int>(seq_len), static_cast<int>(seq_len), static_cast<int>(head_dim_),
                1.0f, aog_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                V_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                0.0f, d_attn_weights.as_gpu_fp32(), static_cast<int>(seq_len));
            launch_causal_mask_zero(d_attn_weights.as_gpu_fp32(), seq_len, CudaContext::instance().stream());
            d_attn_weights.gpu_dirty_ = true;
        } else
#endif
#ifdef USE_CBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim_,
                    1.0f, aogh, head_dim_, vh, head_dim_,
                    0.0f, daw, seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = i + 1; j < seq_len; ++j) {
                daw[i * seq_len + j] = 0.0f;
            }
        }
#else
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                float sum = 0.0f;
                for (size_t d = 0; d < head_dim_; ++d) {
                    sum += aogh[i * head_dim_ + d] * vh[j * head_dim_ + d];
                }
                daw[i * seq_len + j] = sum;
            }
        }
#endif

        Tensor d_V_h({seq_len, head_dim_}, QuantType::FP32);
        float* dvh = d_V_h.as_fp32();
        memset(dvh, 0, d_V_h.data_size_);
        {
            const float* aw_h = aw + h_q * seq_len * seq_len;
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    float s = aw_h[i * seq_len + j];
                    for (size_t d = 0; d < head_dim_; ++d) {
                        dvh[j * head_dim_ + d] += s * aogh[i * head_dim_ + d];
                    }
                }
            }
        }

        Tensor d_scores({seq_len, seq_len}, QuantType::FP32);
        float* dsp = d_scores.as_fp32();
        memset(dsp, 0, d_scores.data_size_);
        {
            const float* aw_h = aw + h_q * seq_len * seq_len;
            for (size_t i = 0; i < seq_len; ++i) {
                float dot = 0.0f;
                for (size_t k = 0; k <= i; ++k) {
                    dot += aw_h[i * seq_len + k] * daw[i * seq_len + k];
                }
                for (size_t j = 0; j <= i; ++j) {
                    float s = aw_h[i * seq_len + j];
                    dsp[i * seq_len + j] = s * (daw[i * seq_len + j] - dot) * inv_scale;
                }
            }
        }

        Tensor d_Q_h({seq_len, head_dim_}, QuantType::FP32);
        Tensor d_K_h({seq_len, head_dim_}, QuantType::FP32);
#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && cache_.q_proj.is_on_gpu()) {
            d_Q_h.to_gpu(); d_K_h.to_gpu();
            CudaContext::instance().sgemm_rowmajor(false, false,
                static_cast<int>(seq_len), static_cast<int>(head_dim_), static_cast<int>(seq_len),
                1.0f, d_scores.as_gpu_fp32(), static_cast<int>(seq_len),
                K_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                0.0f, d_Q_h.as_gpu_fp32(), static_cast<int>(head_dim_));
            CudaContext::instance().sgemm_rowmajor(true, false,
                static_cast<int>(seq_len), static_cast<int>(head_dim_), static_cast<int>(seq_len),
                1.0f, d_scores.as_gpu_fp32(), static_cast<int>(seq_len),
                Q_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                0.0f, d_K_h.as_gpu_fp32(), static_cast<int>(head_dim_));
            d_Q_h.gpu_dirty_ = true;
            d_K_h.gpu_dirty_ = true;
        } else
#endif
#ifdef USE_CBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim_, seq_len,
                    1.0f, dsp, seq_len, kh, head_dim_,
                    0.0f, d_Q_h.as_fp32(), head_dim_);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    seq_len, head_dim_, seq_len,
                    1.0f, dsp, seq_len, qh, head_dim_,
                    0.0f, d_K_h.as_fp32(), head_dim_);
#else
        {
            float* dqh = d_Q_h.as_fp32();
            float* dkh = d_K_h.as_fp32();
            memset(dqh, 0, d_Q_h.data_size_);
            memset(dkh, 0, d_K_h.data_size_);
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    float ds = dsp[i * seq_len + j];
                    for (size_t d = 0; d < head_dim_; ++d) {
                        dqh[i * head_dim_ + d] += ds * kh[j * head_dim_ + d];
                        dkh[j * head_dim_ + d] += ds * qh[i * head_dim_ + d];
                    }
                }
            }
        }
#endif

        {
            const float* dqh = d_Q_h.as_fp32();
            const float* dkh = d_K_h.as_fp32();
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    dqp[i * q_dim + h_q * head_dim_ + d] += dqh[i * head_dim_ + d];
                    dkp[i * kv_dim + kv_h * head_dim_ + d] += dkh[i * head_dim_ + d];
                    dvp[i * kv_dim + kv_h * head_dim_ + d] += dvh[i * head_dim_ + d];
                }
            }
        }
    }

    grads.w_q_weight_grad = linear_backward_weight_impl(cache_.input, d_Q);
    grads.w_q_bias_grad = bias_backward_impl(d_Q);
    grads.w_k_weight_grad = linear_backward_weight_impl(cache_.input, d_K);
    grads.w_k_bias_grad = bias_backward_impl(d_K);
    grads.w_v_weight_grad = linear_backward_weight_impl(cache_.input, d_V);
    grads.w_v_bias_grad = bias_backward_impl(d_V);

    Tensor q_input_grad = linear_backward_input_impl(d_Q, w_q->weight);
    Tensor k_input_grad = linear_backward_input_impl(d_K, w_k->weight);
    Tensor v_input_grad = linear_backward_input_impl(d_V, w_v->weight);

    grads.input_grad = Tensor({seq_len, d_model_}, QuantType::FP32);
    float* ig = grads.input_grad.as_fp32();
    const float* qig = q_input_grad.as_fp32();
    const float* kig = k_input_grad.as_fp32();
    const float* vig = v_input_grad.as_fp32();
    for (size_t i = 0; i < seq_len * d_model_; ++i) {
        ig[i] = pg[i] + qig[i] + kig[i] + vig[i];
    }

    return grads;
}

Tensor CausalSelfAttention::layernorm_backward_impl(const Tensor& input, const Tensor& weight, const Tensor& output_grad, float eps) {
    size_t seq_len = input.shape_[0];
    size_t dim = input.shape_[1];
    Tensor input_grad({seq_len, dim}, QuantType::FP32);
    const float* inp = input.as_fp32();
    const float* w = weight.as_fp32();
    const float* og = output_grad.as_fp32();
    float* ig = input_grad.as_fp32();
    for (size_t i = 0; i < seq_len; ++i) {
        float mean = 0.0f;
        for (size_t d = 0; d < dim; ++d) mean += inp[i * dim + d];
        mean /= dim;
        float var = 0.0f;
        for (size_t d = 0; d < dim; ++d) { float diff = inp[i * dim + d] - mean; var += diff * diff; }
        var /= dim;
        float inv_std = 1.0f / std::sqrt(var + eps);
        float sum_gn = 0.0f, sum_gnx = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float norm = (inp[i * dim + d] - mean) * inv_std;
            float gn = og[i * dim + d] * w[d];
            sum_gn += gn; sum_gnx += gn * norm;
        }
        for (size_t d = 0; d < dim; ++d) {
            float norm = (inp[i * dim + d] - mean) * inv_std;
            float gn = og[i * dim + d] * w[d];
            ig[i * dim + d] = inv_std * (gn - sum_gn / dim - norm * sum_gnx / dim);
        }
    }
    return input_grad;
}

Tensor CausalSelfAttention::linear_backward_weight_impl(const Tensor& input, const Tensor& output_grad) {
    size_t batch = input.shape_[0];
    size_t in_f = input.shape_[1];
    size_t out_f = output_grad.shape_[1];
    Tensor weight_grad({out_f, in_f}, QuantType::FP32);
#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && input.is_on_gpu()) {
        weight_grad.to_gpu();
        CudaContext::instance().sgemm_rowmajor(true, false,
            static_cast<int>(out_f), static_cast<int>(in_f), static_cast<int>(batch),
            1.0f / batch, output_grad.as_gpu_fp32(), static_cast<int>(out_f),
            input.as_gpu_fp32(), static_cast<int>(in_f),
            0.0f, weight_grad.as_gpu_fp32(), static_cast<int>(in_f));
        weight_grad.gpu_dirty_ = true;
        return weight_grad;
    }
#endif
#ifdef USE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                out_f, in_f, batch,
                1.0f / batch, output_grad.as_fp32(), out_f, input.as_fp32(), in_f,
                0.0f, weight_grad.as_fp32(), in_f);
#else
    const float* inp = input.as_fp32();
    const float* og = output_grad.as_fp32();
    float* wg = weight_grad.as_fp32();
    for (size_t i = 0; i < out_f; ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch; ++b) sum += og[b * out_f + i] * inp[b * in_f + j];
            wg[i * in_f + j] = sum / batch;
        }
    }
#endif
    return weight_grad;
}

Tensor CausalSelfAttention::linear_backward_input_impl(const Tensor& output_grad, const Tensor& weight) {
    size_t batch = output_grad.shape_[0];
    size_t out_f = output_grad.shape_[1];
    size_t in_f = weight.shape_[1];
    Tensor input_grad({batch, in_f}, QuantType::FP32);
#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && output_grad.is_on_gpu()) {
        input_grad.to_gpu();
        CudaContext::instance().sgemm_rowmajor(false, false,
            static_cast<int>(batch), static_cast<int>(in_f), static_cast<int>(out_f),
            1.0f, output_grad.as_gpu_fp32(), static_cast<int>(out_f),
            weight.as_gpu_fp32(), static_cast<int>(in_f),
            0.0f, input_grad.as_gpu_fp32(), static_cast<int>(in_f));
        input_grad.gpu_dirty_ = true;
        return input_grad;
    }
#endif
#ifdef USE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch, in_f, out_f,
                1.0f, output_grad.as_fp32(), out_f, weight.as_fp32(), in_f,
                0.0f, input_grad.as_fp32(), in_f);
#else
    const float* og = output_grad.as_fp32();
    const float* w = weight.as_fp32();
    float* ig = input_grad.as_fp32();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t j = 0; j < in_f; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < out_f; ++i) sum += og[b * out_f + i] * w[i * in_f + j];
            ig[b * in_f + j] = sum;
        }
    }
#endif
    return input_grad;
}

Tensor CausalSelfAttention::bias_backward_impl(const Tensor& output_grad) {
    size_t batch = output_grad.shape_[0];
    size_t dim = output_grad.shape_[1];
    Tensor grad({dim}, QuantType::FP32);
    const float* og = output_grad.as_fp32();
    float* g = grad.as_fp32();
    for (size_t j = 0; j < dim; ++j) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch; ++b) sum += og[b * dim + j];
        g[j] = sum / batch;
    }
    return grad;
}

void CausalLMHead::train() {
    training_mode_ = true;
    mode_set_ = true;
    for (auto& attn : attn_layers_) {
        attn->train();
    }
    if (swiglu_) swiglu_->training_mode_ = true;
}

void CausalLMHead::eval() {
    training_mode_ = false;
    mode_set_ = true;
    for (auto& attn : attn_layers_) {
        attn->eval();
    }
    if (swiglu_) swiglu_->training_mode_ = false;
}

void CausalLMHead::set_yarn_scale(float scale_factor) {
    if (scale_factor <= 1.0f) return;
    float temp_scale = std::sqrt(std::log(scale_factor)) + 1.0f;
    for (auto& attn : attn_layers_) {
        if (attn->rope_) {
            attn->rope_->set_yarn_scale(scale_factor);
        }
        attn->yarn_temp_scale_ = temp_scale;
    }
    std::cerr << "[YaRN] CausalLMHead: scale_factor=" << scale_factor
              << ", temp_scale=" << temp_scale << std::endl;
}

void CausalLMHead::tie_weights() {
    if (config_.weight_tying && w_out_) {
        w_out_->weight.shape_ = w_embed_.shape_;
        w_out_->weight.dtype_ = w_embed_.dtype_;
        w_out_->weight.layout_ = w_embed_.layout_;
        w_out_->weight.data_ = w_embed_.data_;
        w_out_->weight.data_size_ = w_embed_.data_size_;
        w_out_->weight.owns_data_ = false;
        w_out_->weight.strides_ = w_embed_.strides_;
    }
}

CausalLMHead::CausalLMHead(const CausalLMConfig& config) : config_(config), sliding_window_drops_(0) {
    w_embed_ = Tensor({config_.vocab_size, config_.d_model}, QuantType::FP32);
    float* we = w_embed_.as_fp32();
    float embed_scale = 1.0f / std::sqrt(static_cast<float>(config_.d_model));
    std::mt19937 embed_rng(config_.vocab_size * 31 + config_.d_model);
    std::uniform_real_distribution<float> dist(-embed_scale, embed_scale);
    for (size_t i = 0; i < w_embed_.numel(); ++i) we[i] = dist(embed_rng);

    w_pos_ = Tensor({config_.max_seq_len, config_.d_model}, QuantType::FP32);
    float* wp = w_pos_.as_fp32();
    for (size_t pos = 0; pos < config_.max_seq_len; ++pos) {
        for (size_t d = 0; d < config_.d_model; ++d) {
            float angle = static_cast<float>(pos) / std::pow(10000.0f, static_cast<float>(d % 2 ? d - 1 : d) / static_cast<float>(config_.d_model));
            wp[pos * config_.d_model + d] = (d % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }

    size_t gate_in = config_.causal_window_size * config_.d_model;
    dw_kernel_ = Tensor({config_.d_model, config_.causal_window_size}, QuantType::FP32);
    {
        float* k = dw_kernel_.as_fp32();
        std::mt19937 dw_rng(42);
        float dw_scale = std::sqrt(2.0f / static_cast<float>(config_.causal_window_size));
        std::uniform_real_distribution<float> dw_dist(-dw_scale, dw_scale);
        for (size_t i = 0; i < dw_kernel_.numel(); ++i) k[i] = dw_dist(dw_rng);
    }
    pw_conv_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);

    sae_w_encode_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);
    sae_w_decode_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);

    ntm_w_read_ = std::make_shared<Linear>(config_.d_model, config_.ntm_memory_slots, false);
    ntm_w_write_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);
    ntm_w_erase_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);
    ntm_memory_ = Tensor({config_.ntm_memory_slots, config_.d_model}, QuantType::FP32);

    w_proj_ = std::make_shared<Linear>(config_.d_model, config_.d_model);
    if (config_.use_bridge) {
        bridge_ = std::make_shared<Linear>(config_.d_model, config_.d_model, true);
    }
    if (config_.use_swiglu) {
        swiglu_ = std::make_unique<SwiGLUFFN>(config_.d_model, config_.swiglu_intermediate_size);
    }
    if (config_.weight_tying) {
        w_out_ = std::make_shared<Linear>(config_.d_model, config_.vocab_size, false);
        w_out_->weight.shape_ = w_embed_.shape_;
        w_out_->weight.dtype_ = w_embed_.dtype_;
        w_out_->weight.layout_ = w_embed_.layout_;
        w_out_->weight.data_ = w_embed_.data_;
        w_out_->weight.data_size_ = w_embed_.data_size_;
        w_out_->weight.owns_data_ = false;
        w_out_->weight.strides_ = w_embed_.strides_;
    } else {
        w_out_ = std::make_shared<Linear>(config_.d_model, config_.vocab_size);
    }
    ln_ = std::make_shared<LayerNorm>(config_.d_model);

    if (config_.use_mla) {
        kv_cache_ = std::make_shared<LatentKVCache>(
            config_.d_model, config_.mla_n_heads,
            config_.mla_latent_dim, config_.mla_max_cache_len);
    }

    last_hidden_ = Tensor({1, config_.d_model}, QuantType::FP32);

    size_t n_kv = config_.n_kv_heads > 0 ? config_.n_kv_heads : config_.num_attn_heads;
    for (size_t i = 0; i < config_.num_attn_layers; ++i) {
        attn_layers_.push_back(std::make_unique<CausalSelfAttention>(
            config_.d_model, config_.num_attn_heads, n_kv, config_.use_rope, config_.max_seq_len, config_.use_qk_norm));
    }
}

Tensor CausalLMHead::embed_lookup(const std::vector<size_t>& ids) {
    size_t seq_len = ids.size();
    Tensor output({seq_len, config_.d_model}, QuantType::FP32);
    float scale = std::sqrt(static_cast<float>(config_.d_model));

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && w_embed_.is_on_gpu()) {
        output.to_gpu();
        std::vector<int> int_ids(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            int_ids[i] = static_cast<int>(ids[i] >= config_.vocab_size ? 1 : ids[i]);
        }
        void* d_ids = CudaContext::instance().alloc(int_ids.size() * sizeof(int));
        CudaContext::instance().copy_h2d(d_ids, int_ids.data(), int_ids.size() * sizeof(int));
        launch_embed_lookup(output.as_gpu_fp32(), w_embed_.as_gpu_fp32(),
                            static_cast<const int*>(d_ids), seq_len, config_.d_model, scale,
                            CudaContext::instance().stream());
        CudaContext::instance().free(d_ids);
        output.gpu_dirty_ = true;
        return output;
    }
#endif

    float* out = output.as_fp32();
    float* embed = w_embed_.as_fp32();

    for (size_t i = 0; i < seq_len; ++i) {
        size_t tid = ids[i];
        if (tid >= config_.vocab_size) {
            std::cerr << "Warning: embed out-of-range token_id=" << tid
                      << ", replacing with UNK" << std::endl;
            tid = 1;
        }
        const float* row = embed + tid * config_.d_model;
        for (size_t d = 0; d < config_.d_model; ++d) {
            out[i * config_.d_model + d] = row[d] * scale;
        }
    }
    return output;
}

Tensor CausalLMHead::positional_encode(const Tensor& x, size_t offset) {
    Tensor result = x.clone();

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && x.is_on_gpu()) {
        result.to_gpu();
        launch_positional_encode(result.as_gpu_fp32(), w_pos_.as_gpu_fp32(),
            x.shape_[0], config_.d_model, static_cast<int>(offset),
            CudaContext::instance().stream());
        result.gpu_dirty_ = true;
        return result;
    }
#endif

    float* out = result.as_fp32();
    const float* pos = w_pos_.as_fp32();

    size_t seq_len = x.shape_[0];
    for (size_t i = 0; i < seq_len; ++i) {
        size_t p = offset + i;
        if (p >= config_.max_seq_len) {
            std::cerr << "Warning: position " << p << " exceeds max_seq_len, using last" << std::endl;
            p = config_.max_seq_len - 1;
        }
        for (size_t d = 0; d < config_.d_model; ++d) {
            out[i * config_.d_model + d] += pos[p * config_.d_model + d];
        }
    }
    return result;
}

Tensor CausalLMHead::causal_window_gate(const Tensor& x) {
    size_t seq_len = x.shape_[0];
    size_t C = config_.d_model;
    size_t K = config_.causal_window_size;
    const float* xp = x.as_fp32();
    const float* kernel = dw_kernel_.as_fp32();

    // In single-step mode (forward_step), pass through — no temporal context available
    if (seq_len == 1) return x.clone();

    Tensor dw_out({seq_len, C}, QuantType::FP32);
    float* dwp = dw_out.as_fp32();

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                if (i >= k) {
                    sum += xp[(i - k) * C + c] * kernel[c * K + k];
                }
            }
            dwp[i * C + c] = sum;
        }
    }

    Tensor pw_out = pw_conv_->forward(dw_out);
    float* gp = pw_out.as_fp32();

    Tensor output({seq_len, C}, QuantType::FP32);
    float* out = output.as_fp32();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t d = 0; d < C; ++d) {
            float g = 1.0f / (1.0f + std::exp(-gp[i * C + d]));
            out[i * C + d] = xp[i * C + d] * g;
        }
    }
    return output;
}

Tensor CausalLMHead::sae_sparse(const Tensor& x) {
    Tensor encoded = sae_w_encode_->forward(x);
    float* data = encoded.as_fp32();
    size_t n = encoded.numel();
    size_t k = config_.sae_k;

    if (k < n) {
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&](size_t a, size_t b) { return std::abs(data[a]) > std::abs(data[b]); });

        std::vector<bool> keep(n, false);
        for (size_t i = 0; i < k; ++i) keep[indices[i]] = true;
        for (size_t i = 0; i < n; ++i) {
            if (!keep[i]) data[i] = 0.0f;
        }
    }

    return sae_w_decode_->forward(encoded);
}

Tensor CausalLMHead::ntm_memory_access(const Tensor& x) {
    size_t batch = x.shape_[0];
    Tensor read_weights = ntm_w_read_->forward(x);
    float* rwp = read_weights.as_fp32();
    size_t slots = config_.ntm_memory_slots;

    for (size_t b = 0; b < batch; ++b) {
        float max_val = rwp[b * slots];
        for (size_t s = 1; s < slots; ++s) {
            max_val = std::max(max_val, rwp[b * slots + s]);
        }
        float sum = 0.0f;
        for (size_t s = 0; s < slots; ++s) {
            rwp[b * slots + s] = std::exp(rwp[b * slots + s] - max_val);
            sum += rwp[b * slots + s];
        }
        for (size_t s = 0; s < slots; ++s) {
            rwp[b * slots + s] /= sum;
        }
    }

    Tensor read_content({batch, config_.d_model}, QuantType::FP32);
    float* rcp = read_content.as_fp32();
    const float* mp = ntm_memory_.as_fp32();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t d = 0; d < config_.d_model; ++d) {
            float val = 0.0f;
            for (size_t s = 0; s < slots; ++s) {
                val += rwp[b * slots + s] * mp[s * config_.d_model + d];
            }
            rcp[b * config_.d_model + d] = val;
        }
    }

    Tensor h({batch, config_.d_model}, QuantType::FP32);
    float* hp = h.as_fp32();
    const float* xp = x.as_fp32();
    for (size_t i = 0; i < batch * config_.d_model; ++i) {
        hp[i] = xp[i] + rcp[i];
    }

    Tensor erase = ntm_w_erase_->forward(h);
    Tensor write = ntm_w_write_->forward(h);

    if (training_mode_) {
        float* ep = erase.as_fp32();
        float* wtp = write.as_fp32();
        float* mmp = (shadow_memory_.numel() > 0) ? shadow_memory_.as_fp32() : ntm_memory_.as_fp32();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < slots; ++s) {
                float rw = rwp[b * slots + s];
                for (size_t d = 0; d < config_.d_model; ++d) {
                    float e = 1.0f / (1.0f + std::exp(-ep[b * config_.d_model + d]));
                    float w = std::tanh(wtp[b * config_.d_model + d]);
                    mmp[s * config_.d_model + d] = mmp[s * config_.d_model + d] * (1.0f - rw * e) + rw * w;
                }
            }
        }
    }

    return h;
}

Tensor CausalLMHead::last_token_pool(const Tensor& x) {
    size_t seq_len = x.shape_[0];
    const float* xp = x.as_fp32();

    last_hidden_ = Tensor({1, config_.d_model}, QuantType::FP32);
    float* out = last_hidden_.as_fp32();
    memcpy(out, xp + (seq_len - 1) * config_.d_model, config_.d_model * sizeof(float));

    return last_hidden_;
}

Tensor CausalLMHead::mean_pool(const Tensor& x) {
    size_t seq_len = x.shape_[0];
    const float* xp = x.as_fp32();

    last_hidden_ = Tensor({1, config_.d_model}, QuantType::FP32);
    float* out = last_hidden_.as_fp32();
    float inv_n = 1.0f / static_cast<float>(seq_len);
    for (size_t d = 0; d < config_.d_model; ++d) {
        float sum = 0.0f;
        for (size_t i = 0; i < seq_len; ++i) {
            sum += xp[i * config_.d_model + d];
        }
        out[d] = sum * inv_n;
    }

    return last_hidden_;
}

Tensor CausalLMHead::pool(const Tensor& x) {
    if (config_.pooling == "mean") {
        return mean_pool(x);
    }
    return last_token_pool(x);
}

Tensor CausalLMHead::make_padding_mask(const std::vector<size_t>& token_ids) const {
    size_t seq_len = token_ids.size();
    Tensor mask({seq_len}, QuantType::FP32);
    float* mp = mask.as_fp32();
    int pad_id = config_.padding_id;
    for (size_t i = 0; i < seq_len; ++i) {
        mp[i] = (pad_id >= 0 && static_cast<int>(token_ids[i]) == pad_id) ? 0.0f : 1.0f;
    }
#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && w_embed_.is_on_gpu()) {
        mask.to_gpu();
    }
#endif
    return mask;
}

Tensor CausalLMHead::forward(const std::vector<size_t>& token_ids) {
    if (token_ids.empty()) {
        Tensor logits({1, config_.vocab_size}, QuantType::FP32);
        memset(logits.as_fp32(), 0, logits.data_size_);
        return logits;
    }
    if (!mode_set_) {
        std::cerr << "[WARNING] forward() called without explicit train()/eval() - using eval mode" << std::endl;
        eval();
    }
    Tensor x = embed_lookup(token_ids);
    x = positional_encode(x, 0);
    Tensor padding_mask = make_padding_mask(token_ids);
    const Tensor* pm_ptr = (config_.padding_id >= 0) ? &padding_mask : nullptr;
    for (auto& attn : attn_layers_) {
        x = attn->forward(x, pm_ptr);
    }
    x = causal_window_gate(x);
    if (config_.use_swiglu && swiglu_) {
        x = swiglu_->forward(x);
    }
    x = sae_sparse(x);
    x = ntm_memory_access(x);
    x = ln_->forward(x);

    Tensor pooled = pool(x);
    if (config_.use_bridge && bridge_) {
        last_projected_ = w_proj_->forward(bridge_->forward(pooled));
    } else {
        last_projected_ = w_proj_->forward(pooled);
    }
    Tensor logits = w_out_->forward(last_projected_);
    return logits;
}

Tensor CausalLMHead::forward_step(size_t token_id, size_t pos) {
    if (!mode_set_) {
        std::cerr << "[WARNING] forward_step() called without explicit train()/eval() - using eval mode" << std::endl;
        eval();
    }
    std::vector<size_t> ids = {token_id};
    Tensor x = embed_lookup(ids);
    x = positional_encode(x, pos);
    for (auto& attn : attn_layers_) {
        x = attn->forward(x, nullptr);
    }
    x = causal_window_gate(x);
    if (config_.use_swiglu && swiglu_) {
        x = swiglu_->forward(x);
    }
    x = sae_sparse(x);
    x = ntm_memory_access(x);
    x = ln_->forward(x);

    if (config_.use_bridge && bridge_) {
        last_projected_ = w_proj_->forward(bridge_->forward(x));
    } else {
        last_projected_ = w_proj_->forward(x);
    }
    last_hidden_ = std::move(x);
    Tensor logits = w_out_->forward(last_projected_);
    return logits;
}

Tensor CausalLMHead::forward_for_training(const std::vector<size_t>& token_ids) {
    if (token_ids.empty()) {
        Tensor logits({1, config_.vocab_size}, QuantType::FP32);
        memset(logits.as_fp32(), 0, logits.data_size_);
        return logits;
    }
    if (!mode_set_) {
        std::cerr << "[WARNING] forward_for_training called without explicit train()/eval() - auto-switching to train mode" << std::endl;
    }
    train();
    shadow_memory_ = ntm_memory_.clone();

    train_cache_.input_ids = token_ids;
    train_cache_.x_embed = embed_lookup(token_ids);
    train_cache_.x_pos = positional_encode(train_cache_.x_embed, 0);

    Tensor x = train_cache_.x_pos;
    train_cache_.attn_inputs.clear();
    train_cache_.attn_outputs.clear();

    Tensor padding_mask = make_padding_mask(token_ids);
    const Tensor* pm_ptr = (config_.padding_id >= 0) ? &padding_mask : nullptr;
    for (auto& attn : attn_layers_) {
        train_cache_.attn_inputs.push_back(x);
        x = attn->forward(x, pm_ptr);
        train_cache_.attn_outputs.push_back(x);
    }

    train_cache_.x_gate_in = x;
    {
        size_t seq_len = x.shape_[0];
        size_t C = config_.d_model;
        size_t K = config_.causal_window_size;
        const float* xp = x.as_fp32();
        const float* kernel = dw_kernel_.as_fp32();

        Tensor dw_out({seq_len, C}, QuantType::FP32);
        float* dwp = dw_out.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (size_t k2 = 0; k2 < K; ++k2) {
                    if (i >= k2) sum += xp[(i - k2) * C + c] * kernel[c * K + k2];
                }
                dwp[i * C + c] = sum;
            }
        }

        train_cache_.x_gate_pre_sigmoid = pw_conv_->forward(dw_out);
        const float* gp = train_cache_.x_gate_pre_sigmoid.as_fp32();

        Tensor output({seq_len, C}, QuantType::FP32);
        float* out = output.as_fp32();
        for (size_t i = 0; i < seq_len * C; ++i) {
            float g = 1.0f / (1.0f + std::exp(-gp[i]));
            out[i] = xp[i] * g;
        }
        x = output;
    }
    train_cache_.x_after_gate = x;

    if (config_.use_swiglu && swiglu_) {
        x = swiglu_->forward(x);
        train_cache_.x_after_swiglu = x;
    }

    train_cache_.x_sae_encoded = sae_w_encode_->forward(x);
    size_t n = train_cache_.x_sae_encoded.numel();
    size_t k = config_.sae_k;

#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && train_cache_.x_sae_encoded.is_on_gpu()) {
        train_cache_.x_sae_encoded.to_gpu();
        launch_sae_topk_mask(train_cache_.x_sae_encoded.as_gpu_fp32(), n, k,
                             CudaContext::instance().stream());
        train_cache_.x_sae_encoded.gpu_dirty_ = true;
    } else
#endif
    {
        float* enc_data = train_cache_.x_sae_encoded.as_fp32();
        if (k < n) {
            std::vector<size_t> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                              [&](size_t a, size_t b) { return std::abs(enc_data[a]) > std::abs(enc_data[b]); });
            std::vector<bool> keep(n, false);
            for (size_t i = 0; i < k; ++i) keep[indices[i]] = true;
            for (size_t i = 0; i < n; ++i) { if (!keep[i]) enc_data[i] = 0.0f; }
        }
    }

    x = sae_w_decode_->forward(train_cache_.x_sae_encoded);
    train_cache_.x_after_sae = x;

    {
        size_t batch = x.shape_[0];
        train_cache_.x_ntm_read_weights = ntm_w_read_->forward(x);
        float* rwp = train_cache_.x_ntm_read_weights.as_fp32();
        size_t slots = config_.ntm_memory_slots;
        for (size_t b = 0; b < batch; ++b) {
            float max_val = rwp[b * slots];
            for (size_t s = 1; s < slots; ++s) max_val = std::max(max_val, rwp[b * slots + s]);
            float sum = 0.0f;
            for (size_t s = 0; s < slots; ++s) { rwp[b * slots + s] = std::exp(rwp[b * slots + s] - max_val); sum += rwp[b * slots + s]; }
            for (size_t s = 0; s < slots; ++s) rwp[b * slots + s] /= sum;
        }
        train_cache_.x_ntm_read_content = Tensor({batch, config_.d_model}, QuantType::FP32);
        float* rcp = train_cache_.x_ntm_read_content.as_fp32();
        const float* mp = ntm_memory_.as_fp32();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t d2 = 0; d2 < config_.d_model; ++d2) {
                float val = 0.0f;
                for (size_t s = 0; s < slots; ++s) val += rwp[b * slots + s] * mp[s * config_.d_model + d2];
                rcp[b * config_.d_model + d2] = val;
            }
        }
        train_cache_.x_ntm_h = Tensor({batch, config_.d_model}, QuantType::FP32);
        float* hp = train_cache_.x_ntm_h.as_fp32();
        const float* xp = x.as_fp32();
        for (size_t i = 0; i < batch * config_.d_model; ++i) hp[i] = xp[i] + rcp[i];

        train_cache_.x_ntm_erase = ntm_w_erase_->forward(train_cache_.x_ntm_h);
        train_cache_.x_ntm_write = ntm_w_write_->forward(train_cache_.x_ntm_h);

        float* ep = train_cache_.x_ntm_erase.as_fp32();
        float* wtp = train_cache_.x_ntm_write.as_fp32();
        float* mmp = (shadow_memory_.numel() > 0) ? shadow_memory_.as_fp32() : ntm_memory_.as_fp32();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < slots; ++s) {
                float rw = rwp[b * slots + s];
                for (size_t d2 = 0; d2 < config_.d_model; ++d2) {
                    float e = 1.0f / (1.0f + std::exp(-ep[b * config_.d_model + d2]));
                    float w = std::tanh(wtp[b * config_.d_model + d2]);
                    mmp[s * config_.d_model + d2] = mmp[s * config_.d_model + d2] * (1.0f - rw * e) + rw * w;
                }
            }
        }
        x = train_cache_.x_ntm_h;
    }
    train_cache_.x_after_ntm = x;

    x = ln_->forward(x);
    train_cache_.x_after_ln = x;

    train_cache_.x_pooled = pool(x);
    if (config_.use_bridge && bridge_) {
        train_cache_.x_bridge = bridge_->forward(train_cache_.x_pooled);
        train_cache_.x_projected = w_proj_->forward(train_cache_.x_bridge);
    } else {
        train_cache_.x_projected = w_proj_->forward(train_cache_.x_pooled);
    }
    Tensor logits = w_out_->forward(train_cache_.x_projected);

    return logits;
}

CausalLMHead::LMGradients CausalLMHead::backward_from_logits(const Tensor& logits_grad) {
    LMGradients grads;
    size_t d_model = config_.d_model;
    size_t seq_len = train_cache_.x_after_ln.shape_[0];

    Tensor proj_grad = lm_head_linear_backward_input(logits_grad, w_out_->weight);

    if (!config_.weight_tying) {
        grads.w_out_weight_grad = lm_head_linear_backward_weight(train_cache_.x_projected, logits_grad);
    }
    grads.w_out_bias_grad = lm_head_bias_backward(logits_grad);

    Tensor proj_input = (config_.use_bridge && bridge_) ? train_cache_.x_bridge : train_cache_.x_pooled;
    Tensor pooled_grad = lm_head_linear_backward_input(proj_grad, w_proj_->weight);
    grads.w_proj_weight_grad = lm_head_linear_backward_weight(proj_input, proj_grad);
    grads.w_proj_bias_grad = lm_head_bias_backward(proj_grad);

    if (config_.use_bridge && bridge_) {
        grads.bridge_weight_grad = lm_head_linear_backward_weight(train_cache_.x_pooled, pooled_grad);
        grads.bridge_bias_grad = lm_head_bias_backward(pooled_grad);
        pooled_grad = lm_head_linear_backward_input(pooled_grad, bridge_->weight);
    }


    Tensor x_grad;
    if (config_.pooling == "mean") {
        x_grad = Tensor({seq_len, d_model}, QuantType::FP32);
        float* xg = x_grad.as_fp32();
        const float* pg = pooled_grad.as_fp32();
        float inv_n = 1.0f / static_cast<float>(seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t d = 0; d < d_model; ++d) {
                xg[i * d_model + d] = pg[d] * inv_n;
            }
        }
    } else {
        x_grad = Tensor({seq_len, d_model}, QuantType::FP32);
        memset(x_grad.as_fp32(), 0, x_grad.data_size_);
        const float* pg = pooled_grad.as_fp32();
        float* xg = x_grad.as_fp32();
        memcpy(xg + (seq_len - 1) * d_model, pg, d_model * sizeof(float));
    }

    Tensor ln_input_grad = ln_backward_impl(train_cache_.x_after_ntm, ln_->weight, x_grad);

    grads.ln_weight_grad = Tensor({d_model}, QuantType::FP32);
    grads.ln_bias_grad = Tensor({d_model}, QuantType::FP32);
    {
        const float* og = x_grad.as_fp32();
        const float* inp = train_cache_.x_after_ntm.as_fp32();
        float* lwg = grads.ln_weight_grad.as_fp32();
        float* lbg = grads.ln_bias_grad.as_fp32();
        memset(lwg, 0, d_model * sizeof(float));
        memset(lbg, 0, d_model * sizeof(float));
        for (size_t i = 0; i < seq_len; ++i) {
            float mean = 0.0f;
            for (size_t d = 0; d < d_model; ++d) mean += inp[i * d_model + d];
            mean /= d_model;
            float var = 0.0f;
            for (size_t d = 0; d < d_model; ++d) { float diff = inp[i * d_model + d] - mean; var += diff * diff; }
            var /= d_model;
            float inv_std = 1.0f / std::sqrt(var + 1e-5f);
            for (size_t d = 0; d < d_model; ++d) {
                float norm = (inp[i * d_model + d] - mean) * inv_std;
                lwg[d] += og[i * d_model + d] * norm / seq_len;
                lbg[d] += og[i * d_model + d] / seq_len;
            }
        }
    }
    x_grad = ln_input_grad;

    {
        size_t batch = seq_len;
        size_t slots = config_.ntm_memory_slots;
        const float* rwp = train_cache_.x_ntm_read_weights.as_fp32();
        const float* mp = shadow_memory_.numel() > 0 ? shadow_memory_.as_fp32() : ntm_memory_.as_fp32();
        const float* dh = x_grad.as_fp32();

        Tensor d_rw({batch, slots}, QuantType::FP32);
        float* drwp = d_rw.as_fp32();
        memset(drwp, 0, d_rw.data_size_);

        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < slots; ++s) {
                float val = 0.0f;
                for (size_t d2 = 0; d2 < d_model; ++d2) {
                    val += dh[b * d_model + d2] * mp[s * d_model + d2];
                }
                drwp[b * slots + s] = val;
            }
        }

        Tensor d_rw_pre_softmax({batch, slots}, QuantType::FP32);
        float* drwps = d_rw_pre_softmax.as_fp32();
        for (size_t b = 0; b < batch; ++b) {
            float dot = 0.0f;
            for (size_t s = 0; s < slots; ++s) dot += rwp[b * slots + s] * drwp[b * slots + s];
            for (size_t s = 0; s < slots; ++s) {
                drwps[b * slots + s] = rwp[b * slots + s] * (drwp[b * slots + s] - dot);
            }
        }

        grads.ntm_read_weight_grad = lm_head_linear_backward_weight(train_cache_.x_after_sae, d_rw_pre_softmax);
        Tensor d_x_from_read = lm_head_linear_backward_input(d_rw_pre_softmax, ntm_w_read_->weight);

        grads.ntm_erase_weight_grad = Tensor(ntm_w_erase_->weight.shape_, QuantType::FP32);
        grads.ntm_write_weight_grad = Tensor(ntm_w_write_->weight.shape_, QuantType::FP32);

        x_grad = Tensor({batch, d_model}, QuantType::FP32);
        float* xgp = x_grad.as_fp32();
        const float* dhr = d_x_from_read.as_fp32();
        for (size_t i = 0; i < batch * d_model; ++i) {
            xgp[i] = dh[i] + dhr[i];
        }
    }

    grads.sae_decode_weight_grad = lm_head_linear_backward_weight(train_cache_.x_sae_encoded, x_grad);
    Tensor sae_dec_input_grad = lm_head_linear_backward_input(x_grad, sae_w_decode_->weight);
    {
        size_t n = sae_dec_input_grad.numel();
        size_t k = config_.sae_k;
#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && sae_dec_input_grad.is_on_gpu()) {
            sae_dec_input_grad.to_gpu();
            train_cache_.x_sae_encoded.to_gpu();
            launch_sae_topk_mask_backward(sae_dec_input_grad.as_gpu_fp32(),
                                          train_cache_.x_sae_encoded.as_gpu_fp32(),
                                          n, k, CudaContext::instance().stream());
            sae_dec_input_grad.gpu_dirty_ = true;
        } else
#endif
        {
            float* enc_g = sae_dec_input_grad.as_fp32();
            const float* enc_data = train_cache_.x_sae_encoded.as_fp32();
            if (k < n) {
                std::vector<size_t> indices(n);
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](size_t a, size_t b) { return std::abs(enc_data[a]) > std::abs(enc_data[b]); });
                std::vector<bool> keep(n, false);
                for (size_t i = 0; i < k; ++i) keep[indices[i]] = true;
                for (size_t i = 0; i < n; ++i) { if (!keep[i]) enc_g[i] = 0.0f; }
            }
        }
    }
    if (config_.use_swiglu && swiglu_) {
        grads.sae_encode_weight_grad = lm_head_linear_backward_weight(train_cache_.x_after_swiglu, sae_dec_input_grad);
        x_grad = lm_head_linear_backward_input(sae_dec_input_grad, sae_w_encode_->weight);

        grads.swiglu_grads = swiglu_->backward(x_grad);
        x_grad = grads.swiglu_grads.input_grad;
    } else {
        grads.sae_encode_weight_grad = lm_head_linear_backward_weight(train_cache_.x_after_gate, sae_dec_input_grad);
        x_grad = lm_head_linear_backward_input(sae_dec_input_grad, sae_w_encode_->weight);
    }


    grads.dw_kernel_grad = Tensor(dw_kernel_.shape_, QuantType::FP32);
    grads.pw_conv_weight_grad = Tensor(pw_conv_->weight.shape_, QuantType::FP32);
    grads.pw_conv_bias_grad = Tensor(pw_conv_->bias.shape_, QuantType::FP32);
    {
        size_t C = d_model;
        size_t K = config_.causal_window_size;
        const float* xp = train_cache_.x_gate_in.as_fp32();
        const float* gp = train_cache_.x_gate_pre_sigmoid.as_fp32();
        const float* xg = x_grad.as_fp32();
        float* dkg = grads.dw_kernel_grad.as_fp32();
        memset(dkg, 0, grads.dw_kernel_grad.data_size_);

        Tensor gate_grad({seq_len, C}, QuantType::FP32);
        {
            float* gg = gate_grad.as_fp32();
            const float* xg2 = x_grad.as_fp32();
            const float* gp2 = train_cache_.x_gate_pre_sigmoid.as_fp32();
            const float* xp2 = train_cache_.x_gate_in.as_fp32();
            for (size_t i = 0; i < seq_len * C; ++i) {
                float sig = 1.0f / (1.0f + std::exp(-gp2[i]));
                gg[i] = xg2[i] * xp2[i] * sig * (1.0f - sig);
            }
        }

        Tensor pw_grad_tensor = lm_head_linear_backward_input(gate_grad, pw_conv_->weight);
        const float* pw_grad = pw_grad_tensor.as_fp32();
        grads.pw_conv_weight_grad = lm_head_linear_backward_weight(train_cache_.x_gate_in, gate_grad);
        grads.pw_conv_bias_grad = lm_head_bias_backward(gate_grad);

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t k = 0; k < K; ++k) {
                    if (i >= k) {
                        dkg[c * K + k] += pw_grad[i * C + c] * xp[(i - k) * C + c];
                    }
                }
            }
        }

        Tensor gate_input_grad({seq_len, C}, QuantType::FP32);
        float* gig = gate_input_grad.as_fp32();
        memset(gig, 0, gate_input_grad.data_size_);
        const float* dkp = dw_kernel_.as_fp32();
        const float* gp2 = train_cache_.x_gate_pre_sigmoid.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t c = 0; c < C; ++c) {
                float val = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    if (i + k < seq_len) {
                        val += pw_grad[(i + k) * C + c] * dkp[c * K + k];
                    }
                }
                float sig = 1.0f / (1.0f + std::exp(-gp2[i * C + c]));
                gig[i * C + c] = x_grad.as_fp32()[i * C + c] * sig + val;
            }
        }
        x_grad = gate_input_grad;
    }


    for (int i = static_cast<int>(attn_layers_.size()) - 1; i >= 0; --i) {
        auto attn_grads = attn_layers_[i]->backward(x_grad);
        bool shape_match = attn_layers_[i]->cache_.input.shape_[0] == attn_grads.input_grad.shape_[0];
        if (shape_match) {
            x_grad = std::move(attn_grads.input_grad);
        }
        grads.attn_grads.insert(grads.attn_grads.begin(), std::move(attn_grads));
    }

    grads.embed_grad = Tensor({seq_len, d_model}, QuantType::FP32);
    grads.used_token_ids = train_cache_.input_ids;

    {
        float* eg = grads.embed_grad.as_fp32();
        const float* xg = x_grad.as_fp32();
        float scale = std::sqrt(static_cast<float>(d_model));
        for (size_t i = 0; i < seq_len; ++i) {
            float norm_val = 0.0f;
            for (size_t d = 0; d < d_model; ++d) norm_val += xg[i * d_model + d] * xg[i * d_model + d];
            if (norm_val < 1e-20f) {
                memset(eg + i * d_model, 0, d_model * sizeof(float));
                continue;
            }
            for (size_t d = 0; d < d_model; ++d) {
                eg[i * d_model + d] = xg[i * d_model + d] * scale / seq_len;
            }
        }
    }

    return grads;
}

void CausalLMHead::apply_lm_gradients(LMGradients& grads, float lr) {
    auto sgd_update = [&](Tensor& param, const Tensor& grad) {
        if (param.shape_ != grad.shape_ || param.numel() == 0 || grad.numel() == 0) return;
#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && param.is_on_gpu()) {
            param.to_gpu();
            grad.to_gpu();
            launch_sgd_update(param.as_gpu_fp32(), grad.as_gpu_fp32(),
                              param.numel(), lr, CudaContext::instance().stream());
            param.gpu_dirty_ = true;
            return;
        }
#endif
        float* p = param.as_fp32();
        const float* g = grad.as_fp32();
        size_t n = param.numel();
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(g[i])) p[i] -= lr * g[i];
        }
    };

    sgd_update(w_proj_->weight, grads.w_proj_weight_grad);
    sgd_update(w_proj_->bias, grads.w_proj_bias_grad);
    if (config_.use_bridge && bridge_) {
        sgd_update(bridge_->weight, grads.bridge_weight_grad);
        sgd_update(bridge_->bias, grads.bridge_bias_grad);
    }
    if (!config_.weight_tying) {
        sgd_update(w_out_->weight, grads.w_out_weight_grad);
    }
    sgd_update(w_out_->bias, grads.w_out_bias_grad);

    sgd_update(ln_->weight, grads.ln_weight_grad);
    sgd_update(ln_->bias, grads.ln_bias_grad);

    sgd_update(ntm_w_read_->weight, grads.ntm_read_weight_grad);
    sgd_update(ntm_w_write_->weight, grads.ntm_write_weight_grad);
    sgd_update(ntm_w_erase_->weight, grads.ntm_erase_weight_grad);

    sgd_update(sae_w_encode_->weight, grads.sae_encode_weight_grad);
    sgd_update(sae_w_decode_->weight, grads.sae_decode_weight_grad);

    if (config_.use_swiglu && swiglu_ && grads.swiglu_grads.w_gate_weight_grad.numel() > 0) {
        sgd_update(swiglu_->w_gate_->weight, grads.swiglu_grads.w_gate_weight_grad);
        sgd_update(swiglu_->w_gate_->bias, grads.swiglu_grads.w_gate_bias_grad);
        sgd_update(swiglu_->w_up_->weight, grads.swiglu_grads.w_up_weight_grad);
        sgd_update(swiglu_->w_up_->bias, grads.swiglu_grads.w_up_bias_grad);
        sgd_update(swiglu_->w_down_->weight, grads.swiglu_grads.w_down_weight_grad);
        sgd_update(swiglu_->w_down_->bias, grads.swiglu_grads.w_down_bias_grad);
    }

    sgd_update(dw_kernel_, grads.dw_kernel_grad);
    sgd_update(pw_conv_->weight, grads.pw_conv_weight_grad);
    sgd_update(pw_conv_->bias, grads.pw_conv_bias_grad);

    {
        const auto& ids = grads.used_token_ids;
        size_t seq_len = ids.size();
        size_t d = config_.d_model;
        size_t vocab_sz = config_.vocab_size;
#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && w_embed_.is_on_gpu()) {
            w_embed_.to_gpu();
            grads.embed_grad.to_gpu();
            std::vector<int> int_ids(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                int_ids[i] = static_cast<int>(ids[i] >= vocab_sz ? 1 : ids[i]);
            }
            void* d_ids = CudaContext::instance().alloc(int_ids.size() * sizeof(int));
            CudaContext::instance().copy_h2d(d_ids, int_ids.data(), int_ids.size() * sizeof(int));
            launch_sparse_embed_update(w_embed_.as_gpu_fp32(), grads.embed_grad.as_gpu_fp32(),
                                       static_cast<const int*>(d_ids), static_cast<int>(seq_len),
                                       static_cast<int>(d), lr, CudaContext::instance().stream());
            CudaContext::instance().free(d_ids);
            w_embed_.gpu_dirty_ = true;
        } else
#endif
        {
            float* embed_data = w_embed_.as_fp32();
            const float* eg = grads.embed_grad.as_fp32();
            for (size_t i = 0; i < seq_len; ++i) {
                size_t tid = ids[i];
                if (tid >= vocab_sz) continue;
                float* row = embed_data + tid * d;
                const float* grad_row = eg + i * d;
                for (size_t j = 0; j < d; ++j) {
                    if (std::isfinite(grad_row[j])) row[j] -= lr * grad_row[j];
                }
            }
        }
    }

    for (size_t i = 0; i < attn_layers_.size() && i < grads.attn_grads.size(); ++i) {
        auto& ag = grads.attn_grads[i];
        sgd_update(attn_layers_[i]->w_q->weight, ag.w_q_weight_grad);
        sgd_update(attn_layers_[i]->w_q->bias, ag.w_q_bias_grad);
        sgd_update(attn_layers_[i]->w_k->weight, ag.w_k_weight_grad);
        sgd_update(attn_layers_[i]->w_k->bias, ag.w_k_bias_grad);
        sgd_update(attn_layers_[i]->w_v->weight, ag.w_v_weight_grad);
        sgd_update(attn_layers_[i]->w_v->bias, ag.w_v_bias_grad);
        sgd_update(attn_layers_[i]->w_out->weight, ag.w_out_weight_grad);
        sgd_update(attn_layers_[i]->w_out->bias, ag.w_out_bias_grad);
    }
}

Tensor CausalLMHead::ln_backward_impl(const Tensor& input, const Tensor& weight, const Tensor& output_grad, float eps) {
    size_t seq_len = input.shape_[0];
    size_t dim = input.shape_[1];
    Tensor input_grad({seq_len, dim}, QuantType::FP32);
    const float* inp = input.as_fp32();
    const float* w = weight.as_fp32();
    const float* og = output_grad.as_fp32();
    float* ig = input_grad.as_fp32();
    for (size_t i = 0; i < seq_len; ++i) {
        float mean = 0.0f;
        for (size_t d = 0; d < dim; ++d) mean += inp[i * dim + d];
        mean /= dim;
        float var = 0.0f;
        for (size_t d = 0; d < dim; ++d) { float diff = inp[i * dim + d] - mean; var += diff * diff; }
        var /= dim;
        float inv_std = 1.0f / std::sqrt(var + eps);
        float sum_gn = 0.0f, sum_gnx = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float norm = (inp[i * dim + d] - mean) * inv_std;
            float gn = og[i * dim + d] * w[d];
            sum_gn += gn; sum_gnx += gn * norm;
        }
        for (size_t d = 0; d < dim; ++d) {
            float norm = (inp[i * dim + d] - mean) * inv_std;
            float gn = og[i * dim + d] * w[d];
            ig[i * dim + d] = inv_std * (gn - sum_gn / dim - norm * sum_gnx / dim);
        }
    }
    return input_grad;
}

Tensor CausalLMHead::lm_head_linear_backward_input(const Tensor& output_grad, const Tensor& weight) {
    size_t batch = output_grad.shape_[0];
    size_t out_f = output_grad.shape_[1];
    size_t in_f = weight.shape_[1];
    Tensor input_grad({batch, in_f}, QuantType::FP32);
#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && output_grad.is_on_gpu()) {
        input_grad.to_gpu();
        CudaContext::instance().sgemm_rowmajor(false, false,
            static_cast<int>(batch), static_cast<int>(in_f), static_cast<int>(out_f),
            1.0f, output_grad.as_gpu_fp32(), static_cast<int>(out_f),
            weight.as_gpu_fp32(), static_cast<int>(in_f),
            0.0f, input_grad.as_gpu_fp32(), static_cast<int>(in_f));
        input_grad.gpu_dirty_ = true;
        return input_grad;
    }
#endif
    const float* og = output_grad.as_fp32();
    const float* w = weight.as_fp32();
    float* ig = input_grad.as_fp32();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t j = 0; j < in_f; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < out_f; ++i) sum += og[b * out_f + i] * w[i * in_f + j];
            ig[b * in_f + j] = sum;
        }
    }
    return input_grad;
}

Tensor CausalLMHead::lm_head_linear_backward_weight(const Tensor& input, const Tensor& output_grad) {
    size_t batch = input.shape_[0];
    size_t in_f = input.shape_[1];
    size_t out_f = output_grad.shape_[1];
    Tensor weight_grad({out_f, in_f}, QuantType::FP32);
#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && input.is_on_gpu()) {
        weight_grad.to_gpu();
        CudaContext::instance().sgemm_rowmajor(true, false,
            static_cast<int>(out_f), static_cast<int>(in_f), static_cast<int>(batch),
            1.0f / batch, output_grad.as_gpu_fp32(), static_cast<int>(out_f),
            input.as_gpu_fp32(), static_cast<int>(in_f),
            0.0f, weight_grad.as_gpu_fp32(), static_cast<int>(in_f));
        weight_grad.gpu_dirty_ = true;
        return weight_grad;
    }
#endif
    const float* inp = input.as_fp32();
    const float* og = output_grad.as_fp32();
    float* wg = weight_grad.as_fp32();
    for (size_t i = 0; i < out_f; ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch; ++b) sum += og[b * out_f + i] * inp[b * in_f + j];
            wg[i * in_f + j] = sum / batch;
        }
    }
    return weight_grad;
}

Tensor CausalLMHead::lm_head_bias_backward(const Tensor& output_grad) {
    size_t batch = output_grad.shape_[0];
    size_t dim = output_grad.shape_[1];
    Tensor grad({dim}, QuantType::FP32);
    const float* og = output_grad.as_fp32();
    float* g = grad.as_fp32();
    for (size_t j = 0; j < dim; ++j) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch; ++b) sum += og[b * dim + j];
        g[j] = sum / batch;
    }
    return grad;
}

void CausalLMHead::clear_cache() {
    if (kv_cache_) {
        kv_cache_->clear_cache();
    }
    memset(ntm_memory_.as_fp32(), 0, ntm_memory_.data_size_);
    sliding_window_drops_ = 0;
}

CacheStats CausalLMHead::cache_stats() const {
    CacheStats stats;
    if (kv_cache_) {
        stats.cache_len = kv_cache_->cache_len;
        stats.memory_bytes = kv_cache_->cache_size_bytes();
        stats.saving_ratio = kv_cache_->memory_saving_ratio();
    }
    stats.sliding_window_drops = sliding_window_drops_;
    return stats;
}

} // namespace neuroflow
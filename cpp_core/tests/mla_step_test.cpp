#include <iostream>
#include <exception>
#include "../include/neuroflow/tensor.hpp"
#include "../include/neuroflow/networks.hpp"

using namespace neuroflow;

int main() {
    try {
        std::cout << "MLA step by step..." << std::endl;
        
        // MLA 参数
        size_t d_model = 64;
        size_t n_heads = 4;
        size_t d_latent = 16;
        size_t head_dim = d_model / n_heads; // 16
        
        std::cout << "d_model=" << d_model << ", n_heads=" << n_heads 
                  << ", d_latent=" << d_latent << ", head_dim=" << head_dim << std::endl;
        
        // 创建 Linear 层
        Linear W_q(d_model, d_model, false);
        Linear W_dkv(d_model, d_latent, false);
        Linear W_uk(d_latent, d_model, false);
        Linear W_uv(d_latent, d_model, false);
        Linear W_o(d_model, d_model, false);
        
        // 输入
        Tensor input({1, d_model});
        for (size_t i = 0; i < input.numel(); ++i) input.as_fp32()[i] = 0.1f * i;
        
        size_t batch = 1;
        size_t seq_len = 1;
        
        std::cout << "1. Q projection..." << std::endl;
        Tensor x_flat = input.reshape({batch * seq_len, d_model});
        std::cout << "   x_flat shape: [" << x_flat.shape[0] << ", " << x_flat.shape[1] << "]" << std::endl;
        
        Tensor q = W_q.forward(x_flat);
        std::cout << "   q shape after linear: [" << q.shape[0] << ", " << q.shape[1] << "]" << std::endl;
        
        // q 需要 reshape 到 {batch, seq_len, n_heads, head_dim}
        std::cout << "   reshaping q to {batch, seq_len, n_heads, head_dim} = {" << batch << ", " << seq_len << ", " << n_heads << ", " << head_dim << "}" << std::endl;
        std::cout << "   q numel: " << q.numel() << std::endl;
        std::cout << "   expected numel: " << (batch * seq_len * n_heads * head_dim) << std::endl;
        
        q = q.reshape({batch, seq_len, n_heads, head_dim});
        std::cout << "   q reshaped: success" << std::endl;
        
        std::cout << "2. KV compression..." << std::endl;
        Tensor c_kv = W_dkv.forward(x_flat);
        std::cout << "   c_kv shape: [" << c_kv.shape[0] << ", " << c_kv.shape[1] << "]" << std::endl;
        c_kv = c_kv.reshape({batch, seq_len, d_latent});
        std::cout << "   c_kv reshaped: success" << std::endl;
        
        std::cout << "3. K/V decompression..." << std::endl;
        size_t total_len = seq_len; // 第一次没有 cache
        
        std::cout << "   creating c_kv_flat {" << (batch * total_len) << ", " << d_latent << "}" << std::endl;
        Tensor c_kv_flat({batch * total_len, d_latent});
        float* ckf = c_kv_flat.as_fp32();
        float* ck = c_kv.as_fp32();
        std::cout << "   copying " << (batch * total_len * d_latent) << " elements" << std::endl;
        for (size_t i = 0; i < batch * total_len * d_latent; ++i) {
            ckf[i] = ck[i];
        }
        std::cout << "   copy done" << std::endl;
        
        Tensor k = W_uk.forward(c_kv_flat);
        Tensor v = W_uv.forward(c_kv_flat);
        std::cout << "   k shape: [" << k.shape[0] << ", " << k.shape[1] << "]" << std::endl;
        std::cout << "   v shape: [" << v.shape[0] << ", " << v.shape[1] << "]" << std::endl;
        
        k = k.reshape({batch, total_len, n_heads, head_dim});
        v = v.reshape({batch, total_len, n_heads, head_dim});
        std::cout << "   k reshaped, v reshaped" << std::endl;
        
        std::cout << "4. Attention computation..." << std::endl;
        Tensor output({batch, seq_len, d_model});
        float* out = output.as_fp32();
        float* qp = q.as_fp32();
        float* kp = k.as_fp32();
        float* vp = v.as_fp32();
        
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < n_heads; ++h) {
                for (size_t s = 0; s < seq_len; ++s) {
                    std::vector<float> scores(total_len);
                    for (size_t t = 0; t < total_len; ++t) {
                        float dot = 0;
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t q_idx = b * seq_len * n_heads * head_dim + s * n_heads * head_dim + h * head_dim + d;
                            size_t k_idx = b * total_len * n_heads * head_dim + t * n_heads * head_dim + h * head_dim + d;
                            std::cout << "   q_idx=" << q_idx << ", k_idx=" << k_idx << std::endl;
                            dot += qp[q_idx] * kp[k_idx];
                        }
                        scores[t] = dot * scale;
                    }
                    // softmax...
                    // output...
                }
            }
        }
        
        std::cout << "Success!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}

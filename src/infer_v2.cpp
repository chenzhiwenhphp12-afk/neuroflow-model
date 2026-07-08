#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "neuroflow/generative.hpp"
#include "neuroflow/model.hpp"

using namespace neuroflow;

int main(int argc, char* argv[]) {
    std::string config_path = "configs/config_distill_small.json";
    std::string tokenizer_path = "configs/tokenizer_cn_013.json";
    std::string lm_head_path = "";
    int max_new_tokens = 64;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    bool use_cuda = false;
    std::string strategy_name = "top_k";
    int yarn_max_seq_len = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--lm-head" && i + 1 < argc) lm_head_path = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) max_new_tokens = std::atoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::atof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) top_p = std::atof(argv[++i]);
        else if (arg == "--repetition-penalty" && i + 1 < argc) repetition_penalty = std::atof(argv[++i]);
        else if (arg == "--strategy" && i + 1 < argc) strategy_name = argv[++i];
        else if (arg == "--max-seq-len" && i + 1 < argc) yarn_max_seq_len = std::atoi(argv[++i]);
        else if (arg == "--use-cuda") use_cuda = true;
    }

    if (lm_head_path.empty()) {
        std::cerr << "用法: neuroflow_infer --lm-head <path> [--config <path>] [--tokenizer <path>]"
                  << " [--max-tokens <N>] [--temperature <f>] [--top-k <N>] [--use-cuda]" << std::endl;
        return 1;
    }

    NeuroFlowModel::Config model_cfg;
    {
        std::ifstream jf(config_path);
        if (jf) {
            std::string json_str((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
            auto ex_num = [&](const std::string& key, size_t def) -> size_t {
                std::string pat = "\"" + key + "\"";
                size_t pos = json_str.find(pat);
                if (pos == std::string::npos) return def;
                pos = json_str.find(':', pos + pat.size());
                if (pos == std::string::npos) return def;
                pos++;
                while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t')) pos++;
                return std::stoull(json_str.substr(pos));
            };
            auto ex_str = [&](const std::string& key, const std::string& def) -> std::string {
                std::string pat = "\"" + key + "\"";
                size_t pos = json_str.find(pat);
                if (pos == std::string::npos) return def;
                pos = json_str.find(':', pos + pat.size());
                if (pos == std::string::npos) return def;
                size_t q1 = json_str.find('"', pos);
                if (q1 == std::string::npos) return def;
                size_t q2 = json_str.find('"', q1 + 1);
                if (q2 == std::string::npos) return def;
                return json_str.substr(q1 + 1, q2 - q1 - 1);
            };
            model_cfg.input_dim = ex_num("d_model", model_cfg.input_dim);
            model_cfg.hidden_dim = ex_num("hidden_dim", model_cfg.hidden_dim);
            model_cfg.output_dim = ex_num("output_dim", model_cfg.output_dim);
            model_cfg.vocab_size = ex_num("vocab_size", model_cfg.vocab_size);
            model_cfg.max_seq_len = ex_num("max_seq_len", model_cfg.max_seq_len);
            model_cfg.causal_window_size = ex_num("causal_window_size", model_cfg.causal_window_size);
            model_cfg.sae_k = ex_num("sae_k", model_cfg.sae_k);
            model_cfg.ntm_memory_slots = ex_num("ntm_memory_slots", model_cfg.ntm_memory_slots);
            model_cfg.lm_num_attn_layers = ex_num("lm_num_attn_layers", model_cfg.lm_num_attn_layers);
            model_cfg.lm_pooling = ex_str("lm_pooling", "mean");
        }
    }

    CausalLMConfig lm_cfg;
    lm_cfg.vocab_size = model_cfg.vocab_size;
    lm_cfg.d_model = model_cfg.hidden_dim;
    lm_cfg.max_seq_len = model_cfg.max_seq_len;
    lm_cfg.causal_window_size = model_cfg.causal_window_size;
    lm_cfg.sae_k = model_cfg.sae_k;
    lm_cfg.ntm_memory_slots = model_cfg.ntm_memory_slots;
    lm_cfg.use_mla = model_cfg.use_mla;
    lm_cfg.mla_latent_dim = model_cfg.mla_latent_dim;
    lm_cfg.mla_n_heads = model_cfg.mla_n_heads;
    lm_cfg.mla_max_cache_len = 4096;
    lm_cfg.weight_tying = true;
    lm_cfg.num_attn_layers = model_cfg.lm_num_attn_layers;
    lm_cfg.num_attn_heads = 4;
    lm_cfg.pooling = model_cfg.lm_pooling;

    CausalLMHead lm_head(lm_cfg);
    if (lm_cfg.weight_tying) lm_head.tie_weights();

    std::cerr << "加载 LM Head: " << lm_head_path << std::endl;
    {
        std::ifstream ifs(lm_head_path, std::ios::binary);
        if (!ifs) {
            std::cerr << "错误: 无法打开 " << lm_head_path << std::endl;
            return 1;
        }
        char magic[5] = {0};
        ifs.read(magic, 4);
        if (std::string(magic) != "LMH2" && std::string(magic) != "LMH1") {
            std::cerr << "错误: 格式不匹配 " << std::string(magic) << std::endl;
            return 1;
        }

        std::unordered_map<std::string, Tensor*> tensor_map;
        tensor_map["w_embed"] = &lm_head.w_embed_;
        tensor_map["w_pos"] = &lm_head.w_pos_;
        tensor_map["dw_kernel"] = &lm_head.dw_kernel_;
        tensor_map["pw_conv.weight"] = &lm_head.pw_conv_->weight;
        tensor_map["sae_encode.weight"] = &lm_head.sae_w_encode_->weight;
        tensor_map["sae_decode.weight"] = &lm_head.sae_w_decode_->weight;
        tensor_map["ntm_read.weight"] = &lm_head.ntm_w_read_->weight;
        tensor_map["ntm_write.weight"] = &lm_head.ntm_w_write_->weight;
        tensor_map["ntm_erase.weight"] = &lm_head.ntm_w_erase_->weight;
        tensor_map["ntm_memory"] = &lm_head.ntm_memory_;
        tensor_map["w_proj.weight"] = &lm_head.w_proj_->weight;
        tensor_map["w_proj.bias"] = &lm_head.w_proj_->bias;
        tensor_map["w_out.weight"] = &lm_head.w_out_->weight;
        tensor_map["w_out.bias"] = &lm_head.w_out_->bias;
        tensor_map["ln.weight"] = &lm_head.ln_->weight;
        tensor_map["ln.bias"] = &lm_head.ln_->bias;
        for (size_t i = 0; i < lm_head.attn_layers_.size(); ++i) {
            std::string p = "attn" + std::to_string(i) + ".";
            tensor_map[p + "w_q.weight"] = &lm_head.attn_layers_[i]->w_q->weight;
            tensor_map[p + "w_q.bias"] = &lm_head.attn_layers_[i]->w_q->bias;
            tensor_map[p + "w_k.weight"] = &lm_head.attn_layers_[i]->w_k->weight;
            tensor_map[p + "w_k.bias"] = &lm_head.attn_layers_[i]->w_k->bias;
            tensor_map[p + "w_v.weight"] = &lm_head.attn_layers_[i]->w_v->weight;
            tensor_map[p + "w_v.bias"] = &lm_head.attn_layers_[i]->w_v->bias;
            tensor_map[p + "w_out.weight"] = &lm_head.attn_layers_[i]->w_out->weight;
            tensor_map[p + "w_out.bias"] = &lm_head.attn_layers_[i]->w_out->bias;
            tensor_map[p + "norm.weight"] = &lm_head.attn_layers_[i]->norm->weight;
            tensor_map[p + "norm.bias"] = &lm_head.attn_layers_[i]->norm->bias;
        }

        size_t loaded = 0;
        while (ifs) {
            uint32_t nl = 0;
            ifs.read((char*)&nl, 4);
            if (nl == 0 || ifs.eof()) break;
            if (nl > 256) { std::cerr << "异常名称长度: " << nl << ", 可能文件损坏" << std::endl; break; }
            std::string name(nl, '\0'); ifs.read(&name[0], nl);
            uint32_t nd = 0; ifs.read((char*)&nd, 4);
            if (nd > 10) { std::cerr << "异常维度: " << nd << ", name=" << name << std::endl; break; }
            std::vector<size_t> shape(nd);
            for (size_t i = 0; i < nd; ++i) { uint32_t d = 0; ifs.read((char*)&d, 4); shape[i] = d; }
            uint32_t ds = 0; ifs.read((char*)&ds, 4);
            Tensor t(shape, QuantType::FP32);
            if (t.data_size_ == ds) {
                ifs.read((char*)t.data_.get(), ds);
            } else {
                ifs.seekg(ds, std::ios::cur);
                continue;
            }
            auto it = tensor_map.find(name);
            if (it != tensor_map.end() && it->second->shape_ == t.shape_) {
                memcpy(it->second->data_.get(), t.data_.get(), t.data_size_);
                loaded++;
            } else if (name.find("w_qkv.") != std::string::npos) {
                std::string base = name.substr(0, name.find("w_qkv."));
                std::string suffix = name.substr(name.find("w_qkv.") + 6);
                for (size_t i = 0; i < lm_head.attn_layers_.size(); ++i) {
                    std::string p = "attn" + std::to_string(i) + ".";
                    if (base != p) continue;
                    size_t d_model = lm_head.config_.d_model;
                    size_t head_dim = d_model / lm_head.config_.num_attn_heads;
                    size_t n_q = lm_head.attn_layers_[i]->n_q_heads_;
                    size_t n_kv = lm_head.attn_layers_[i]->n_kv_heads_;
                    if (suffix == "weight" && t.shape_.size() == 2 && t.shape_[0] == 3 * d_model) {
                        const float* src = t.as_fp32();
                        float* dq = lm_head.attn_layers_[i]->w_q->weight.as_fp32();
                        float* dk = lm_head.attn_layers_[i]->w_k->weight.as_fp32();
                        float* dv = lm_head.attn_layers_[i]->w_v->weight.as_fp32();
                        for (size_t r = 0; r < d_model; ++r) {
                            memcpy(dq + r * n_q * head_dim, src + r * 3 * d_model, n_q * head_dim * sizeof(float));
                            memcpy(dk + r * n_kv * head_dim, src + r * 3 * d_model + d_model, n_kv * head_dim * sizeof(float));
                            memcpy(dv + r * n_kv * head_dim, src + r * 3 * d_model + 2 * d_model, n_kv * head_dim * sizeof(float));
                        }
                        loaded++;
                    } else if (suffix == "bias" && t.shape_[0] == 3 * d_model) {
                        const float* src = t.as_fp32();
                        float* bq = lm_head.attn_layers_[i]->w_q->bias.as_fp32();
                        float* bk = lm_head.attn_layers_[i]->w_k->bias.as_fp32();
                        float* bv = lm_head.attn_layers_[i]->w_v->bias.as_fp32();
                        memcpy(bq, src, n_q * head_dim * sizeof(float));
                        memcpy(bk, src + d_model, n_kv * head_dim * sizeof(float));
                        memcpy(bv, src + 2 * d_model, n_kv * head_dim * sizeof(float));
                        loaded++;
                    }
                }
            }
        }
        ifs.close();
        if (lm_cfg.weight_tying) lm_head.tie_weights();
        std::cerr << "已加载 " << loaded << " 个张量" << std::endl;
    }

    BPETokenizer tokenizer(tokenizer_path);
    std::cerr << "词表: " << tokenizer.vocab_size() << " tokens" << std::endl;

#ifdef USE_CUDA
    bool cuda_active = false;
    if (use_cuda) {
        if (CudaContext::instance().initialize(0)) {
            cuda_active = true;
            std::cerr << "GPU后端: 已启用" << std::endl;

            lm_head.w_embed_.to_gpu();
            lm_head.w_pos_.to_gpu();
            lm_head.dw_kernel_.to_gpu();
            lm_head.pw_conv_->weight.to_gpu();
            lm_head.sae_w_encode_->weight.to_gpu();
            lm_head.sae_w_decode_->weight.to_gpu();
            lm_head.ntm_w_read_->weight.to_gpu();
            lm_head.ntm_w_write_->weight.to_gpu();
            lm_head.ntm_w_erase_->weight.to_gpu();
            lm_head.ntm_memory_.to_gpu();
            lm_head.w_proj_->weight.to_gpu();
            lm_head.w_proj_->bias.to_gpu();
            lm_head.w_out_->weight.to_gpu();
            if (lm_head.w_out_->bias.data_) lm_head.w_out_->bias.to_gpu();
            lm_head.ln_->weight.to_gpu();
            lm_head.ln_->bias.to_gpu();
            for (auto& attn : lm_head.attn_layers_) {
                attn->w_q->weight.to_gpu();
                attn->w_q->bias.to_gpu();
                attn->w_k->weight.to_gpu();
                attn->w_k->bias.to_gpu();
                attn->w_v->weight.to_gpu();
                attn->w_v->bias.to_gpu();
                attn->w_out->weight.to_gpu();
                attn->w_out->bias.to_gpu();
                attn->norm->weight.to_gpu();
                attn->norm->bias.to_gpu();
            }
            CudaContext::instance().synchronize();
            size_t free_mem = CudaContext::instance().free_memory();
            size_t total_mem = CudaContext::instance().total_memory();
            std::cerr << "GPU显存: " << (total_mem - free_mem) / (1024*1024)
                      << " MB 已用 / " << total_mem / (1024*1024) << " MB 总计" << std::endl;
        } else {
            std::cerr << "[CUDA WARNING] GPU初始化失败，回退CPU后端" << std::endl;
            use_cuda = false;
        }
    } else {
        std::cerr << "GPU后端: 未启用 (使用--use-cuda启用)" << std::endl;
    }
#else
    if (use_cuda) {
        std::cerr << "[CUDA WARNING] 编译时未启用CUDA支持 (NEUROFLOW_USE_CUDA=OFF)，回退CPU后端" << std::endl;
        use_cuda = false;
    }
#endif

    lm_head.eval();

    if (yarn_max_seq_len > 0 && static_cast<size_t>(yarn_max_seq_len) > lm_cfg.max_seq_len) {
        float scale_factor = static_cast<float>(yarn_max_seq_len) / static_cast<float>(lm_cfg.max_seq_len);
        lm_head.set_yarn_scale(scale_factor);
    }

    std::cerr << "\n=== NeuroFlow 推理模式 ===" << std::endl;
    std::cerr << "输入提示（Ctrl+C退出）:" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "quit" || line == "exit") break;

        auto token_ids = tokenizer.encode(line, lm_cfg.max_seq_len);
        if (token_ids.empty()) {
            std::cerr << "(空token序列)" << std::endl;
            continue;
        }

        lm_head.clear_cache();

        std::vector<size_t> prefix(token_ids.begin(), token_ids.end() - 1);
        Tensor logits;
        if (prefix.size() > 0) {
            logits = lm_head.forward(prefix);
#ifdef USE_CUDA
            if (cuda_active && logits.is_on_gpu()) {
                logits.to_cpu();
            }
#endif
        }

        std::mt19937 rng(42);
        std::vector<size_t> generated;
        size_t last_id = token_ids.back();

        GenerateConfig gen_cfg;
        gen_cfg.max_new_tokens = static_cast<size_t>(max_new_tokens);
        gen_cfg.temperature = temperature;
        gen_cfg.top_k = static_cast<size_t>(top_k);
        gen_cfg.top_p = top_p;
        gen_cfg.repetition_penalty = repetition_penalty;
        gen_cfg.eos_id = 0;

        std::unique_ptr<SamplingStrategy> sampler;
        if (strategy_name == "greedy") {
            sampler = std::make_unique<GreedyDecoding>();
        } else if (strategy_name == "top_p") {
            sampler = std::make_unique<TopPSampling>();
        } else if (strategy_name == "top_k_top_p") {
            sampler = std::make_unique<TopKTopPSampling>();
        } else {
            sampler = std::make_unique<TopKSampling>();
        }

        for (int step = 0; step < max_new_tokens; ++step) {
            size_t pos = token_ids.size() - 1 + step;
            logits = lm_head.forward_step(last_id, pos);

#ifdef USE_CUDA
            if (cuda_active && logits.is_on_gpu()) {
                int d_sampled_token = 0;
                int h_sampled_token = 0;
                cudaMalloc(&d_sampled_token, sizeof(int));

                unsigned int seed = static_cast<unsigned int>(step * 7919 + 42);
                bool ok = launch_topk_topp_sampling(
                    logits.as_gpu_fp32(), d_sampled_token,
                    static_cast<int>(lm_cfg.vocab_size),
                    top_k, top_p, temperature, seed,
                    CudaContext::instance().stream());

                if (ok) {
                    CudaContext::instance().synchronize();
                    cudaMemcpy(&h_sampled_token, d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaFree(d_sampled_token);

                    size_t chosen = static_cast<size_t>(h_sampled_token);
                    generated.push_back(chosen);
                    last_id = chosen;

                    if (chosen == 0 || chosen == 1) break;
                    continue;
                } else {
                    cudaFree(d_sampled_token);
                    std::cerr << "[INFER WARNING] GPU sampling failed, falling back to CPU sampling" << std::endl;
                }

                logits.to_cpu();
            }
#endif

            Tensor probs = sampler->apply(std::move(logits), gen_cfg, generated);
            size_t chosen = sampler->sample(probs, rng);

            generated.push_back(chosen);
            last_id = chosen;

            if (chosen == 0 || chosen == 1) break;
        }

        std::string output = tokenizer.decode(generated);
        std::cout << output << std::endl;
    }

    return 0;
}
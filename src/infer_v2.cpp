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
    bool use_cuda = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--lm-head" && i + 1 < argc) lm_head_path = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) max_new_tokens = std::atoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::atof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) top_k = std::atoi(argv[++i]);
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
            tensor_map[p + "w_qkv.weight"] = &lm_head.attn_layers_[i]->w_qkv->weight;
            tensor_map[p + "w_qkv.bias"] = &lm_head.attn_layers_[i]->w_qkv->bias;
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
                attn->w_qkv->weight.to_gpu();
                attn->w_qkv->bias.to_gpu();
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

        for (int step = 0; step < max_new_tokens; ++step) {
            size_t pos = token_ids.size() - 1 + step;
            logits = lm_head.forward_step(last_id, pos);

#ifdef USE_CUDA
            if (cuda_active && logits.is_on_gpu()) {
                logits.to_cpu();
            }
#endif

            const float* pred = logits.as_fp32();
            size_t vocab_sz = lm_cfg.vocab_size;

            float max_val = -1e30f;
            for (size_t j = 0; j < vocab_sz; ++j) {
                if (pred[j] > max_val) max_val = pred[j];
            }

            std::vector<std::pair<float, size_t>> scored;
            float sum_exp = 0.0f;
            for (size_t j = 0; j < vocab_sz; ++j) {
                float val = std::exp((pred[j] - max_val) / temperature);
                sum_exp += val;
                scored.emplace_back(val, j);
            }

            std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

            float cumsum = 0.0f;
            size_t k = std::min(static_cast<size_t>(top_k), scored.size());
            std::discrete_distribution<size_t> dist;
            std::vector<float> weights(k);
            for (size_t i = 0; i < k; ++i) weights[i] = scored[i].first / sum_exp;
            std::discrete_distribution<size_t> topk_dist(weights.begin(), weights.end());
            size_t chosen = scored[topk_dist(rng)].second;

            generated.push_back(chosen);
            last_id = chosen;

            if (chosen == 0 || chosen == 1) break;
        }

        std::string output = tokenizer.decode(generated);
        std::cout << output << std::endl;
    }

    return 0;
}
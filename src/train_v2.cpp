#ifndef NOMINMAX
#define NOMINMAX
#endif

// C 系统头文件
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

// C++ 标准库
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// 第三方库
#ifdef _OPENMP
#include <omp.h>
#endif

// 项目头文件
#include "neuroflow/backprop.hpp"
#include "neuroflow/generative.hpp"
#include "neuroflow/model.hpp"
#include "neuroflow/online_learning.hpp"
#include "weight_io.hpp"

using neuroflow::BPETokenizer;
using neuroflow::CausalLMConfig;
using neuroflow::CausalLMHead;
using neuroflow::FullBackpropEngine;
using neuroflow::InitStrategy;
using neuroflow::NeuroFlowModel;
using neuroflow::QuantType;
using neuroflow::Tensor;
using neuroflow::WeightInitializer;

namespace {

Tensor lm_linear_backward_input(const Tensor& output_grad, const Tensor& weight) {
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

    memset(input_grad.as_fp32(), 0, input_grad.data_size_);

    if (out_f > 8192 && batch == 1) {
        const float* og = output_grad.as_fp32();
        const float* w = weight.as_fp32();
        float* ig = input_grad.as_fp32();
        float threshold = 1e-4f;
        size_t n_active = 0;
        for (size_t i = 0; i < out_f; ++i) {
            if (std::abs(og[i]) >= threshold) n_active++;
        }
        if (n_active < out_f / 4) {
            for (size_t i = 0; i < out_f; ++i) {
                float g = og[i];
                if (std::abs(g) < threshold) continue;
                const float* w_row = w + i * in_f;
                for (size_t j = 0; j < in_f; ++j) {
                    ig[j] += g * w_row[j];
                }
            }
            return input_grad;
        }
    }

#ifdef USE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch, in_f, out_f,
                1.0f, output_grad.as_fp32(), out_f, weight.as_fp32(), in_f,
                0.0f, input_grad.as_fp32(), in_f);
#else
    const float* og2 = output_grad.as_fp32();
    const float* w2 = weight.as_fp32();
    float* ig2 = input_grad.as_fp32();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t j = 0; j < in_f; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < out_f; ++i) {
                sum += og2[b * out_f + i] * w2[i * in_f + j];
            }
            ig2[b * in_f + j] = sum;
        }
    }
#endif
    return input_grad;
}

Tensor lm_linear_backward_weight(const Tensor& input, const Tensor& output_grad) {
    size_t batch = input.shape_[0];
    size_t in_f = input.shape_[1];
    size_t out_f = output_grad.shape_[1];
    Tensor weight_grad({out_f, in_f}, QuantType::FP32);
#ifdef USE_CUDA
    if (CudaContext::instance().is_available() && output_grad.is_on_gpu()) {
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
            for (size_t b = 0; b < batch; ++b) {
                sum += og[b * out_f + i] * inp[b * in_f + j];
            }
            wg[i * in_f + j] = sum / batch;
        }
    }
#endif
    return weight_grad;
}

}

struct TrainConfig {
    std::string config_path;
    std::string tokenizer_path;
    std::string data_path;
    std::string output_dir = "./output";
    int epochs = 10;
    float learning_rate = 3e-5f;
    uint32_t seed = 42;
    std::string resume_path = "";
    std::string init_strategy = "xavier";
    int batch_size = 256;
    int log_interval = 50;
    int save_interval = 5000;
    float grad_clip = 4.0f;
    bool use_adam = false;
    int grad_accum_steps = 1;
    std::string vocab_mask_path = "";
    int replay_buffer_size = 10000;
    float replay_ratio = 0.25f;
    bool use_cuda = false;
    bool verbose = false;
};

TrainConfig parse_args(int argc, char* argv[]) {
    TrainConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) cfg.config_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) cfg.tokenizer_path = argv[++i];
        else if (arg == "--data" && i + 1 < argc) cfg.data_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) cfg.output_dir = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) cfg.epochs = std::atoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) cfg.learning_rate = std::atof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) cfg.seed = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--resume" && i + 1 < argc) cfg.resume_path = argv[++i];
        else if (arg == "--init-weights" && i + 1 < argc) cfg.init_strategy = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) cfg.batch_size = std::atoi(argv[++i]);
        else if (arg == "--log-interval" && i + 1 < argc) cfg.log_interval = std::atoi(argv[++i]);
        else if (arg == "--save-interval" && i + 1 < argc) cfg.save_interval = std::atoi(argv[++i]);
        else if (arg == "--grad-clip" && i + 1 < argc) cfg.grad_clip = std::atof(argv[++i]);
        else if (arg == "--adam") cfg.use_adam = true;
        else if (arg == "--grad-accum" && i + 1 < argc) cfg.grad_accum_steps = std::atoi(argv[++i]);
        else if (arg == "--vocab-mask" && i + 1 < argc) cfg.vocab_mask_path = argv[++i];
        else if (arg == "--replay-buffer" && i + 1 < argc) cfg.replay_buffer_size = std::atoi(argv[++i]);
        else if (arg == "--replay-ratio" && i + 1 < argc) cfg.replay_ratio = std::atof(argv[++i]);
        else if (arg == "--use-cuda") cfg.use_cuda = true;
        else if (arg == "--verbose") cfg.verbose = true;
    }
    return cfg;
}

namespace {

std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return "";
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size() || json[pos] != '"') return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

size_t extract_json_number(const std::string& json, const std::string& key, size_t default_val = 0) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    size_t end = pos;
    while (end < json.size() && (json[end] >= '0' && json[end] <= '9')) end++;
    if (end == pos) return default_val;
    return std::stoul(json.substr(pos, end - pos));
}

bool extract_json_bool(const std::string& json, const std::string& key, bool default_val = false) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos + 3 < json.size() && json.substr(pos, 4) == "true") return true;
    if (pos + 4 < json.size() && json.substr(pos, 5) == "false") return false;
    return default_val;
}

}

NeuroFlowModel::Config load_model_config(const std::string& json_path) {
    NeuroFlowModel::Config cfg;
    std::string json = read_file(json_path);
    if (json.empty()) {
        std::cerr << "配置加载: " << json_path << " (使用默认配置)" << std::endl;
        return cfg;
    }

    cfg.input_dim = extract_json_number(json, "d_model", cfg.input_dim);
    cfg.hidden_dim = extract_json_number(json, "hidden_dim", cfg.hidden_dim);
    cfg.output_dim = extract_json_number(json, "output_dim", cfg.output_dim);
    cfg.memory_dim = extract_json_number(json, "memory_dim", cfg.memory_dim);
    cfg.memory_slots = extract_json_number(json, "memory_slots", cfg.memory_slots);
    cfg.num_layers = extract_json_number(json, "num_layers", cfg.num_layers);
    cfg.num_associations = extract_json_number(json, "num_associations", cfg.num_associations);
    cfg.use_quantization = extract_json_bool(json, "use_quantization", cfg.use_quantization);
    cfg.use_mla = extract_json_bool(json, "use_mla", cfg.use_mla);
 cfg.use_causal_lm = extract_json_bool(json, "use_causal_lm", cfg.use_causal_lm);
    cfg.mla_latent_dim = extract_json_number(json, "mla_latent_dim", cfg.mla_latent_dim);
    cfg.vocab_size = extract_json_number(json, "vocab_size", cfg.vocab_size);
    cfg.max_seq_len = extract_json_number(json, "max_seq_len", cfg.max_seq_len);
    cfg.causal_window_size = extract_json_number(json, "causal_window_size", cfg.causal_window_size);
    cfg.lm_num_attn_layers = extract_json_number(json, "lm_num_attn_layers", cfg.lm_num_attn_layers);
    cfg.lm_pooling = extract_json_string(json, "lm_pooling");
    if (cfg.lm_pooling.empty()) cfg.lm_pooling = "mean";

    std::cerr << "配置加载: " << json_path << std::endl;
    std::cerr << "  d_model=" << cfg.input_dim << " hidden_dim=" << cfg.hidden_dim
              << " output_dim=" << cfg.output_dim << " vocab_size=" << cfg.vocab_size << std::endl;
    return cfg;
}

struct ResumeState {
    bool found = false;
    size_t step = 0;
    int epoch = 0;
    float loss = 0.0f;
    float lr = 0.0f;
    size_t config_d_model = 0;
    size_t config_output_dim = 0;
};

ResumeState load_checkpoint(NeuroFlowModel& model, const std::string& ckpt_path) {
    ResumeState state;

    std::string model_path, state_path;
    if (ckpt_path.size() >= 5 && ckpt_path.substr(ckpt_path.size() - 5) == ".nfv1") {
        model_path = ckpt_path;
        state_path = ckpt_path.substr(0, ckpt_path.size() - 5) + "/../training_state.json";
        std::string dir = ckpt_path.substr(0, ckpt_path.find_last_of("/\\"));
        state_path = dir + "/training_state.json";
    } else {
        model_path = ckpt_path + "/model.nfv1";
        state_path = ckpt_path + "/training_state.json";
    }

    std::ifstream ifs(model_path, std::ios::binary);
    if (!ifs) return state;
    char magic[5] = {0};
    ifs.read(magic, 4);
    if (std::string(magic) != "NFv1") return state;
    ifs.close();

    std::ifstream state_ifs(state_path);
    if (state_ifs) {
        std::string content((std::istreambuf_iterator<char>(state_ifs)),
                            std::istreambuf_iterator<char>());
        state.step = extract_json_number(content, "step", 0);
        state.epoch = static_cast<int>(extract_json_number(content, "epoch", 0));
        state.loss = std::atof(extract_json_string(content, "loss").c_str());
        state.lr = std::atof(extract_json_string(content, "learning_rate").c_str());
        state.config_d_model = extract_json_number(content, "config_d_model", 0);
        state.config_output_dim = extract_json_number(content, "config_output_dim", 0);
    }

    if (state.config_d_model > 0 && state.config_d_model != model.config.input_dim) {
        std::cerr << "警告: checkpoint的d_model=" << state.config_d_model
                  << " 与当前config的d_model=" << model.config.input_dim
                  << " 不匹配，可能导致维度错误！" << std::endl;
    }
    if (state.config_output_dim > 0 && state.config_output_dim != model.config.output_dim) {
        std::cerr << "警告: checkpoint的output_dim=" << state.config_output_dim
                  << " 与当前config的output_dim=" << model.config.output_dim
                  << " 不匹配！" << std::endl;
    }

    model.load(model_path);
    state.found = true;
    return state;
}

bool load_lm_head(CausalLMHead& lm_head, const std::string& lm_path) {
    std::ifstream ifs(lm_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "LM Head checkpoint未找到: " << lm_path << std::endl;
        return false;
    }

    char magic[5] = {0};
    ifs.read(magic, 4);
    if (std::string(magic) != "LMH2" && std::string(magic) != "LMH1") {
        std::cerr << "LM Head格式不匹配: " << std::string(magic) << " (期望LMH2/LMH1)" << std::endl;
        ifs.close();
        return false;
    }

    auto read_named_tensor = [&]() -> std::pair<std::string, Tensor> {
        uint32_t nl = 0; ifs.read((char*)&nl, 4);
        if (nl == 0 || ifs.eof()) return {"", Tensor()};
        std::string name(nl, '\0'); ifs.read(&name[0], nl);
        uint32_t nd = 0; ifs.read((char*)&nd, 4);
        std::vector<size_t> shape(nd);
        for (size_t i = 0; i < nd; ++i) { uint32_t d = 0; ifs.read((char*)&d, 4); shape[i] = d; }
        uint32_t ds = 0; ifs.read((char*)&ds, 4);
        Tensor t(shape, QuantType::FP32);
        if (t.data_size_ == ds) {
            ifs.read((char*)t.data_.get(), ds);
        } else {
            ifs.seekg(ds, std::ios::cur);
            t = Tensor();
        }
        return {name, std::move(t)};
    };

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
        auto [name, tensor] = read_named_tensor();
        if (name.empty()) break;
        auto it = tensor_map.find(name);
        if (it != tensor_map.end() && tensor.numel() > 0) {
            if (it->second->shape_ == tensor.shape_) {
                memcpy(it->second->data_.get(), tensor.data_.get(), tensor.data_size_);
                loaded++;
            } else {
                std::cerr << "  跳过 '" << name << "': 形状不匹配" << std::endl;
            }
        }
    }

    ifs.close();
    if (lm_head.config_.weight_tying) lm_head.tie_weights();
    std::cerr << "LM Head已恢复: " << loaded << " 个张量从 " << lm_path << std::endl;
    return loaded > 0;
}

NeuroFlowModel build_model(const NeuroFlowModel::Config& cfg, const TrainConfig& train_cfg,
                            ResumeState& resume_state) {
    NeuroFlowModel model(cfg);
    if (!train_cfg.resume_path.empty()) {
        resume_state = load_checkpoint(model, train_cfg.resume_path);
        if (resume_state.found) {
            std::cerr << "从checkpoint恢复: " << train_cfg.resume_path
                      << " (step=" << resume_state.step
                      << ", epoch=" << resume_state.epoch
                      << ", loss=" << resume_state.loss << ")" << std::endl;
        } else {
            throw std::runtime_error("无法加载checkpoint: " + train_cfg.resume_path);
        }
    } else {
        InitStrategy strategy = InitStrategy::XAVIER_UNIFORM;
        if (train_cfg.init_strategy == "kaiming") strategy = InitStrategy::KAIMING_NORMAL;
        else if (train_cfg.init_strategy == "zeros") strategy = InitStrategy::ZEROS;
        WeightInitializer::init_model_weights(model, strategy, train_cfg.seed);
        std::cerr << "权重初始化: " << train_cfg.init_strategy << std::endl;
    }
    return model;
}

struct LMSample {
    std::vector<size_t> token_ids;
};

struct TrainingSample {
    std::vector<float> input;
    std::vector<float> target;
};

class DataLoader {
public:
    enum class Mode { LM, CLASSIFICATION, REGRESSION };

    DataLoader(const std::string& path, int batch_size, size_t input_dim, size_t output_dim,
               Mode mode = Mode::LM, BPETokenizer* tokenizer = nullptr,
               size_t max_seq_len = 128, size_t max_samples = 500000,
               size_t max_memory_mb = 4096)
        : path_(path), batch_size_(batch_size), input_dim_(input_dim), output_dim_(output_dim),
          current_idx_(0), mode_(mode), tokenizer_(tokenizer), max_seq_len_(max_seq_len),
          max_samples_(max_samples), max_memory_bytes_(max_memory_mb * 1024ULL * 1024ULL) {
        load_data();
    }

    void load_data() {
        if (path_.empty()) {
            throw std::runtime_error("DataLoader: 数据路径为空，请指定--data参数");
        }

        namespace fs = std::filesystem;
        fs::path p(path_);

        if (!fs::exists(p)) {
            throw std::runtime_error("DataLoader: 路径不存在: " + path_);
        }

        if (fs::is_directory(p)) {
            load_directory(p.string());
        } else {
            load_file(p.string());
        }

        if (lm_samples_.empty() && samples_.empty()) {
            throw std::runtime_error(
                "DataLoader: 未能从 " + path_ + " 加载任何训练数据。"
                "请检查文件格式（支持txt/json/jsonl/csv/tsv）和内容是否有效。");
        }

        if (!lm_samples_.empty()) {
            std::cerr << "DataLoader: " << lm_samples_.size() << " LM样本已加载" << std::endl;
        } else {
            std::cerr << "DataLoader: " << samples_.size() << " 数值样本已加载" << std::endl;
        }
    }

    void load_directory(const std::string& dir_path) {
        namespace fs = std::filesystem;
        size_t file_count = 0;
        size_t skipped = 0;

        std::cerr << "DataLoader: 扫描目录 " << dir_path << " ..." << std::endl;
        std::vector<fs::path> files;
        for (auto& entry : fs::recursive_directory_iterator(dir_path, fs::directory_options::follow_directory_symlink)) {
            if (!entry.is_regular_file()) continue;
            files.push_back(entry.path());
        }
        std::cerr << "DataLoader: 发现 " << files.size() << " 个文件" << std::endl;
        std::sort(files.begin(), files.end());

        for (size_t fi = 0; fi < files.size(); ++fi) {
            auto& fpath = files[fi];
            std::string ext = fpath.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".json" || ext == ".jsonl" || ext == ".txt" ||
                ext == ".csv" || ext == ".tsv" || ext == ".tok1" || ext == ".bin") {
                size_t file_size_mb = 0;
                try { file_size_mb = fs::file_size(fpath) / 1024 / 1024; } catch (...) {}
                std::cerr << "[" << (fi + 1) << "/" << files.size() << "] 加载: "
                          << fpath.filename().string()
                          << " (" << file_size_mb << " MB)" << std::endl;

                auto load_start = std::chrono::steady_clock::now();
                try {
                    load_file(fpath.string());
                    file_count++;
                } catch (const std::exception& e) {
                    std::cerr << "  跳过: " << e.what() << std::endl;
                    skipped++;
                }
                auto load_end = std::chrono::steady_clock::now();
                double load_sec = std::chrono::duration<double>(load_end - load_start).count();

                size_t total = lm_samples_.size() + samples_.size();
                std::cerr << "  完成: " << load_sec << "s, 累计 "
                          << total << " 样本" << std::endl;

                if (total >= max_samples_) {
                    std::cerr << "DataLoader: 已达采样上限 " << max_samples_
                              << "，停止遍历" << std::endl;
                    break;
                }
            } else if (ext == ".parquet") {
                skipped++;
            }
        }
        std::cerr << "DataLoader: 遍历 " << file_count << " 个文件, 跳过 " << skipped << std::endl;
    }

    void load_file(const std::string& file_path) {
        namespace fs = std::filesystem;
        std::string ext = fs::path(file_path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        {
            std::ifstream ifs(file_path, std::ios::binary);
            if (ifs) {
                char magic[5] = {0};
                ifs.read(magic, 4);
                ifs.close();
                if (std::string(magic) == "TOK1") {
                    load_tok1(file_path);
                    return;
                }
            }
        }

        if (mode_ == Mode::LM && tokenizer_) {
            if (ext == ".json") {
                size_t file_size = fs::file_size(file_path);
                if (file_size > 512 * 1024 * 1024) {
                    load_json_mmap(file_path);
                } else {
                    load_json(file_path);
                }
            }
            else if (ext == ".jsonl") load_jsonl(file_path);
            else if (ext == ".txt") load_txt(file_path);
            else if (ext == ".csv" || ext == ".tsv") load_txt(file_path);
            else if (ext == ".tok1" || ext == ".bin") load_tok1(file_path);
            else if (ext == ".parquet") {
                throw std::runtime_error("Parquet格式需Python预处理: python3 scripts/preprocess_corpus.py split --input <path>");
            } else {
                throw std::runtime_error("不支持的文件格式: " + ext);
            }
        } else {
            if (ext == ".csv" || ext == ".tsv") load_csv(file_path);
            else if (ext == ".tok1" || ext == ".bin") load_tok1(file_path);
            else {
                throw std::runtime_error("数值模式仅支持csv/tsv/tok1格式，收到: " + ext);
            }
        }
    }

    void load_tok1(const std::string& file_path) {
        std::ifstream ifs(file_path, std::ios::binary);
        if (!ifs) throw std::runtime_error("无法打开TOK1文件: " + file_path);

        char magic[5] = {0};
        ifs.read(magic, 4);
        if (std::string(magic) != "TOK1") {
            throw std::runtime_error("TOK1格式Magic不匹配: " + std::string(magic));
        }

        uint16_t version = 0;
        ifs.read(reinterpret_cast<char*>(&version), 2);

        uint32_t vocab_size = 0, max_seq = 0, total_samples = 0;
        ifs.read(reinterpret_cast<char*>(&vocab_size), 4);
        ifs.read(reinterpret_cast<char*>(&max_seq), 4);
        ifs.read(reinterpret_cast<char*>(&total_samples), 4);

        std::cerr << "TOK1: version=" << version << " vocab=" << vocab_size
                  << " max_seq=" << max_seq << " samples=" << total_samples << std::endl;

        size_t loaded = 0;
        for (size_t i = 0; i < total_samples; ++i) {
            uint16_t seq_len = 0;
            ifs.read(reinterpret_cast<char*>(&seq_len), 2);
            if (!ifs) break;

            LMSample sample;
            sample.token_ids.resize(seq_len);
            ifs.read(reinterpret_cast<char*>(sample.token_ids.data()), seq_len * 4);
            if (!ifs) break;

            lm_samples_.push_back(std::move(sample));
            loaded++;

            if (lm_samples_.size() >= max_samples_) {
                std::cerr << "TOK1: 达到采样上限 " << max_samples_ << std::endl;
                break;
            }
        }

        std::cerr << "TOK1: 加载 " << loaded << " 样本" << std::endl;
    }

    void load_json_mmap(const std::string& file_path) {
#ifdef _WIN32
        load_json_chunked(file_path);
#else
        int fd = open(file_path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("无法mmap: " + file_path);

        size_t file_size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);

        void* addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap失败: " + file_path);
        }

        const char* data = static_cast<const char*>(addr);
        std::cerr << "mmap: 映射 " << file_path << " (" << file_size / 1024 / 1024 << " MB)" << std::endl;

        size_t pos = 0;
        size_t brace_depth = 0;
        std::string record;
        const char* fields[] = {"\"text\"", "\"content\"", "\"title\"", "\"question\"", "\"answer\""};

        while (pos < file_size && lm_samples_.size() < max_samples_) {
            char ch = data[pos];
            if (ch == '{') {
                if (brace_depth == 0) record.clear();
                brace_depth++;
            } else if (ch == '}') {
                brace_depth--;
                if (brace_depth == 0 && !record.empty()) {
                    record += '}';
                    extract_text_from_json_record(record);
                    record.clear();
                }
            }
            if (brace_depth > 0) record += ch;
            pos++;

            if (lm_samples_.size() % 100000 == 0 && lm_samples_.size() > 0 && pos % (100 * 1024 * 1024) == 0) {
                std::cerr << "  mmap进度: " << pos * 100 / file_size << "%, "
                          << lm_samples_.size() << " 样本" << std::endl;
            }
        }

        munmap(addr, file_size);
        close(fd);
        std::cerr << "mmap: 完成, " << lm_samples_.size() << " 样本" << std::endl;
#endif
    }

    void load_json_chunked(const std::string& file_path) {
        const size_t chunk_size = 256 * 1024 * 1024;
        std::ifstream ifs(file_path, std::ios::binary);
        if (!ifs) throw std::runtime_error("无法打开: " + file_path);

        size_t file_size = std::filesystem::file_size(file_path);
        std::cerr << "分块读取: " << file_path << " (" << file_size / 1024 / 1024 << " MB)" << std::endl;

        size_t offset = 0;
        size_t brace_depth = 0;
        std::string record;
        std::string overlap;

        while (offset < file_size && lm_samples_.size() < max_samples_) {
            size_t read_size = std::min(chunk_size, file_size - offset);
            std::string buffer;
            buffer.resize(read_size);
            ifs.read(&buffer[0], read_size);
            if (!ifs) break;

            if (!overlap.empty()) {
                buffer = overlap + buffer;
                overlap.clear();
            }

            size_t pos = 0;
            while (pos < buffer.size() && lm_samples_.size() < max_samples_) {
                char ch = buffer[pos];
                if (ch == '{') {
                    if (brace_depth == 0) record.clear();
                    brace_depth++;
                } else if (ch == '}') {
                    brace_depth--;
                    if (brace_depth == 0 && !record.empty()) {
                        record += '}';
                        extract_text_from_json_record(record);
                        record.clear();
                    }
                }
                if (brace_depth > 0 && record.size() < 1048576) record += ch;
                pos++;
            }

            if (brace_depth > 0) {
                overlap = record;
            }

            offset += read_size;
            std::cerr << "  分块进度: " << offset * 100 / file_size << "%, "
                      << lm_samples_.size() << " 样本" << std::endl;
        }

        if (!record.empty() && brace_depth == 0) {
            extract_text_from_json_record(record);
        }
    }

    void load_json(const std::string& file_path) {
        std::ifstream ifs(file_path);
        if (!ifs) throw std::runtime_error("无法打开: " + file_path);

        std::string line;
        size_t brace_depth = 0;
        std::string record;
        size_t records_found = 0;

        while (std::getline(ifs, line)) {
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] == '{') {
                    if (brace_depth == 0) record.clear();
                    brace_depth++;
                } else if (line[i] == '}') {
                    brace_depth--;
                    if (brace_depth == 0 && !record.empty()) {
                        record += '}';
                        extract_text_from_json_record(record);
                        record.clear();
                        records_found++;
                        if (lm_samples_.size() >= max_samples_) return;
                    }
                }
            }
            if (brace_depth > 0) {
                if (record.size() < 1048576) {
                    record += line;
                    record += '\n';
                }
            }
        }

        if (records_found == 0) {
            std::cerr << "  警告: 流式record解析未找到数据，尝试逐行文本提取" << std::endl;
            ifs.clear();
            ifs.seekg(0);
            std::string line;
            while (std::getline(ifs, line)) {
                if (line.size() < 20) continue;
                size_t q1 = line.find('"');
                if (q1 == std::string::npos) continue;
                size_t q2 = line.rfind('"');
                if (q2 <= q1 + 1) continue;
                std::string text = line.substr(q1 + 1, q2 - q1 - 1);
                unescape_json(text);
                if (text.size() >= 10) add_lm_sample(text);
                if (lm_samples_.size() >= max_samples_) return;
            }
        }
    }

    void extract_text_from_json_record(const std::string& record) {
        const char* fields[] = {"\"text\"", "\"content\"", "\"title\"", "\"question\"", "\"answer\""};
        for (auto& field : fields) {
            size_t pos = record.find(field);
            if (pos == std::string::npos) continue;

            size_t colon = record.find(':', pos + strlen(field));
            if (colon == std::string::npos) continue;
            colon++;
            while (colon < record.size() && record[colon] != '"') colon++;
            if (colon >= record.size()) continue;
            colon++;

            size_t end = colon;
            while (end < record.size() && record[end] != '"') {
                if (record[end] == '\\') end++;
                end++;
            }

            std::string text = record.substr(colon, end - colon);
            unescape_json(text);
            if (text.size() >= 10) add_lm_sample(text);
        }
    }

    void extract_all_text_fields(const std::string& content) {
        size_t pos = 0;
        const char* fields[] = {"\"text\"", "\"content\"", "\"title\"", "\"question\"", "\"answer\""};
        while (pos < content.size()) {
            size_t best_pos = std::string::npos;
            for (auto& field : fields) {
                size_t p = content.find(field, pos);
                if (p != std::string::npos && (best_pos == std::string::npos || p < best_pos)) {
                    best_pos = p;
                }
            }
            if (best_pos == std::string::npos) break;

            size_t colon = content.find(':', best_pos);
            if (colon == std::string::npos) break;
            colon++;
            while (colon < content.size() && content[colon] != '"') colon++;
            if (colon >= content.size()) break;
            colon++;

            size_t end = colon;
            while (end < content.size() && content[end] != '"') {
                if (content[end] == '\\') end++;
                end++;
            }
            if (end >= content.size()) break;

            std::string text = content.substr(colon, end - colon);
            unescape_json(text);
            add_lm_sample(text);

            pos = end + 1;
            if (lm_samples_.size() >= max_samples_) return;
        }
    }

    void load_jsonl(const std::string& file_path) {
        std::ifstream ifs(file_path);
        if (!ifs) throw std::runtime_error("无法打开: " + file_path);
        std::string line;
        size_t line_num = 0;
        while (std::getline(ifs, line)) {
            line_num++;
            if (line.empty() || line[0] == '#') continue;

            const char* fields[] = {"\"text\"", "\"content\"", "\"title\"", "\"question\"", "\"answer\""};
            bool found = false;
            for (auto& field : fields) {
                size_t text_pos = line.find(field);
                if (text_pos == std::string::npos) continue;

                size_t colon = line.find(':', text_pos);
                if (colon == std::string::npos) continue;
                colon++;
                while (colon < line.size() && line[colon] != '"') colon++;
                if (colon >= line.size()) continue;
                colon++;

                size_t end = colon;
                while (end < line.size() && line[end] != '"') {
                    if (line[end] == '\\') end++;
                    end++;
                }

                std::string text = line.substr(colon, end - colon);
                unescape_json(text);
                add_lm_sample(text);
                found = true;
                break;
            }
            if (!found && line_num <= 5) {
                std::cerr << "  jsonl行" << line_num << ": 未找到text/content字段" << std::endl;
            }
            if (lm_samples_.size() >= max_samples_) return;
        }
    }

    void load_txt(const std::string& file_path) {
        std::ifstream ifs(file_path);
        if (!ifs) throw std::runtime_error("无法打开: " + file_path);
        std::string line;
        std::string paragraph;
        size_t line_count = 0;

        while (std::getline(ifs, line)) {
            line_count++;
            if (line.empty()) {
                if (paragraph.size() >= 10) {
                    add_lm_sample(paragraph);
                    paragraph.clear();
                    if (lm_samples_.size() >= max_samples_) return;
                }
            } else {
                if (!paragraph.empty()) paragraph += '\n';
                paragraph += line;
                if (paragraph.size() > 10000) {
                    add_lm_sample(paragraph);
                    paragraph.clear();
                    if (lm_samples_.size() >= max_samples_) return;
                }
            }
            if (line_count % 500000 == 0) {
                std::cerr << "  load_txt: " << line_count << " 行, "
                          << lm_samples_.size() << " 样本" << std::endl;
            }
        }
        if (paragraph.size() >= 10) add_lm_sample(paragraph);
    }

    void load_csv(const std::string& file_path) {
        std::ifstream ifs(file_path);
        if (!ifs) throw std::runtime_error("无法打开: " + file_path);
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;
            TrainingSample s;
            std::stringstream ss(line);
            std::string val;
            char delim = ',';
            if (line.find('\t') != std::string::npos && line.find(',') == std::string::npos) {
                delim = '\t';
            }
            bool in_target = false;
            while (std::getline(ss, val, delim)) {
                float v = std::atof(val.c_str());
                if (!in_target) {
                    s.input.push_back(v);
                    if (s.input.size() == input_dim_) in_target = true;
                } else {
                    s.target.push_back(v);
                }
            }
            if (!s.input.empty() && !s.target.empty()) {
                samples_.push_back(std::move(s));
            }
            if (samples_.size() >= max_samples_) return;
        }
    }

    void add_lm_sample(const std::string& text) {
    if (text.size() < 10) return;
    if (!tokenizer_) return;
    if (lm_samples_.size() >= max_samples_) return;
    if (memory_usage_bytes() >= max_memory_bytes_) {
    if (lm_samples_.size() % 100000 == 1) {
    std::cerr << " 内存预算达到 " << max_memory_bytes_ / 1024 / 1024
    << " MB，停止加载" << std::endl;
    }
    return;
    }
    // BPE安全截断: 限制输入长度, 防止超长段落卡死encode
    // max_seq_len_已由DataLoader构造时设定(默认128),
    // 但text字符数可能远大于token数(中文3字节/token)
    // 安全上限: max_seq_len_ * 4 字符 (留足UTF-8余量)
    const size_t max_text_len = max_seq_len_ * 4;
    if (text.size() > max_text_len) {
    // 截断到安全长度, 保留前max_text_len字符
    auto ids = tokenizer_->encode(text.substr(0, max_text_len), max_seq_len_);
    if (ids.size() >= 4) {
    lm_samples_.push_back(LMSample{std::move(ids)});
    }
    return;
    }
    auto ids = tokenizer_->encode(text, max_seq_len_);
    if (ids.size() < 4) return;
    lm_samples_.push_back(LMSample{std::move(ids)});
    }

    static void unescape_json(std::string& s) {
        size_t pos = 0;
        while ((pos = s.find('\\', pos)) != std::string::npos) {
            if (pos + 1 < s.size()) {
                char c = s[pos + 1];
                if (c == 'n') { s.replace(pos, 2, "\n"); pos++; }
                else if (c == 't') { s.replace(pos, 2, "\t"); pos++; }
                else if (c == 'r') { s.replace(pos, 2, "\r"); pos++; }
                else if (c == '"') { s.replace(pos, 2, "\""); pos++; }
                else if (c == '\\') { s.replace(pos, 2, "\\"); pos++; }
                else if (c == '/') { s.replace(pos, 2, "/"); pos++; }
                else if (c == 'u' && pos + 5 < s.size()) { pos += 6; }
                else { pos += 2; }
            } else { pos++; }
        }
    }

    bool has_next() const {
        if (mode_ == Mode::LM) return current_idx_ < lm_samples_.size();
        return current_idx_ < samples_.size();
    }

    std::vector<TrainingSample> next_batch() {
        std::vector<TrainingSample> batch;
        if (mode_ == Mode::LM) {
            float vocab_scale = 1.0f;
            if (tokenizer_) vocab_scale = 1.0f / static_cast<float>(tokenizer_->vocab_size());

            for (int i = 0; i < batch_size_ && current_idx_ < lm_samples_.size(); ++i) {
                auto& sample = lm_samples_[current_idx_++];
                TrainingSample s;
                s.input.resize(input_dim_, 0.0f);
                s.target.resize(output_dim_, 0.0f);
                size_t n = std::min(sample.token_ids.size(), input_dim_);

                for (size_t j = 0; j < n; ++j) {
                    s.input[j] = static_cast<float>(sample.token_ids[j]) * vocab_scale;
                }

                for (size_t j = 0; j + 1 < n; ++j) {
                    size_t next_id = sample.token_ids[j + 1];
                    if (next_id < output_dim_) {
                        s.target[next_id] = 1.0f;
                    }
                }
                batch.push_back(std::move(s));
            }
        } else {
            for (int i = 0; i < batch_size_ && current_idx_ < samples_.size(); ++i) {
                batch.push_back(samples_[current_idx_++]);
            }
        }
        return batch;
    }

    std::vector<LMSample> next_lm_batch_raw() {
        std::vector<LMSample> batch;
        for (int i = 0; i < batch_size_ && current_idx_ < lm_samples_.size(); ++i) {
            batch.push_back(lm_samples_[current_idx_++]);
        }
        return batch;
    }

    void reset() { current_idx_ = 0; }
    void shuffle(std::mt19937& rng) {
        if (mode_ == Mode::LM) {
            std::shuffle(lm_samples_.begin(), lm_samples_.end(), rng);
        } else {
            std::shuffle(samples_.begin(), samples_.end(), rng);
        }
    }
    size_t total_samples() const {
        if (mode_ == Mode::LM) return lm_samples_.size();
        return samples_.size();
    }
    Mode mode() const { return mode_; }
    size_t memory_usage_bytes() const {
        size_t total = 0;
        for (auto& s : lm_samples_) total += s.token_ids.size() * sizeof(size_t);
        for (auto& s : samples_) total += (s.input.size() + s.target.size()) * sizeof(float);
        return total;
    }

private:
    std::string path_;
    int batch_size_;
    size_t input_dim_;
    size_t output_dim_;
    size_t current_idx_;
    Mode mode_;
    BPETokenizer* tokenizer_;
    size_t max_seq_len_;
    size_t max_samples_;
    size_t max_memory_bytes_;
    std::vector<TrainingSample> samples_;
    std::vector<LMSample> lm_samples_;
};

struct TrainMetrics {
    float loss = 0.0f;
    float lr = 0.0f;
    size_t step = 0;
    int epoch = 0;
    float elapsed_seconds = 0.0f;
    float grad_norm = 0.0f;
};

class MetricsLogger {
public:
    MetricsLogger() : start_time_(std::chrono::steady_clock::now()) {}

    void log_step(const TrainMetrics& m, int interval) {
        if (m.step % interval != 0) return;
        std::cerr << "[Epoch " << m.epoch << "][Step " << m.step
                  << "] loss=" << m.loss << " lr=" << m.lr
                  << " grad_norm=" << m.grad_norm
                  << " elapsed=" << m.elapsed_seconds << "s" << std::endl;
    }

    void log_epoch(int epoch, float avg_loss, float elapsed) {
        std::cerr << "=== Epoch " << epoch << " 完成, avg_loss=" << avg_loss
                  << ", elapsed=" << elapsed << "s ===" << std::endl;
    }

    void save_training_log(const std::string& path, const std::vector<float>& losses) {
        std::ofstream ofs(path);
        if (!ofs) return;
        ofs << "{\n  \"losses\": [";
        for (size_t i = 0; i < losses.size(); ++i) {
            if (i > 0) ofs << ", ";
            ofs << losses[i];
        }
        ofs << "],\n  \"num_epochs\": " << losses.size() << "\n}";
        ofs.close();
    }

    double total_elapsed() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

private:
    std::chrono::steady_clock::time_point start_time_;
};

void ensure_dir_exists(const std::string& dir) {
    std::filesystem::create_directories(dir);
}

void save_checkpoint(const NeuroFlowModel& model, const TrainMetrics& metrics,
 const std::string& output_dir, size_t step, int epoch,
 const TrainConfig& cfg) {
 // ═══════════════════════════════════════════════════════
 // RAM Disk加速: 先写/dev/shm (1.3GB/s), 再后台拷贝到D盘 (125MB/s)
 // 307MB checkpoint: shm写0.2s vs D盘写2.5s — 节省2.3s/次
 // ═══════════════════════════════════════════════════════
 std::string ckpt_name = "checkpoint_step" + std::to_string(step);
 std::string ckpt_dir = output_dir + "/" + ckpt_name;
 ensure_dir_exists(ckpt_dir);

 // 1) 快速写入RAM Disk
 std::string shm_dir = "/dev/shm/nf_ckpt/" + ckpt_name;
 ensure_dir_exists(shm_dir);
 model.save(shm_dir + "/model.nfv1");

 std::ofstream shm_ofs(shm_dir + "/training_state.json");
 if (shm_ofs) {
 shm_ofs << "{\n"
 << " \"step\": " << step << ",\n"
 << " \"epoch\": " << epoch << ",\n"
 << " \"loss\": " << metrics.loss << ",\n"
 << " \"learning_rate\": " << metrics.lr << ",\n"
 << " \"grad_norm\": " << metrics.grad_norm << ",\n"
 << " \"config_d_model\": " << model.config.input_dim << ",\n"
 << " \"config_hidden_dim\": " << model.config.hidden_dim << ",\n"
 << " \"config_output_dim\": " << model.config.output_dim << ",\n"
 << " \"config_vocab_size\": " << model.config.vocab_size << ",\n"
 << " \"init_strategy\": \"" << cfg.init_strategy << "\",\n"
 << " \"seed\": " << cfg.seed << "\n"
 << "}\n";
 shm_ofs.close();
 }

 // 2) 后台拷贝到持久目录 (不阻塞训练)
#ifndef _WIN32
 pid_t pid = fork();
 if (pid == 0) {
 // 子进程: 拷贝后退出
 execlp("cp", "cp", "-r", shm_dir.c_str(), ckpt_dir.c_str(),  static_cast<char*>(nullptr));
 _exit(1); // execlp失败
 }
 // 父进程不等待, 继续训练
 std::cerr << " [ckpt] RAM→D盘后台拷贝: " << ckpt_name
 << " (pid=" << pid << ")" << std::endl;
#else
 // Windows: 直接拷贝(无fork)
 std::filesystem::copy(shm_dir, ckpt_dir,
 std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
#endif

    std::cerr << "Checkpoint已保存: step=" << step << " epoch=" << epoch
              << " loss=" << metrics.loss << std::endl;
}


std::unique_ptr<BPETokenizer> load_tokenizer(const std::string& tokenizer_path) {
    auto tok = std::make_unique<BPETokenizer>(tokenizer_path);
    std::cerr << "词表加载成功: " << tokenizer_path
              << " (vocab_size=" << tok->vocab_size() << ")" << std::endl;
    return tok;
}

void train_loop(NeuroFlowModel& model, CausalLMHead& lm_head, DataLoader& loader, MetricsLogger& logger,
                const TrainConfig& cfg, const ResumeState& resume_state,
                const std::vector<uint8_t>& vocab_mask = {}) {
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    std::cerr << "OpenMP最大线程数: " << max_threads << std::endl;
#else
    std::cerr << "警告: 未启用OpenMP，非GEMM运算为单线程" << std::endl;
#endif

    size_t vocab_sz = lm_head.config_.vocab_size;
    size_t d_model = lm_head.config_.d_model;
    float learning_rate = cfg.learning_rate;

    std::cerr << "训练模式: CausalLMHead完整forward (embed→PE→Attn×"
              << lm_head.config_.num_attn_layers << "→Gate→SAE→NTM→LN→Pool→Proj→Out)" << std::endl;
    std::cerr << "  d_model=" << d_model << " vocab=" << vocab_sz
              << " pooling=" << lm_head.config_.pooling << std::endl;

    std::vector<float> epoch_losses;
    int start_epoch = resume_state.found ? resume_state.epoch : 0;
    size_t start_step = resume_state.found ? resume_state.step : 0;

    if (start_epoch > 0) {
        std::cerr << "续训: 从epoch " << start_epoch + 1 << " 开始, 已完成 "
                  << start_step << " 步" << std::endl;
    }

    auto save_lm_head = [&](const std::string& path) {
#ifdef USE_CUDA
        auto sync_to_cpu = [](const Tensor& t) {
            if (t.is_on_gpu()) { const_cast<Tensor&>(t).to_cpu(); }
        };
        sync_to_cpu(lm_head.w_embed_);
        sync_to_cpu(lm_head.w_pos_);
        sync_to_cpu(lm_head.dw_kernel_);
        sync_to_cpu(lm_head.pw_conv_->weight);
        sync_to_cpu(lm_head.sae_w_encode_->weight);
        sync_to_cpu(lm_head.sae_w_decode_->weight);
        sync_to_cpu(lm_head.ntm_w_read_->weight);
        sync_to_cpu(lm_head.ntm_w_write_->weight);
        sync_to_cpu(lm_head.ntm_w_erase_->weight);
        sync_to_cpu(lm_head.ntm_memory_);
        sync_to_cpu(lm_head.w_proj_->weight);
        sync_to_cpu(lm_head.w_proj_->bias);
        sync_to_cpu(lm_head.w_out_->weight);
        sync_to_cpu(lm_head.w_out_->bias);
        sync_to_cpu(lm_head.ln_->weight);
        sync_to_cpu(lm_head.ln_->bias);
        for (auto& attn : lm_head.attn_layers_) {
            sync_to_cpu(attn->w_qkv->weight);
            sync_to_cpu(attn->w_qkv->bias);
            sync_to_cpu(attn->w_out->weight);
            sync_to_cpu(attn->w_out->bias);
            sync_to_cpu(attn->norm->weight);
            sync_to_cpu(attn->norm->bias);
        }
#endif
        auto sl = [](std::ofstream& o, const std::string& n, const Tensor& t) {
            uint32_t nl = n.size(); o.write((char*)&nl, 4); o.write(n.data(), nl);
            uint32_t nd = t.shape_.size(); o.write((char*)&nd, 4);
            for (auto d : t.shape_) { uint32_t dd = d; o.write((char*)&dd, 4); }
            uint32_t ds = t.data_size_; o.write((char*)&ds, 4);
            o.write((char*)t.data_.get(), ds);
        };
        std::ofstream o(path, std::ios::binary);
        o.write("LMH2", 4);
        sl(o, "w_embed", lm_head.w_embed_);
        sl(o, "w_pos", lm_head.w_pos_);
        sl(o, "dw_kernel", lm_head.dw_kernel_);
        sl(o, "pw_conv.weight", lm_head.pw_conv_->weight);
        sl(o, "sae_encode.weight", lm_head.sae_w_encode_->weight);
        sl(o, "sae_decode.weight", lm_head.sae_w_decode_->weight);
        sl(o, "ntm_read.weight", lm_head.ntm_w_read_->weight);
        sl(o, "ntm_write.weight", lm_head.ntm_w_write_->weight);
        sl(o, "ntm_erase.weight", lm_head.ntm_w_erase_->weight);
        sl(o, "ntm_memory", lm_head.ntm_memory_);
        sl(o, "w_proj.weight", lm_head.w_proj_->weight);
        sl(o, "w_proj.bias", lm_head.w_proj_->bias);
        sl(o, "w_out.weight", lm_head.w_out_->weight);
        if (lm_head.w_out_->bias.data_) sl(o, "w_out.bias", lm_head.w_out_->bias);
        sl(o, "ln.weight", lm_head.ln_->weight);
        sl(o, "ln.bias", lm_head.ln_->bias);
        for (size_t i = 0; i < lm_head.attn_layers_.size(); ++i) {
            std::string p = "attn" + std::to_string(i) + ".";
            sl(o, p + "w_qkv.weight", lm_head.attn_layers_[i]->w_qkv->weight);
            sl(o, p + "w_qkv.bias", lm_head.attn_layers_[i]->w_qkv->bias);
            sl(o, p + "w_out.weight", lm_head.attn_layers_[i]->w_out->weight);
            sl(o, p + "w_out.bias", lm_head.attn_layers_[i]->w_out->bias);
            sl(o, p + "norm.weight", lm_head.attn_layers_[i]->norm->weight);
            sl(o, p + "norm.bias", lm_head.attn_layers_[i]->norm->bias);
        }
        uint32_t z = 0; o.write((char*)&z, 4); o.close();
    };

    for (int epoch = start_epoch; epoch < cfg.epochs; ++epoch) {
        auto epoch_start = std::chrono::steady_clock::now();
        loader.reset();
        std::mt19937 shuffle_rng(cfg.seed + epoch);
        loader.shuffle(shuffle_rng);
        float epoch_loss = 0.0f;
        size_t step_count = 0;
        size_t global_step = start_step + epoch * ((loader.total_samples() + cfg.batch_size - 1) / cfg.batch_size);

        while (loader.has_next()) {
            auto batch = loader.next_lm_batch_raw();
            if (batch.empty()) break;

            float batch_loss = 0.0f;
            float batch_grad_norm = 0.0f;
            size_t batch_count = 0;

            for (auto& sample : batch) {
                if (sample.token_ids.size() < 2) continue;

                std::vector<size_t> input_ids(sample.token_ids.begin(), sample.token_ids.end() - 1);
                size_t target_id = sample.token_ids.back();
                if (target_id >= vocab_sz) target_id = 1;

                Tensor logits = lm_head.forward_for_training(input_ids);

                float loss = 0.0f;
                float grad_norm = 0.0f;
                Tensor logits_grad({1, vocab_sz}, QuantType::FP32);

#ifdef USE_CUDA
                if (cfg.use_cuda && logits.is_on_gpu()) {
                    logits_grad.to_gpu();
                    launch_cross_entropy_backward(logits_grad.as_gpu_fp32(), logits.as_gpu_fp32(),
                        static_cast<int>(target_id), static_cast<int>(vocab_sz), CudaContext::instance().stream());
                    logits_grad.gpu_dirty_ = true;

                    Tensor d_loss({1}, QuantType::FP32);
                    d_loss.to_gpu();
                    launch_cross_entropy(d_loss.as_gpu_fp32(), logits.as_gpu_fp32(),
                        static_cast<int>(target_id), static_cast<int>(vocab_sz), CudaContext::instance().stream());
                    CudaContext::instance().synchronize();
                    float loss_arr[1];
                    CudaContext::instance().copy_d2h(loss_arr, d_loss.as_gpu_fp32(), sizeof(float));
                    CudaContext::instance().synchronize();
                    loss = loss_arr[0];

                    if (!std::isfinite(loss)) {
                        std::cerr << "[WARN] NaN loss at step " << global_step << ", skipping" << std::endl;
                        continue;
                    }

                    grad_norm = 0.0f;
                } else
#endif
                {
                    const float* pred = logits.as_fp32();
                    float max_val = -1e30f;
                    for (size_t j = 0; j < vocab_sz; ++j) {
                        if (pred[j] > max_val) max_val = pred[j];
                    }
                    float sum_exp = 0.0f;
                    for (size_t j = 0; j < vocab_sz; ++j) {
                        sum_exp += std::exp(pred[j] - max_val);
                    }
                    float log_sum_exp = max_val + std::log(sum_exp);
                    loss = -(pred[target_id] - log_sum_exp);

                    if (!std::isfinite(loss)) {
                        std::cerr << "[WARN] NaN loss at step " << global_step << ", skipping" << std::endl;
                        continue;
                    }

                    float* lg = logits_grad.as_fp32();
                    for (size_t j = 0; j < vocab_sz; ++j) {
                        float softmax_val = std::exp(pred[j] - max_val) / sum_exp;
                        lg[j] = softmax_val;
                        if (j == target_id) lg[j] -= 1.0f;
                        grad_norm += lg[j] * lg[j];
                    }
                }

                auto lm_grads = lm_head.backward_from_logits(logits_grad);

                float clip_val = cfg.grad_clip;
                float gn = std::sqrt(grad_norm);
                float clip_scale = 1.0f;
                if (!std::isfinite(gn) || gn > clip_val && clip_val > 0.0f) {
                    clip_scale = clip_val / gn;
                }

                lm_head.apply_lm_gradients(lm_grads, learning_rate * clip_scale);


                batch_loss += loss;
                batch_grad_norm += grad_norm;
                batch_count++;
            }

            if (batch_count > 0) {
                epoch_loss += batch_loss / batch_count;
                step_count++;
                global_step++;

                TrainMetrics metrics;
                metrics.loss = batch_loss / batch_count;
                metrics.lr = cfg.learning_rate;
                metrics.step = global_step;
                metrics.epoch = epoch + 1;
                metrics.grad_norm = std::sqrt(batch_grad_norm / batch_count);
                metrics.elapsed_seconds = static_cast<float>(logger.total_elapsed());
                logger.log_step(metrics, cfg.log_interval);

                if (cfg.save_interval > 0 && global_step % cfg.save_interval == 0) {
                    std::string cdir = cfg.output_dir + "/checkpoint_step" + std::to_string(global_step);
                    ensure_dir_exists(cdir);
                    model.save(cdir + "/model.nfv1");
                    save_lm_head(cdir + "/lm_head.nfv1");
                    std::cerr << "Checkpoint: step=" << global_step << " loss=" << metrics.loss << std::endl;
                }
            }
        }

        float avg_loss = (step_count > 0) ? epoch_loss / step_count : 0.0f;
        epoch_losses.push_back(avg_loss);

        auto epoch_end = std::chrono::steady_clock::now();
        float epoch_elapsed = static_cast<float>(
            std::chrono::duration<double>(epoch_end - epoch_start).count());
        logger.log_epoch(epoch + 1, avg_loss, epoch_elapsed);

        std::string cdir = cfg.output_dir + "/checkpoint_epoch" + std::to_string(epoch + 1);
        ensure_dir_exists(cdir);
        model.save(cdir + "/model.nfv1");
        save_lm_head(cdir + "/lm_head.nfv1");
    }

    logger.save_training_log(cfg.output_dir + "/training_log.json", epoch_losses);
    save_lm_head(cfg.output_dir + "/lm_head_final.nfv1");
    std::cerr << "训练完成, LM Head已保存: " << cfg.output_dir << "/lm_head_final.nfv1" << std::endl;
}

int main(int argc, char* argv[]) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::ios::sync_with_stdio(false);

    #ifdef USE_CBLAS
    int ncpu = 1;
    #ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); ncpu = si.dwNumberOfProcessors;
    #else
    ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    #endif
    if (ncpu < 1) ncpu = 1;
    int blas_threads = std::max(1, ncpu / 2);
    std::string nt = std::to_string(blas_threads);
    #ifdef _WIN32
    _putenv_s("OPENBLAS_NUM_THREADS", nt.c_str());
    _putenv_s("GOTO_NUM_THREADS", nt.c_str());
    #else
    setenv("OPENBLAS_NUM_THREADS", nt.c_str(), 1);
    setenv("GOTO_NUM_THREADS", nt.c_str(), 1);
    #endif
    std::cerr << "GEMM后端: OpenBLAS (多线程), BLAS线程: " << blas_threads << std::endl;
    #else
    std::cerr << "GEMM后端: 内置AVX2 (单线程!) — 安装libopenblas-dev可大幅提升性能" << std::endl;
    #endif

    TrainConfig cfg = parse_args(argc, argv);

    if (cfg.config_path.empty() || cfg.tokenizer_path.empty()) {
        std::cerr << "用法: neuroflow_train_v2 --config <config.json> --tokenizer <tokenizer.json>"
                  << " [--data <data>] [--output <dir>] [--epochs <N>] [--lr <lr>]"
                  << " [--seed <N>] [--resume <path>] [--init-weights <strategy>]"
                  << " [--batch-size <N>] [--log-interval <N>] [--save-interval <N>]"
                  << " [--grad-clip <val>] [--adam] [--grad-accum <N>]"
                  << " [--use-cuda] [--verbose]" << std::endl;
        return 1;
    }

    std::cerr << "======================================" << std::endl;
    std::cerr << "NeuroFlow Train v2 (Full Backprop)" << std::endl;
    std::cerr << "======================================" << std::endl;
    std::cerr << "配置: " << cfg.config_path << std::endl;
    std::cerr << "词表: " << cfg.tokenizer_path << std::endl;
    std::cerr << "数据: " << (cfg.data_path.empty() ? "(随机)" : cfg.data_path) << std::endl;
    std::cerr << "输出: " << cfg.output_dir << std::endl;
    std::cerr << "Epochs: " << cfg.epochs << " LR: " << cfg.learning_rate
              << " Batch: " << cfg.batch_size
              << " GradAccum: " << cfg.grad_accum_steps
              << " (等效batch=" << cfg.batch_size * cfg.grad_accum_steps << ")" << std::endl;

#ifdef USE_CUDA
    bool cuda_active = false;
    if (cfg.use_cuda) {
        if (CudaContext::instance().initialize(0)) {
            cuda_active = true;
            std::cerr << "GPU后端: 已启用 (RTX 3090)" << std::endl;
        } else {
            std::cerr << "[CUDA WARNING] GPU初始化失败，回退CPU后端" << std::endl;
            cfg.use_cuda = false;
        }
    } else {
        std::cerr << "GPU后端: 未启用 (使用--use-cuda启用)" << std::endl;
    }
#else
    if (cfg.use_cuda) {
        std::cerr << "[CUDA WARNING] 编译时未启用CUDA支持 (NEUROFLOW_USE_CUDA=OFF)，回退CPU后端" << std::endl;
        cfg.use_cuda = false;
    }
#endif

    auto model_cfg = load_model_config(cfg.config_path);
    ensure_dir_exists(cfg.output_dir);
    ResumeState resume_state;
    auto model = build_model(model_cfg, cfg, resume_state);
    auto tokenizer = load_tokenizer(cfg.tokenizer_path);

    auto stats = model.get_stats();
    std::cerr << "模型参数: " << stats.total_params << " 内存: " << stats.memory_bytes << " bytes" << std::endl;

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
    if (!cfg.resume_path.empty()) {
        std::string lm_ckpt_path;
        if (cfg.resume_path.size() >= 5 && cfg.resume_path.substr(cfg.resume_path.size() - 5) == ".nfv1") {
            std::string dir = cfg.resume_path.substr(0, cfg.resume_path.find_last_of("/\\"));
            lm_ckpt_path = dir + "/lm_head.nfv1";
        } else {
            lm_ckpt_path = cfg.resume_path + "/lm_head.nfv1";
        }
        load_lm_head(lm_head, lm_ckpt_path);
    }
    std::cerr << "CausalLMHead: d_model=" << lm_cfg.d_model
              << " vocab=" << lm_cfg.vocab_size
              << " attn_layers=" << lm_cfg.num_attn_layers
              << " pooling=" << lm_cfg.pooling << std::endl;

#ifdef USE_CUDA
    if (cfg.use_cuda && cuda_active) {
        std::cerr << "传输模型参数到GPU..." << std::endl;
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
    }
#endif

    DataLoader loader(cfg.data_path, cfg.batch_size, model_cfg.input_dim, model_cfg.vocab_size,
                      DataLoader::Mode::LM, tokenizer.get(), model_cfg.max_seq_len,
                      500000, 4096);
    std::cerr << "DataLoader内存预算: 4096 MB, 采样上限: 500000" << std::endl;

    std::vector<uint8_t> vocab_mask;
    if (!cfg.vocab_mask_path.empty()) {
        std::ifstream vmf(cfg.vocab_mask_path);
        if (vmf) {
            std::string vm_content((std::istreambuf_iterator<char>(vmf)), std::istreambuf_iterator<char>());
            size_t mask_start = vm_content.find("\"mask\"");
            if (mask_start != std::string::npos) {
                size_t arr_start = vm_content.find('[', mask_start);
                size_t arr_end = vm_content.find(']', arr_start);
                if (arr_start != std::string::npos && arr_end != std::string::npos) {
                    std::string arr_str = vm_content.substr(arr_start + 1, arr_end - arr_start - 1);
                    size_t pos = 0;
                    while (pos < arr_str.size()) {
                        while (pos < arr_str.size() && (arr_str[pos] == ' ' || arr_str[pos] == ',')) pos++;
                        if (pos >= arr_str.size()) break;
                        vocab_mask.push_back(static_cast<uint8_t>(arr_str[pos] - '0'));
                        pos++;
                    }
                }
            }
            size_t active = 0;
            for (auto m : vocab_mask) active += m;
            std::cerr << "Vocab mask: " << cfg.vocab_mask_path
                      << " (" << active << "/" << vocab_mask.size() << " active)" << std::endl;
        } else {
            std::cerr << "Warning: vocab mask not found: " << cfg.vocab_mask_path << std::endl;
        }
    }

    MetricsLogger metrics_logger;

    train_loop(model, lm_head, loader, metrics_logger, cfg, resume_state, vocab_mask);

    model.save(cfg.output_dir + "/model_final.nfv1");
    std::cerr << "NF模型已保存: " << cfg.output_dir << "/model_final.nfv1" << std::endl;

    return 0;
}

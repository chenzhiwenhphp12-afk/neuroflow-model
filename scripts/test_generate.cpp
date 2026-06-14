#include "neuroflow/model.hpp"
#include "neuroflow/generative.hpp"
#include "weight_io.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

using namespace neuroflow;

NeuroFlowModel::Config load_config(const std::string& path) {
    NeuroFlowModel::Config cfg;
    std::ifstream f(path);
    if (!f) { std::cerr << "无法加载配置文件: " << path << std::endl; return cfg; }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto extract_num = [&](const std::string& key, size_t def = 0) {
        size_t p = json.find("\"" + key + "\"");
        if (p == std::string::npos) return def;
        p = json.find(':', p + key.size() + 2);
        while (p < json.size() && !std::isdigit(json[p])) p++;
        size_t e = p;
        while (e < json.size() && std::isdigit(json[e])) e++;
        return (e > p) ? std::stoul(json.substr(p, e - p)) : def;
    };

    cfg.vocab_size = extract_num("vocab_size", 5000);
    cfg.input_dim = extract_num("input_dim", 128);
    cfg.hidden_dim = extract_num("hidden_dim", 256);
    cfg.output_dim = extract_num("output_dim", cfg.vocab_size);
    cfg.num_layers = extract_num("num_layers", 2);
    cfg.memory_slots = extract_num("memory_slots", 64);
    cfg.memory_dim = extract_num("memory_dim", 128);
    cfg.num_associations = extract_num("num_associations", 8);
    cfg.use_causal_lm = true;
    cfg.max_seq_len = extract_num("max_seq_len", 128);
    cfg.causal_window_size = extract_num("causal_window_size", 32);

    std::cerr << "配置加载完成:" << std::endl;
    std::cerr << "  vocab=" << cfg.vocab_size << " d_model=" << cfg.input_dim
              << " hidden=" << cfg.hidden_dim << " output=" << cfg.output_dim << std::endl;
    return cfg;
}

// 判断是否为有效token（非特殊，且在词表范围内）
bool is_valid_token(size_t id, size_t vocab_actual) {
    return id >= 4 && id < vocab_actual;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <config.json> <model.nfv1>" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string model_path = argv[2];

    // 哲学相关测试提示词
    std::vector<std::string> prompts = {
        "哲学",
        "辩证法",
        "唯物主义",
        "认识论",
        "存在",
        "意识",
        "真理",
        "实践",
    };

    std::cerr << "加载配置: " << config_path << std::endl;
    auto cfg = load_config(config_path);

    std::cerr << "构建模型..." << std::endl;
    NeuroFlowModel model(cfg);

    std::cerr << "加载权重: " << model_path << std::endl;
    model.load(model_path);

    auto stats = model.get_stats();
    std::cerr << "模型参数: " << stats.total_params << " 内存: " << stats.memory_bytes / 1024 / 1024 << " MB" << std::endl;

    std::string tok_path = config_path.substr(0, config_path.find_last_of("/\\")) + "/tokenizer_cn_013.json";
    std::cerr << "加载词表: " << tok_path << std::endl;
    BPETokenizer tokenizer(tok_path);
    size_t vocab_actual = tokenizer.vocab_size();
    std::cerr << "词表大小: " << vocab_actual << std::endl;

    float scale = 1.0f / cfg.vocab_size;
    std::mt19937 rng(42);

    for (auto& prompt : prompts) {
        std::cerr << "\n========================================\n";
        std::cerr << "提示词: " << prompt << std::endl;
        std::cerr << "========================================" << std::endl;

        std::vector<size_t> input_ids = tokenizer.encode(prompt);
        std::cerr << "输入tokens: ";
        for (auto id : input_ids) std::cerr << id << " ";
        std::cerr << std::endl;

        // 前向传播
        size_t seq_len = std::min(input_ids.size(), (size_t)cfg.max_seq_len);
        Tensor input({1, cfg.input_dim}, QuantType::FP32);
        float* inp = input.as_fp32();
        for (size_t j = 0; j < seq_len && j < cfg.input_dim; ++j) {
            inp[j] = static_cast<float>(input_ids[j]) * scale;
        }

        auto output = model.forward(input);
        const float* logits = output.output.as_fp32();

        // Top-10（只显示有效token）
        std::vector<std::pair<float, size_t>> scored;
        for (size_t i = 0; i < cfg.output_dim; ++i) {
            if (is_valid_token(i, vocab_actual))
                scored.push_back({logits[i], i});
        }
        std::sort(scored.begin(), scored.end(), std::greater<>());

        std::cout << "\n-- Top-10 有效token预测 --" << std::endl;
        for (int i = 0; i < std::min(10, (int)scored.size()); ++i) {
            size_t id = scored[i].second;
            float score = scored[i].first;
            std::string token = tokenizer.decode({id});
            std::cout << "  [" << id << "] \"" << token << "\" score=" << score << std::endl;
        }

        // 自回归生成（排除特殊token，使用温度采样）
        std::cout << "\n-- 自回归生成 --" << std::endl;
        std::vector<size_t> generated = input_ids;
        size_t last_id = input_ids.back();
        float temperature = 1.0f;

        for (int step = 0; step < 30; ++step) {
            Tensor step_input({1, cfg.input_dim}, QuantType::FP32);
            float* si = step_input.as_fp32();
            size_t ctx = std::min(generated.size(), (size_t)cfg.max_seq_len);
            size_t start = generated.size() - ctx;
            for (size_t j = 0; j < cfg.input_dim; ++j) {
                si[j] = (j < ctx) ? static_cast<float>(generated[start + j]) * scale : 0.0f;
            }

            auto out = model.forward(step_input);
            float* log = out.output.as_fp32();

            // 温度采样（只从有效token中选）
            float max_val = -1e30f;
            for (size_t j = 4; j < vocab_actual; ++j)
                if (log[j] > max_val) max_val = log[j];

            float sum_exp = 0.0f;
            std::vector<float> probs(cfg.output_dim, 0.0f);
            for (size_t j = 4; j < vocab_actual; ++j) {
                probs[j] = std::exp((log[j] - max_val) / temperature);
                sum_exp += probs[j];
            }
            for (size_t j = 4; j < vocab_actual; ++j)
                probs[j] /= sum_exp;

            // 累积采样
            float r = std::uniform_real_distribution<float>(0, 1)(rng);
            float cum = 0;
            size_t next_id = 4;
            for (size_t j = 4; j < vocab_actual; ++j) {
                cum += probs[j];
                if (r <= cum) { next_id = j; break; }
            }

            generated.push_back(next_id);
            std::string token = tokenizer.decode({next_id});
            std::cout << token;

            if (next_id == 2) break; // </s>
        }
        std::cout << std::endl;
    }

    return 0;
}

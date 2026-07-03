#include "weight_io.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace neuroflow {

void WeightInitializer::xavier_uniform(Tensor& weight, size_t fan_in, size_t fan_out, std::mt19937& rng) {
    if (weight.numel() == 0) return;
    if (fan_in == 0 || fan_out == 0) {
        std::cerr << "Warning: xavier_uniform with fan_in=" << fan_in << " fan_out=" << fan_out << ", skipping" << std::endl;
        return;
    }
    float a = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-a, a);
    float* data = weight.as_fp32();
    for (size_t i = 0; i < weight.numel(); ++i) {
        data[i] = dist(rng);
    }
}

void WeightInitializer::kaiming_normal(Tensor& weight, size_t fan_in, std::mt19937& rng) {
    if (weight.numel() == 0) return;
    if (fan_in == 0) {
        std::cerr << "Warning: kaiming_normal with fan_in=0, skipping" << std::endl;
        return;
    }
    float std_dev = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::normal_distribution<float> dist(0.0f, std_dev);
    float* data = weight.as_fp32();
    for (size_t i = 0; i < weight.numel(); ++i) {
        data[i] = dist(rng);
    }
}

void WeightInitializer::zeros(Tensor& tensor) {
    if (tensor.numel() == 0) return;
    std::memset(tensor.as_fp32(), 0, tensor.data_size_);
}

void WeightInitializer::random_normal(Tensor& tensor, float mean, float std_dev, std::mt19937& rng) {
    if (tensor.numel() == 0) return;
    if (std_dev <= 0.0f) {
        std::cerr << "Warning: random_normal with std_dev<=0, using 0.01" << std::endl;
        std_dev = 0.01f;
    }
    std::normal_distribution<float> dist(mean, std_dev);
    float* data = tensor.as_fp32();
    for (size_t i = 0; i < tensor.numel(); ++i) {
        data[i] = dist(rng);
    }
}

void WeightInitializer::init_model_weights(NeuroFlowModel& model, InitStrategy strategy, uint32_t seed) {
    std::mt19937 rng(seed);

    auto init_linear = [&](std::shared_ptr<Linear>& layer, size_t fan_in, size_t fan_out) {
        if (!layer) return;
        switch (strategy) {
            case InitStrategy::XAVIER_UNIFORM:
                xavier_uniform(layer->weight, fan_in, fan_out, rng);
                break;
            case InitStrategy::KAIMING_NORMAL:
                kaiming_normal(layer->weight, fan_in, rng);
                break;
            case InitStrategy::ZEROS:
                zeros(layer->weight);
                break;
            case InitStrategy::RANDOM_NORMAL:
                random_normal(layer->weight, 0.0f, 0.02f, rng);
                break;
            default:
                xavier_uniform(layer->weight, fan_in, fan_out, rng);
                break;
        }
        zeros(layer->bias);
    };

    auto& cfg = model.config;
    size_t half = cfg.hidden_dim / 2;

    // === Input Projection ===
    init_linear(model.input_proj_linear, cfg.input_dim, cfg.hidden_dim);

    // === ECN (Executive Control Network) ===
    // dlPFC: first layer input_dim=hidden_dim, subsequent=hidden_dim
    for (size_t i = 0; i < model.ecn->dlpfc_linear.size(); ++i) {
        size_t fan_in = (i == 0) ? cfg.hidden_dim : cfg.hidden_dim;
        init_linear(model.ecn->dlpfc_linear[i], fan_in, cfg.hidden_dim);
    }
    // OFC: hidden_dim -> half -> 1
    init_linear(model.ecn->ofc1, cfg.hidden_dim, half);
    init_linear(model.ecn->ofc2, half, 1);
    // vmPFC: hidden_dim -> half -> output_dim
    init_linear(model.ecn->vmpfc1, cfg.hidden_dim, half);
    init_linear(model.ecn->vmpfc2, half, cfg.hidden_dim);

    // === DMN (Default Mode Network) ===
    // mem_encoder: memory_dim -> latent_dim*2 -> latent_dim
    // latent_dim = hidden_dim/2
    size_t latent_dim = half;
    init_linear(model.dmn->mem_encoder1, cfg.memory_dim, latent_dim * 2);
    init_linear(model.dmn->mem_encoder2, latent_dim * 2, latent_dim);
    // association heads: latent_dim -> latent_dim (each)
    for (auto& [h1, h2] : model.dmn->association_heads) {
        init_linear(h1, latent_dim, latent_dim);
        init_linear(h2, latent_dim, latent_dim);
    }
    // future_proj: latent_dim * num_assoc -> latent_dim * 2
    init_linear(model.dmn->future_proj1, latent_dim * cfg.num_associations, latent_dim * 2);

    // === SN (Salience Network) ===
    size_t sn_hidden = half;
    init_linear(model.sn->saliency1, cfg.hidden_dim, sn_hidden);
    init_linear(model.sn->saliency2, sn_hidden, sn_hidden / 2);
    init_linear(model.sn->saliency3, sn_hidden / 2, 1);
    init_linear(model.sn->gate1, cfg.hidden_dim, sn_hidden);
    init_linear(model.sn->gate2, sn_hidden, 2);
    init_linear(model.sn->anomaly1, cfg.hidden_dim, sn_hidden);
    init_linear(model.sn->anomaly2, sn_hidden, 1);

    // === Memory Consolidation Module ===
    init_linear(model.memory->encode_proj, cfg.hidden_dim, cfg.memory_dim);
    init_linear(model.memory->retrieve_proj, cfg.memory_dim, cfg.hidden_dim);
    init_linear(model.memory->query_proj, cfg.hidden_dim, cfg.memory_dim);
    zeros(model.memory->memory_bank);

    // === Manifold Projection ===
    size_t manifold_in = cfg.hidden_dim + half;
    init_linear(model.manifold_proj1, manifold_in, cfg.hidden_dim);
    init_linear(model.manifold_proj2, cfg.hidden_dim, 32);

    // === Output Fusion (低秩因式分解) ===
    size_t fusion_in = cfg.hidden_dim * 3;
    size_t bn = cfg.fusion_bottleneck_dim;
    init_linear(model.output_fusion_down, fusion_in, bn);
    init_linear(model.output_fusion_up, bn, cfg.hidden_dim);
}

ValidationResult WeightInitializer::validate_dimensions(const NeuroFlowModel& model) {
    ValidationResult result;
    auto& cfg = model.config;
    size_t half = cfg.hidden_dim / 2;
    size_t latent_dim = half;

    auto check2d = [&](const std::string& name, const Tensor& t, size_t expected_rows, size_t expected_cols) {
        if (t.shape_.size() < 2 || t.shape_[0] != expected_rows || t.shape_[1] != expected_cols) {
            result.all_passed = false;
            std::string actual = (t.shape_.size() >= 2)
                ? "[" + std::to_string(t.shape_[0]) + "," + std::to_string(t.shape_[1]) + "]"
                : "ndim=" + std::to_string(t.shape_.size());
            result.failures.push_back(name + ": 期望[" + std::to_string(expected_rows) + "," +
                std::to_string(expected_cols) + "], 实际" + actual);
        }
    };

    auto check1d = [&](const std::string& name, const Tensor& t, size_t expected_dim) {
        if (t.shape_.size() < 1 || t.shape_[0] != expected_dim) {
            result.all_passed = false;
            result.failures.push_back(name + ": 期望[" + std::to_string(expected_dim) + "], 实际dim不匹配");
        }
    };

    // Input Projection
    check2d("input_proj.weight", model.input_proj_linear->weight, cfg.hidden_dim, cfg.input_dim);
    check1d("input_proj.bias", model.input_proj_linear->bias, cfg.hidden_dim);

    // ECN dlPFC
    for (size_t i = 0; i < model.ecn->dlpfc_linear.size(); ++i) {
        check2d("ecn.dlpfc" + std::to_string(i) + ".weight",
                model.ecn->dlpfc_linear[i]->weight, cfg.hidden_dim, cfg.hidden_dim);
    }
    check2d("ecn.ofc1.weight", model.ecn->ofc1->weight, half, cfg.hidden_dim);
    check2d("ecn.ofc2.weight", model.ecn->ofc2->weight, 1, half);
    check2d("ecn.vmpfc1.weight", model.ecn->vmpfc1->weight, half, cfg.hidden_dim);
    check2d("ecn.vmpfc2.weight", model.ecn->vmpfc2->weight, cfg.hidden_dim, half);

    // DMN
    check2d("dmn.mem_encoder1.weight", model.dmn->mem_encoder1->weight, latent_dim * 2, cfg.memory_dim);
    check2d("dmn.mem_encoder2.weight", model.dmn->mem_encoder2->weight, latent_dim, latent_dim * 2);
    for (size_t i = 0; i < model.dmn->association_heads.size(); ++i) {
        auto& [h1, h2] = model.dmn->association_heads[i];
        check2d("dmn.head" + std::to_string(i) + ".1.weight", h1->weight, latent_dim, latent_dim);
        check2d("dmn.head" + std::to_string(i) + ".2.weight", h2->weight, latent_dim, latent_dim);
    }
    check2d("dmn.future_proj1.weight", model.dmn->future_proj1->weight, latent_dim * 2, latent_dim * cfg.num_associations);

    // SN
    size_t sn_hidden = half;
    check2d("sn.saliency1.weight", model.sn->saliency1->weight, sn_hidden, cfg.hidden_dim);
    check2d("sn.saliency2.weight", model.sn->saliency2->weight, sn_hidden / 2, sn_hidden);
    check2d("sn.saliency3.weight", model.sn->saliency3->weight, 1, sn_hidden / 2);
    check2d("sn.gate1.weight", model.sn->gate1->weight, sn_hidden, cfg.hidden_dim);
    check2d("sn.gate2.weight", model.sn->gate2->weight, 2, sn_hidden);
    check2d("sn.anomaly1.weight", model.sn->anomaly1->weight, sn_hidden, cfg.hidden_dim);
    check2d("sn.anomaly2.weight", model.sn->anomaly2->weight, 1, sn_hidden);

    // Memory
    check2d("memory.encode_proj.weight", model.memory->encode_proj->weight, cfg.memory_dim, cfg.hidden_dim);
    check2d("memory.retrieve_proj.weight", model.memory->retrieve_proj->weight, cfg.hidden_dim, cfg.memory_dim);
    check2d("memory.query_proj.weight", model.memory->query_proj->weight, cfg.memory_dim, cfg.hidden_dim);

    // Manifold
    size_t manifold_in = cfg.hidden_dim + half;
    check2d("manifold_proj1.weight", model.manifold_proj1->weight, cfg.hidden_dim, manifold_in);
    check2d("manifold_proj2.weight", model.manifold_proj2->weight, 32, cfg.hidden_dim);

    // Output Fusion (低秩因式分解)
    size_t bn = cfg.fusion_bottleneck_dim;
    check2d("output_fusion.down.weight", model.output_fusion_down->weight, bn, cfg.hidden_dim * 3);
    check1d("output_fusion.down.bias", model.output_fusion_down->bias, bn);
    check2d("output_fusion.up.weight", model.output_fusion_up->weight, cfg.hidden_dim, bn);
    check1d("output_fusion.up.bias", model.output_fusion_up->bias, cfg.hidden_dim);

    return result;
}

void save_binary(const NeuroFlowModel& model, const std::string& path) {
    model.save(path);
}

void load_binary(NeuroFlowModel& model, const std::string& path) {
    model.load(path);
}

void save_npz(const NeuroFlowModel& model, const std::string& path) {
    // NPZ格式需要zlib/miniz依赖
    // 当前实现：将每个权重层保存为独立的.raw文件 + 一个manifest.json索引
    // 这与NumPy的.npz格式（ZIP存档）兼容性有限
    // 完整NPZ实现需引入cnpy库: https://github.com/rogersce/cnpy
    //
    // 回退策略：使用NFv1二进制格式
    model.save(path + ".nfv1");
    std::cerr << "注意: NPZ格式保存需要cnpy/zlib依赖，当前回退到NFv1格式" << std::endl;
    std::cerr << "  如需完整NPZ支持，请安装cnpy: https://github.com/rogersce/cnpy" << std::endl;
}

void load_npz(NeuroFlowModel& model, const std::string& path) {
    // 同save_npz，回退到NFv1
    model.load(path + ".nfv1");
}

void save_metadata(const NeuroFlowModel& model, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Warning: cannot open metadata file: " << path << std::endl;
        return;
    }

    auto& cfg = model.config;
    size_t half = cfg.hidden_dim / 2;
    size_t latent_dim = half;

    auto count_params = [](const Tensor& t) -> size_t { return t.numel(); };

    ofs << "{\n";
    ofs << "  \"model_type\": \"NeuroFlow\",\n";
    ofs << "  \"format\": \"NFv1\",\n";
    ofs << "  \"config\": {\n";
    ofs << "    \"input_dim\": " << cfg.input_dim << ",\n";
    ofs << "    \"hidden_dim\": " << cfg.hidden_dim << ",\n";
    ofs << "    \"output_dim\": " << cfg.output_dim << ",\n";
    ofs << "    \"memory_dim\": " << cfg.memory_dim << ",\n";
    ofs << "    \"memory_slots\": " << cfg.memory_slots << ",\n";
    ofs << "    \"num_layers\": " << cfg.num_layers << ",\n";
    ofs << "    \"num_associations\": " << cfg.num_associations << ",\n";
    ofs << "    \"vocab_size\": " << cfg.vocab_size << ",\n";
    ofs << "    \"max_seq_len\": " << cfg.max_seq_len << "\n";
    ofs << "  },\n";

    ofs << "  \"layers\": [\n";
    auto add_layer = [&](const std::string& name, const std::string& type,
                         const Tensor& weight, const Tensor& bias, bool& first) {
        if (!first) ofs << ",\n";
        first = false;
        ofs << "    {\"name\": \"" << name << "\", \"type\": \"" << type << "\", ";
        ofs << "\"weight_shape\": [";
        for (size_t i = 0; i < weight.shape_.size(); ++i) {
            if (i > 0) ofs << ", ";
            ofs << weight.shape_[i];
        }
        ofs << "], \"params\": " << count_params(weight) + (bias.data_ ? count_params(bias) : 0) << "}";
    };

    bool first = true;
    add_layer("input_proj", "Linear", model.input_proj_linear->weight, model.input_proj_linear->bias, first);
    for (size_t i = 0; i < model.ecn->dlpfc_linear.size(); ++i) {
        add_layer("ecn.dlpfc" + std::to_string(i), "Linear",
                  model.ecn->dlpfc_linear[i]->weight, model.ecn->dlpfc_linear[i]->bias, first);
    }
    add_layer("ecn.ofc1", "Linear", model.ecn->ofc1->weight, model.ecn->ofc1->bias, first);
    add_layer("ecn.ofc2", "Linear", model.ecn->ofc2->weight, model.ecn->ofc2->bias, first);
    add_layer("ecn.vmpfc1", "Linear", model.ecn->vmpfc1->weight, model.ecn->vmpfc1->bias, first);
    add_layer("ecn.vmpfc2", "Linear", model.ecn->vmpfc2->weight, model.ecn->vmpfc2->bias, first);
    add_layer("dmn.mem_encoder1", "Linear", model.dmn->mem_encoder1->weight, model.dmn->mem_encoder1->bias, first);
    add_layer("dmn.mem_encoder2", "Linear", model.dmn->mem_encoder2->weight, model.dmn->mem_encoder2->bias, first);
    for (size_t i = 0; i < model.dmn->association_heads.size(); ++i) {
        auto& [h1, h2] = model.dmn->association_heads[i];
        add_layer("dmn.head" + std::to_string(i) + ".1", "Linear", h1->weight, h1->bias, first);
        add_layer("dmn.head" + std::to_string(i) + ".2", "Linear", h2->weight, h2->bias, first);
    }
    add_layer("dmn.future_proj1", "Linear", model.dmn->future_proj1->weight, model.dmn->future_proj1->bias, first);
    add_layer("sn.saliency1", "Linear", model.sn->saliency1->weight, model.sn->saliency1->bias, first);
    add_layer("sn.saliency2", "Linear", model.sn->saliency2->weight, model.sn->saliency2->bias, first);
    add_layer("sn.saliency3", "Linear", model.sn->saliency3->weight, model.sn->saliency3->bias, first);
    add_layer("sn.gate1", "Linear", model.sn->gate1->weight, model.sn->gate1->bias, first);
    add_layer("sn.gate2", "Linear", model.sn->gate2->weight, model.sn->gate2->bias, first);
    add_layer("sn.anomaly1", "Linear", model.sn->anomaly1->weight, model.sn->anomaly1->bias, first);
    add_layer("sn.anomaly2", "Linear", model.sn->anomaly2->weight, model.sn->anomaly2->bias, first);
    add_layer("memory.encode_proj", "Linear", model.memory->encode_proj->weight, model.memory->encode_proj->bias, first);
    add_layer("memory.retrieve_proj", "Linear", model.memory->retrieve_proj->weight, model.memory->retrieve_proj->bias, first);
    add_layer("memory.query_proj", "Linear", model.memory->query_proj->weight, model.memory->query_proj->bias, first);
    add_layer("manifold_proj1", "Linear", model.manifold_proj1->weight, model.manifold_proj1->bias, first);
    add_layer("manifold_proj2", "Linear", model.manifold_proj2->weight, model.manifold_proj2->bias, first);
    add_layer("output_fusion.down", "Linear", model.output_fusion_down->weight, model.output_fusion_down->bias, first);
    add_layer("output_fusion.up", "Linear", model.output_fusion_up->weight, model.output_fusion_up->bias, first);
    ofs << "\n  ],\n";

    size_t total_params = 0;
    auto count_layer = [&](const Tensor& w, const Tensor& b) {
        total_params += count_params(w) + (b.data_ ? count_params(b) : 0);
    };
    count_layer(model.input_proj_linear->weight, model.input_proj_linear->bias);
    for (auto& l : model.ecn->dlpfc_linear) count_layer(l->weight, l->bias);
    count_layer(model.ecn->ofc1->weight, model.ecn->ofc1->bias);
    count_layer(model.ecn->ofc2->weight, model.ecn->ofc2->bias);
    count_layer(model.ecn->vmpfc1->weight, model.ecn->vmpfc1->bias);
    count_layer(model.ecn->vmpfc2->weight, model.ecn->vmpfc2->bias);
    count_layer(model.dmn->mem_encoder1->weight, model.dmn->mem_encoder1->bias);
    count_layer(model.dmn->mem_encoder2->weight, model.dmn->mem_encoder2->bias);
    for (auto& [h1, h2] : model.dmn->association_heads) {
        count_layer(h1->weight, h1->bias);
        count_layer(h2->weight, h2->bias);
    }
    count_layer(model.dmn->future_proj1->weight, model.dmn->future_proj1->bias);
    count_layer(model.sn->saliency1->weight, model.sn->saliency1->bias);
    count_layer(model.sn->saliency2->weight, model.sn->saliency2->bias);
    count_layer(model.sn->saliency3->weight, model.sn->saliency3->bias);
    count_layer(model.sn->gate1->weight, model.sn->gate1->bias);
    count_layer(model.sn->gate2->weight, model.sn->gate2->bias);
    count_layer(model.sn->anomaly1->weight, model.sn->anomaly1->bias);
    count_layer(model.sn->anomaly2->weight, model.sn->anomaly2->bias);
    count_layer(model.memory->encode_proj->weight, model.memory->encode_proj->bias);
    count_layer(model.memory->retrieve_proj->weight, model.memory->retrieve_proj->bias);
    count_layer(model.memory->query_proj->weight, model.memory->query_proj->bias);
    count_layer(model.manifold_proj1->weight, model.manifold_proj1->bias);
    count_layer(model.manifold_proj2->weight, model.manifold_proj2->bias);
    count_layer(model.output_fusion_down->weight, model.output_fusion_down->bias);
    count_layer(model.output_fusion_up->weight, model.output_fusion_up->bias);

    ofs << "  \"total_params\": " << total_params << ",\n";
    ofs << "  \"memory_bank_slots\": " << cfg.memory_slots << ",\n";
    ofs << "  \"memory_bank_dim\": " << cfg.memory_dim << "\n";
    ofs << "}\n";
    ofs.close();
}

}

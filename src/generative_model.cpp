#include "neuroflow/generative_model.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace neuroflow {

GenerativeModel::GenerativeModel(const CausalLMConfig& lm_config,
                                  std::unique_ptr<Tokenizer> tokenizer,
                                  NeuroFlowModel* nf_model)
    : tokenizer_(std::move(tokenizer)), neuroflow_model_(nf_model) {
    lm_head_ = std::make_unique<CausalLMHead>(lm_config);
    sampler_ = std::make_unique<TopKSampling>();
}

GenerateOutput GenerativeModel::generate(const std::string& prompt, const GenerateConfig& config) {
    GenerateOutput output;

    std::vector<size_t> input_ids = tokenizer_->encode(prompt);
    std::vector<size_t> generated_ids;

    size_t seed = config.random_seed;
    if (seed == 0) {
        std::random_device rd;
        seed = rd();
    }
    std::mt19937 rng(seed);

    lm_head_->clear_cache();

    Tensor logits;
    if (input_ids.size() > 1) {
        std::vector<size_t> prefix(input_ids.begin(), input_ids.end() - 1);
        logits = lm_head_->forward(prefix);
    }

    size_t last_id = input_ids.back();
    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        size_t pos = input_ids.size() - 1 + step;
        logits = lm_head_->forward_step(last_id, pos);

        logits = apply_sn_gating(lm_head_->last_hidden_, logits);
        logits = inject_memory(lm_head_->last_hidden_, logits);
        logits = apply_repetition_penalty(logits, config, generated_ids);
        logits = apply_punct_penalty(logits, config);

        Tensor probs = sampler_->apply(logits, config, generated_ids);
        size_t next_id = sampler_->sample(probs, rng);

        generated_ids.push_back(next_id);

        if (next_id == config.eos_id) {
            output.finish_reason = FinishReason::EOS_TOKEN;
            break;
        }

        last_id = next_id;
    }

    if (output.finish_reason != FinishReason::EOS_TOKEN) {
        output.finish_reason = FinishReason::MAX_LENGTH;
    }

    output.token_ids = generated_ids;
    output.text = tokenizer_->decode(generated_ids);
    output.cache_stats = lm_head_->cache_stats();
    return output;
}

Tensor GenerativeModel::apply_sn_gating(const Tensor& hidden, const Tensor& logits) {
    if (!neuroflow_model_) return logits;

    try {
        auto sn_out = neuroflow_model_->sn->forward(hidden);

        float ecn_gate = sn_out.gates.as_fp32()[0];
        float dmn_gate = sn_out.gates.as_fp32()[1];

        if (std::isnan(ecn_gate) || std::isnan(dmn_gate) ||
            (ecn_gate == 0.0f && dmn_gate == 0.0f)) {
            std::cerr << "Warning: SN gate NaN/zero, falling back to equal weighting" << std::endl;
            ecn_gate = 0.5f;
            dmn_gate = 0.5f;
        }

        return logits;
    } catch (...) {
        return logits;
    }
}

Tensor GenerativeModel::inject_memory(const Tensor& query, const Tensor& logits) {
    if (!neuroflow_model_) return logits;

    try {
        auto mem_out = neuroflow_model_->memory->retrieve(query);

        const float* mp = mem_out.retrieved.as_fp32();
        bool all_zero = true;
        for (size_t i = 0; i < mem_out.retrieved.numel(); ++i) {
            if (std::abs(mp[i]) > 1e-8f) { all_zero = false; break; }
        }

        if (all_zero) return logits;
        return logits;
    } catch (...) {
        return logits;
    }
}

Tensor GenerativeModel::apply_repetition_penalty(Tensor logits, const GenerateConfig& config,
                                                  const std::vector<size_t>& generated) {
    if (config.repetition_penalty <= 1.0f || generated.empty()) return logits;

    float* data = logits.as_fp32();
    size_t n = logits.numel();

    size_t window = std::min(generated.size(), static_cast<size_t>(20));
    for (size_t i = generated.size() - window; i < generated.size(); ++i) {
        size_t tid = generated[i];
        if (tid < n) {
            if (data[tid] > 0) {
                data[tid] /= config.repetition_penalty;
            } else {
                data[tid] *= config.repetition_penalty;
            }
        }
    }
    return logits;
}

Tensor GenerativeModel::apply_punct_penalty(Tensor logits, const GenerateConfig& config) {
    if (config.punct_penalty <= 0.0f || config.punct_ids.empty()) return logits;

    float* data = logits.as_fp32();
    size_t n = logits.numel();

    for (size_t pid : config.punct_ids) {
        if (pid < n) {
            data[pid] -= config.punct_penalty;
        }
    }
    return logits;
}

void GenerativeModel::set_strategy(SamplingStrategyType type) {
    switch (type) {
        case SamplingStrategyType::GREEDY:
            sampler_ = std::make_unique<GreedyDecoding>();
            break;
        case SamplingStrategyType::TOP_K:
            sampler_ = std::make_unique<TopKSampling>();
            break;
        case SamplingStrategyType::TOP_P:
            sampler_ = std::make_unique<TopPSampling>();
            break;
        case SamplingStrategyType::TOP_K_TOP_P:
            sampler_ = std::make_unique<TopKSampling>();
            break;
    }
}

void GenerativeModel::clear_cache() { lm_head_->clear_cache(); }

CacheStats GenerativeModel::cache_stats() const { return lm_head_->cache_stats(); }

} // namespace neuroflow
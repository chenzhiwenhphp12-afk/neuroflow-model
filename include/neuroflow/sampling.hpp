#ifndef NEUROFLOW_SAMPLING_HPP
#define NEUROFLOW_SAMPLING_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include "tensor.hpp"

namespace neuroflow {

enum class SamplingStrategyType : uint8_t {
    GREEDY = 0,
    TOP_K = 1,
    TOP_P = 2,
    TOP_K_TOP_P = 3
};

struct GenerateConfig {
    size_t max_new_tokens = 50;
    float temperature = 1.0f;
    size_t top_k = 40;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    float punct_penalty = 0.0f;
    std::vector<size_t> punct_ids;
    size_t random_seed = 0;
    SamplingStrategyType strategy = SamplingStrategyType::TOP_K;
    size_t eos_id = 3;
};

class SamplingStrategy {
public:
    virtual ~SamplingStrategy() = default;
    virtual Tensor apply(Tensor logits, const GenerateConfig& config,
                         const std::vector<size_t>& generated) = 0;
    virtual size_t sample(const Tensor& probs, std::mt19937& rng) const = 0;
};

class GreedyDecoding : public SamplingStrategy {
public:
    Tensor apply(Tensor logits, const GenerateConfig& config,
                 const std::vector<size_t>& generated) override;
    size_t sample(const Tensor& probs, std::mt19937& rng) const override;
};

class TopKSampling : public SamplingStrategy {
public:
    Tensor apply(Tensor logits, const GenerateConfig& config,
                 const std::vector<size_t>& generated) override;
    size_t sample(const Tensor& probs, std::mt19937& rng) const override;
};

class TopPSampling : public SamplingStrategy {
public:
    Tensor apply(Tensor logits, const GenerateConfig& config,
                 const std::vector<size_t>& generated) override;
    size_t sample(const Tensor& probs, std::mt19937& rng) const override;
};

class TopKTopPSampling : public SamplingStrategy {
public:
    Tensor apply(Tensor logits, const GenerateConfig& config,
                 const std::vector<size_t>& generated) override;
    size_t sample(const Tensor& probs, std::mt19937& rng) const override;
};

} // namespace neuroflow

#endif // NEUROFLOW_SAMPLING_HPP
#include "neuroflow/sampling.hpp"
#include <cstring>
#include <unordered_set>

namespace neuroflow {

static void apply_repetition_penalty(float* data, size_t n, float penalty,
                                      const std::vector<size_t>& generated) {
    if (penalty <= 1.0f || generated.empty()) return;
    std::unordered_set<size_t> seen(generated.begin(), generated.end());
    for (size_t id : seen) {
        if (id >= n) continue;
        if (data[id] > 0.0f) {
            data[id] /= penalty;
        } else {
            data[id] *= penalty;
        }
    }
}

Tensor GreedyDecoding::apply(Tensor logits, const GenerateConfig& config,
                              const std::vector<size_t>& generated) {
    float* data = logits.as_fp32();
    size_t n = logits.numel();
    apply_repetition_penalty(data, n, config.repetition_penalty, generated);
    return logits;
}

size_t GreedyDecoding::sample(const Tensor& probs, std::mt19937& rng) const {
    const float* data = probs.as_fp32_const();
    size_t n = probs.numel();
    size_t best = 0;
    float best_val = data[0];
    for (size_t i = 1; i < n; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}

Tensor TopKSampling::apply(Tensor logits, const GenerateConfig& config,
                            const std::vector<size_t>& generated) {
    float* data = logits.as_fp32();
    size_t n = logits.numel();

    apply_repetition_penalty(data, n, config.repetition_penalty, generated);

    float temp = config.temperature;
    if (temp <= 0.0f) {
        size_t best = 0;
        float best_val = data[0];
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > best_val) { best_val = data[i]; best = i; }
        }
        memset(data, 0, n * sizeof(float));
        data[best] = 1.0f;
        return logits;
    }

    for (size_t i = 0; i < n; ++i) data[i] /= temp;

    size_t k = config.top_k;
    if (k > n) k = n;
    if (k == 0) k = 1;

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](size_t a, size_t b) { return data[a] > data[b]; });

    float threshold = data[indices[k - 1]];
    for (size_t i = 0; i < n; ++i) {
        if (data[i] < threshold) data[i] = -INFINITY;
    }

    float max_val = data[indices[0]];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (size_t i = 0; i < n; ++i) data[i] /= sum;

    return logits;
}

size_t TopKSampling::sample(const Tensor& probs, std::mt19937& rng) const {
    const float* data = probs.as_fp32_const();
    size_t n = probs.numel();

    std::vector<float> weights(data, data + n);
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return dist(rng);
}

Tensor TopPSampling::apply(Tensor logits, const GenerateConfig& config,
                            const std::vector<size_t>& generated) {
    float* data = logits.as_fp32();
    size_t n = logits.numel();

    apply_repetition_penalty(data, n, config.repetition_penalty, generated);

    float temp = config.temperature;
    if (temp <= 0.0f) {
        size_t best = 0;
        float best_val = data[0];
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > best_val) { best_val = data[i]; best = i; }
        }
        memset(data, 0, n * sizeof(float));
        data[best] = 1.0f;
        return logits;
    }

    for (size_t i = 0; i < n; ++i) data[i] /= temp;

    float max_val = *std::max_element(data, data + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (size_t i = 0; i < n; ++i) data[i] /= sum;

    float p = config.top_p;
    if (p <= 0.0f) p = 1e-6f;
    if (p > 1.0f) p = 1.0f;

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return data[a] > data[b]; });

    float cumsum = 0.0f;
    size_t cutoff = n;
    for (size_t i = 0; i < n; ++i) {
        cumsum += data[indices[i]];
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    std::vector<bool> keep(n, false);
    for (size_t i = 0; i < cutoff; ++i) keep[indices[i]] = true;

    sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) data[i] = 0.0f;
        sum += data[i];
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i < n; ++i) data[i] /= sum;
    }

    return logits;
}

size_t TopPSampling::sample(const Tensor& probs, std::mt19937& rng) const {
    const float* data = probs.as_fp32_const();
    size_t n = probs.numel();

    std::vector<float> weights(data, data + n);
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return dist(rng);
}

Tensor TopKTopPSampling::apply(Tensor logits, const GenerateConfig& config,
                                const std::vector<size_t>& generated) {
    float* data = logits.as_fp32();
    size_t n = logits.numel();

    apply_repetition_penalty(data, n, config.repetition_penalty, generated);

    float temp = config.temperature;
    if (temp <= 0.0f) {
        size_t best = 0;
        float best_val = data[0];
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > best_val) { best_val = data[i]; best = i; }
        }
        memset(data, 0, n * sizeof(float));
        data[best] = 1.0f;
        return logits;
    }

    for (size_t i = 0; i < n; ++i) data[i] /= temp;

    size_t k = config.top_k;
    if (k > n) k = n;
    if (k == 0) k = 1;

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](size_t a, size_t b) { return data[a] > data[b]; });

    float threshold = data[indices[k - 1]];
    for (size_t i = 0; i < n; ++i) {
        if (data[i] < threshold) data[i] = -INFINITY;
    }

    float max_val = data[indices[0]];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (size_t i = 0; i < n; ++i) data[i] /= sum;

    float p = config.top_p;
    if (p <= 0.0f) p = 1e-6f;
    if (p > 1.0f) p = 1.0f;

    std::vector<size_t> sorted(n);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(),
              [&](size_t a, size_t b) { return data[a] > data[b]; });

    float cumsum = 0.0f;
    size_t cutoff = n;
    for (size_t i = 0; i < n; ++i) {
        cumsum += data[sorted[i]];
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    std::vector<bool> keep(n, false);
    for (size_t i = 0; i < cutoff; ++i) keep[sorted[i]] = true;

    sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) data[i] = 0.0f;
        sum += data[i];
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i < n; ++i) data[i] /= sum;
    }

    return logits;
}

size_t TopKTopPSampling::sample(const Tensor& probs, std::mt19937& rng) const {
    const float* data = probs.as_fp32_const();
    size_t n = probs.numel();

    std::vector<float> weights(data, data + n);
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return dist(rng);
}

} // namespace neuroflow

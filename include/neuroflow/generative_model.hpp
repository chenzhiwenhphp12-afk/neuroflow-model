#ifndef NEUROFLOW_GENERATIVE_MODEL_HPP
#define NEUROFLOW_GENERATIVE_MODEL_HPP

#include <memory>
#include <random>
#include <string>
#include <vector>
#include "model.hpp"
#include "tensor.hpp"
#include "causal_lm.hpp"
#include "tokenizer.hpp"
#include "sampling.hpp"

namespace neuroflow {

class GenerativeModel {
public:
    std::unique_ptr<CausalLMHead> lm_head_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<SamplingStrategy> sampler_;
    NeuroFlowModel* neuroflow_model_;

    GenerativeModel(const CausalLMConfig& lm_config,
                    std::unique_ptr<Tokenizer> tokenizer,
                    NeuroFlowModel* nf_model = nullptr);

    GenerateOutput generate(const std::string& prompt, const GenerateConfig& config);

    Tensor apply_sn_gating(const Tensor& hidden, const Tensor& logits);
    Tensor inject_memory(const Tensor& query, const Tensor& logits);

    Tensor apply_repetition_penalty(Tensor logits, const GenerateConfig& config,
                                     const std::vector<size_t>& generated);
    Tensor apply_punct_penalty(Tensor logits, const GenerateConfig& config);

    void set_strategy(SamplingStrategyType type);

    void clear_cache();
    CacheStats cache_stats() const;
};

} // namespace neuroflow

#endif // NEUROFLOW_GENERATIVE_MODEL_HPP
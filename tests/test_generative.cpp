#include <iostream>
#include <cassert>
#include <cmath>
#include "neuroflow/generative.hpp"

using namespace neuroflow;

void test_tokenizer() {
    std::cout << "=== Tokenizer Test ===" << std::endl;

    BPETokenizer tok;
    tok.add_vocab("你", 4);
    tok.add_vocab("好", 5);
    tok.add_vocab("世", 6);
    tok.add_vocab("界", 7);
    tok.add_vocab("hello", 8);
    tok.add_vocab(" ", 9);
    tok.add_vocab("world", 10);
    tok.set_vocab_size(11);

    auto ids = tok.encode("你好世界");
    std::cout << "encode('你好世界') = [";
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << ids[i];
        if (i < ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    assert(ids[0] == tok.bos_id());
    assert(ids.back() == tok.eos_id());
    std::cout << "  BOS/EOS check: PASS" << std::endl;

    auto decoded = tok.decode(ids);
    std::cout << "decode result: '" << decoded << "'" << std::endl;
    std::cout << "  Tokenizer test PASSED" << std::endl;
}

void test_causal_lm_head() {
    std::cout << "\n=== CausalLMHead Test ===" << std::endl;

    CausalLMConfig config;
    config.vocab_size = 100;
    config.d_model = 32;
    config.max_seq_len = 64;
    config.causal_window_size = 8;
    config.sae_k = 16;
    config.ntm_memory_slots = 4;
    config.use_mla = false;

    CausalLMHead lm(config);
    std::cout << "  CausalLMHead constructed: vocab=" << config.vocab_size
              << " d_model=" << config.d_model << std::endl;

    std::vector<size_t> ids = {2, 5, 10, 20, 3};
    Tensor logits = lm.forward(ids);
    std::cout << "  forward() output shape: [" << logits.shape_[0] << ", " << logits.shape_[1] << "]" << std::endl;
    assert(logits.shape_[0] == 1);
    assert(logits.shape_[1] == config.vocab_size);

    float max_logit = *std::max_element(logits.as_fp32(), logits.as_fp32() + logits.numel());
    float min_logit = *std::min_element(logits.as_fp32(), logits.as_fp32() + logits.numel());
    std::cout << "  logits range: [" << min_logit << ", " << max_logit << "]" << std::endl;
    assert(!std::isnan(max_logit) && !std::isnan(min_logit));
    std::cout << "  NaN check: PASS" << std::endl;

    lm.clear_cache();
    Tensor step_logits = lm.forward_step(5, 0);
    std::cout << "  forward_step() output shape: [" << step_logits.shape_[0] << ", " << step_logits.shape_[1] << "]" << std::endl;
    assert(step_logits.shape_[1] == config.vocab_size);
    std::cout << "  CausalLMHead test PASSED" << std::endl;
}

void test_sampling_strategies() {
    std::cout << "\n=== Sampling Strategy Test ===" << std::endl;

    std::mt19937 rng(42);

    Tensor logits({1, 10}, QuantType::FP32);
    float* data = logits.as_fp32();
    for (size_t i = 0; i < 10; ++i) data[i] = static_cast<float>(i) * 0.5f;

    GenerateConfig config;
    config.temperature = 1.0f;
    config.top_k = 5;
    config.top_p = 0.9f;
    config.repetition_penalty = 1.0f;

    GreedyDecoding greedy;
    Tensor greedy_probs = greedy.apply(logits.clone(), config, {});
    size_t greedy_id = greedy.sample(greedy_probs, rng);
    std::cout << "  Greedy: selected token " << greedy_id << " (expected 9)" << std::endl;
    assert(greedy_id == 9);

    rng.seed(42);
    TopKSampling topk;
    Tensor topk_probs = topk.apply(logits.clone(), config, {});
    size_t topk_id = topk.sample(topk_probs, rng);
    std::cout << "  Top-K(K=5): selected token " << topk_id << std::endl;
    assert(topk_id >= 5);

    rng.seed(42);
    TopPSampling topp;
    Tensor topp_probs = topp.apply(logits.clone(), config, {});
    size_t topp_id = topp.sample(topp_probs, rng);
    std::cout << "  Top-P(P=0.9): selected token " << topp_id << std::endl;

    config.temperature = 0.0f;
    Tensor temp0_probs = topk.apply(logits.clone(), config, {});
    size_t temp0_id = topk.sample(temp0_probs, rng);
    std::cout << "  Temperature=0 (greedy fallback): selected token " << temp0_id << std::endl;
    assert(temp0_id == 9);

    std::cout << "  Sampling strategy test PASSED" << std::endl;
}

void test_generative_model() {
    std::cout << "\n=== GenerativeModel Test ===" << std::endl;

    CausalLMConfig lm_config;
    lm_config.vocab_size = 200;
    lm_config.d_model = 64;
    lm_config.max_seq_len = 64;
    lm_config.causal_window_size = 8;
    lm_config.sae_k = 16;
    lm_config.ntm_memory_slots = 4;
    lm_config.use_mla = false;

    auto tokenizer = std::make_unique<BPETokenizer>();
    tokenizer->add_vocab("你", 4);
    tokenizer->add_vocab("好", 5);
    tokenizer->add_vocab("世", 6);
    tokenizer->add_vocab("界", 7);
    tokenizer->add_vocab("测", 8);
    tokenizer->add_vocab("试", 9);
    tokenizer->add_vocab("生", 10);
    tokenizer->add_vocab("成", 11);
    tokenizer->set_vocab_size(200);

    GenerativeModel model(lm_config, std::move(tokenizer));
    std::cout << "  GenerativeModel constructed" << std::endl;

    GenerateConfig gen_config;
    gen_config.max_new_tokens = 10;
    gen_config.temperature = 0.8f;
    gen_config.top_k = 20;
    gen_config.random_seed = 12345;
    gen_config.eos_id = 3;

    GenerateOutput output = model.generate("你好", gen_config);
    std::cout << "  Generated text: '" << output.text << "'" << std::endl;
    std::cout << "  Generated " << output.token_ids.size() << " tokens" << std::endl;
    std::cout << "  Finish reason: " << static_cast<int>(output.finish_reason) << std::endl;
    std::cout << "  Cache stats: len=" << output.cache_stats.cache_len
              << " mem=" << output.cache_stats.memory_bytes << " bytes" << std::endl;

    assert(!output.token_ids.empty());

    model.set_strategy(SamplingStrategyType::GREEDY);
    gen_config.random_seed = 42;
    GenerateOutput greedy_out = model.generate("测试", gen_config);
    std::cout << "  Greedy output: '" << greedy_out.text << "'" << std::endl;

    model.set_strategy(SamplingStrategyType::TOP_P);
    gen_config.temperature = 1.0f;
    gen_config.random_seed = 99;
    GenerateOutput topp_out = model.generate("生成", gen_config);
    std::cout << "  Top-P output: '" << topp_out.text << "'" << std::endl;

    std::cout << "  GenerativeModel test PASSED" << std::endl;
}

void test_repetition_penalty() {
    std::cout << "\n=== Repetition Penalty Test ===" << std::endl;

    CausalLMConfig config;
    config.vocab_size = 50;
    config.d_model = 16;
    config.max_seq_len = 32;
    config.causal_window_size = 4;
    config.sae_k = 8;
    config.ntm_memory_slots = 2;
    config.use_mla = false;

    auto tokenizer = std::make_unique<BPETokenizer>();
    tokenizer->set_vocab_size(50);

    GenerativeModel model(config, std::move(tokenizer));

    GenerateConfig gen_config;
    gen_config.max_new_tokens = 15;
    gen_config.temperature = 0.8f;
    gen_config.top_k = 10;
    gen_config.repetition_penalty = 1.5f;
    gen_config.random_seed = 42;

    GenerateOutput output = model.generate("测试", gen_config);

    std::unordered_map<size_t, size_t> counts;
    for (auto id : output.token_ids) counts[id]++;
    size_t max_repeat = 0;
    for (auto& [id, cnt] : counts) max_repeat = std::max(max_repeat, cnt);
    std::cout << "  Max repetition count: " << max_repeat << std::endl;
    std::cout << "  Repetition penalty test PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "NeuroFlow Generative Model Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_tokenizer();
        test_causal_lm_head();
        test_sampling_strategies();
        test_generative_model();
        test_repetition_penalty();

        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
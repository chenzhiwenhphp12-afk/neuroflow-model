#ifndef NEUROFLOW_GENERATIVE_HPP
#define NEUROFLOW_GENERATIVE_HPP

/**
 * NeuroFlow 生成式语言模型模块
 *
 * 为类脑模块化网络增加因果语言模型能力：
 * - CausalLMHead: 词嵌入+因果门控+SAE+NTM+投影输出
 * - Tokenizer: BPE/WordPiece分词器
 * - SamplingStrategy: 贪心/Top-K/Top-P采样
 * - GenerativeModel: 编排层，与ECN/DMN/SN协同
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "memory.hpp"
#include "model.hpp"
#include "networks.hpp"
#include "tensor.hpp"

#ifdef USE_CUDA
#include "cuda_kernels.hpp"
#endif

namespace neuroflow {



enum class SamplingStrategyType : uint8_t {
    GREEDY = 0,
    TOP_K = 1,
    TOP_P = 2,
    TOP_K_TOP_P = 3
};

enum class FinishReason : uint8_t {
    EOS_TOKEN = 0,
    MAX_LENGTH = 1,
    GEN_ERROR = 2
};

struct CausalLMConfig {
    size_t vocab_size = 128000;
    size_t d_model = 256;
    size_t max_seq_len = 128;
    size_t causal_window_size = 32;
    size_t sae_k = 64;
    size_t ntm_memory_slots = 16;
    bool use_mla = true;
    size_t mla_latent_dim = 32;
    size_t mla_n_heads = 8;
    size_t mla_max_cache_len = 4096;
    bool use_quantization = false;
    bool weight_tying = true;
    size_t num_attn_layers = 2;
    size_t num_attn_heads = 4;
    std::string pooling = "mean";
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

struct CacheStats {
    size_t cache_len = 0;
    size_t memory_bytes = 0;
    float saving_ratio = 0.0f;
    size_t sliding_window_drops = 0;
};

struct GenerateOutput {
    std::string text;
    std::vector<size_t> token_ids;
    std::vector<Tensor> logits_history;
    FinishReason finish_reason = FinishReason::MAX_LENGTH;
    CacheStats cache_stats;
};

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<size_t> encode(const std::string& text, size_t max_len = 0) = 0;
    virtual std::string decode(const std::vector<size_t>& ids) = 0;

    size_t vocab_size() const { return vocab_size_; }
    size_t pad_id() const { return pad_id_; }
    size_t unk_id() const { return unk_id_; }
    size_t bos_id() const { return bos_id_; }
    size_t eos_id() const { return eos_id_; }

protected:
    size_t vocab_size_ = 128000;
    size_t pad_id_ = 0;
    size_t unk_id_ = 1;
    size_t bos_id_ = 2;
    size_t eos_id_ = 3;
    size_t max_input_len_ = 128;
};

class BPETokenizer : public Tokenizer {
public:
    BPETokenizer(const std::string& config_path) {
        std::ifstream f(config_path);
        if (!f) throw std::runtime_error("Cannot open tokenizer config: " + config_path);
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        f.close();
        parse_config(content);
    }

    BPETokenizer() {}

    std::vector<size_t> encode(const std::string& text, size_t max_len = 0) override {
        size_t limit = (max_len > 0) ? max_len : max_input_len_;
        std::vector<uint8_t> bytes = utf8_to_bytes(text);

        std::vector<size_t> ids;
        ids.push_back(bos_id_);

        size_t i = 0;
        while (i < bytes.size() && ids.size() < limit - 1) {
            std::string byte_seq;
            size_t byte_len = 1;
            if ((bytes[i] & 0xE0) == 0xC0) byte_len = 2;
            else if ((bytes[i] & 0xF0) == 0xE0) byte_len = 3;
            else if ((bytes[i] & 0xF8) == 0xF0) byte_len = 4;

            for (size_t j = 0; j < byte_len && i + j < bytes.size(); ++j) {
                byte_seq += static_cast<char>(bytes[i + j]);
            }

            std::string bpe_result = apply_bpe(byte_seq);
            auto it = vocab_.find(bpe_result);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else {
                for (char c : bpe_result) {
                    std::string single(1, c);
                    auto sit = vocab_.find(single);
                    ids.push_back((sit != vocab_.end()) ? sit->second : unk_id_);
                }
            }
            i += byte_len;
        }

        ids.push_back(eos_id_);
        if (ids.size() > limit) {
            ids.resize(limit - 1);
            ids.push_back(eos_id_);
        }
        return ids;
    }

    std::string decode(const std::vector<size_t>& ids) override {
        std::string result;
        for (auto id : ids) {
            if (id == pad_id_ || id == unk_id_ || id == bos_id_ || id == eos_id_) continue;
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
                result += it->second;
            }
        }
        return bytes_to_utf8(result);
    }

    void add_vocab(const std::string& token, size_t id) {
        vocab_[token] = id;
        id_to_token_[id] = token;
    }

    void set_vocab_size(size_t vs) { vocab_size_ = vs; }

private:
    std::unordered_map<std::string, size_t> vocab_;
    std::unordered_map<size_t, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, size_t> merge_ranks_;

    std::vector<uint8_t> utf8_to_bytes(const std::string& text) {
        std::vector<uint8_t> bytes;
        for (unsigned char c : text) bytes.push_back(c);
        return bytes;
    }

    std::string bytes_to_utf8(const std::string& raw) {
        return raw;
    }

    std::string apply_bpe(const std::string& token) {
    if (token.size() <= 1 || merge_ranks_.empty()) return token;

    // ═══════════════════════════════════════════════════════
    // BPE 优先队列算法 — O(K·M) 替代原始 O(K·M²)
    // K=合并轮数, M=初始符号数(=token长度)
    //
    // 原始: 每轮遍历所有邻接对 O(M²), 删除O(M) → 总 O(K·M²)
    // 优化: 优先队列缓存对 → 每轮弹出O(logH) + 懒验证O(1)
    //       合并后仅更新受影响的2个邻接对 → 总 O(K·logH)
    // 1000字token: ~1000x 加速
    // ═══════════════════════════════════════════════════════

    struct PairEntry {
    size_t rank;
    size_t pos;
    bool operator>(const PairEntry& o) const {
    return rank > o.rank; // min-heap
    }
    };
    std::priority_queue<PairEntry, std::vector<PairEntry>,
    std::greater<PairEntry>> pq;

    // 双向链表: next[i]/prev[i], END=末尾
    const size_t END = SIZE_MAX;
    const size_t HEAD = 0;
    std::vector<size_t> next_link(token.size());
    std::vector<size_t> prev_link(token.size());
    std::vector<std::string> symbols(token.size());
    for (size_t i = 0; i < token.size(); ++i) {
    symbols[i] = std::string(1, token[i]);
    next_link[i] = (i + 1 < token.size()) ? i + 1 : END;
    prev_link[i] = (i > 0) ? i - 1 : END;
    }

    // 入堆: 检查(i, next[i])对是否在merge表
    auto push_pair = [&](size_t i) {
    size_t j = next_link[i];
    if (j == END) return;
    auto it = merge_ranks_.find(symbols[i] + symbols[j]);
    if (it != merge_ranks_.end()) {
    pq.push({it->second, i});
    }
    };

    // 初始扫描
    for (size_t i = HEAD; i != END; i = next_link[i]) {
    push_pair(i);
    }

    // 主循环: 弹出最小rank → 懒验证 → 合并 → 更新邻接
    while (!pq.empty()) {
    PairEntry top = pq.top();
    pq.pop();

    size_t i = top.pos;
    size_t j = next_link[i];
    if (j == END) continue; // 右邻已消失

    // 懒验证: pair字符串+rank必须匹配(否则是过期条目)
    auto it = merge_ranks_.find(symbols[i] + symbols[j]);
    if (it == merge_ranks_.end() || it->second != top.rank) continue;

    // 执行合并: i吞并j, 从链表删除j
    size_t k = next_link[j]; // j的右邻
    symbols[i] += symbols[j];
    next_link[i] = k;
    if (k != END) prev_link[k] = i; // k回指i
    // j逻辑删除: 无需清理, 不会再被访问

    // 更新受影响的邻接对:
    // 1) (prev, i): i内容变了, 左邻可能与i产生新merge
    // 2) (i, k):   i内容变了, i可能与k产生新merge
    if (prev_link[i] != END) push_pair(prev_link[i]);
    push_pair(i);
    }

    // 链表遍历收集结果
    std::string result;
    for (size_t i = HEAD; i != END; i = next_link[i]) {
    result += symbols[i];
    }
    return result;
    }

    void parse_config(const std::string& content) {
        size_t vocab_pos = content.find("\"vocab\":{");
        if (vocab_pos == std::string::npos) vocab_pos = content.find("\"vocab\": {");
        if (vocab_pos == std::string::npos) return;

        size_t obj_start = content.find('{', vocab_pos);
        size_t brace_depth = 0;
        size_t obj_end = obj_start;
        bool in_string = false;
        bool escape_next = false;
        for (; obj_end < content.size(); ++obj_end) {
            char c = content[obj_end];
            if (escape_next) { escape_next = false; continue; }
            if (c == '\\') { escape_next = true; continue; }
            if (c == '"') { in_string = !in_string; continue; }
            if (in_string) continue;
            if (c == '{') brace_depth++;
            else if (c == '}') {
                if (--brace_depth == 0) break;
            }
        }
        if (obj_start == std::string::npos || brace_depth != 0) return;

        size_t id = 0;
        size_t pos = obj_start + 1;
        size_t obj_limit = obj_end;
        size_t parse_errors = 0;
        while (pos < obj_limit) {
            size_t key_start = content.find('"', pos);
            if (key_start == std::string::npos || key_start >= obj_limit) break;
            size_t key_end = content.find('"', key_start + 1);
            if (key_end == std::string::npos || key_end >= obj_limit) break;

            std::string key = content.substr(key_start + 1, key_end - key_start - 1);

            size_t colon = content.find(':', key_end);
            if (colon == std::string::npos || colon >= obj_limit) break;

            size_t val_start = colon + 1;
            while (val_start < obj_limit &&
                   (content[val_start] == ' ' || content[val_start] == '\t' ||
                    content[val_start] == '\n' || content[val_start] == '\r'))
                ++val_start;

            size_t val_end = val_start;
            while (val_end < obj_limit &&
                   content[val_end] != ',' && content[val_end] != '}' &&
                   content[val_end] != '\n' && content[val_end] != '\r')
                ++val_end;

            std::string val_str = content.substr(val_start, val_end - val_start);
            try {
                size_t token_id = std::stoul(val_str);
                vocab_[key] = token_id;
                id_to_token_[token_id] = key;
                id = token_id + 1;
            } catch (...) {
                parse_errors++;
                if (parse_errors <= 3) {
                    std::cerr << "[Tokenizer] Parse error at key=" << key << " val=" << val_str << std::endl;
                }
            }
            pos = val_end + 1;
        }
        vocab_size_ = id;
        if (parse_errors > 0) {
            std::cerr << "[Tokenizer] Total parse errors: " << parse_errors << std::endl;
        }

        size_t merges_pos = content.find("\"merges\":");
        if (merges_pos == std::string::npos) merges_pos = content.find("\"merges\" :");
        if (merges_pos != std::string::npos) {
            size_t arr_start = content.find('[', merges_pos);
            pos = arr_start + 1;
            size_t rank = 0;
            while (pos < content.size()) {
                size_t q1 = content.find('\"', pos);
                if (q1 == std::string::npos) break;
                size_t q2 = content.find('\"', q1 + 1);
                if (q2 == std::string::npos) break;

                std::string merge_str = content.substr(q1 + 1, q2 - q1 - 1);
                size_t space = merge_str.find(' ');
                if (space != std::string::npos) {
                    std::string left = merge_str.substr(0, space);
                    std::string right = merge_str.substr(space + 1);
                    merges_.push_back({left, right});
                    merge_ranks_[left + right] = rank++;
                }
                pos = q2 + 1;
            }
        }
    }
};

class WordPieceTokenizer : public Tokenizer {
public:
    WordPieceTokenizer() {}

    WordPieceTokenizer(const std::string& config_path) {
        std::ifstream f(config_path);
        if (!f) throw std::runtime_error("Cannot open tokenizer config: " + config_path);
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        f.close();
        parse_vocab(content);
    }

    std::vector<size_t> encode(const std::string& text, size_t max_len = 0) override {
        size_t limit = (max_len > 0) ? max_len : max_input_len_;
        std::vector<size_t> ids;
        ids.push_back(bos_id_);

        size_t i = 0;
        while (i < text.size() && ids.size() < limit - 1) {
            size_t char_len = 1;
            if ((text[i] & 0xE0) == 0xC0) char_len = 2;
            else if ((text[i] & 0xF0) == 0xE0) char_len = 3;
            else if ((text[i] & 0xF8) == 0xF0) char_len = 4;

            bool matched = false;
            for (size_t len = char_len; len <= std::min(text.size() - i, static_cast<size_t>(20)); ++len) {
                std::string sub = (len == char_len) ?
                    text.substr(i, len) : ("##" + text.substr(i, len));
                auto it = vocab_.find(sub);
                if (it != vocab_.end()) {
                    ids.push_back(it->second);
                    i += len;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                ids.push_back(unk_id_);
                i += char_len;
            }
        }

        ids.push_back(eos_id_);
        if (ids.size() > limit) {
            ids.resize(limit - 1);
            ids.push_back(eos_id_);
        }
        return ids;
    }

    std::string decode(const std::vector<size_t>& ids) override {
        std::string result;
        for (auto id : ids) {
            if (id == pad_id_ || id == unk_id_ || id == bos_id_ || id == eos_id_) continue;
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
                std::string token = it->second;
                if (token.size() > 2 && token[0] == '#' && token[1] == '#') {
                    result += token.substr(2);
                } else {
                    result += token;
                }
            }
        }
        return result;
    }

    void add_vocab(const std::string& token, size_t id) {
        vocab_[token] = id;
        id_to_token_[id] = token;
    }

    void set_vocab_size(size_t vs) { vocab_size_ = vs; }

private:
    std::unordered_map<std::string, size_t> vocab_;
    std::unordered_map<size_t, std::string> id_to_token_;

    void parse_vocab(const std::string& content) {
        size_t id = 0;
        size_t pos = 0;
        while (pos < content.size()) {
            size_t line_end = content.find('\n', pos);
            if (line_end == std::string::npos) line_end = content.size();
            std::string token = content.substr(pos, line_end - pos);
            if (!token.empty()) {
                vocab_[token] = id;
                id_to_token_[id] = token;
                ++id;
            }
            pos = line_end + 1;
        }
        vocab_size_ = id;
    }
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
                 const std::vector<size_t>& generated) override {
        return logits;
    }

    size_t sample(const Tensor& probs, std::mt19937& rng) const override {
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
};

class TopKSampling : public SamplingStrategy {
public:
    Tensor apply(Tensor logits, const GenerateConfig& config,
                 const std::vector<size_t>& generated) override {
        float* data = logits.as_fp32();
        size_t n = logits.numel();

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

        if (temp < 0.0f) {
            std::cerr << "Warning: negative temperature, using absolute value" << std::endl;
            temp = -temp;
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

    size_t sample(const Tensor& probs, std::mt19937& rng) const override {
        const float* data = probs.as_fp32_const();
        size_t n = probs.numel();

        std::vector<float> weights(data, data + n);
        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        return dist(rng);
    }
};

class TopPSampling : public SamplingStrategy {
public:
    Tensor apply(Tensor logits, const GenerateConfig& config,
                 const std::vector<size_t>& generated) override {
        float* data = logits.as_fp32();
        size_t n = logits.numel();

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
        if (p <= 0.0f) {
            std::cerr << "Warning: top_p <= 0, setting to 1e-6" << std::endl;
            p = 1e-6f;
        }
        if (p > 1.0f) {
            std::cerr << "Warning: top_p > 1, setting to 1.0" << std::endl;
            p = 1.0f;
        }

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

    size_t sample(const Tensor& probs, std::mt19937& rng) const override {
        const float* data = probs.as_fp32_const();
        size_t n = probs.numel();

        std::vector<float> weights(data, data + n);
        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        return dist(rng);
    }
};

class CausalSelfAttention {
public:
    size_t n_heads_;
    size_t d_model_;
    size_t head_dim_;

    std::shared_ptr<Linear> w_qkv;
    std::shared_ptr<Linear> w_out;
    std::shared_ptr<LayerNorm> norm;

    struct Cache {
        Tensor input;
        Tensor qkv;
        Tensor attn_weights;
        Tensor attn_output;
        Tensor w_out_input;
        Tensor residual;
    };
    Cache cache_;

    CausalSelfAttention(size_t d_model, size_t n_heads)
        : n_heads_(n_heads), d_model_(d_model), head_dim_(d_model / n_heads) {
        w_qkv = std::make_shared<Linear>(d_model, d_model * 3, true);
        w_out = std::make_shared<Linear>(d_model, d_model, true);
        norm = std::make_shared<LayerNorm>(d_model);
    }

    Tensor forward(const Tensor& x) {
        cache_.input = x.clone();
        size_t seq_len = x.shape_[0];

        cache_.qkv = w_qkv->forward(x);

        cache_.attn_weights = Tensor({n_heads_, seq_len, seq_len}, QuantType::FP32);
        cache_.attn_output = Tensor({seq_len, d_model_}, QuantType::FP32);

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && x.is_on_gpu()) {
            cache_.attn_weights.to_gpu();
            cache_.attn_output.to_gpu();
            const float* qkvp = cache_.qkv.as_gpu_fp32();
            float* aw = cache_.attn_weights.as_gpu_fp32();
            float* ao = cache_.attn_output.as_gpu_fp32();
            auto stream = CudaContext::instance().stream();

            launch_fill_zero(ao, seq_len * d_model_, stream);

            for (size_t h = 0; h < n_heads_; ++h) {
                size_t q_off = h * head_dim_;
                size_t k_off = d_model_ + h * head_dim_;
                size_t v_off = 2 * d_model_ + h * head_dim_;

                Tensor Q_h({seq_len, head_dim_}, QuantType::FP32);
                Tensor K_h({seq_len, head_dim_}, QuantType::FP32);
                Tensor V_h({seq_len, head_dim_}, QuantType::FP32);
                Q_h.to_gpu(); K_h.to_gpu(); V_h.to_gpu();

                launch_extract_qkv(Q_h.as_gpu_fp32(), K_h.as_gpu_fp32(), V_h.as_gpu_fp32(),
                                    qkvp, seq_len, d_model_, head_dim_, h, stream);

                float* aw_h = aw + h * seq_len * seq_len;
                CudaContext::instance().sgemm_rowmajor(false, true,
                    static_cast<int>(seq_len), static_cast<int>(seq_len), static_cast<int>(head_dim_),
                    scale, Q_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                    K_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                    0.0f, aw_h, static_cast<int>(seq_len));

                launch_causal_softmax(aw_h, seq_len, stream);

                float* ao_ptr = ao;
                CudaContext::instance().sgemm_rowmajor(false, false,
                    static_cast<int>(seq_len), static_cast<int>(head_dim_), static_cast<int>(seq_len),
                    1.0f, aw_h, static_cast<int>(seq_len),
                    V_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                    1.0f, ao_ptr, static_cast<int>(d_model_));

                cache_.attn_weights.gpu_dirty_ = true;
                cache_.attn_output.gpu_dirty_ = true;
            }

            cache_.w_out_input = cache_.attn_output.clone();
            Tensor projected = w_out->forward(cache_.attn_output);

            cache_.residual = Tensor({seq_len, d_model_}, QuantType::FP32);
            cache_.residual.to_gpu();
            launch_add(cache_.residual.as_gpu_fp32(), projected.as_gpu_fp32(), cache_.input.as_gpu_fp32(), seq_len * d_model_, stream);
            cache_.residual.gpu_dirty_ = true;

            return norm->forward(cache_.residual);
        }
#endif

        const float* qkvp = cache_.qkv.as_fp32();
        float* aw = cache_.attn_weights.as_fp32();
        float* ao = cache_.attn_output.as_fp32();
        memset(ao, 0, cache_.attn_output.data_size_);
        const float* xp = x.as_fp32();

        for (size_t h = 0; h < n_heads_; ++h) {
            size_t q_off = h * head_dim_;
            size_t k_off = d_model_ + h * head_dim_;
            size_t v_off = 2 * d_model_ + h * head_dim_;

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim_; ++d) {
                        dot += qkvp[i * 3 * d_model_ + q_off + d]
                             * qkvp[j * 3 * d_model_ + k_off + d];
                    }
                    aw[h * seq_len * seq_len + i * seq_len + j] = dot * scale;
                }
            }

            for (size_t i = 0; i < seq_len; ++i) {
                float max_val = -1e30f;
                for (size_t j = 0; j <= i; ++j) {
                    if (aw[h * seq_len * seq_len + i * seq_len + j] > max_val)
                        max_val = aw[h * seq_len * seq_len + i * seq_len + j];
                }
                float sum = 0.0f;
                for (size_t j = 0; j <= i; ++j) {
                    aw[h * seq_len * seq_len + i * seq_len + j] = std::exp(aw[h * seq_len * seq_len + i * seq_len + j] - max_val);
                    sum += aw[h * seq_len * seq_len + i * seq_len + j];
                }
                for (size_t j = 0; j <= i; ++j) {
                    aw[h * seq_len * seq_len + i * seq_len + j] /= sum;
                }
                for (size_t j = i + 1; j < seq_len; ++j) {
                    aw[h * seq_len * seq_len + i * seq_len + j] = 0.0f;
                }
            }

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    float val = 0.0f;
                    for (size_t j = 0; j <= i; ++j) {
                        val += aw[h * seq_len * seq_len + i * seq_len + j] * qkvp[j * 3 * d_model_ + v_off + d];
                    }
                    ao[i * d_model_ + h * head_dim_ + d] = val;
                }
            }
        }

        cache_.w_out_input = cache_.attn_output.clone();
        Tensor projected = w_out->forward(cache_.attn_output);

        cache_.residual = Tensor({seq_len, d_model_}, QuantType::FP32);
        float* rp = cache_.residual.as_fp32();
        const float* pp = projected.as_fp32();
        const float* xpp = xp;
        for (size_t i = 0; i < seq_len * d_model_; ++i) {
            rp[i] = pp[i] + xpp[i];
        }

        return norm->forward(cache_.residual);
    }

    struct Gradients {
        Tensor w_qkv_weight_grad;
        Tensor w_qkv_bias_grad;
        Tensor w_out_weight_grad;
        Tensor w_out_bias_grad;
        Tensor input_grad;
    };

    Gradients backward(const Tensor& output_grad) {
        Gradients grads;
        size_t seq_len = cache_.input.shape_[0];
        const float* og = output_grad.as_fp32();
        const float* aw = cache_.attn_weights.as_fp32();
        const float* qkvp = cache_.qkv.as_fp32();

        Tensor residual_grad = layernorm_backward_impl(cache_.residual, norm->weight, output_grad);

        const float* rg = residual_grad.as_fp32();

        Tensor proj_grad({seq_len, d_model_}, QuantType::FP32);
        float* pg = proj_grad.as_fp32();
        memcpy(pg, rg, proj_grad.data_size_);

        grads.w_out_weight_grad = linear_backward_weight_impl(cache_.w_out_input, proj_grad);
        grads.w_out_bias_grad = bias_backward_impl(proj_grad);

        Tensor attn_out_grad = linear_backward_input_impl(proj_grad, w_out->weight);

        const float* aog = attn_out_grad.as_fp32();

        Tensor d_qkv({seq_len, 3 * d_model_}, QuantType::FP32);
        float* dqkvp = d_qkv.as_fp32();
        memset(dqkvp, 0, d_qkv.data_size_);

        float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

        for (size_t h = 0; h < n_heads_; ++h) {
            size_t q_off = h * head_dim_;
            size_t k_off = d_model_ + h * head_dim_;
            size_t v_off = 2 * d_model_ + h * head_dim_;

            // Extract per-head Q, K, V from cached qkv: [seq_len, head_dim]
            Tensor Q_h({seq_len, head_dim_}, QuantType::FP32);
            Tensor K_h({seq_len, head_dim_}, QuantType::FP32);
            Tensor V_h({seq_len, head_dim_}, QuantType::FP32);
            float* qh = Q_h.as_fp32();
            float* kh = K_h.as_fp32();
            float* vh = V_h.as_fp32();
            for (size_t i = 0; i < seq_len; ++i) {
                memcpy(qh + i * head_dim_, qkvp + i * 3 * d_model_ + q_off, head_dim_ * sizeof(float));
                memcpy(kh + i * head_dim_, qkvp + i * 3 * d_model_ + k_off, head_dim_ * sizeof(float));
                memcpy(vh + i * head_dim_, qkvp + i * 3 * d_model_ + v_off, head_dim_ * sizeof(float));
            }

            // Extract per-head attn_out_grad: [seq_len, head_dim]
            Tensor aog_h({seq_len, head_dim_}, QuantType::FP32);
            float* aogh = aog_h.as_fp32();
            for (size_t i = 0; i < seq_len; ++i) {
                memcpy(aogh + i * head_dim_, aog + i * d_model_ + h * head_dim_, head_dim_ * sizeof(float));
            }

            // d_attn_weights = aog_h @ V_h^T: [seq_len, seq_len]
            // But with causal mask: only j <= i
            Tensor d_attn_weights({seq_len, seq_len}, QuantType::FP32);
            float* daw = d_attn_weights.as_fp32();
            memset(daw, 0, d_attn_weights.data_size_);

#ifdef USE_CUDA
            if (CudaContext::instance().is_available() && cache_.qkv.is_on_gpu()) {
                d_attn_weights.to_gpu();
                CudaContext::instance().sgemm_rowmajor(false, true,
                    static_cast<int>(seq_len), static_cast<int>(seq_len), static_cast<int>(head_dim_),
                    1.0f, aog_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                    V_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                    0.0f, d_attn_weights.as_gpu_fp32(), static_cast<int>(seq_len));
                launch_causal_mask_zero(d_attn_weights.as_gpu_fp32(), seq_len, CudaContext::instance().stream());
                d_attn_weights.gpu_dirty_ = true;
            } else
#endif
#ifdef USE_CBLAS
            // Full matmul then zero out future positions
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, seq_len, head_dim_,
                        1.0f, aogh, head_dim_, vh, head_dim_,
                        0.0f, daw, seq_len);
            // Apply causal mask: zero j > i
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = i + 1; j < seq_len; ++j) {
                    daw[i * seq_len + j] = 0.0f;
                }
            }
#else
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < head_dim_; ++d) {
                        sum += aogh[i * head_dim_ + d] * vh[j * head_dim_ + d];
                    }
                    daw[i * seq_len + j] = sum;
                }
            }
#endif

            // d_V = attn_weights[h]^T @ aog_h: [seq_len, head_dim]
            // With causal: d_V[j,d] += aw[h,i,j] * aogh[i,d] for i >= j
            Tensor d_V_h({seq_len, head_dim_}, QuantType::FP32);
            float* dvh = d_V_h.as_fp32();
            memset(dvh, 0, d_V_h.data_size_);
            {
                const float* aw_h = aw + h * seq_len * seq_len;
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        float s = aw_h[i * seq_len + j];
                        for (size_t d = 0; d < head_dim_; ++d) {
                            dvh[j * head_dim_ + d] += s * aogh[i * head_dim_ + d];
                        }
                    }
                }
            }

            // Softmax backward: d_score[i,j] = s[i,j] * (daw[i,j] - sum_k(s[i,k] * daw[i,k]))
            // Then d_score *= inv_scale
            Tensor d_scores({seq_len, seq_len}, QuantType::FP32);
            float* dsp = d_scores.as_fp32();
            memset(dsp, 0, d_scores.data_size_);
            {
                const float* aw_h = aw + h * seq_len * seq_len;
                for (size_t i = 0; i < seq_len; ++i) {
                    float dot = 0.0f;
                    for (size_t k = 0; k <= i; ++k) {
                        dot += aw_h[i * seq_len + k] * daw[i * seq_len + k];
                    }
                    for (size_t j = 0; j <= i; ++j) {
                        float s = aw_h[i * seq_len + j];
                        dsp[i * seq_len + j] = s * (daw[i * seq_len + j] - dot) * inv_scale;
                    }
                }
            }

            // d_Q = d_scores @ K_h: [seq_len, head_dim]
            // d_K = d_scores^T @ Q_h: [seq_len, head_dim]
            Tensor d_Q_h({seq_len, head_dim_}, QuantType::FP32);
            Tensor d_K_h({seq_len, head_dim_}, QuantType::FP32);
#ifdef USE_CUDA
            if (CudaContext::instance().is_available() && cache_.qkv.is_on_gpu()) {
                d_Q_h.to_gpu(); d_K_h.to_gpu();
                CudaContext::instance().sgemm_rowmajor(false, false,
                    static_cast<int>(seq_len), static_cast<int>(head_dim_), static_cast<int>(seq_len),
                    1.0f, d_scores.as_gpu_fp32(), static_cast<int>(seq_len),
                    K_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                    0.0f, d_Q_h.as_gpu_fp32(), static_cast<int>(head_dim_));
                CudaContext::instance().sgemm_rowmajor(true, false,
                    static_cast<int>(seq_len), static_cast<int>(head_dim_), static_cast<int>(seq_len),
                    1.0f, d_scores.as_gpu_fp32(), static_cast<int>(seq_len),
                    Q_h.as_gpu_fp32(), static_cast<int>(head_dim_),
                    0.0f, d_K_h.as_gpu_fp32(), static_cast<int>(head_dim_));
                d_Q_h.gpu_dirty_ = true;
                d_K_h.gpu_dirty_ = true;
            } else
#endif
#ifdef USE_CBLAS
            // d_Q = d_scores @ K
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len, head_dim_, seq_len,
                        1.0f, dsp, seq_len, kh, head_dim_,
                        0.0f, d_Q_h.as_fp32(), head_dim_);
            // d_K = d_scores^T @ Q
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        seq_len, head_dim_, seq_len,
                        1.0f, dsp, seq_len, qh, head_dim_,
                        0.0f, d_K_h.as_fp32(), head_dim_);
#else
            {
                float* dqh = d_Q_h.as_fp32();
                float* dkh = d_K_h.as_fp32();
                memset(dqh, 0, d_Q_h.data_size_);
                memset(dkh, 0, d_K_h.data_size_);
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        float ds = dsp[i * seq_len + j];
                        for (size_t d = 0; d < head_dim_; ++d) {
                            dqh[i * head_dim_ + d] += ds * kh[j * head_dim_ + d];
                            dkh[j * head_dim_ + d] += ds * qh[i * head_dim_ + d];
                        }
                    }
                }
            }
#endif

            // Scatter d_Q, d_K, d_V back into d_qkv
            {
                const float* dqh = d_Q_h.as_fp32();
                const float* dkh = d_K_h.as_fp32();
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t d = 0; d < head_dim_; ++d) {
                        dqkvp[i * 3 * d_model_ + q_off + d] += dqh[i * head_dim_ + d];
                        dqkvp[i * 3 * d_model_ + k_off + d] += dkh[i * head_dim_ + d];
                        dqkvp[i * 3 * d_model_ + v_off + d] += dvh[i * head_dim_ + d];
                    }
                }
            }
        }

        grads.w_qkv_weight_grad = linear_backward_weight_impl(cache_.input, d_qkv);
        grads.w_qkv_bias_grad = bias_backward_impl(d_qkv);

        Tensor qkv_input_grad = linear_backward_input_impl(d_qkv, w_qkv->weight);

        grads.input_grad = Tensor({seq_len, d_model_}, QuantType::FP32);
        float* ig = grads.input_grad.as_fp32();
        const float* qig = qkv_input_grad.as_fp32();
        for (size_t i = 0; i < seq_len * d_model_; ++i) {
            ig[i] = pg[i] + qig[i];
        }

        return grads;
    }

private:
    Tensor layernorm_backward_impl(const Tensor& input, const Tensor& weight, const Tensor& output_grad, float eps = 1e-5f) {
        size_t seq_len = input.shape_[0];
        size_t dim = input.shape_[1];
        Tensor input_grad({seq_len, dim}, QuantType::FP32);
        const float* inp = input.as_fp32();
        const float* w = weight.as_fp32();
        const float* og = output_grad.as_fp32();
        float* ig = input_grad.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            float mean = 0.0f;
            for (size_t d = 0; d < dim; ++d) mean += inp[i * dim + d];
            mean /= dim;
            float var = 0.0f;
            for (size_t d = 0; d < dim; ++d) { float diff = inp[i * dim + d] - mean; var += diff * diff; }
            var /= dim;
            float inv_std = 1.0f / std::sqrt(var + eps);
            float sum_gn = 0.0f, sum_gnx = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float norm = (inp[i * dim + d] - mean) * inv_std;
                float gn = og[i * dim + d] * w[d];
                sum_gn += gn; sum_gnx += gn * norm;
            }
            for (size_t d = 0; d < dim; ++d) {
                float norm = (inp[i * dim + d] - mean) * inv_std;
                float gn = og[i * dim + d] * w[d];
                ig[i * dim + d] = inv_std * (gn - sum_gn / dim - norm * sum_gnx / dim);
            }
        }
        return input_grad;
    }

    Tensor linear_backward_weight_impl(const Tensor& input, const Tensor& output_grad) {
        size_t batch = input.shape_[0];
        size_t in_f = input.shape_[1];
        size_t out_f = output_grad.shape_[1];
        Tensor weight_grad({out_f, in_f}, QuantType::FP32);
#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && input.is_on_gpu()) {
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
                for (size_t b = 0; b < batch; ++b) sum += og[b * out_f + i] * inp[b * in_f + j];
                wg[i * in_f + j] = sum / batch;
            }
        }
#endif
        return weight_grad;
    }

    Tensor linear_backward_input_impl(const Tensor& output_grad, const Tensor& weight) {
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
#ifdef USE_CBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    batch, in_f, out_f,
                    1.0f, output_grad.as_fp32(), out_f, weight.as_fp32(), in_f,
                    0.0f, input_grad.as_fp32(), in_f);
#else
        const float* og = output_grad.as_fp32();
        const float* w = weight.as_fp32();
        float* ig = input_grad.as_fp32();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < in_f; ++j) {
                float sum = 0.0f;
                for (size_t i = 0; i < out_f; ++i) sum += og[b * out_f + i] * w[i * in_f + j];
                ig[b * in_f + j] = sum;
            }
        }
#endif
        return input_grad;
    }

    Tensor bias_backward_impl(const Tensor& output_grad) {
        size_t batch = output_grad.shape_[0];
        size_t dim = output_grad.shape_[1];
        Tensor grad({dim}, QuantType::FP32);
        const float* og = output_grad.as_fp32();
        float* g = grad.as_fp32();
        for (size_t j = 0; j < dim; ++j) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch; ++b) sum += og[b * dim + j];
            g[j] = sum / batch;
        }
        return grad;
    }
};

class CausalLMHead {
public:
    CausalLMConfig config_;

    Tensor w_embed_;
    Tensor w_pos_;
    Tensor dw_kernel_;
    std::shared_ptr<Linear> pw_conv_;
    std::shared_ptr<Linear> sae_w_encode_;
    std::shared_ptr<Linear> sae_w_decode_;
    std::shared_ptr<Linear> ntm_w_read_;
    std::shared_ptr<Linear> ntm_w_write_;
    std::shared_ptr<Linear> ntm_w_erase_;
    Tensor ntm_memory_;
    std::shared_ptr<Linear> w_proj_;
    std::shared_ptr<Linear> w_out_;
    std::shared_ptr<LayerNorm> ln_;
    std::shared_ptr<LatentKVCache> kv_cache_;
    Tensor last_hidden_;
    Tensor last_projected_;
    std::vector<std::unique_ptr<CausalSelfAttention>> attn_layers_;

    struct TrainingCache {
        std::vector<size_t> input_ids;
        Tensor x_embed;
        Tensor x_pos;
        std::vector<Tensor> attn_inputs;
        std::vector<Tensor> attn_qkv;
        std::vector<Tensor> attn_outputs;
        std::vector<Tensor> attn_weights;
        Tensor x_gate_in;
        Tensor x_after_gate;
        Tensor x_sae_encoded;
        Tensor x_after_sae;
        Tensor x_after_ntm;
        Tensor x_after_ln;
        Tensor x_pooled;
        Tensor x_projected;
    };
    TrainingCache train_cache_;

    size_t sliding_window_drops_;

    void tie_weights() {
        if (config_.weight_tying && w_out_) {
            w_out_->weight.shape_ = w_embed_.shape_;
            w_out_->weight.dtype_ = w_embed_.dtype_;
            w_out_->weight.layout_ = w_embed_.layout_;
            w_out_->weight.data_ = w_embed_.data_;
            w_out_->weight.data_size_ = w_embed_.data_size_;
            w_out_->weight.owns_data_ = false;
            w_out_->weight.strides_ = w_embed_.strides_;
        }
    }

    CausalLMHead(const CausalLMConfig& config) : config_(config), sliding_window_drops_(0) {
        w_embed_ = Tensor({config_.vocab_size, config_.d_model}, QuantType::FP32);
        float* we = w_embed_.as_fp32();
        float embed_scale = 1.0f / std::sqrt(static_cast<float>(config_.d_model));
        std::mt19937 embed_rng(config_.vocab_size * 31 + config_.d_model);
        std::uniform_real_distribution<float> dist(-embed_scale, embed_scale);
        for (size_t i = 0; i < w_embed_.numel(); ++i) we[i] = dist(embed_rng);

        w_pos_ = Tensor({config_.max_seq_len, config_.d_model}, QuantType::FP32);
        float* wp = w_pos_.as_fp32();
        for (size_t pos = 0; pos < config_.max_seq_len; ++pos) {
            for (size_t d = 0; d < config_.d_model; ++d) {
                float angle = static_cast<float>(pos) / std::pow(10000.0f, static_cast<float>(d % 2 ? d - 1 : d) / static_cast<float>(config_.d_model));
                wp[pos * config_.d_model + d] = (d % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
        }

        size_t gate_in = config_.causal_window_size * config_.d_model;
        dw_kernel_ = Tensor({config_.d_model, config_.causal_window_size}, QuantType::FP32);
        {
            float* k = dw_kernel_.as_fp32();
            std::mt19937 dw_rng(42);
            float dw_scale = std::sqrt(2.0f / static_cast<float>(config_.causal_window_size));
            std::uniform_real_distribution<float> dw_dist(-dw_scale, dw_scale);
            for (size_t i = 0; i < dw_kernel_.numel(); ++i) k[i] = dw_dist(dw_rng);
        }
        pw_conv_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);

        sae_w_encode_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);
        sae_w_decode_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);

        ntm_w_read_ = std::make_shared<Linear>(config_.d_model, config_.ntm_memory_slots, false);
        ntm_w_write_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);
        ntm_w_erase_ = std::make_shared<Linear>(config_.d_model, config_.d_model, false);
        ntm_memory_ = Tensor({config_.ntm_memory_slots, config_.d_model}, QuantType::FP32);

        w_proj_ = std::make_shared<Linear>(config_.d_model, config_.d_model);
        if (config_.weight_tying) {
            w_out_ = std::make_shared<Linear>(config_.d_model, config_.vocab_size, false);
            w_out_->weight.shape_ = w_embed_.shape_;
            w_out_->weight.dtype_ = w_embed_.dtype_;
            w_out_->weight.layout_ = w_embed_.layout_;
            w_out_->weight.data_ = w_embed_.data_;
            w_out_->weight.data_size_ = w_embed_.data_size_;
            w_out_->weight.owns_data_ = false;
            w_out_->weight.strides_ = w_embed_.strides_;
        } else {
            w_out_ = std::make_shared<Linear>(config_.d_model, config_.vocab_size);
        }
        ln_ = std::make_shared<LayerNorm>(config_.d_model);

        if (config_.use_mla) {
            kv_cache_ = std::make_shared<LatentKVCache>(
                config_.d_model, config_.mla_n_heads,
                config_.mla_latent_dim, config_.mla_max_cache_len);
        }

        last_hidden_ = Tensor({1, config_.d_model}, QuantType::FP32);

        for (size_t i = 0; i < config_.num_attn_layers; ++i) {
            attn_layers_.push_back(std::make_unique<CausalSelfAttention>(
                config_.d_model, config_.num_attn_heads));
        }
    }

    Tensor embed_lookup(const std::vector<size_t>& ids) {
        size_t seq_len = ids.size();
        Tensor output({seq_len, config_.d_model}, QuantType::FP32);
        float scale = std::sqrt(static_cast<float>(config_.d_model));

#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && w_embed_.is_on_gpu()) {
            output.to_gpu();
            std::vector<int> int_ids(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                int_ids[i] = static_cast<int>(ids[i] >= config_.vocab_size ? 1 : ids[i]);
            }
            void* d_ids = CudaContext::instance().alloc(int_ids.size() * sizeof(int));
            CudaContext::instance().copy_h2d(d_ids, int_ids.data(), int_ids.size() * sizeof(int));
            launch_embed_lookup(output.as_gpu_fp32(), w_embed_.as_gpu_fp32(),
                                static_cast<const int*>(d_ids), seq_len, config_.d_model, scale,
                                CudaContext::instance().stream());
            CudaContext::instance().free(d_ids);
            output.gpu_dirty_ = true;
            return output;
        }
#endif

        float* out = output.as_fp32();
        float* embed = w_embed_.as_fp32();

        for (size_t i = 0; i < seq_len; ++i) {
            size_t tid = ids[i];
            if (tid >= config_.vocab_size) {
                std::cerr << "Warning: embed out-of-range token_id=" << tid
                          << ", replacing with UNK" << std::endl;
                tid = 1;
            }
            const float* row = embed + tid * config_.d_model;
            for (size_t d = 0; d < config_.d_model; ++d) {
                out[i * config_.d_model + d] = row[d] * scale;
            }
        }
        return output;
    }

    Tensor positional_encode(const Tensor& x, size_t offset = 0) {
        Tensor result = x.clone();

#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && x.is_on_gpu()) {
            result.to_gpu();
            launch_positional_encode(result.as_gpu_fp32(), w_pos_.as_gpu_fp32(),
                x.shape_[0], config_.d_model, static_cast<int>(offset),
                CudaContext::instance().stream());
            result.gpu_dirty_ = true;
            return result;
        }
#endif

        float* out = result.as_fp32();
        const float* pos = w_pos_.as_fp32();

        size_t seq_len = x.shape_[0];
        for (size_t i = 0; i < seq_len; ++i) {
            size_t p = offset + i;
            if (p >= config_.max_seq_len) {
                std::cerr << "Warning: position " << p << " exceeds max_seq_len, using last" << std::endl;
                p = config_.max_seq_len - 1;
            }
            for (size_t d = 0; d < config_.d_model; ++d) {
                out[i * config_.d_model + d] += pos[p * config_.d_model + d];
            }
        }
        return result;
    }

    Tensor causal_window_gate(const Tensor& x) {
        size_t seq_len = x.shape_[0];
        size_t C = config_.d_model;
        size_t K = config_.causal_window_size;
        const float* xp = x.as_fp32();
        const float* kernel = dw_kernel_.as_fp32();

        Tensor dw_out({seq_len, C}, QuantType::FP32);
        float* dwp = dw_out.as_fp32();

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    if (i >= k) {
                        sum += xp[(i - k) * C + c] * kernel[c * K + k];
                    }
                }
                dwp[i * C + c] = sum;
            }
        }

        Tensor pw_out = pw_conv_->forward(dw_out);
        float* gp = pw_out.as_fp32();

        Tensor output({seq_len, C}, QuantType::FP32);
        float* out = output.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t d = 0; d < C; ++d) {
                float g = 1.0f / (1.0f + std::exp(-gp[i * C + d]));
                out[i * C + d] = xp[i * C + d] * g;
            }
        }
        return output;
    }

    Tensor sae_sparse(const Tensor& x) {
        Tensor encoded = sae_w_encode_->forward(x);
        float* data = encoded.as_fp32();
        size_t n = encoded.numel();
        size_t k = config_.sae_k;

        if (k < n) {
            std::vector<size_t> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                              [&](size_t a, size_t b) { return std::abs(data[a]) > std::abs(data[b]); });

            std::vector<bool> keep(n, false);
            for (size_t i = 0; i < k; ++i) keep[indices[i]] = true;
            for (size_t i = 0; i < n; ++i) {
                if (!keep[i]) data[i] = 0.0f;
            }
        }

        return sae_w_decode_->forward(encoded);
    }

    Tensor ntm_memory_access(const Tensor& x) {
        size_t batch = x.shape_[0];
        Tensor read_weights = ntm_w_read_->forward(x);
        float* rwp = read_weights.as_fp32();
        size_t slots = config_.ntm_memory_slots;

        for (size_t b = 0; b < batch; ++b) {
            float max_val = rwp[b * slots];
            for (size_t s = 1; s < slots; ++s) {
                max_val = std::max(max_val, rwp[b * slots + s]);
            }
            float sum = 0.0f;
            for (size_t s = 0; s < slots; ++s) {
                rwp[b * slots + s] = std::exp(rwp[b * slots + s] - max_val);
                sum += rwp[b * slots + s];
            }
            for (size_t s = 0; s < slots; ++s) {
                rwp[b * slots + s] /= sum;
            }
        }

        Tensor read_content({batch, config_.d_model}, QuantType::FP32);
        float* rcp = read_content.as_fp32();
        const float* mp = ntm_memory_.as_fp32();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t d = 0; d < config_.d_model; ++d) {
                float val = 0.0f;
                for (size_t s = 0; s < slots; ++s) {
                    val += rwp[b * slots + s] * mp[s * config_.d_model + d];
                }
                rcp[b * config_.d_model + d] = val;
            }
        }

        Tensor h({batch, config_.d_model}, QuantType::FP32);
        float* hp = h.as_fp32();
        const float* xp = x.as_fp32();
        for (size_t i = 0; i < batch * config_.d_model; ++i) {
            hp[i] = xp[i] + rcp[i];
        }

        Tensor erase = ntm_w_erase_->forward(h);
        Tensor write = ntm_w_write_->forward(h);
        float* ep = erase.as_fp32();
        float* wtp = write.as_fp32();
        float* mmp = ntm_memory_.as_fp32();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < slots; ++s) {
                float rw = rwp[b * slots + s];
                for (size_t d = 0; d < config_.d_model; ++d) {
                    float e = 1.0f / (1.0f + std::exp(-ep[b * config_.d_model + d]));
                    float w = std::tanh(wtp[b * config_.d_model + d]);
                    mmp[s * config_.d_model + d] = mmp[s * config_.d_model + d] * (1.0f - rw * e) + rw * w;
                }
            }
        }

        return h;
    }

    Tensor last_token_pool(const Tensor& x) {
        size_t seq_len = x.shape_[0];
        const float* xp = x.as_fp32();

        Tensor output({1, config_.d_model}, QuantType::FP32);
        float* out = output.as_fp32();
        memcpy(out, xp + (seq_len - 1) * config_.d_model, config_.d_model * sizeof(float));

        last_hidden_ = output.clone();
        return output;
    }

    Tensor mean_pool(const Tensor& x) {
        size_t seq_len = x.shape_[0];
        const float* xp = x.as_fp32();

        Tensor output({1, config_.d_model}, QuantType::FP32);
        float* out = output.as_fp32();
        float inv_n = 1.0f / static_cast<float>(seq_len);
        for (size_t d = 0; d < config_.d_model; ++d) {
            float sum = 0.0f;
            for (size_t i = 0; i < seq_len; ++i) {
                sum += xp[i * config_.d_model + d];
            }
            out[d] = sum * inv_n;
        }

        last_hidden_ = output.clone();
        return output;
    }

    Tensor pool(const Tensor& x) {
        if (config_.pooling == "mean") {
            return mean_pool(x);
        }
        return last_token_pool(x);
    }

    Tensor forward(const std::vector<size_t>& token_ids) {
        Tensor x = embed_lookup(token_ids);
        x = positional_encode(x, 0);
        for (auto& attn : attn_layers_) {
            x = attn->forward(x);
        }
        x = causal_window_gate(x);
        x = sae_sparse(x);
        x = ntm_memory_access(x);
        x = ln_->forward(x);

        Tensor pooled = pool(x);
        last_projected_ = w_proj_->forward(pooled);
        Tensor logits = w_out_->forward(last_projected_);
        return logits;
    }

    Tensor forward_step(size_t token_id, size_t pos) {
        std::vector<size_t> ids = {token_id};
        Tensor x = embed_lookup(ids);
        x = positional_encode(x, pos);
        for (auto& attn : attn_layers_) {
            x = attn->forward(x);
        }
        x = causal_window_gate(x);
        x = sae_sparse(x);
        x = ntm_memory_access(x);
        x = ln_->forward(x);

        last_hidden_ = x.clone();
        last_projected_ = w_proj_->forward(x);
        Tensor logits = w_out_->forward(last_projected_);
        return logits;
    }

    Tensor forward_for_training(const std::vector<size_t>& token_ids) {
        train_cache_.input_ids = token_ids;
        train_cache_.x_embed = embed_lookup(token_ids);
        train_cache_.x_pos = positional_encode(train_cache_.x_embed, 0);

        Tensor x = train_cache_.x_pos;
        train_cache_.attn_inputs.clear();
        train_cache_.attn_qkv.clear();
        train_cache_.attn_outputs.clear();
        train_cache_.attn_weights.clear();

        for (auto& attn : attn_layers_) {
            train_cache_.attn_inputs.push_back(x.clone());
            x = attn->forward(x);
            train_cache_.attn_outputs.push_back(x.clone());
        }

        train_cache_.x_gate_in = x.clone();
        x = causal_window_gate(x);
        train_cache_.x_after_gate = x.clone();

        train_cache_.x_sae_encoded = sae_w_encode_->forward(x);
        size_t n = train_cache_.x_sae_encoded.numel();
        size_t k = config_.sae_k;

#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && train_cache_.x_sae_encoded.is_on_gpu()) {
            train_cache_.x_sae_encoded.to_gpu();
            launch_sae_topk_mask(train_cache_.x_sae_encoded.as_gpu_fp32(), n, k,
                                 CudaContext::instance().stream());
            train_cache_.x_sae_encoded.gpu_dirty_ = true;
        } else
#endif
        {
            float* enc_data = train_cache_.x_sae_encoded.as_fp32();
            if (k < n) {
                std::vector<size_t> indices(n);
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                                  [&](size_t a, size_t b) { return std::abs(enc_data[a]) > std::abs(enc_data[b]); });
                std::vector<bool> keep(n, false);
                for (size_t i = 0; i < k; ++i) keep[indices[i]] = true;
                for (size_t i = 0; i < n; ++i) { if (!keep[i]) enc_data[i] = 0.0f; }
            }
        }

        x = sae_w_decode_->forward(train_cache_.x_sae_encoded);
        train_cache_.x_after_sae = x.clone();

        x = ntm_memory_access(x);
        train_cache_.x_after_ntm = x.clone();

        x = ln_->forward(x);
        train_cache_.x_after_ln = x.clone();

        train_cache_.x_pooled = pool(x);
        train_cache_.x_projected = w_proj_->forward(train_cache_.x_pooled);
        Tensor logits = w_out_->forward(train_cache_.x_projected);
        return logits;
    }

    struct LMGradients {
        std::vector<CausalSelfAttention::Gradients> attn_grads;
        Tensor w_proj_weight_grad;
        Tensor w_proj_bias_grad;
        Tensor w_out_weight_grad;
        Tensor w_out_bias_grad;
        Tensor embed_grad;
        std::vector<size_t> used_token_ids;
        Tensor ln_weight_grad;
        Tensor ln_bias_grad;
        Tensor ntm_read_weight_grad;
        Tensor ntm_write_weight_grad;
        Tensor ntm_erase_weight_grad;
        Tensor sae_encode_weight_grad;
        Tensor sae_decode_weight_grad;
        Tensor dw_kernel_grad;
        Tensor pw_conv_weight_grad;
        Tensor pw_conv_bias_grad;
    };

    LMGradients backward_from_logits(const Tensor& logits_grad) {
        LMGradients grads;
        size_t d_model = config_.d_model;
        size_t seq_len = train_cache_.x_after_ln.shape_[0];

        Tensor proj_grad = lm_head_linear_backward_input(logits_grad, w_out_->weight);

        if (!config_.weight_tying) {
            grads.w_out_weight_grad = lm_head_linear_backward_weight(train_cache_.x_projected, logits_grad);
        }
        grads.w_out_bias_grad = lm_head_bias_backward(logits_grad);

        Tensor pooled_grad = lm_head_linear_backward_input(proj_grad, w_proj_->weight);
        grads.w_proj_weight_grad = lm_head_linear_backward_weight(train_cache_.x_pooled, proj_grad);
        grads.w_proj_bias_grad = lm_head_bias_backward(proj_grad);


        Tensor x_grad;
        if (config_.pooling == "mean") {
            x_grad = Tensor({seq_len, d_model}, QuantType::FP32);
            float* xg = x_grad.as_fp32();
            const float* pg = pooled_grad.as_fp32();
            float inv_n = 1.0f / static_cast<float>(seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < d_model; ++d) {
                    xg[i * d_model + d] = pg[d] * inv_n;
                }
            }
        } else {
            x_grad = Tensor({seq_len, d_model}, QuantType::FP32);
            memset(x_grad.as_fp32(), 0, x_grad.data_size_);
            const float* pg = pooled_grad.as_fp32();
            float* xg = x_grad.as_fp32();
            memcpy(xg + (seq_len - 1) * d_model, pg, d_model * sizeof(float));
        }

        // LN backward
        Tensor ln_input_grad = ln_backward_impl(train_cache_.x_after_ntm, ln_->weight, x_grad);

        grads.ln_weight_grad = Tensor({d_model}, QuantType::FP32);
        grads.ln_bias_grad = Tensor({d_model}, QuantType::FP32);
        {
            const float* og = x_grad.as_fp32();
            const float* inp = train_cache_.x_after_ntm.as_fp32();
            float* lwg = grads.ln_weight_grad.as_fp32();
            float* lbg = grads.ln_bias_grad.as_fp32();
            memset(lwg, 0, d_model * sizeof(float));
            memset(lbg, 0, d_model * sizeof(float));
            for (size_t i = 0; i < seq_len; ++i) {
                float mean = 0.0f;
                for (size_t d = 0; d < d_model; ++d) mean += inp[i * d_model + d];
                mean /= d_model;
                float var = 0.0f;
                for (size_t d = 0; d < d_model; ++d) { float diff = inp[i * d_model + d] - mean; var += diff * diff; }
                var /= d_model;
                float inv_std = 1.0f / std::sqrt(var + 1e-5f);
                for (size_t d = 0; d < d_model; ++d) {
                    float norm = (inp[i * d_model + d] - mean) * inv_std;
                    lwg[d] += og[i * d_model + d] * norm / seq_len;
                    lbg[d] += og[i * d_model + d] / seq_len;
                }
            }
        }
        x_grad = ln_input_grad;

        // NTM backward: h = x_after_sae + read_content, so dx_after_sae = x_grad (skip NTM internal grads)
        grads.ntm_read_weight_grad = Tensor(ntm_w_read_->weight.shape_, QuantType::FP32);
        grads.ntm_write_weight_grad = Tensor(ntm_w_write_->weight.shape_, QuantType::FP32);
        grads.ntm_erase_weight_grad = Tensor(ntm_w_erase_->weight.shape_, QuantType::FP32);
        // x_grad already flows through since h = x + read_content

        // SAE backward
        grads.sae_decode_weight_grad = lm_head_linear_backward_weight(train_cache_.x_sae_encoded, x_grad);
        Tensor sae_dec_input_grad = lm_head_linear_backward_input(x_grad, sae_w_decode_->weight);
        {
            size_t n = sae_dec_input_grad.numel();
            size_t k = config_.sae_k;
#ifdef USE_CUDA
            if (CudaContext::instance().is_available() && sae_dec_input_grad.is_on_gpu()) {
                sae_dec_input_grad.to_gpu();
                train_cache_.x_sae_encoded.to_gpu();
                launch_sae_topk_mask_backward(sae_dec_input_grad.as_gpu_fp32(),
                                              train_cache_.x_sae_encoded.as_gpu_fp32(),
                                              n, k, CudaContext::instance().stream());
                sae_dec_input_grad.gpu_dirty_ = true;
            } else
#endif
            {
                float* enc_g = sae_dec_input_grad.as_fp32();
                const float* enc_data = train_cache_.x_sae_encoded.as_fp32();
                if (k < n) {
                    std::vector<size_t> indices(n);
                    std::iota(indices.begin(), indices.end(), 0);
                    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                        [&](size_t a, size_t b) { return std::abs(enc_data[a]) > std::abs(enc_data[b]); });
                    std::vector<bool> keep(n, false);
                    for (size_t i = 0; i < k; ++i) keep[indices[i]] = true;
                    for (size_t i = 0; i < n; ++i) { if (!keep[i]) enc_g[i] = 0.0f; }
                }
            }
        }
        grads.sae_encode_weight_grad = lm_head_linear_backward_weight(train_cache_.x_after_gate, sae_dec_input_grad);
        x_grad = lm_head_linear_backward_input(sae_dec_input_grad, sae_w_encode_->weight);


        // Causal gate backward (depthwise separable conv)
        grads.dw_kernel_grad = Tensor(dw_kernel_.shape_, QuantType::FP32);
        grads.pw_conv_weight_grad = Tensor(pw_conv_->weight.shape_, QuantType::FP32);
        grads.pw_conv_bias_grad = Tensor(pw_conv_->bias.shape_, QuantType::FP32);
        {
            size_t C = d_model;
            size_t K = config_.causal_window_size;
            const float* xp = train_cache_.x_gate_in.as_fp32();
            const float* gp = train_cache_.x_after_gate.as_fp32();
            const float* xg = x_grad.as_fp32();
            float* dkg = grads.dw_kernel_grad.as_fp32();
            memset(dkg, 0, grads.dw_kernel_grad.data_size_);

            Tensor gate_grad({seq_len, C}, QuantType::FP32);
#ifdef USE_CUDA
            if (CudaContext::instance().is_available() && x_grad.is_on_gpu()) {
                gate_grad.to_gpu();
                launch_sigmoid_backward(gate_grad.as_gpu_fp32(),
                                        train_cache_.x_after_gate.as_gpu_fp32(),
                                        train_cache_.x_gate_in.as_gpu_fp32(),
                                        x_grad.as_gpu_fp32(),
                                        seq_len * C, CudaContext::instance().stream());
                gate_grad.gpu_dirty_ = true;
            } else
#endif
            {
                float* gg = gate_grad.as_fp32();
                const float* xg = x_grad.as_fp32();
                const float* gp = train_cache_.x_after_gate.as_fp32();
                const float* xp = train_cache_.x_gate_in.as_fp32();
                for (size_t i = 0; i < seq_len * C; ++i) {
                    float g = 1.0f / (1.0f + std::exp(-gp[i]));
                    gg[i] = xg[i] * g * (1.0f - g) * xp[i] + xg[i] * g;
                }
            }

            Tensor pw_grad_tensor = lm_head_linear_backward_input(gate_grad, pw_conv_->weight);
            const float* pw_grad = pw_grad_tensor.as_fp32();
            grads.pw_conv_weight_grad = lm_head_linear_backward_weight(train_cache_.x_gate_in, gate_grad);
            grads.pw_conv_bias_grad = lm_head_bias_backward(gate_grad);

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t k = 0; k < K; ++k) {
                        if (i >= k) {
                            dkg[c * K + k] += pw_grad[i * C + c] * xp[(i - k) * C + c];
                        }
                    }
                }
            }

            Tensor gate_input_grad({seq_len, C}, QuantType::FP32);
            float* gig = gate_input_grad.as_fp32();
            memset(gig, 0, gate_input_grad.data_size_);
            const float* dkp = dw_kernel_.as_fp32();
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t c = 0; c < C; ++c) {
                    float val = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        if (i + k < seq_len) {
                            val += pw_grad[(i + k) * C + c] * dkp[c * K + k];
                        }
                    }
                    gig[i * C + c] = val + gg[i * C + c];
                }
            }
            x_grad = gate_input_grad;
        }


        // Attention layers backward
        for (int i = static_cast<int>(attn_layers_.size()) - 1; i >= 0; --i) {
            auto attn_grads = attn_layers_[i]->backward(x_grad);
            grads.attn_grads.insert(grads.attn_grads.begin(), std::move(attn_grads));
            x_grad = attn_layers_[i]->cache_.input.shape_[0] == grads.attn_grads.front().input_grad.shape_[0]
                ? grads.attn_grads.front().input_grad.clone()
                : x_grad;
        }

        // Embed gradient: sparse update only for used token IDs
        grads.embed_grad = Tensor({seq_len, d_model}, QuantType::FP32);
        grads.used_token_ids = train_cache_.input_ids;

        {
            float* eg = grads.embed_grad.as_fp32();
            const float* xg = x_grad.as_fp32();
            float scale = std::sqrt(static_cast<float>(d_model));
            for (size_t i = 0; i < seq_len; ++i) {
                float norm_val = 0.0f;
                for (size_t d = 0; d < d_model; ++d) norm_val += xg[i * d_model + d] * xg[i * d_model + d];
                if (norm_val < 1e-20f) {
                    memset(eg + i * d_model, 0, d_model * sizeof(float));
                    continue;
                }
                for (size_t d = 0; d < d_model; ++d) {
                    eg[i * d_model + d] = xg[i * d_model + d] * scale / seq_len;
                }
            }
        }

        return grads;
    }

    void apply_lm_gradients(LMGradients& grads, float lr) {
        auto sgd_update = [&](Tensor& param, const Tensor& grad) {
            if (param.shape_ != grad.shape_ || param.numel() == 0 || grad.numel() == 0) return;
#ifdef USE_CUDA
            if (CudaContext::instance().is_available() && param.is_on_gpu()) {
                param.to_gpu();
                grad.to_gpu();
                launch_sgd_update(param.as_gpu_fp32(), grad.as_gpu_fp32(),
                                  param.numel(), lr, CudaContext::instance().stream());
                param.gpu_dirty_ = true;
                return;
            }
#endif
            float* p = param.as_fp32();
            const float* g = grad.as_fp32();
            size_t n = param.numel();
            for (size_t i = 0; i < n; ++i) {
                if (std::isfinite(g[i])) p[i] -= lr * g[i];
            }
        };

        sgd_update(w_proj_->weight, grads.w_proj_weight_grad);
        sgd_update(w_proj_->bias, grads.w_proj_bias_grad);
        if (!config_.weight_tying) {
            sgd_update(w_out_->weight, grads.w_out_weight_grad);
        }
        sgd_update(w_out_->bias, grads.w_out_bias_grad);

        sgd_update(ln_->weight, grads.ln_weight_grad);
        sgd_update(ln_->bias, grads.ln_bias_grad);

        sgd_update(ntm_w_read_->weight, grads.ntm_read_weight_grad);
        sgd_update(ntm_w_write_->weight, grads.ntm_write_weight_grad);
        sgd_update(ntm_w_erase_->weight, grads.ntm_erase_weight_grad);

        sgd_update(sae_w_encode_->weight, grads.sae_encode_weight_grad);
        sgd_update(sae_w_decode_->weight, grads.sae_decode_weight_grad);

        sgd_update(dw_kernel_, grads.dw_kernel_grad);
        sgd_update(pw_conv_->weight, grads.pw_conv_weight_grad);
        sgd_update(pw_conv_->bias, grads.pw_conv_bias_grad);

        // Sparse embed update: only update rows for used token IDs
        {
            const auto& ids = grads.used_token_ids;
            size_t seq_len = ids.size();
            size_t d = config_.d_model;
            size_t vocab_sz = config_.vocab_size;
#ifdef USE_CUDA
            if (CudaContext::instance().is_available() && w_embed_.is_on_gpu()) {
                w_embed_.to_gpu();
                grads.embed_grad.to_gpu();
                std::vector<int> int_ids(ids.size());
                for (size_t i = 0; i < ids.size(); ++i) {
                    int_ids[i] = static_cast<int>(ids[i] >= vocab_sz ? 1 : ids[i]);
                }
                void* d_ids = CudaContext::instance().alloc(int_ids.size() * sizeof(int));
                CudaContext::instance().copy_h2d(d_ids, int_ids.data(), int_ids.size() * sizeof(int));
                launch_sparse_embed_update(w_embed_.as_gpu_fp32(), grads.embed_grad.as_gpu_fp32(),
                                           static_cast<const int*>(d_ids), static_cast<int>(seq_len),
                                           static_cast<int>(d), lr, CudaContext::instance().stream());
                CudaContext::instance().free(d_ids);
                w_embed_.gpu_dirty_ = true;
            } else
#endif
            {
                float* embed_data = w_embed_.as_fp32();
                const float* eg = grads.embed_grad.as_fp32();
                for (size_t i = 0; i < seq_len; ++i) {
                    size_t tid = ids[i];
                    if (tid >= vocab_sz) continue;
                    float* row = embed_data + tid * d;
                    const float* grad_row = eg + i * d;
                    for (size_t j = 0; j < d; ++j) {
                        if (std::isfinite(grad_row[j])) row[j] -= lr * grad_row[j];
                    }
                }
            }
        }

        for (size_t i = 0; i < attn_layers_.size() && i < grads.attn_grads.size(); ++i) {
            auto& ag = grads.attn_grads[i];
            sgd_update(attn_layers_[i]->w_qkv->weight, ag.w_qkv_weight_grad);
            sgd_update(attn_layers_[i]->w_qkv->bias, ag.w_qkv_bias_grad);
            sgd_update(attn_layers_[i]->w_out->weight, ag.w_out_weight_grad);
            sgd_update(attn_layers_[i]->w_out->bias, ag.w_out_bias_grad);
        }
    }

private:
    Tensor ln_backward_impl(const Tensor& input, const Tensor& weight, const Tensor& output_grad, float eps = 1e-5f) {
        size_t seq_len = input.shape_[0];
        size_t dim = input.shape_[1];
        Tensor input_grad({seq_len, dim}, QuantType::FP32);
        const float* inp = input.as_fp32();
        const float* w = weight.as_fp32();
        const float* og = output_grad.as_fp32();
        float* ig = input_grad.as_fp32();
        for (size_t i = 0; i < seq_len; ++i) {
            float mean = 0.0f;
            for (size_t d = 0; d < dim; ++d) mean += inp[i * dim + d];
            mean /= dim;
            float var = 0.0f;
            for (size_t d = 0; d < dim; ++d) { float diff = inp[i * dim + d] - mean; var += diff * diff; }
            var /= dim;
            float inv_std = 1.0f / std::sqrt(var + eps);
            float sum_gn = 0.0f, sum_gnx = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float norm = (inp[i * dim + d] - mean) * inv_std;
                float gn = og[i * dim + d] * w[d];
                sum_gn += gn; sum_gnx += gn * norm;
            }
            for (size_t d = 0; d < dim; ++d) {
                float norm = (inp[i * dim + d] - mean) * inv_std;
                float gn = og[i * dim + d] * w[d];
                ig[i * dim + d] = inv_std * (gn - sum_gn / dim - norm * sum_gnx / dim);
            }
        }
        return input_grad;
    }

    Tensor lm_head_linear_backward_input(const Tensor& output_grad, const Tensor& weight) {
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
        const float* og = output_grad.as_fp32();
        const float* w = weight.as_fp32();
        float* ig = input_grad.as_fp32();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < in_f; ++j) {
                float sum = 0.0f;
                for (size_t i = 0; i < out_f; ++i) sum += og[b * out_f + i] * w[i * in_f + j];
                ig[b * in_f + j] = sum;
            }
        }
        return input_grad;
    }

    Tensor lm_head_linear_backward_weight(const Tensor& input, const Tensor& output_grad) {
        size_t batch = input.shape_[0];
        size_t in_f = input.shape_[1];
        size_t out_f = output_grad.shape_[1];
        Tensor weight_grad({out_f, in_f}, QuantType::FP32);
#ifdef USE_CUDA
        if (CudaContext::instance().is_available() && input.is_on_gpu()) {
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
        const float* inp = input.as_fp32();
        const float* og = output_grad.as_fp32();
        float* wg = weight_grad.as_fp32();
        for (size_t i = 0; i < out_f; ++i) {
            for (size_t j = 0; j < in_f; ++j) {
                float sum = 0.0f;
                for (size_t b = 0; b < batch; ++b) sum += og[b * out_f + i] * inp[b * in_f + j];
                wg[i * in_f + j] = sum / batch;
            }
        }
        return weight_grad;
    }

    Tensor lm_head_bias_backward(const Tensor& output_grad) {
        size_t batch = output_grad.shape_[0];
        size_t dim = output_grad.shape_[1];
        Tensor grad({dim}, QuantType::FP32);
        const float* og = output_grad.as_fp32();
        float* g = grad.as_fp32();
        for (size_t j = 0; j < dim; ++j) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch; ++b) sum += og[b * dim + j];
            g[j] = sum / batch;
        }
        return grad;
    }

public:
    void clear_cache() {
        if (kv_cache_) {
            kv_cache_->clear_cache();
        }
        memset(ntm_memory_.as_fp32(), 0, ntm_memory_.data_size_);
        sliding_window_drops_ = 0;
    }

    CacheStats cache_stats() const {
        CacheStats stats;
        if (kv_cache_) {
            stats.cache_len = kv_cache_->cache_len;
            stats.memory_bytes = kv_cache_->cache_size_bytes();
            stats.saving_ratio = kv_cache_->memory_saving_ratio();
        }
        stats.sliding_window_drops = sliding_window_drops_;
        return stats;
    }
};

class GenerativeModel {
public:
    std::unique_ptr<CausalLMHead> lm_head_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<SamplingStrategy> sampler_;
    NeuroFlowModel* neuroflow_model_;

    GenerativeModel(const CausalLMConfig& lm_config,
                    std::unique_ptr<Tokenizer> tokenizer,
                    NeuroFlowModel* nf_model = nullptr)
        : tokenizer_(std::move(tokenizer)), neuroflow_model_(nf_model) {
        lm_head_ = std::make_unique<CausalLMHead>(lm_config);
        sampler_ = std::make_unique<TopKSampling>();
    }

    GenerateOutput generate(const std::string& prompt, const GenerateConfig& config) {
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

    Tensor apply_sn_gating(const Tensor& hidden, const Tensor& logits) {
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

    Tensor inject_memory(const Tensor& query, const Tensor& logits) {
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

    Tensor apply_repetition_penalty(Tensor logits, const GenerateConfig& config,
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

    Tensor apply_punct_penalty(Tensor logits, const GenerateConfig& config) {
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

    void set_strategy(SamplingStrategyType type) {
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

    void clear_cache() { lm_head_->clear_cache(); }
    CacheStats cache_stats() const { return lm_head_->cache_stats(); }
};

} // namespace neuroflow

#endif // NEUROFLOW_GENERATIVE_HPP
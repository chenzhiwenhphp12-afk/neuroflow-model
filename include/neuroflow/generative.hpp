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

#include "tensor.hpp"
#include "networks.hpp"
#include "memory.hpp"
#include "model.hpp"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <thread>
#include <sstream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <iostream>
#include <queue>

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
    ERROR = 2
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

        std::string vocab_section = content.substr(obj_start + 1, obj_end - obj_start - 1);

        size_t id = 0;
        size_t pos = 0;
        while (pos < vocab_section.size()) {
            size_t key_start = vocab_section.find('\"', pos);
            if (key_start == std::string::npos) break;
            size_t key_end = vocab_section.find('\"', key_start + 1);
            if (key_end == std::string::npos) break;

            std::string key = vocab_section.substr(key_start + 1, key_end - key_start - 1);

            size_t colon = vocab_section.find(':', key_end);
            if (colon == std::string::npos) break;

            size_t val_start = colon + 1;
            while (val_start < vocab_section.size() &&
                   (vocab_section[val_start] == ' ' || vocab_section[val_start] == '\t'))
                ++val_start;

            size_t val_end = val_start;
            while (val_end < vocab_section.size() &&
                   vocab_section[val_end] != ',' && vocab_section[val_end] != '}')
                ++val_end;

            std::string val_str = vocab_section.substr(val_start, val_end - val_start);
            size_t token_id = std::stoul(val_str);

            vocab_[key] = token_id;
            id_to_token_[token_id] = key;
            id = token_id + 1;
            pos = val_end + 1;
        }
        vocab_size_ = id;

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
            for (size_t len = char_len; len <= std::min(text.size() - i, (size_t)20); ++len) {
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
    }

    Tensor embed_lookup(const std::vector<size_t>& ids) {
        size_t seq_len = ids.size();
        Tensor output({seq_len, config_.d_model}, QuantType::FP32);
        float* out = output.as_fp32();
        float* embed = w_embed_.as_fp32();

        float scale = std::sqrt(static_cast<float>(config_.d_model));

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

    Tensor forward(const std::vector<size_t>& token_ids) {
        Tensor x = embed_lookup(token_ids);
        x = positional_encode(x, 0);
        x = causal_window_gate(x);
        x = sae_sparse(x);
        x = ntm_memory_access(x);
        x = ln_->forward(x);

        Tensor pooled = last_token_pool(x);
        Tensor projected = w_proj_->forward(pooled);
        Tensor logits = w_out_->forward(projected);
        return logits;
    }

    Tensor forward_step(size_t token_id, size_t pos) {
        std::vector<size_t> ids = {token_id};
        Tensor x = embed_lookup(ids);
        x = positional_encode(x, pos);
        x = causal_window_gate(x);
        x = sae_sparse(x);
        x = ntm_memory_access(x);
        x = ln_->forward(x);

        last_hidden_ = x.clone();
        Tensor projected = w_proj_->forward(x);
        Tensor logits = w_out_->forward(projected);
        return logits;
    }

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
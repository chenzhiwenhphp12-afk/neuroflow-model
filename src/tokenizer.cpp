#include "neuroflow/tokenizer.hpp"
#include <cstdint>
#include <stdexcept>

namespace neuroflow {

// ═══════════════════════════════════════════════════════════
// BPETokenizer
// ═══════════════════════════════════════════════════════════

BPETokenizer::BPETokenizer(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f) throw std::runtime_error("Cannot open tokenizer config: " + config_path);
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    f.close();
    parse_config(content);
}

BPETokenizer::BPETokenizer() {}

std::vector<size_t> BPETokenizer::encode(const std::string& text, size_t max_len) {
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

std::string BPETokenizer::decode(const std::vector<size_t>& ids) {
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

void BPETokenizer::add_vocab(const std::string& token, size_t id) {
    vocab_[token] = id;
    id_to_token_[id] = token;
}

void BPETokenizer::set_vocab_size(size_t vs) { vocab_size_ = vs; }

std::vector<uint8_t> BPETokenizer::utf8_to_bytes(const std::string& text) {
    std::vector<uint8_t> bytes;
    for (unsigned char c : text) bytes.push_back(c);
    return bytes;
}

std::string BPETokenizer::bytes_to_utf8(const std::string& raw) {
    return raw;
}

std::string BPETokenizer::apply_bpe(const std::string& token) {
    if (token.size() <= 1 || merge_ranks_.empty()) return token;

    struct PairEntry {
        size_t rank;
        size_t pos;
        bool operator>(const PairEntry& o) const {
            return rank > o.rank;
        }
    };
    std::priority_queue<PairEntry, std::vector<PairEntry>,
    std::greater<PairEntry>> pq;

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

    auto push_pair = [&](size_t i) {
        size_t j = next_link[i];
        if (j == END) return;
        auto it = merge_ranks_.find(symbols[i] + symbols[j]);
        if (it != merge_ranks_.end()) {
            pq.push({it->second, i});
        }
    };

    for (size_t i = HEAD; i != END; i = next_link[i]) {
        push_pair(i);
    }

    while (!pq.empty()) {
        PairEntry top = pq.top();
        pq.pop();

        size_t i = top.pos;
        size_t j = next_link[i];
        if (j == END) continue;

        auto it = merge_ranks_.find(symbols[i] + symbols[j]);
        if (it == merge_ranks_.end() || it->second != top.rank) continue;

        size_t k = next_link[j];
        symbols[i] += symbols[j];
        next_link[i] = k;
        if (k != END) prev_link[k] = i;

        if (prev_link[i] != END) push_pair(prev_link[i]);
        push_pair(i);
    }

    std::string result;
    for (size_t i = HEAD; i != END; i = next_link[i]) {
        result += symbols[i];
    }
    return result;
}

void BPETokenizer::parse_config(const std::string& content) {
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
            size_t q1 = content.find('"', pos);
            if (q1 == std::string::npos) break;
            size_t q2 = content.find('"', q1 + 1);
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

// ═══════════════════════════════════════════════════════════
// WordPieceTokenizer
// ═══════════════════════════════════════════════════════════

WordPieceTokenizer::WordPieceTokenizer() {}

WordPieceTokenizer::WordPieceTokenizer(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f) throw std::runtime_error("Cannot open tokenizer config: " + config_path);
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    f.close();
    parse_vocab(content);
}

std::vector<size_t> WordPieceTokenizer::encode(const std::string& text, size_t max_len) {
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

std::string WordPieceTokenizer::decode(const std::vector<size_t>& ids) {
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

void WordPieceTokenizer::add_vocab(const std::string& token, size_t id) {
    vocab_[token] = id;
    id_to_token_[id] = token;
}

void WordPieceTokenizer::set_vocab_size(size_t vs) { vocab_size_ = vs; }

void WordPieceTokenizer::parse_vocab(const std::string& content) {
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

} // namespace neuroflow
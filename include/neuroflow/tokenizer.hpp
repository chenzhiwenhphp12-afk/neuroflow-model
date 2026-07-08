#ifndef NEUROFLOW_TOKENIZER_HPP
#define NEUROFLOW_TOKENIZER_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuroflow {

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
    BPETokenizer(const std::string& config_path);
    BPETokenizer();

    std::vector<size_t> encode(const std::string& text, size_t max_len = 0) override;
    std::string decode(const std::vector<size_t>& ids) override;

    void add_vocab(const std::string& token, size_t id);
    void set_vocab_size(size_t vs);

private:
    std::unordered_map<std::string, size_t> vocab_;
    std::unordered_map<size_t, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, size_t> merge_ranks_;

    std::vector<uint8_t> utf8_to_bytes(const std::string& text);
    std::string bytes_to_utf8(const std::string& raw);
    std::string apply_bpe(const std::string& token);
    void parse_config(const std::string& content);
};

class WordPieceTokenizer : public Tokenizer {
public:
    WordPieceTokenizer();
    WordPieceTokenizer(const std::string& config_path);

    std::vector<size_t> encode(const std::string& text, size_t max_len = 0) override;
    std::string decode(const std::vector<size_t>& ids) override;

    void add_vocab(const std::string& token, size_t id);
    void set_vocab_size(size_t vs);

private:
    std::unordered_map<std::string, size_t> vocab_;
    std::unordered_map<size_t, std::string> id_to_token_;

    void parse_vocab(const std::string& content);
};

} // namespace neuroflow

#endif // NEUROFLOW_TOKENIZER_HPP
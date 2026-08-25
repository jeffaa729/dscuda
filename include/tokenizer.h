#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dscuda {

// Loads the tokenizer produced by tools/train_tokenizer.py and applies the
// same byte-run BPE rules for prompts and generated token decoding.
class Tokenizer {
public:
    explicit Tokenizer(const std::string& path);

    std::vector<int> encode(
        std::string_view text,
        bool add_bos = false,
        bool add_eos = false) const;
    std::string decode(
        const std::vector<int>& tokens,
        bool skip_special = true) const;

    int vocabulary_size() const { return vocabulary_size_; }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

private:
    struct MergeRule {
        int rank;
        int result;
    };

    static std::uint64_t pair_key(int left, int right);
    std::vector<int> encode_piece(std::string_view piece) const;

    int vocabulary_size_ = 0;
    int bos_id_ = 0;
    int eos_id_ = 0;
    std::unordered_map<std::uint64_t, MergeRule> merge_rules_;
    std::vector<std::string> token_bytes_;
};

}  // namespace dscuda

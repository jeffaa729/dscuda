#include "tokenizer.h"

#include <array>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace dscuda {
namespace {

constexpr std::array<char, 8> kMagic = {'D', 'S', 'B', 'P', 'E', '0', '1', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr int kByteVocabularySize = 256;
constexpr int kFirstMergeId = 258;

std::uint32_t read_u32(std::ifstream& file) {
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!file) {
        throw std::runtime_error("tokenizer file is truncated");
    }
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

int byte_class(unsigned char value) {
    if ((value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z')) {
        return 0;
    }
    if (value >= '0' && value <= '9') {
        return 1;
    }
    if (value == '\t' || value == '\n' || value == '\v'
        || value == '\f' || value == '\r' || value == ' ') {
        return 2;
    }
    return 3;
}

}  // namespace

std::uint64_t Tokenizer::pair_key(int left, int right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32)
        | static_cast<std::uint32_t>(right);
}

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open tokenizer: " + path);
    }

    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());
    if (!file || magic != kMagic) {
        throw std::runtime_error("unsupported tokenizer magic");
    }

    const std::uint32_t version = read_u32(file);
    vocabulary_size_ = static_cast<int>(read_u32(file));
    const int merge_count = static_cast<int>(read_u32(file));
    bos_id_ = static_cast<int>(read_u32(file));
    eos_id_ = static_cast<int>(read_u32(file));
    read_u32(file);  // Reserved.

    if (version != kVersion
        || vocabulary_size_ != kFirstMergeId + merge_count
        || bos_id_ != 256
        || eos_id_ != 257) {
        throw std::runtime_error("unsupported tokenizer layout");
    }

    token_bytes_.reserve(vocabulary_size_);
    for (int value = 0; value < kByteVocabularySize; ++value) {
        token_bytes_.emplace_back(1, static_cast<char>(value));
    }
    token_bytes_.emplace_back();
    token_bytes_.emplace_back();

    for (int rank = 0; rank < merge_count; ++rank) {
        const int left = static_cast<int>(read_u32(file));
        const int right = static_cast<int>(read_u32(file));
        const int result = static_cast<int>(read_u32(file));
        if (left >= result || right >= result || result != kFirstMergeId + rank) {
            throw std::runtime_error("invalid tokenizer merge table");
        }
        merge_rules_[pair_key(left, right)] = {rank, result};
        token_bytes_.push_back(token_bytes_[left] + token_bytes_[right]);
    }
}

std::vector<int> Tokenizer::encode_piece(std::string_view piece) const {
    if (piece.empty()) {
        return {};
    }

    std::vector<int> tokens(piece.size());
    std::vector<int> previous(piece.size());
    std::vector<int> following(piece.size());
    std::vector<bool> alive(piece.size(), true);
    for (int index = 0; index < static_cast<int>(piece.size()); ++index) {
        tokens[index] = static_cast<unsigned char>(piece[index]);
        previous[index] = index - 1;
        following[index] = index + 1;
    }
    following.back() = -1;

    using Candidate = std::pair<int, int>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;

    auto push_pair = [&](int left_index) {
        if (left_index < 0 || !alive[left_index]) {
            return;
        }
        const int right_index = following[left_index];
        if (right_index < 0) {
            return;
        }
        const auto rule = merge_rules_.find(
            pair_key(tokens[left_index], tokens[right_index]));
        if (rule != merge_rules_.end()) {
            queue.emplace(rule->second.rank, left_index);
        }
    };

    for (int index = 0; index + 1 < static_cast<int>(tokens.size()); ++index) {
        push_pair(index);
    }

    while (!queue.empty()) {
        const auto [rank, left_index] = queue.top();
        queue.pop();
        if (!alive[left_index]) {
            continue;
        }
        const int right_index = following[left_index];
        if (right_index < 0 || !alive[right_index]) {
            continue;
        }

        const auto rule = merge_rules_.find(
            pair_key(tokens[left_index], tokens[right_index]));
        if (rule == merge_rules_.end() || rule->second.rank != rank) {
            continue;
        }

        tokens[left_index] = rule->second.result;
        alive[right_index] = false;
        following[left_index] = following[right_index];
        if (following[right_index] >= 0) {
            previous[following[right_index]] = left_index;
        }
        push_pair(previous[left_index]);
        push_pair(left_index);
    }

    std::vector<int> encoded;
    encoded.reserve(piece.size());
    for (int index = 0; index < static_cast<int>(tokens.size()); ++index) {
        if (alive[index]) {
            encoded.push_back(tokens[index]);
        }
    }
    return encoded;
}

std::vector<int> Tokenizer::encode(
    std::string_view text,
    bool add_bos,
    bool add_eos) const {
    std::vector<int> encoded;
    if (add_bos) {
        encoded.push_back(bos_id_);
    }

    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = start + 1;
        const int current_class = byte_class(
            static_cast<unsigned char>(text[start]));
        while (end < text.size()
               && byte_class(static_cast<unsigned char>(text[end]))
                   == current_class) {
            ++end;
        }
        const std::vector<int> piece = encode_piece(text.substr(start, end - start));
        encoded.insert(encoded.end(), piece.begin(), piece.end());
        start = end;
    }

    if (add_eos) {
        encoded.push_back(eos_id_);
    }
    return encoded;
}

std::string Tokenizer::decode(
    const std::vector<int>& tokens,
    bool skip_special) const {
    std::string text;
    for (const int token : tokens) {
        if (token == bos_id_ || token == eos_id_) {
            if (!skip_special) {
                text += token == bos_id_ ? "<bos>" : "<eos>";
            }
            continue;
        }
        if (token < 0 || token >= static_cast<int>(token_bytes_.size())) {
            throw std::runtime_error("token ID is outside the vocabulary");
        }
        text += token_bytes_[token];
    }
    return text;
}

}  // namespace dscuda

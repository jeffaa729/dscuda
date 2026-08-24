#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dscuda {

// Owns a pretokenized little-endian uint16 stream and creates contiguous
// next-token-prediction batches ready to copy into the embedding input.
class TokenDataset {
public:
    explicit TokenDataset(const std::string& path);

    void get_batch(
        std::size_t start_token,
        int batch_size,
        int sequence_length,
        std::vector<int>& inputs,
        std::vector<int>& targets) const;

    std::size_t token_count() const { return tokens_.size(); }
    std::size_t batch_count(int batch_size, int sequence_length) const;

private:
    std::vector<std::uint16_t> tokens_;
};

}  // namespace dscuda

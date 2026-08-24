#include "dataset.h"

#include <fstream>
#include <stdexcept>

namespace dscuda {

TokenDataset::TokenDataset(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open token dataset: " + path);
    }
    const std::streamsize bytes = file.tellg();
    if (bytes < static_cast<std::streamsize>(2 * sizeof(std::uint16_t))
        || bytes % sizeof(std::uint16_t) != 0) {
        throw std::runtime_error("token dataset has an invalid size");
    }

    tokens_.resize(static_cast<std::size_t>(bytes) / sizeof(std::uint16_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(tokens_.data()), bytes);
    if (!file) {
        throw std::runtime_error("failed to read token dataset");
    }
}

void TokenDataset::get_batch(
    std::size_t start_token,
    int batch_size,
    int sequence_length,
    std::vector<int>& inputs,
    std::vector<int>& targets) const {
    const std::size_t elements =
        static_cast<std::size_t>(batch_size) * sequence_length;
    if (batch_size <= 0 || sequence_length <= 0
        || start_token + elements >= tokens_.size()) {
        throw std::runtime_error("requested batch is outside the token dataset");
    }

    inputs.resize(elements);
    targets.resize(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        inputs[index] = tokens_[start_token + index];
        targets[index] = tokens_[start_token + index + 1];
    }
}

std::size_t TokenDataset::batch_count(
    int batch_size,
    int sequence_length) const {
    if (batch_size <= 0 || sequence_length <= 0) {
        return 0;
    }
    const std::size_t elements =
        static_cast<std::size_t>(batch_size) * sequence_length;
    return (tokens_.size() - 1) / elements;
}

}  // namespace dscuda

#pragma once

#include <cstdint>
#include <vector>

namespace dscuda {

struct GenerationConfig {
    int maximum_new_tokens = 64;
    float temperature = 0.8F;
    int top_k = 40;
    std::uint64_t seed = 1337;
};

// Model backends expose only the final-position logits required by the shared
// sampler. Dense GPT, MLA, and compressed-attention models can implement this
// interface with different cache strategies without changing generation.
class AutoregressiveModel {
public:
    virtual ~AutoregressiveModel() = default;
    virtual int vocabulary_size() const = 0;
    virtual int maximum_context_length() const = 0;
    virtual std::vector<float> forward_last_logits(
        const std::vector<int>& tokens) = 0;
};

int sample_token(
    const std::vector<float>& logits,
    float temperature,
    int top_k,
    std::uint64_t& random_state);

std::vector<int> generate_tokens(
    AutoregressiveModel& model,
    const std::vector<int>& prompt_tokens,
    int eos_token,
    const GenerationConfig& config);

}  // namespace dscuda

// Implements model-independent autoregressive generation with stable temperature scaling and top-k multinomial sampling.
// The model interface supplies final-position logits, leaving dense KV, MLA, and future compressed caches to their backends.

#include "generation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace dscuda {
namespace {

std::uint32_t random_u32(std::uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return static_cast<std::uint32_t>(
        (state * 0x2545F4914F6CDD1DULL) >> 32);
}

float random_f32(std::uint64_t& state) {
    return static_cast<float>(random_u32(state) >> 8) / 16777216.0F;
}

}  // namespace

int sample_token(
    const std::vector<float>& logits,
    float temperature,
    int top_k,
    std::uint64_t& random_state) {
    if (logits.empty()) {
        throw std::runtime_error("cannot sample from empty logits");
    }
    if (temperature <= 0.0F) {
        return static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<int> candidates(logits.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    if (top_k > 0 && top_k < static_cast<int>(candidates.size())) {
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + top_k,
            candidates.end(),
            [&](int left, int right) {
                return logits[left] > logits[right];
            });
        candidates.resize(top_k);
    }

    float maximum = logits[candidates[0]] / temperature;
    for (const int token : candidates) {
        maximum = std::max(maximum, logits[token] / temperature);
    }
    std::vector<float> weights(candidates.size());
    double total = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        weights[index] = std::exp(
            logits[candidates[index]] / temperature - maximum);
        total += weights[index];
    }

    const double coin = random_f32(random_state) * total;
    double cumulative = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        cumulative += weights[index];
        if (coin < cumulative) {
            return candidates[index];
        }
    }
    return candidates.back();
}

std::vector<int> generate_tokens(
    AutoregressiveModel& model,
    const std::vector<int>& prompt_tokens,
    int eos_token,
    const GenerationConfig& config) {
    if (prompt_tokens.empty()) {
        throw std::runtime_error("generation requires at least one prompt token");
    }
    std::vector<int> tokens = prompt_tokens;
    std::uint64_t random_state = config.seed;
    for (int generated = 0;
         generated < config.maximum_new_tokens
         && static_cast<int>(tokens.size()) < model.maximum_context_length();
         ++generated) {
        const std::vector<float> logits = model.forward_last_logits(tokens);
        if (static_cast<int>(logits.size()) != model.vocabulary_size()) {
            throw std::runtime_error("model returned the wrong logits size");
        }
        const int next = sample_token(
            logits, config.temperature, config.top_k, random_state);
        tokens.push_back(next);
        if (next == eos_token) {
            break;
        }
    }
    return tokens;
}

}  // namespace dscuda

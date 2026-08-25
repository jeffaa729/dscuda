// Verifies the reusable generation loop, greedy termination, deterministic sampling, and top-k candidate restriction.
// A synthetic model keeps this test independent of any particular transformer architecture or CUDA backend.

#include "generation.h"

#include <cstdio>
#include <vector>

namespace {

class ScriptedModel : public dscuda::AutoregressiveModel {
public:
    int vocabulary_size() const override { return 4; }
    int maximum_context_length() const override { return 8; }

    std::vector<float> forward_last_logits(
        const std::vector<int>& tokens) override {
        std::vector<float> logits(4, -10.0F);
        const int next = tokens.size() < 3
            ? static_cast<int>(tokens.size())
            : 3;
        logits[next] = 10.0F;
        return logits;
    }
};

}  // namespace

int main() {
    ScriptedModel model;
    dscuda::GenerationConfig config;
    config.maximum_new_tokens = 6;
    config.temperature = 0.0F;
    const std::vector<int> generated =
        dscuda::generate_tokens(model, {0}, 3, config);
    bool passed = generated == std::vector<int>({0, 1, 2, 3});

    std::uint64_t random_state = 17;
    const int top_one = dscuda::sample_token(
        {0.0F, 4.0F, 3.0F, 2.0F}, 1.0F, 1, random_state);
    passed &= top_one == 1;
    std::printf("Generation test: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

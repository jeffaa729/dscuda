// Exercises a two-layer DeepSeek-V3 language model with sequential MTP through loss, full backward, AdamW, routing-bias balancing, and sampler logits.
// Lower-level CPU comparisons cover exact mathematics; this test verifies that main and MTP modules share embedding/head gradients and checkpointable runtime state correctly.

#include "deepseek_v3_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const dscuda::DeepSeekV3Config config{
        1, 4, 32, 2, 32, 2, 32, 32, 16, 8, 16,
        16, 4, 1, 2, 1.0e-5F, 1.0F, 0.01F,
        1.0e-3F, 1, 0.1F, 1, 48};
    dscuda::DeepSeekV3Model model(config);
    model.initialize(1234);
    const std::vector<int> input = {1, 4, 7, 3};
    const std::vector<int> target = {4, 7, 3, 2};

    const float loss = model.forward(input, target);
    model.zero_gradients();
    model.backward();
    const std::vector<float> gradients = model.gradients_to_host();
    const bool finite_loss = std::isfinite(loss) && loss > 0.0F;
    const bool finite_gradients = std::all_of(
        gradients.begin(), gradients.end(),
        [](float value) { return std::isfinite(value); });
    const float maximum_gradient = *std::max_element(
        gradients.begin(), gradients.end(),
        [](float left, float right) {
            return std::abs(left) < std::abs(right);
        });
    const bool nonzero_gradient = std::abs(maximum_gradient) > 1.0e-8F;
    const auto& mtp = model.parameter_layout().mtp_modules.front();
    const bool mtp_gradient = std::any_of(
        gradients.begin() + mtp.hidden_norm,
        gradients.end(),
        [](float value) { return std::abs(value) > 1.0e-8F; });

    const std::vector<float> parameters_before = model.parameters_to_host();
    const std::vector<float> bias_before = model.routing_bias_to_host();
    const dscuda::AdamWConfig optimizer{
        3.0e-4F, 0.9F, 0.95F, 1.0e-8F, 0.0F};
    const dscuda::TrainStepResult step =
        model.train_step(input, target, 1, optimizer, 1.0F);
    const std::vector<float> parameters_after = model.parameters_to_host();
    const std::vector<float> bias_after = model.routing_bias_to_host();
    const bool parameters_changed = parameters_before != parameters_after;
    const bool bias_changed = bias_before != bias_after;
    const std::vector<float> generation_logits =
        model.forward_last_logits({1, 4});
    const bool generation_shape = generation_logits.size() == 32;
    const bool generation_finite = std::all_of(
        generation_logits.begin(), generation_logits.end(),
        [](float value) { return std::isfinite(value); });

    model.forward(input, target);
    const std::vector<float> full_logits = model.logits_to_host();
    float cached_decode_error = 0.0F;
    for (std::size_t position = 0; position < input.size(); ++position) {
        const std::vector<int> prefix(
            input.begin(), input.begin() + position + 1);
        const std::vector<float> cached_logits =
            model.forward_last_logits(prefix);
        for (int token = 0; token < config.vocabulary_size; ++token) {
            cached_decode_error = std::max(
                cached_decode_error,
                std::abs(
                    cached_logits[token]
                    - full_logits[position * config.vocabulary_size + token]));
        }
    }
    const bool cached_decode_matches = cached_decode_error < 2.0e-4F;
    const bool compressed_cache_size =
        model.kv_cache_bytes_per_token() == 160;

    dscuda::DeepSeekV3Config all_dense_config = config;
    all_dense_config.layers = 1;
    all_dense_config.dense_layers = 1;
    all_dense_config.mtp_depth = 0;
    all_dense_config.mtp_loss_weight = 0.0F;
    dscuda::DeepSeekV3Model all_dense_model(all_dense_config);
    all_dense_model.initialize(4321);
    const bool empty_routing_state =
        all_dense_model.routing_bias_to_host().empty()
        && std::isfinite(all_dense_model.forward(input, target));

    std::printf("DeepSeek-V3 model integration test\n");
    std::printf("  %-24s %s (%.6f)\n", "finite loss", finite_loss ? "PASS" : "FAIL", loss);
    std::printf("  %-24s %s\n", "finite gradients", finite_gradients ? "PASS" : "FAIL");
    std::printf("  %-24s %s\n", "nonzero gradients", nonzero_gradient ? "PASS" : "FAIL");
    std::printf("  %-24s %s\n", "MTP gradients", mtp_gradient ? "PASS" : "FAIL");
    std::printf("  %-24s %s\n", "parameters updated", parameters_changed ? "PASS" : "FAIL");
    std::printf("  %-24s %s\n", "routing bias updated", bias_changed ? "PASS" : "FAIL");
    std::printf("  %-24s %s (%.6f)\n", "train step finite", std::isfinite(step.loss) ? "PASS" : "FAIL", step.loss);
    std::printf("  %-24s %s\n", "generation logits", generation_shape && generation_finite ? "PASS" : "FAIL");
    std::printf(
        "  %-24s %s (%.3e)\n",
        "compressed cached decode",
        cached_decode_matches ? "PASS" : "FAIL",
        cached_decode_error);
    std::printf(
        "  %-24s %s (%zu bytes/token)\n",
        "compressed KV cache",
        compressed_cache_size ? "PASS" : "FAIL",
        model.kv_cache_bytes_per_token());
    std::printf(
        "  %-24s %s\n",
        "all-dense empty routing",
        empty_routing_state ? "PASS" : "FAIL");
    return finite_loss && finite_gradients && nonzero_gradient && mtp_gradient
            && parameters_changed && bias_changed && std::isfinite(step.loss)
            && generation_shape && generation_finite && cached_decode_matches
            && compressed_cache_size && empty_routing_state
        ? 0
        : 1;
}

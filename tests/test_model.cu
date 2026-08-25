// Compares a complete one-layer tied-embedding language model against the scalar CPU graph in forward and backward.
// This test covers repeated-model wiring, saved activation ownership, tied head gradients, and flattened parameter offsets.

#include "cuda_common.h"
#include "model.h"
#include "model_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr dscuda::ModelConfig kConfig{
    1, 8, 32, 1, 16, 2, 24, 8, 1.0e-5F};

float max_error(
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    float error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance) {
    const float error = max_error(expected, actual);
    const bool passed = error < tolerance;
    std::printf(
        "  %-24s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

void make_rope_tables(
    std::vector<float>& cosine,
    std::vector<float>& sine) {
    const int pairs = kConfig.rotary_size / 2;
    cosine.resize(kConfig.sequence_length * pairs);
    sine.resize(kConfig.sequence_length * pairs);
    for (int position = 0; position < kConfig.sequence_length; ++position) {
        for (int pair = 0; pair < pairs; ++pair) {
            const float inverse_frequency = std::pow(
                10000.0F,
                -2.0F * static_cast<float>(pair) / kConfig.rotary_size);
            const float angle = position * inverse_frequency;
            cosine[position * pairs + pair] = std::cos(angle);
            sine[position * pairs + pair] = std::sin(angle);
        }
    }
}

bool run_test() {
    const int rows = kConfig.batch_size * kConfig.sequence_length;
    std::vector<int> inputs(rows);
    std::vector<int> targets(rows);
    for (int index = 0; index < rows; ++index) {
        inputs[index] = (index * 7 + 3) % kConfig.vocabulary_size;
        targets[index] = (index * 11 + 5) % kConfig.vocabulary_size;
    }

    dscuda::DenseGptModel model(kConfig);
    model.initialize(2026);
    const std::vector<float> parameters = model.parameters_to_host();
    const dscuda::ModelParameterLayout layout = model.parameter_layout();

    std::vector<float> cosine;
    std::vector<float> sine;
    make_rope_tables(cosine, sine);
    dscuda::DenseGptCpuCache cpu_cache;
    const float cpu_loss = dscuda::dense_gpt_forward_cpu(
        cpu_cache,
        inputs.data(),
        targets.data(),
        parameters.data(),
        layout,
        cosine,
        sine,
        kConfig);
    std::vector<float> cpu_gradients(layout.elements, 0.0F);
    dscuda::dense_gpt_backward_cpu(
        cpu_gradients.data(),
        cpu_cache,
        inputs.data(),
        targets.data(),
        parameters.data(),
        layout,
        cosine,
        sine,
        kConfig);

    const float gpu_loss = model.forward(inputs, targets);
    const std::vector<float> gpu_logits = model.logits_to_host();
    model.zero_gradients();
    model.backward();
    const std::vector<float> gpu_gradients = model.gradients_to_host();

    bool passed = true;
    const float loss_error = std::abs(cpu_loss - gpu_loss);
    std::printf(
        "  %-24s error = %.3e  %s\n",
        "mean loss",
        loss_error,
        loss_error < 2.0e-5F ? "PASS" : "FAIL");
    passed &= loss_error < 2.0e-5F;
    passed &= check("logits", cpu_cache.logits, gpu_logits, 3.0e-4F);
    passed &= check(
        "all parameter gradients",
        cpu_gradients,
        gpu_gradients,
        8.0e-4F);

    const dscuda::ModelMemoryReport memory = model.memory_report();
    std::printf(
        "  parameters = %zu, allocated model memory = %.2f MiB\n",
        memory.parameter_elements,
        static_cast<double>(memory.total_bytes) / (1024.0 * 1024.0));
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Dense GPT model test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Dense GPT model test failed: %s\n", error.what());
        return 1;
    }
}

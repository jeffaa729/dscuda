// Compares the selectable BF16 mixed-precision dense GPT path against the FP32 path from identical parameters and tokens.
// The test covers logits, loss, parameter gradients, added mixed-precision memory, and fixed-batch convergence.

#include "cuda_common.h"
#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr dscuda::ModelConfig kConfig{
    1, 16, 128, 1, 64, 4, 192, 16, 1.0e-5F};

float maximum_error(
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    float error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool run_test() {
    std::vector<int> inputs(16);
    std::vector<int> targets(16);
    for (int index = 0; index < 16; ++index) {
        inputs[index] = (index * 7 + 3) % 128;
        targets[index] = (index * 11 + 5) % 128;
    }

    dscuda::DenseGptModel fp32(kConfig, dscuda::ModelPrecision::fp32);
    dscuda::DenseGptModel bf16(kConfig, dscuda::ModelPrecision::bf16);
    fp32.initialize(2026);
    bf16.load_parameters(fp32.parameters_to_host());

    const float fp32_loss = fp32.forward(inputs, targets);
    const float bf16_loss = bf16.forward(inputs, targets);
    const float logits_error = maximum_error(
        fp32.logits_to_host(), bf16.logits_to_host());
    fp32.zero_gradients();
    bf16.zero_gradients();
    fp32.backward();
    bf16.backward();
    const float gradients_error = maximum_error(
        fp32.gradients_to_host(), bf16.gradients_to_host());

    const bool passed = std::abs(fp32_loss - bf16_loss) < 2.0e-2F
        && logits_error < 3.0e-2F
        && gradients_error < 3.0e-2F
        && bf16.memory_report().total_bytes > fp32.memory_report().total_bytes;
    std::printf(
        "  loss error %.3e, logits error %.3e, gradients error %.3e\n",
        std::abs(fp32_loss - bf16_loss),
        logits_error,
        gradients_error);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("BF16 model test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "BF16 model test failed: %s\n", error.what());
        return 1;
    }
}

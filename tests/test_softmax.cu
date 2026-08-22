// Compares scaled causal softmax forward and backward results between the scalar CPU reference and CUDA implementation.
// The test exercises varying visible-prefix lengths and verifies both probabilities and scale-aware logits gradients.

#include "cuda_common.h"
#include "softmax.h"
#include "softmax_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kBatchSize = 2;
constexpr int kHeads = 4;
constexpr int kSequenceLength = 64;
constexpr int kElements =
    kBatchSize * kHeads * kSequenceLength * kSequenceLength;
constexpr float kScale = 0.125F;

float max_error(const std::vector<float>& expected, const std::vector<float>& actual) {
    float error = 0.0F;
    for (int index = 0; index < static_cast<int>(expected.size()); ++index) {
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
    std::printf("  %-18s max error = %.3e  %s\n", name, error, passed ? "PASS" : "FAIL");
    return passed;
}

bool run_test() {
    std::vector<float> logits(kElements);
    std::vector<float> probabilities_gradient(kElements);
    for (int index = 0; index < kElements; ++index) {
        logits[index] = static_cast<float>((index * 17) % 101 - 50) / 8.0F;
        probabilities_gradient[index] =
            static_cast<float>((index * 29) % 89 - 44) / 16.0F;
    }

    std::vector<float> cpu_probabilities(kElements);
    std::vector<float> cpu_logits_gradient(kElements, 0.0F);
    dscuda::causal_softmax_forward_cpu(
        cpu_probabilities.data(),
        logits.data(),
        kBatchSize,
        kHeads,
        kSequenceLength,
        kScale);
    dscuda::causal_softmax_backward_cpu(
        cpu_logits_gradient.data(),
        probabilities_gradient.data(),
        cpu_probabilities.data(),
        kBatchSize,
        kHeads,
        kSequenceLength,
        kScale);

    auto* gpu_logits = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_probabilities_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_probabilities =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_logits_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        gpu_logits, logits.data(), kElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_probabilities_gradient,
        probabilities_gradient.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpu_logits_gradient, 0, kElements * sizeof(float)));

    dscuda::causal_softmax_forward_cuda(
        gpu_probabilities,
        gpu_logits,
        kBatchSize,
        kHeads,
        kSequenceLength,
        kScale);
    dscuda::causal_softmax_backward_cuda(
        gpu_logits_gradient,
        gpu_probabilities_gradient,
        gpu_probabilities,
        kBatchSize,
        kHeads,
        kSequenceLength,
        kScale);
    dscuda::synchronize();

    std::vector<float> gpu_probabilities_host(kElements);
    std::vector<float> gpu_logits_gradient_host(kElements);
    CUDA_CHECK(cudaMemcpy(
        gpu_probabilities_host.data(),
        gpu_probabilities,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_logits_gradient_host.data(),
        gpu_logits_gradient,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_logits_gradient);
    dscuda::device_free(gpu_probabilities);
    dscuda::device_free(gpu_probabilities_gradient);
    dscuda::device_free(gpu_logits);

    bool passed = true;
    passed &= check("probabilities", cpu_probabilities, gpu_probabilities_host, 1.0e-5F);
    passed &= check("logits gradient", cpu_logits_gradient, gpu_logits_gradient_host, 2.0e-5F);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Causal softmax test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Causal softmax test failed: %s\n", error.what());
        return 1;
    }
}

// Compares vectorized embedding lookup and accumulated table gradients against the scalar CPU reference.
// Deliberately repeated token IDs verify atomic collision handling while untouched vocabulary rows test accumulation boundaries.

#include "cuda_common.h"
#include "embedding.h"
#include "embedding_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kTokenCount = 96;
constexpr int kVocabularySize = 23;
constexpr int kHiddenSize = 128;
constexpr int kActivationElements = kTokenCount * kHiddenSize;
constexpr int kWeightElements = kVocabularySize * kHiddenSize;

float max_error(
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
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
    std::printf(
        "  %-24s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool run_test() {
    std::vector<int> token_ids(kTokenCount);
    std::vector<float> weight(kWeightElements);
    std::vector<float> output_gradient(kActivationElements);
    std::vector<float> initial_weight_gradient(kWeightElements);
    for (int token = 0; token < kTokenCount; ++token) {
        token_ids[token] = (token * token + 3 * token + 7) % 11;
    }
    for (int index = 0; index < kWeightElements; ++index) {
        weight[index] = static_cast<float>((index * 17) % 101 - 50) / 71.0F;
        initial_weight_gradient[index] =
            static_cast<float>((index * 19) % 47 - 23) / 131.0F;
    }
    for (int index = 0; index < kActivationElements; ++index) {
        output_gradient[index] =
            static_cast<float>((index * 29) % 89 - 44) / 97.0F;
    }

    std::vector<float> cpu_output(kActivationElements);
    std::vector<float> cpu_weight_gradient = initial_weight_gradient;
    dscuda::embedding_forward_cpu(
        cpu_output.data(),
        token_ids.data(),
        weight.data(),
        kTokenCount,
        kHiddenSize);
    dscuda::embedding_backward_cpu(
        cpu_weight_gradient.data(),
        output_gradient.data(),
        token_ids.data(),
        kTokenCount,
        kHiddenSize);

    auto* gpu_token_ids = static_cast<int*>(
        dscuda::device_malloc(kTokenCount * sizeof(int)));
    auto* gpu_weight = static_cast<float*>(
        dscuda::device_malloc(kWeightElements * sizeof(float)));
    auto* gpu_output_gradient = static_cast<float*>(
        dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_output = static_cast<float*>(
        dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_weight_gradient = static_cast<float*>(
        dscuda::device_malloc(kWeightElements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        gpu_token_ids,
        token_ids.data(),
        kTokenCount * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_weight,
        weight.data(),
        kWeightElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        kActivationElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_weight_gradient,
        initial_weight_gradient.data(),
        kWeightElements * sizeof(float),
        cudaMemcpyHostToDevice));

    dscuda::embedding_forward_cuda(
        gpu_output,
        gpu_token_ids,
        gpu_weight,
        kTokenCount,
        kHiddenSize);
    dscuda::embedding_backward_cuda(
        gpu_weight_gradient,
        gpu_output_gradient,
        gpu_token_ids,
        kTokenCount,
        kHiddenSize);
    dscuda::synchronize();

    std::vector<float> gpu_output_host(kActivationElements);
    std::vector<float> gpu_weight_gradient_host(kWeightElements);
    CUDA_CHECK(cudaMemcpy(
        gpu_output_host.data(),
        gpu_output,
        kActivationElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_weight_gradient_host.data(),
        gpu_weight_gradient,
        kWeightElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_weight_gradient);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_weight);
    dscuda::device_free(gpu_token_ids);

    bool passed = true;
    passed &= check("output", cpu_output, gpu_output_host, 1.0e-7F);
    passed &= check(
        "embedding gradient",
        cpu_weight_gradient,
        gpu_weight_gradient_host,
        2.0e-6F);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Embedding test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Embedding test failed: %s\n", error.what());
        return 1;
    }
}

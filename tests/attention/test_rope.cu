// Compares interleaved FP32 RoPE forward and inverse-rotation backward results between CPU and CUDA implementations.
// The test covers repeated positions across batches, multiple heads, and an unchanged suffix outside the rotary dimensions.

#include "cuda_common.h"
#include "rope.h"
#include "rope_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kBatchSize = 2;
constexpr int kSequenceLength = 32;
constexpr int kHeads = 4;
constexpr int kHeadSize = 64;
constexpr int kRotarySize = 32;
constexpr int kElements = kBatchSize * kSequenceLength * kHeads * kHeadSize;
constexpr int kFrequencies = kSequenceLength * kRotarySize / 2;

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
    std::vector<float> input(kElements);
    std::vector<float> output_gradient(kElements);
    std::vector<float> cosine(kFrequencies);
    std::vector<float> sine(kFrequencies);

    for (int index = 0; index < kElements; ++index) {
        input[index] = static_cast<float>((index * 17) % 101 - 50) / 32.0F;
        output_gradient[index] = static_cast<float>((index * 29) % 89 - 44) / 32.0F;
    }
    for (int position = 0; position < kSequenceLength; ++position) {
        for (int pair = 0; pair < kRotarySize / 2; ++pair) {
            const int index = position * kRotarySize / 2 + pair;
            const float angle = static_cast<float>(position * (pair + 1)) / 128.0F;
            cosine[index] = std::cos(angle);
            sine[index] = std::sin(angle);
        }
    }

    std::vector<float> cpu_output(kElements);
    std::vector<float> cpu_input_gradient(kElements, 0.0F);
    dscuda::rope_forward_cpu(
        cpu_output.data(),
        input.data(),
        cosine.data(),
        sine.data(),
        kBatchSize,
        kSequenceLength,
        kHeads,
        kHeadSize,
        kRotarySize);
    dscuda::rope_backward_cpu(
        cpu_input_gradient.data(),
        output_gradient.data(),
        cosine.data(),
        sine.data(),
        kBatchSize,
        kSequenceLength,
        kHeads,
        kHeadSize,
        kRotarySize);

    auto* gpu_input = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_output_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_cosine =
        static_cast<float*>(dscuda::device_malloc(kFrequencies * sizeof(float)));
    auto* gpu_sine = static_cast<float*>(dscuda::device_malloc(kFrequencies * sizeof(float)));
    auto* gpu_output = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_input_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(gpu_input, input.data(), kElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_cosine, cosine.data(), kFrequencies * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_sine, sine.data(), kFrequencies * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpu_input_gradient, 0, kElements * sizeof(float)));

    dscuda::rope_forward_cuda(
        gpu_output,
        gpu_input,
        gpu_cosine,
        gpu_sine,
        kBatchSize,
        kSequenceLength,
        kHeads,
        kHeadSize,
        kRotarySize);
    dscuda::rope_backward_cuda(
        gpu_input_gradient,
        gpu_output_gradient,
        gpu_cosine,
        gpu_sine,
        kBatchSize,
        kSequenceLength,
        kHeads,
        kHeadSize,
        kRotarySize);
    dscuda::synchronize();

    std::vector<float> gpu_output_host(kElements);
    std::vector<float> gpu_input_gradient_host(kElements);
    CUDA_CHECK(cudaMemcpy(
        gpu_output_host.data(), gpu_output, kElements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_input_gradient_host.data(),
        gpu_input_gradient,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_input_gradient);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_sine);
    dscuda::device_free(gpu_cosine);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_input);

    bool passed = true;
    passed &= check("forward output", cpu_output, gpu_output_host, 3.0e-6F);
    passed &= check("input gradient", cpu_input_gradient, gpu_input_gradient_host, 3.0e-6F);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("RoPE test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RoPE test failed: %s\n", error.what());
        return 1;
    }
}

// Compares composed dense causal attention forward and backward against the scalar CPU reference.
// The test covers [B,T,H,D] layout transforms, strided-batched transpose GEMMs, masking, and accumulated input gradients.

#include "attention.h"
#include "attention_cpu.h"
#include "cuda_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kBatchSize = 2;
constexpr int kSequenceLength = 48;
constexpr int kHeads = 3;
constexpr int kHeadSize = 32;
constexpr int kActivationElements =
    kBatchSize * kSequenceLength * kHeads * kHeadSize;
constexpr int kProbabilityElements =
    kBatchSize * kHeads * kSequenceLength * kSequenceLength;
constexpr float kScale = 0.176776695F;

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
        "  %-20s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool run_test() {
    std::vector<float> query(kActivationElements);
    std::vector<float> key(kActivationElements);
    std::vector<float> value(kActivationElements);
    std::vector<float> output_gradient(kActivationElements);
    for (int index = 0; index < kActivationElements; ++index) {
        query[index] = static_cast<float>((index * 17) % 101 - 50) / 64.0F;
        key[index] = static_cast<float>((index * 23) % 97 - 48) / 61.0F;
        value[index] = static_cast<float>((index * 31) % 89 - 44) / 59.0F;
        output_gradient[index] =
            static_cast<float>((index * 37) % 83 - 41) / 67.0F;
    }

    std::vector<float> cpu_output(kActivationElements);
    std::vector<float> cpu_probabilities(kProbabilityElements);
    std::vector<float> cpu_query_gradient(kActivationElements);
    std::vector<float> cpu_key_gradient(kActivationElements);
    std::vector<float> cpu_value_gradient(kActivationElements);
    for (int index = 0; index < kActivationElements; ++index) {
        cpu_query_gradient[index] =
            static_cast<float>((index * 7) % 17 - 8) / 113.0F;
        cpu_key_gradient[index] =
            static_cast<float>((index * 11) % 19 - 9) / 127.0F;
        cpu_value_gradient[index] =
            static_cast<float>((index * 13) % 23 - 11) / 131.0F;
    }

    dscuda::dense_attention_forward_cpu(
        cpu_output.data(),
        cpu_probabilities.data(),
        query.data(),
        key.data(),
        value.data(),
        kBatchSize,
        kSequenceLength,
        kHeads,
        kHeadSize,
        kScale);
    dscuda::dense_attention_backward_cpu(
        cpu_query_gradient.data(),
        cpu_key_gradient.data(),
        cpu_value_gradient.data(),
        output_gradient.data(),
        cpu_probabilities.data(),
        query.data(),
        key.data(),
        value.data(),
        kBatchSize,
        kSequenceLength,
        kHeads,
        kHeadSize,
        kScale);

    auto* gpu_query =
        static_cast<float*>(dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_key =
        static_cast<float*>(dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_value =
        static_cast<float*>(dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_output_gradient =
        static_cast<float*>(dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_output =
        static_cast<float*>(dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_probabilities = static_cast<float*>(
        dscuda::device_malloc(kProbabilityElements * sizeof(float)));
    auto* gpu_query_gradient =
        static_cast<float*>(dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_key_gradient =
        static_cast<float*>(dscuda::device_malloc(kActivationElements * sizeof(float)));
    auto* gpu_value_gradient =
        static_cast<float*>(dscuda::device_malloc(kActivationElements * sizeof(float)));

    const std::size_t forward_workspace_elements =
        dscuda::dense_attention_forward_workspace_elements(
            kBatchSize, kSequenceLength, kHeads, kHeadSize);
    const std::size_t backward_workspace_elements =
        dscuda::dense_attention_backward_workspace_elements(
            kBatchSize, kSequenceLength, kHeads, kHeadSize);
    auto* gpu_forward_workspace = static_cast<float*>(
        dscuda::device_malloc(forward_workspace_elements * sizeof(float)));
    auto* gpu_backward_workspace = static_cast<float*>(
        dscuda::device_malloc(backward_workspace_elements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        gpu_query,
        query.data(),
        kActivationElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_key,
        key.data(),
        kActivationElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_value,
        value.data(),
        kActivationElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        kActivationElements * sizeof(float),
        cudaMemcpyHostToDevice));
    std::vector<float> initial_query_gradient(kActivationElements);
    std::vector<float> initial_key_gradient(kActivationElements);
    std::vector<float> initial_value_gradient(kActivationElements);
    for (int index = 0; index < kActivationElements; ++index) {
        initial_query_gradient[index] =
            static_cast<float>((index * 7) % 17 - 8) / 113.0F;
        initial_key_gradient[index] =
            static_cast<float>((index * 11) % 19 - 9) / 127.0F;
        initial_value_gradient[index] =
            static_cast<float>((index * 13) % 23 - 11) / 131.0F;
    }
    CUDA_CHECK(cudaMemcpy(
        gpu_query_gradient,
        initial_query_gradient.data(),
        kActivationElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_key_gradient,
        initial_key_gradient.data(),
        kActivationElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_value_gradient,
        initial_value_gradient.data(),
        kActivationElements * sizeof(float),
        cudaMemcpyHostToDevice));

    dscuda::dense_attention_forward_cuda(
        gpu_output,
        gpu_probabilities,
        gpu_query,
        gpu_key,
        gpu_value,
        gpu_forward_workspace,
        kBatchSize,
        kSequenceLength,
        kHeads,
        kHeadSize,
        kScale);
    dscuda::dense_attention_backward_cuda(
        gpu_query_gradient,
        gpu_key_gradient,
        gpu_value_gradient,
        gpu_output_gradient,
        gpu_probabilities,
        gpu_query,
        gpu_key,
        gpu_value,
        gpu_backward_workspace,
        kBatchSize,
        kSequenceLength,
        kHeads,
        kHeadSize,
        kScale);
    dscuda::synchronize();

    std::vector<float> gpu_output_host(kActivationElements);
    std::vector<float> gpu_probabilities_host(kProbabilityElements);
    std::vector<float> gpu_query_gradient_host(kActivationElements);
    std::vector<float> gpu_key_gradient_host(kActivationElements);
    std::vector<float> gpu_value_gradient_host(kActivationElements);
    CUDA_CHECK(cudaMemcpy(
        gpu_output_host.data(),
        gpu_output,
        kActivationElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_probabilities_host.data(),
        gpu_probabilities,
        kProbabilityElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_query_gradient_host.data(),
        gpu_query_gradient,
        kActivationElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_key_gradient_host.data(),
        gpu_key_gradient,
        kActivationElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_value_gradient_host.data(),
        gpu_value_gradient,
        kActivationElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_backward_workspace);
    dscuda::device_free(gpu_forward_workspace);
    dscuda::device_free(gpu_value_gradient);
    dscuda::device_free(gpu_key_gradient);
    dscuda::device_free(gpu_query_gradient);
    dscuda::device_free(gpu_probabilities);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_value);
    dscuda::device_free(gpu_key);
    dscuda::device_free(gpu_query);

    bool passed = true;
    passed &= check("probabilities", cpu_probabilities, gpu_probabilities_host, 3.0e-5F);
    passed &= check("output", cpu_output, gpu_output_host, 2.0e-4F);
    passed &= check("query gradient", cpu_query_gradient, gpu_query_gradient_host, 5.0e-4F);
    passed &= check("key gradient", cpu_key_gradient, gpu_key_gradient_host, 5.0e-4F);
    passed &= check("value gradient", cpu_value_gradient, gpu_value_gradient_host, 5.0e-4F);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Dense attention test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Dense attention test failed: %s\n", error.what());
        return 1;
    }
}

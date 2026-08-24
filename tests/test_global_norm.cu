// Compares the two-pass CUDA global gradient norm and in-place clipping with a double-accumulation CPU reference.
// The test verifies the reported norm, every clipped element, and the final norm at the requested threshold.

#include "cuda_common.h"
#include "global_norm.h"
#include "global_norm_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kElements = 65536;
constexpr float kMaxNorm = 10.0F;

float max_error(
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    float error = 0.0F;
    for (int index = 0; index < kElements; ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool run_test() {
    std::vector<float> gradients(kElements);
    for (int index = 0; index < kElements; ++index) {
        gradients[index] =
            static_cast<float>((index * 31) % 257 - 128) / 113.0F;
    }
    const float cpu_norm =
        dscuda::global_norm_cpu(gradients.data(), kElements);
    std::vector<float> cpu_clipped = gradients;
    dscuda::clip_gradients_cpu(
        cpu_clipped.data(), kElements, cpu_norm, kMaxNorm);

    auto* gpu_gradients = static_cast<float*>(
        dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_norm = static_cast<float*>(
        dscuda::device_malloc(sizeof(float)));
    const std::size_t workspace_elements =
        dscuda::global_norm_workspace_elements(kElements);
    auto* gpu_workspace = static_cast<float*>(
        dscuda::device_malloc(workspace_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        gpu_gradients,
        gradients.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));

    dscuda::global_norm_cuda(
        gpu_norm,
        gpu_gradients,
        gpu_workspace,
        kElements);
    dscuda::clip_gradients_cuda(
        gpu_gradients,
        gpu_norm,
        kElements,
        kMaxNorm);
    dscuda::synchronize();

    float gpu_norm_host = 0.0F;
    std::vector<float> gpu_clipped(kElements);
    CUDA_CHECK(cudaMemcpy(
        &gpu_norm_host,
        gpu_norm,
        sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_clipped.data(),
        gpu_gradients,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_workspace);
    dscuda::device_free(gpu_norm);
    dscuda::device_free(gpu_gradients);

    const float norm_error = std::abs(cpu_norm - gpu_norm_host);
    const float clipped_error = max_error(cpu_clipped, gpu_clipped);
    const float final_norm =
        dscuda::global_norm_cpu(gpu_clipped.data(), kElements);
    const float threshold_error = std::abs(final_norm - kMaxNorm);
    const bool passed = norm_error < 2.0e-4F &&
        clipped_error < 2.0e-6F && threshold_error < 2.0e-5F;
    std::printf(
        "  %-22s error = %.3e  %s\n",
        "global norm",
        norm_error,
        norm_error < 2.0e-4F ? "PASS" : "FAIL");
    std::printf(
        "  %-22s max error = %.3e  %s\n",
        "clipped gradients",
        clipped_error,
        clipped_error < 2.0e-6F ? "PASS" : "FAIL");
    std::printf(
        "  %-22s error = %.3e  %s\n",
        "clipped norm",
        threshold_error,
        threshold_error < 2.0e-5F ? "PASS" : "FAIL");
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf(
            "Global norm test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Global norm test failed: %s\n", error.what());
        return 1;
    }
}

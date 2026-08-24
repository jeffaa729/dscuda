// Compares vectorized residual forward and accumulated backward results against the CPU reference.
// Nonzero initial gradients verify that both identity branches preserve graph accumulation semantics.

#include "cuda_common.h"
#include "residual.h"
#include "residual_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kElements = 8192;

float max_error(
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    float error = 0.0F;
    for (int index = 0; index < kElements; ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    const float error = max_error(expected, actual);
    const bool passed = error < 1.0e-6F;
    std::printf(
        "  %-18s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool run_test() {
    std::vector<float> input(kElements);
    std::vector<float> branch(kElements);
    std::vector<float> output_gradient(kElements);
    std::vector<float> cpu_input_gradient(kElements);
    std::vector<float> cpu_branch_gradient(kElements);
    for (int index = 0; index < kElements; ++index) {
        input[index] = static_cast<float>((index * 17) % 101 - 50) / 37.0F;
        branch[index] = static_cast<float>((index * 23) % 97 - 48) / 41.0F;
        output_gradient[index] =
            static_cast<float>((index * 29) % 89 - 44) / 43.0F;
        cpu_input_gradient[index] =
            static_cast<float>((index * 7) % 31 - 15) / 47.0F;
        cpu_branch_gradient[index] =
            static_cast<float>((index * 11) % 37 - 18) / 53.0F;
    }
    const std::vector<float> initial_input_gradient = cpu_input_gradient;
    const std::vector<float> initial_branch_gradient = cpu_branch_gradient;

    std::vector<float> cpu_output(kElements);
    dscuda::residual_forward_cpu(
        cpu_output.data(), input.data(), branch.data(), kElements);
    dscuda::residual_backward_cpu(
        cpu_input_gradient.data(),
        cpu_branch_gradient.data(),
        output_gradient.data(),
        kElements);

    auto* gpu_input =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_branch =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_output_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_output =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_input_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_branch_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        gpu_input,
        input.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_branch,
        branch.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_input_gradient,
        initial_input_gradient.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_branch_gradient,
        initial_branch_gradient.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));

    dscuda::residual_forward_cuda(
        gpu_output, gpu_input, gpu_branch, kElements);
    dscuda::residual_backward_cuda(
        gpu_input_gradient,
        gpu_branch_gradient,
        gpu_output_gradient,
        kElements);
    dscuda::synchronize();

    std::vector<float> gpu_output_host(kElements);
    std::vector<float> gpu_input_gradient_host(kElements);
    std::vector<float> gpu_branch_gradient_host(kElements);
    CUDA_CHECK(cudaMemcpy(
        gpu_output_host.data(),
        gpu_output,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_input_gradient_host.data(),
        gpu_input_gradient,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_branch_gradient_host.data(),
        gpu_branch_gradient,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_branch_gradient);
    dscuda::device_free(gpu_input_gradient);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_branch);
    dscuda::device_free(gpu_input);

    bool passed = true;
    passed &= check("output", cpu_output, gpu_output_host);
    passed &= check("input gradient", cpu_input_gradient, gpu_input_gradient_host);
    passed &= check("branch gradient", cpu_branch_gradient, gpu_branch_gradient_host);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Residual test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Residual test failed: %s\n", error.what());
        return 1;
    }
}

// Runs the same fixed RMSNorm forward and backward calculation on the CPU and GPU, then compares every result tensor.
// The test validates the CUDA reduction kernels against the simple CPU reference.

#include "cuda_common.h"
#include "rmsnorm.h"
#include "rmsnorm_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kRows = 8;
constexpr int kHiddenSize = 512;
constexpr int kElements = kRows * kHiddenSize;
constexpr float kEpsilon = 1.0e-6F;

float max_error(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    float error = 0.0F;
    for (int index = 0; index < static_cast<int>(cpu.size()); ++index) {
        error = std::max(error, std::abs(cpu[index] - gpu[index]));
    }
    return error;
}

bool check(const char* name, const std::vector<float>& cpu, const std::vector<float>& gpu, float tolerance) {
    const float error = max_error(cpu, gpu);
    const bool passed = error < tolerance;
    std::printf("  %-18s max error = %.3e  %s\n", name, error, passed ? "PASS" : "FAIL");
    return passed;
}

bool run_test() {
    std::vector<float> input(kElements);
    std::vector<float> weight(kHiddenSize);
    std::vector<float> output_gradient(kElements);

    for (int index = 0; index < kElements; ++index) {
        input[index] = static_cast<float>((index * 17) % 101 - 50) / 50.0F;
        output_gradient[index] = static_cast<float>((index * 29) % 97 - 48) / 48.0F;
    }
    for (int index = 0; index < kHiddenSize; ++index) {
        weight[index] = 0.5F + static_cast<float>((index * 13) % 100) / 100.0F;
    }

    // 1. Compute the expected forward and backward results on the CPU.
    std::vector<float> cpu_output(kElements);
    std::vector<float> cpu_inverse_rms(kRows);
    std::vector<float> cpu_input_gradient(kElements, 0.0F);
    std::vector<float> cpu_weight_gradient(kHiddenSize, 0.0F);

    dscuda::rmsnorm_forward_cpu(
        cpu_output.data(),
        cpu_inverse_rms.data(),
        input.data(),
        weight.data(),
        kRows,
        kHiddenSize,
        kEpsilon);
    dscuda::rmsnorm_backward_cpu(
        cpu_input_gradient.data(),
        cpu_weight_gradient.data(),
        output_gradient.data(),
        input.data(),
        weight.data(),
        cpu_inverse_rms.data(),
        kRows,
        kHiddenSize);

    // 2. Allocate GPU buffers and copy the inputs to the device.
    auto* gpu_input = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_weight = static_cast<float*>(dscuda::device_malloc(kHiddenSize * sizeof(float)));
    auto* gpu_output_gradient = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_output = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_inverse_rms = static_cast<float*>(dscuda::device_malloc(kRows * sizeof(float)));
    auto* gpu_input_gradient = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_weight_gradient = static_cast<float*>(dscuda::device_malloc(kHiddenSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(gpu_input, input.data(), kElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_weight, weight.data(), kHiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpu_input_gradient, 0, kElements * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_weight_gradient, 0, kHiddenSize * sizeof(float)));

    // 3. Run the CUDA forward and backward kernels.
    dscuda::rmsnorm_forward_cuda(
        gpu_output,
        gpu_inverse_rms,
        gpu_input,
        gpu_weight,
        kRows,
        kHiddenSize,
        kEpsilon);
    dscuda::rmsnorm_backward_cuda(
        gpu_input_gradient,
        gpu_weight_gradient,
        gpu_output_gradient,
        gpu_input,
        gpu_weight,
        gpu_inverse_rms,
        kRows,
        kHiddenSize);
    dscuda::synchronize();

    // 4. Copy the GPU results back to the host.
    std::vector<float> gpu_output_host(kElements);
    std::vector<float> gpu_inverse_rms_host(kRows);
    std::vector<float> gpu_input_gradient_host(kElements);
    std::vector<float> gpu_weight_gradient_host(kHiddenSize);

    CUDA_CHECK(cudaMemcpy(
        gpu_output_host.data(), gpu_output, kElements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_inverse_rms_host.data(), gpu_inverse_rms, kRows * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_input_gradient_host.data(),
        gpu_input_gradient,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_weight_gradient_host.data(),
        gpu_weight_gradient,
        kHiddenSize * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_weight_gradient);
    dscuda::device_free(gpu_input_gradient);
    dscuda::device_free(gpu_inverse_rms);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_weight);
    dscuda::device_free(gpu_input);

    // 5. Compare CPU and GPU results.
    bool passed = true;
    passed &= check("forward output", cpu_output, gpu_output_host, 2.0e-5F);
    passed &= check("inverse RMS", cpu_inverse_rms, gpu_inverse_rms_host, 2.0e-5F);
    passed &= check("input gradient", cpu_input_gradient, gpu_input_gradient_host, 3.0e-5F);
    passed &= check("weight gradient", cpu_weight_gradient, gpu_weight_gradient_host, 3.0e-4F);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();

        const bool passed = run_test();

        std::printf("RMSNorm test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RMSNorm test failed: %s\n", error.what());
        return 1;
    }
}

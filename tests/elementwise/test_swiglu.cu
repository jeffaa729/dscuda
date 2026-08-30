// Runs identical SwiGLU forward and backward calculations on the CPU and GPU, then compares every output tensor.
// The test covers the fused CUDA activation and both analytical input-gradient branches.

#include "cuda_common.h"
#include "swiglu.h"
#include "swiglu_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kElements = 4096;

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
    std::printf("  %-16s max error = %.3e  %s\n", name, error, passed ? "PASS" : "FAIL");
    return passed;
}

bool run_test() {
    std::vector<float> gate(kElements);
    std::vector<float> up(kElements);
    std::vector<float> output_gradient(kElements);

    for (int index = 0; index < kElements; ++index) {
        gate[index] = static_cast<float>((index * 17) % 101 - 50) / 25.0F;
        up[index] = static_cast<float>((index * 23) % 97 - 48) / 32.0F;
        output_gradient[index] = static_cast<float>((index * 29) % 89 - 44) / 24.0F;
    }

    std::vector<float> cpu_output(kElements);
    std::vector<float> cpu_gate_gradient(kElements);
    std::vector<float> cpu_up_gradient(kElements);
    dscuda::swiglu_forward_cpu(cpu_output.data(), gate.data(), up.data(), kElements);
    dscuda::swiglu_backward_cpu(
        cpu_gate_gradient.data(),
        cpu_up_gradient.data(),
        output_gradient.data(),
        gate.data(),
        up.data(),
        kElements);

    auto* gpu_gate = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_up = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_output_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_output = static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_gate_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_up_gradient =
        static_cast<float*>(dscuda::device_malloc(kElements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(gpu_gate, gate.data(), kElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_up, up.data(), kElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));

    dscuda::swiglu_forward_cuda(gpu_output, gpu_gate, gpu_up, kElements);
    dscuda::swiglu_backward_cuda(
        gpu_gate_gradient,
        gpu_up_gradient,
        gpu_output_gradient,
        gpu_gate,
        gpu_up,
        kElements);
    dscuda::synchronize();

    std::vector<float> gpu_output_host(kElements);
    std::vector<float> gpu_gate_gradient_host(kElements);
    std::vector<float> gpu_up_gradient_host(kElements);
    CUDA_CHECK(cudaMemcpy(
        gpu_output_host.data(), gpu_output, kElements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_gate_gradient_host.data(),
        gpu_gate_gradient,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_up_gradient_host.data(),
        gpu_up_gradient,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_up_gradient);
    dscuda::device_free(gpu_gate_gradient);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_up);
    dscuda::device_free(gpu_gate);

    bool passed = true;
    passed &= check("forward output", cpu_output, gpu_output_host, 2.0e-6F);
    passed &= check("gate gradient", cpu_gate_gradient, gpu_gate_gradient_host, 3.0e-6F);
    passed &= check("up gradient", cpu_up_gradient, gpu_up_gradient_host, 2.0e-6F);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("SwiGLU test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "SwiGLU test failed: %s\n", error.what());
        return 1;
    }
}

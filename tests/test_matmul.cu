// Compares aligned FP32 CUDA matrix-multiplication forward and backward results with the scalar CPU reference.
// The rectangular test shape catches indexing and transpose errors in output, left-gradient, and right-gradient calculations.

#include "cuda_common.h"
#include "matmul.h"
#include "matmul_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kM = 128;
constexpr int kN = 256;
constexpr int kK = 64;
constexpr int kLeftElements = kM * kK;
constexpr int kRightElements = kK * kN;
constexpr int kOutputElements = kM * kN;

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
    std::vector<float> left(kLeftElements);
    std::vector<float> right(kRightElements);
    std::vector<float> output_gradient(kOutputElements);

    for (int index = 0; index < kLeftElements; ++index) {
        left[index] = static_cast<float>((index * 17) % 101 - 50) / 64.0F;
    }
    for (int index = 0; index < kRightElements; ++index) {
        right[index] = static_cast<float>((index * 23) % 97 - 48) / 64.0F;
    }
    for (int index = 0; index < kOutputElements; ++index) {
        output_gradient[index] = static_cast<float>((index * 29) % 89 - 44) / 64.0F;
    }

    std::vector<float> cpu_output(kOutputElements);
    std::vector<float> cpu_left_gradient(kLeftElements, 0.0F);
    std::vector<float> cpu_right_gradient(kRightElements, 0.0F);
    dscuda::matmul_forward_cpu(
        cpu_output.data(), left.data(), right.data(), kM, kN, kK);
    dscuda::matmul_backward_cpu(
        cpu_left_gradient.data(),
        cpu_right_gradient.data(),
        output_gradient.data(),
        left.data(),
        right.data(),
        kM,
        kN,
        kK);

    auto* gpu_left = static_cast<float*>(dscuda::device_malloc(kLeftElements * sizeof(float)));
    auto* gpu_right = static_cast<float*>(dscuda::device_malloc(kRightElements * sizeof(float)));
    auto* gpu_output_gradient =
        static_cast<float*>(dscuda::device_malloc(kOutputElements * sizeof(float)));
    auto* gpu_output =
        static_cast<float*>(dscuda::device_malloc(kOutputElements * sizeof(float)));
    auto* gpu_left_gradient =
        static_cast<float*>(dscuda::device_malloc(kLeftElements * sizeof(float)));
    auto* gpu_right_gradient =
        static_cast<float*>(dscuda::device_malloc(kRightElements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        gpu_left, left.data(), kLeftElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_right, right.data(), kRightElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        kOutputElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpu_left_gradient, 0, kLeftElements * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_right_gradient, 0, kRightElements * sizeof(float)));

    dscuda::matmul_forward_cuda(
        gpu_output, gpu_left, gpu_right, kM, kN, kK);
    dscuda::matmul_backward_cuda(
        gpu_left_gradient,
        gpu_right_gradient,
        gpu_output_gradient,
        gpu_left,
        gpu_right,
        kM,
        kN,
        kK);
    dscuda::synchronize();

    std::vector<float> gpu_output_host(kOutputElements);
    std::vector<float> gpu_left_gradient_host(kLeftElements);
    std::vector<float> gpu_right_gradient_host(kRightElements);
    CUDA_CHECK(cudaMemcpy(
        gpu_output_host.data(),
        gpu_output,
        kOutputElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_left_gradient_host.data(),
        gpu_left_gradient,
        kLeftElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_right_gradient_host.data(),
        gpu_right_gradient,
        kRightElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_right_gradient);
    dscuda::device_free(gpu_left_gradient);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_right);
    dscuda::device_free(gpu_left);

    bool passed = true;
    passed &= check("forward output", cpu_output, gpu_output_host, 2.0e-4F);
    passed &= check("left gradient", cpu_left_gradient, gpu_left_gradient_host, 3.0e-4F);
    passed &= check("right gradient", cpu_right_gradient, gpu_right_gradient_host, 3.0e-4F);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Matmul test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Matmul test failed: %s\n", error.what());
        return 1;
    }
}

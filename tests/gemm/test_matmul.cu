// Compares native FP32 CUDA Core and BF16 Tensor Core matrix multiplication with matching scalar CPU references.
// A rectangular aligned shape exercises transpose modes, while nonzero FP32 gradients verify accumulation semantics.

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
constexpr int kK = 128;
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
        left[index] = static_cast<float>((index * 17) % 101 - 50) / 63.0F;
    }
    for (int index = 0; index < kRightElements; ++index) {
        right[index] = static_cast<float>((index * 23) % 97 - 48) / 61.0F;
    }
    for (int index = 0; index < kOutputElements; ++index) {
        output_gradient[index] = static_cast<float>((index * 29) % 89 - 44) / 59.0F;
    }

    std::vector<__nv_bfloat16> bf16_left(kLeftElements);
    std::vector<__nv_bfloat16> bf16_right(kRightElements);
    std::vector<__nv_bfloat16> bf16_output_gradient(kOutputElements);
    std::vector<float> quantized_left(kLeftElements);
    std::vector<float> quantized_right(kRightElements);
    std::vector<float> quantized_output_gradient(kOutputElements);
    for (int index = 0; index < kLeftElements; ++index) {
        bf16_left[index] = __float2bfloat16(left[index]);
        quantized_left[index] = __bfloat162float(bf16_left[index]);
    }
    for (int index = 0; index < kRightElements; ++index) {
        bf16_right[index] = __float2bfloat16(right[index]);
        quantized_right[index] = __bfloat162float(bf16_right[index]);
    }
    for (int index = 0; index < kOutputElements; ++index) {
        bf16_output_gradient[index] = __float2bfloat16(output_gradient[index]);
        quantized_output_gradient[index] = __bfloat162float(bf16_output_gradient[index]);
    }

    std::vector<float> cpu_output(kOutputElements);
    std::vector<float> initial_left_gradient(kLeftElements, 0.25F);
    std::vector<float> initial_right_gradient(kRightElements, -0.125F);
    std::vector<float> cpu_left_gradient = initial_left_gradient;
    std::vector<float> cpu_right_gradient = initial_right_gradient;
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

    std::vector<float> bf16_cpu_output(kOutputElements);
    std::vector<float> bf16_cpu_left_gradient = initial_left_gradient;
    std::vector<float> bf16_cpu_right_gradient = initial_right_gradient;
    dscuda::matmul_forward_cpu(
        bf16_cpu_output.data(),
        quantized_left.data(),
        quantized_right.data(),
        kM,
        kN,
        kK);
    dscuda::matmul_backward_cpu(
        bf16_cpu_left_gradient.data(),
        bf16_cpu_right_gradient.data(),
        quantized_output_gradient.data(),
        quantized_left.data(),
        quantized_right.data(),
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
    auto* gpu_bf16_left = static_cast<__nv_bfloat16*>(
        dscuda::device_malloc(kLeftElements * sizeof(__nv_bfloat16)));
    auto* gpu_bf16_right = static_cast<__nv_bfloat16*>(
        dscuda::device_malloc(kRightElements * sizeof(__nv_bfloat16)));
    auto* gpu_bf16_output_gradient = static_cast<__nv_bfloat16*>(
        dscuda::device_malloc(kOutputElements * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(
        gpu_left, left.data(), kLeftElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_right, right.data(), kRightElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        kOutputElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_bf16_left,
        bf16_left.data(),
        kLeftElements * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_bf16_right,
        bf16_right.data(),
        kRightElements * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_bf16_output_gradient,
        bf16_output_gradient.data(),
        kOutputElements * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_left_gradient,
        initial_left_gradient.data(),
        kLeftElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_right_gradient,
        initial_right_gradient.data(),
        kRightElements * sizeof(float),
        cudaMemcpyHostToDevice));

    dscuda::matmul_fp32_forward_cuda(
        gpu_output, gpu_left, gpu_right, kM, kN, kK);
    dscuda::matmul_fp32_backward_cuda(
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

    CUDA_CHECK(cudaMemcpy(
        gpu_left_gradient,
        initial_left_gradient.data(),
        kLeftElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_right_gradient,
        initial_right_gradient.data(),
        kRightElements * sizeof(float),
        cudaMemcpyHostToDevice));

    dscuda::matmul_bf16_forward_cuda(
        gpu_output, gpu_bf16_left, gpu_bf16_right, kM, kN, kK);
    dscuda::matmul_bf16_backward_cuda(
        gpu_left_gradient,
        gpu_right_gradient,
        gpu_bf16_output_gradient,
        gpu_bf16_left,
        gpu_bf16_right,
        kM,
        kN,
        kK);
    dscuda::synchronize();

    std::vector<float> bf16_output(kOutputElements);
    std::vector<float> bf16_left_gradient(kLeftElements);
    std::vector<float> bf16_right_gradient(kRightElements);
    CUDA_CHECK(cudaMemcpy(
        bf16_output.data(),
        gpu_output,
        kOutputElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        bf16_left_gradient.data(),
        gpu_left_gradient,
        kLeftElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        bf16_right_gradient.data(),
        gpu_right_gradient,
        kRightElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_bf16_output_gradient);
    dscuda::device_free(gpu_bf16_right);
    dscuda::device_free(gpu_bf16_left);
    dscuda::device_free(gpu_right_gradient);
    dscuda::device_free(gpu_left_gradient);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_right);
    dscuda::device_free(gpu_left);

    bool passed = true;
    passed &= check("FP32 forward", cpu_output, gpu_output_host, 2.0e-4F);
    passed &= check("FP32 left grad", cpu_left_gradient, gpu_left_gradient_host, 3.0e-4F);
    passed &= check("FP32 right grad", cpu_right_gradient, gpu_right_gradient_host, 3.0e-4F);
    passed &= check("BF16 forward", bf16_cpu_output, bf16_output, 2.0e-4F);
    passed &= check("BF16 left grad", bf16_cpu_left_gradient, bf16_left_gradient, 3.0e-4F);
    passed &= check("BF16 right grad", bf16_cpu_right_gradient, bf16_right_gradient, 3.0e-4F);
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

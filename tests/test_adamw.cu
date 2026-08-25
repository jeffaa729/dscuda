// Compares several vectorized CUDA AdamW steps with the scalar CPU optimizer state and parameters.
// Zero-gradient lanes provide an independent check that weight decay is decoupled from both moment estimates.

#include "cuda_common.h"
#include "optimizer.h"
#include "optimizer_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kElements = 8192;
constexpr int kSteps = 5;
constexpr dscuda::AdamWConfig kConfig{
    3.0e-4F, 0.9F, 0.95F, 1.0e-8F, 0.1F};

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
    const std::vector<float>& actual,
    float tolerance) {
    const float error = max_error(expected, actual);
    const bool passed = error < tolerance;
    std::printf(
        "  %-22s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool run_test() {
    std::vector<float> parameters(kElements);
    std::vector<float> gradients(kElements);
    std::vector<float> first_moment(kElements, 0.0F);
    std::vector<float> second_moment(kElements, 0.0F);
    for (int index = 0; index < kElements; ++index) {
        parameters[index] =
            static_cast<float>((index * 17) % 101 - 50) / 37.0F;
    }
    const std::vector<float> initial_parameters = parameters;

    auto* gpu_parameters = static_cast<float*>(
        dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_gradients = static_cast<float*>(
        dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_first_moment = static_cast<float*>(
        dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_second_moment = static_cast<float*>(
        dscuda::device_malloc(kElements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        gpu_parameters,
        parameters.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(
        gpu_first_moment, 0, kElements * sizeof(float)));
    CUDA_CHECK(cudaMemset(
        gpu_second_moment, 0, kElements * sizeof(float)));

    for (int step = 1; step <= kSteps; ++step) {
        for (int index = 0; index < kElements; ++index) {
            gradients[index] = index < 4
                ? 0.0F
                : static_cast<float>(
                      (index * (23 + step * 2)) % 113 - 56) /
                      (83.0F + step);
        }
        dscuda::adamw_step_cpu(
            parameters.data(),
            first_moment.data(),
            second_moment.data(),
            gradients.data(),
            kElements,
            step,
            kConfig);
        CUDA_CHECK(cudaMemcpy(
            gpu_gradients,
            gradients.data(),
            kElements * sizeof(float),
            cudaMemcpyHostToDevice));
        dscuda::adamw_step_cuda(
            gpu_parameters,
            gpu_first_moment,
            gpu_second_moment,
            gpu_gradients,
            kElements,
            step,
            kConfig);
    }
    dscuda::synchronize();

    std::vector<float> gpu_parameters_host(kElements);
    std::vector<float> gpu_first_moment_host(kElements);
    std::vector<float> gpu_second_moment_host(kElements);
    CUDA_CHECK(cudaMemcpy(
        gpu_parameters_host.data(),
        gpu_parameters,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_first_moment_host.data(),
        gpu_first_moment,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_second_moment_host.data(),
        gpu_second_moment,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_second_moment);
    dscuda::device_free(gpu_first_moment);
    dscuda::device_free(gpu_gradients);
    dscuda::device_free(gpu_parameters);

    const float expected_decay = initial_parameters[0] * std::pow(
        1.0F - kConfig.learning_rate * kConfig.weight_decay,
        kSteps);
    const float decay_error = std::abs(parameters[0] - expected_decay);
    const bool decay_passed = decay_error < 1.0e-6F &&
        first_moment[0] == 0.0F && second_moment[0] == 0.0F;

    bool passed = true;
    passed &= check(
        "parameters", parameters, gpu_parameters_host, 2.0e-6F);
    passed &= check(
        "first moment", first_moment, gpu_first_moment_host, 2.0e-6F);
    passed &= check(
        "second moment", second_moment, gpu_second_moment_host, 2.0e-6F);
    std::printf(
        "  %-22s error = %.3e  %s\n",
        "decoupled decay",
        decay_error,
        decay_passed ? "PASS" : "FAIL");
    passed &= decay_passed;
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("AdamW test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "AdamW test failed: %s\n", error.what());
        return 1;
    }
}

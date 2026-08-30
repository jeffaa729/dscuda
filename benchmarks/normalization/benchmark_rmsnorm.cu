// Runs one RMSNorm forward and backward workload for external measurement by Nsight Compute.
// Timing and hardware metrics are intentionally collected by the profiler rather than CUDA events in this executable.

#include "cuda_common.h"
#include "rmsnorm.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

constexpr int kHiddenSize = 512;
constexpr float kEpsilon = 1.0e-6F;

}  // namespace

int main(int argc, char** argv) {
    try {
        const int rows = argc > 1 ? std::atoi(argv[1]) : 2048;
        const int elements = rows * kHiddenSize;
        std::vector<float> input(elements, 0.5F);
        std::vector<float> weight(kHiddenSize, 1.0F);
        std::vector<float> output_gradient(elements, 0.25F);

        auto* gpu_input = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_weight =
            static_cast<float*>(dscuda::device_malloc(kHiddenSize * sizeof(float)));
        auto* gpu_output_gradient =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_output = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_inverse_rms = static_cast<float*>(dscuda::device_malloc(rows * sizeof(float)));
        auto* gpu_input_gradient =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_weight_gradient =
            static_cast<float*>(dscuda::device_malloc(kHiddenSize * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            gpu_input, input.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_weight, weight.data(), kHiddenSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_output_gradient,
            output_gradient.data(),
            elements * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(gpu_input_gradient, 0, elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpu_weight_gradient, 0, kHiddenSize * sizeof(float)));

        std::printf("RMSNorm workload: rows=%d hidden=%d\n", rows, kHiddenSize);
        dscuda::rmsnorm_forward_cuda(
            gpu_output,
            gpu_inverse_rms,
            gpu_input,
            gpu_weight,
            rows,
            kHiddenSize,
            kEpsilon);
        dscuda::rmsnorm_backward_cuda(
            gpu_input_gradient,
            gpu_weight_gradient,
            gpu_output_gradient,
            gpu_input,
            gpu_weight,
            gpu_inverse_rms,
            rows,
            kHiddenSize);
        dscuda::synchronize();

        dscuda::device_free(gpu_weight_gradient);
        dscuda::device_free(gpu_input_gradient);
        dscuda::device_free(gpu_inverse_rms);
        dscuda::device_free(gpu_output);
        dscuda::device_free(gpu_output_gradient);
        dscuda::device_free(gpu_weight);
        dscuda::device_free(gpu_input);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RMSNorm benchmark failed: %s\n", error.what());
        return 1;
    }
}

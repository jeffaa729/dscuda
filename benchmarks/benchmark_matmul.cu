// Runs one FP32 matrix-multiplication forward and backward workload for external measurement by Nsight Compute.
// Timing and hardware metrics are intentionally collected by the profiler rather than CUDA events in this executable.

#include "cuda_common.h"
#include "matmul.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int M = argc > 1 ? std::atoi(argv[1]) : 1024;
        const int N = argc > 2 ? std::atoi(argv[2]) : 1024;
        const int K = argc > 3 ? std::atoi(argv[3]) : 1024;
        const size_t left_elements = static_cast<size_t>(M) * K;
        const size_t right_elements = static_cast<size_t>(K) * N;
        const size_t output_elements = static_cast<size_t>(M) * N;

        std::vector<float> left(left_elements, 0.5F);
        std::vector<float> right(right_elements, 0.25F);
        std::vector<float> output_gradient(output_elements, 0.125F);

        auto* gpu_left = static_cast<float*>(dscuda::device_malloc(left_elements * sizeof(float)));
        auto* gpu_right =
            static_cast<float*>(dscuda::device_malloc(right_elements * sizeof(float)));
        auto* gpu_output_gradient =
            static_cast<float*>(dscuda::device_malloc(output_elements * sizeof(float)));
        auto* gpu_output =
            static_cast<float*>(dscuda::device_malloc(output_elements * sizeof(float)));
        auto* gpu_left_gradient =
            static_cast<float*>(dscuda::device_malloc(left_elements * sizeof(float)));
        auto* gpu_right_gradient =
            static_cast<float*>(dscuda::device_malloc(right_elements * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            gpu_left, left.data(), left_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_right, right.data(), right_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_output_gradient,
            output_gradient.data(),
            output_elements * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(gpu_left_gradient, 0, left_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpu_right_gradient, 0, right_elements * sizeof(float)));

        std::printf("Matmul workload: M=%d N=%d K=%d\n", M, N, K);
        dscuda::matmul_forward_cuda(
            gpu_output, gpu_left, gpu_right, M, N, K);
        dscuda::matmul_backward_cuda(
            gpu_left_gradient,
            gpu_right_gradient,
            gpu_output_gradient,
            gpu_left,
            gpu_right,
            M,
            N,
            K);
        dscuda::synchronize();

        dscuda::device_free(gpu_right_gradient);
        dscuda::device_free(gpu_left_gradient);
        dscuda::device_free(gpu_output);
        dscuda::device_free(gpu_output_gradient);
        dscuda::device_free(gpu_right);
        dscuda::device_free(gpu_left);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Matmul benchmark failed: %s\n", error.what());
        return 1;
    }
}

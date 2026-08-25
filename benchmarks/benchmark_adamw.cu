// Runs one vectorized FP32 AdamW parameter update for external Nsight Compute measurement.
// The benchmark profiles parameter, gradient, and two moment streams without using CUDA events or internal timing loops.

#include "cuda_common.h"
#include "optimizer.h"

#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char** argv) {
    try {
        const int elements = argc > 1 ? std::atoi(argv[1]) : 1 << 22;
        const std::size_t bytes =
            static_cast<std::size_t>(elements) * sizeof(float);
        const dscuda::AdamWConfig config{
            3.0e-4F, 0.9F, 0.95F, 1.0e-8F, 0.1F};

        auto* parameters =
            static_cast<float*>(dscuda::device_malloc(bytes));
        auto* gradients =
            static_cast<float*>(dscuda::device_malloc(bytes));
        auto* first_moment =
            static_cast<float*>(dscuda::device_malloc(bytes));
        auto* second_moment =
            static_cast<float*>(dscuda::device_malloc(bytes));
        CUDA_CHECK(cudaMemset(parameters, 0x3f, bytes));
        CUDA_CHECK(cudaMemset(gradients, 0x3d, bytes));
        CUDA_CHECK(cudaMemset(first_moment, 0, bytes));
        CUDA_CHECK(cudaMemset(second_moment, 0, bytes));

        std::printf("AdamW workload: elements=%d\n", elements);
        CUDA_CHECK(cudaProfilerStart());
        dscuda::adamw_step_cuda(
            parameters,
            first_moment,
            second_moment,
            gradients,
            elements,
            1,
            config);
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(second_moment);
        dscuda::device_free(first_moment);
        dscuda::device_free(gradients);
        dscuda::device_free(parameters);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "AdamW benchmark failed: %s\n", error.what());
        return 1;
    }
}

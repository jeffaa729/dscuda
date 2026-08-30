// Runs flattened global gradient norm reduction and in-place clipping for external Nsight Compute measurement.
// The norm remains on the device between reduction and clipping so the benchmark includes only the GPU operator path.

#include "cuda_common.h"
#include "global_norm.h"

#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char** argv) {
    try {
        const int elements = argc > 1 ? std::atoi(argv[1]) : 1 << 22;
        const std::size_t bytes =
            static_cast<std::size_t>(elements) * sizeof(float);
        auto* gradients =
            static_cast<float*>(dscuda::device_malloc(bytes));
        auto* norm = static_cast<float*>(
            dscuda::device_malloc(sizeof(float)));
        const std::size_t workspace_elements =
            dscuda::global_norm_workspace_elements(elements);
        auto* workspace = static_cast<float*>(
            dscuda::device_malloc(workspace_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(gradients, 0x3d, bytes));

        std::printf("Global norm workload: elements=%d\n", elements);
        CUDA_CHECK(cudaProfilerStart());
        dscuda::global_norm_cuda(
            norm, gradients, workspace, elements);
        dscuda::clip_gradients_cuda(
            gradients, norm, elements, 1.0F);
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(workspace);
        dscuda::device_free(norm);
        dscuda::device_free(gradients);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "Global norm benchmark failed: %s\n", error.what());
        return 1;
    }
}

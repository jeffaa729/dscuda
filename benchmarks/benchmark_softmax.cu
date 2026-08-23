// Runs one FP32 scaled causal-softmax forward and backward workload for external measurement by Nsight Compute.
// Timing and hardware metrics are intentionally collected by the profiler rather than CUDA events in this executable.

#include "cuda_common.h"
#include "softmax.h"

#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 4;
        const int heads = argc > 2 ? std::atoi(argv[2]) : 16;
        const int sequence_length = argc > 3 ? std::atoi(argv[3]) : 512;
        const float scale = argc > 4 ? std::strtof(argv[4], nullptr) : 0.125F;
        const size_t elements = static_cast<size_t>(batch_size) * heads *
                                sequence_length * sequence_length;

        std::vector<float> logits(elements, 0.5F);
        std::vector<float> probabilities_gradient(elements, 0.125F);

        auto* gpu_logits =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_probabilities =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_probabilities_gradient =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_logits_gradient =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            gpu_logits, logits.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_probabilities_gradient,
            probabilities_gradient.data(),
            elements * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(gpu_logits_gradient, 0, elements * sizeof(float)));

        std::printf(
            "Causal softmax workload: batch=%d heads=%d sequence=%d scale=%g\n",
            batch_size,
            heads,
            sequence_length,
            scale);
        CUDA_CHECK(cudaProfilerStart());
        dscuda::causal_softmax_forward_cuda(
            gpu_probabilities,
            gpu_logits,
            batch_size,
            heads,
            sequence_length,
            scale);
        dscuda::causal_softmax_backward_cuda(
            gpu_logits_gradient,
            gpu_probabilities_gradient,
            gpu_probabilities,
            batch_size,
            heads,
            sequence_length,
            scale);
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(gpu_logits_gradient);
        dscuda::device_free(gpu_probabilities_gradient);
        dscuda::device_free(gpu_probabilities);
        dscuda::device_free(gpu_logits);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Causal softmax benchmark failed: %s\n", error.what());
        return 1;
    }
}

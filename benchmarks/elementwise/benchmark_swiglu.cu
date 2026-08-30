// Runs one SwiGLU forward and backward workload for external measurement by Nsight Compute.
// Timing and hardware metrics are intentionally collected by the profiler rather than CUDA events in this executable.

#include "cuda_common.h"
#include "swiglu.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int elements = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
        std::vector<float> gate(elements, 0.5F);
        std::vector<float> up(elements, 0.25F);
        std::vector<float> output_gradient(elements, 1.0F);

        auto* gpu_gate = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_up = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_output_gradient =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_output = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_gate_gradient =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
        auto* gpu_up_gradient =
            static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            gpu_gate, gate.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_up, up.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_output_gradient,
            output_gradient.data(),
            elements * sizeof(float),
            cudaMemcpyHostToDevice));

        std::printf("SwiGLU workload: elements=%d\n", elements);
        dscuda::swiglu_forward_cuda(gpu_output, gpu_gate, gpu_up, elements);
        dscuda::swiglu_backward_cuda(
            gpu_gate_gradient,
            gpu_up_gradient,
            gpu_output_gradient,
            gpu_gate,
            gpu_up,
            elements);
        dscuda::synchronize();

        dscuda::device_free(gpu_up_gradient);
        dscuda::device_free(gpu_gate_gradient);
        dscuda::device_free(gpu_output);
        dscuda::device_free(gpu_output_gradient);
        dscuda::device_free(gpu_up);
        dscuda::device_free(gpu_gate);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "SwiGLU benchmark failed: %s\n", error.what());
        return 1;
    }
}

// Runs one FP32 Rotary Positional Embedding forward and backward workload for external measurement by Nsight Compute.
// Timing and hardware metrics are intentionally collected by the profiler rather than CUDA events in this executable.

#include "cuda_common.h"
#include "rope.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 4;
        const int sequence_length = argc > 2 ? std::atoi(argv[2]) : 512;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 16;
        const int head_size = argc > 4 ? std::atoi(argv[4]) : 128;
        const int rotary_size = argc > 5 ? std::atoi(argv[5]) : 64;
        const size_t activation_elements =
            static_cast<size_t>(batch_size) * sequence_length * heads * head_size;
        const size_t frequency_elements =
            static_cast<size_t>(sequence_length) * (rotary_size / 2);

        std::vector<float> input(activation_elements, 0.5F);
        std::vector<float> output_gradient(activation_elements, 0.125F);
        std::vector<float> cosine(frequency_elements);
        std::vector<float> sine(frequency_elements);
        for (size_t index = 0; index < frequency_elements; ++index) {
            const float angle = 0.001F * static_cast<float>(index);
            cosine[index] = std::cos(angle);
            sine[index] = std::sin(angle);
        }

        auto* gpu_input =
            static_cast<float*>(dscuda::device_malloc(activation_elements * sizeof(float)));
        auto* gpu_output =
            static_cast<float*>(dscuda::device_malloc(activation_elements * sizeof(float)));
        auto* gpu_output_gradient =
            static_cast<float*>(dscuda::device_malloc(activation_elements * sizeof(float)));
        auto* gpu_input_gradient =
            static_cast<float*>(dscuda::device_malloc(activation_elements * sizeof(float)));
        auto* gpu_cosine =
            static_cast<float*>(dscuda::device_malloc(frequency_elements * sizeof(float)));
        auto* gpu_sine =
            static_cast<float*>(dscuda::device_malloc(frequency_elements * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            gpu_input, input.data(), activation_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_output_gradient,
            output_gradient.data(),
            activation_elements * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_cosine, cosine.data(), frequency_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_sine, sine.data(), frequency_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(gpu_input_gradient, 0, activation_elements * sizeof(float)));

        std::printf(
            "RoPE workload: batch=%d sequence=%d heads=%d head_size=%d rotary_size=%d\n",
            batch_size,
            sequence_length,
            heads,
            head_size,
            rotary_size);
        dscuda::rope_forward_cuda(
            gpu_output,
            gpu_input,
            gpu_cosine,
            gpu_sine,
            batch_size,
            sequence_length,
            heads,
            head_size,
            rotary_size);
        dscuda::rope_backward_cuda(
            gpu_input_gradient,
            gpu_output_gradient,
            gpu_cosine,
            gpu_sine,
            batch_size,
            sequence_length,
            heads,
            head_size,
            rotary_size);
        dscuda::synchronize();

        dscuda::device_free(gpu_sine);
        dscuda::device_free(gpu_cosine);
        dscuda::device_free(gpu_input_gradient);
        dscuda::device_free(gpu_output_gradient);
        dscuda::device_free(gpu_output);
        dscuda::device_free(gpu_input);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RoPE benchmark failed: %s\n", error.what());
        return 1;
    }
}

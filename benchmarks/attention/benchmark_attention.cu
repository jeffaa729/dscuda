// Runs one composed FP32 dense-attention forward and backward workload for Nsight Compute.
// Timing and memory metrics are collected externally, so this executable only brackets the kernels with profiler control.

#include "attention.h"
#include "cuda_common.h"

#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 2;
        const int sequence_length = argc > 2 ? std::atoi(argv[2]) : 256;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 8;
        const int head_size = argc > 4 ? std::atoi(argv[4]) : 64;
        const float scale = argc > 5
            ? std::strtof(argv[5], nullptr)
            : 1.0F / std::sqrt(static_cast<float>(head_size));
        const std::size_t activations =
            static_cast<std::size_t>(batch_size) * sequence_length * heads *
            head_size;
        const std::size_t probabilities =
            static_cast<std::size_t>(batch_size) * heads * sequence_length *
            sequence_length;

        std::vector<float> query(activations, 0.125F);
        std::vector<float> key(activations, -0.0625F);
        std::vector<float> value(activations, 0.25F);
        std::vector<float> output_gradient(activations, 0.03125F);

        auto* gpu_query =
            static_cast<float*>(dscuda::device_malloc(activations * sizeof(float)));
        auto* gpu_key =
            static_cast<float*>(dscuda::device_malloc(activations * sizeof(float)));
        auto* gpu_value =
            static_cast<float*>(dscuda::device_malloc(activations * sizeof(float)));
        auto* gpu_output_gradient =
            static_cast<float*>(dscuda::device_malloc(activations * sizeof(float)));
        auto* gpu_output =
            static_cast<float*>(dscuda::device_malloc(activations * sizeof(float)));
        auto* gpu_probabilities =
            static_cast<float*>(dscuda::device_malloc(probabilities * sizeof(float)));
        auto* gpu_query_gradient =
            static_cast<float*>(dscuda::device_malloc(activations * sizeof(float)));
        auto* gpu_key_gradient =
            static_cast<float*>(dscuda::device_malloc(activations * sizeof(float)));
        auto* gpu_value_gradient =
            static_cast<float*>(dscuda::device_malloc(activations * sizeof(float)));

        const std::size_t forward_workspace_elements =
            dscuda::dense_attention_forward_workspace_elements(
                batch_size, sequence_length, heads, head_size);
        const std::size_t backward_workspace_elements =
            dscuda::dense_attention_backward_workspace_elements(
                batch_size, sequence_length, heads, head_size);
        auto* gpu_forward_workspace = static_cast<float*>(
            dscuda::device_malloc(forward_workspace_elements * sizeof(float)));
        auto* gpu_backward_workspace = static_cast<float*>(
            dscuda::device_malloc(backward_workspace_elements * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            gpu_query,
            query.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_key,
            key.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_value,
            value.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_output_gradient,
            output_gradient.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(gpu_query_gradient, 0, activations * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpu_key_gradient, 0, activations * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpu_value_gradient, 0, activations * sizeof(float)));

        std::printf(
            "Dense attention workload: batch=%d sequence=%d heads=%d head_size=%d scale=%g\n",
            batch_size,
            sequence_length,
            heads,
            head_size,
            scale);
        CUDA_CHECK(cudaProfilerStart());
        dscuda::dense_attention_forward_cuda(
            gpu_output,
            gpu_probabilities,
            gpu_query,
            gpu_key,
            gpu_value,
            gpu_forward_workspace,
            batch_size,
            sequence_length,
            heads,
            head_size,
            scale);
        dscuda::dense_attention_backward_cuda(
            gpu_query_gradient,
            gpu_key_gradient,
            gpu_value_gradient,
            gpu_output_gradient,
            gpu_probabilities,
            gpu_query,
            gpu_key,
            gpu_value,
            gpu_backward_workspace,
            batch_size,
            sequence_length,
            heads,
            head_size,
            scale);
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(gpu_backward_workspace);
        dscuda::device_free(gpu_forward_workspace);
        dscuda::device_free(gpu_value_gradient);
        dscuda::device_free(gpu_key_gradient);
        dscuda::device_free(gpu_query_gradient);
        dscuda::device_free(gpu_probabilities);
        dscuda::device_free(gpu_output);
        dscuda::device_free(gpu_output_gradient);
        dscuda::device_free(gpu_value);
        dscuda::device_free(gpu_key);
        dscuda::device_free(gpu_query);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Dense attention benchmark failed: %s\n", error.what());
        return 1;
    }
}

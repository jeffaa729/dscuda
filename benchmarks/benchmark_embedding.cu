// Runs token-embedding lookup and accumulated table-gradient workloads for external Nsight Compute measurement.
// The token stream intentionally reuses a working vocabulary subset so the backward profile includes realistic atomic collisions.

#include "cuda_common.h"
#include "embedding.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int token_count = argc > 1 ? std::atoi(argv[1]) : 2048;
        const int hidden_size = argc > 2 ? std::atoi(argv[2]) : 1024;
        const int vocabulary_size = argc > 3 ? std::atoi(argv[3]) : 32768;
        const int active_vocabulary = std::min(vocabulary_size, 2048);
        const std::size_t activation_elements =
            static_cast<std::size_t>(token_count) * hidden_size;
        const std::size_t weight_elements =
            static_cast<std::size_t>(vocabulary_size) * hidden_size;

        std::vector<int> token_ids(token_count);
        for (int token = 0; token < token_count; ++token) {
            token_ids[token] = (token * 7919 + 17) % active_vocabulary;
        }

        auto* gpu_token_ids = static_cast<int*>(
            dscuda::device_malloc(token_count * sizeof(int)));
        auto* gpu_weight = static_cast<float*>(
            dscuda::device_malloc(weight_elements * sizeof(float)));
        auto* gpu_output_gradient = static_cast<float*>(
            dscuda::device_malloc(activation_elements * sizeof(float)));
        auto* gpu_output = static_cast<float*>(
            dscuda::device_malloc(activation_elements * sizeof(float)));
        auto* gpu_weight_gradient = static_cast<float*>(
            dscuda::device_malloc(weight_elements * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            gpu_token_ids,
            token_ids.data(),
            token_count * sizeof(int),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(
            gpu_weight, 0, weight_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            gpu_output_gradient, 0, activation_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            gpu_weight_gradient, 0, weight_elements * sizeof(float)));

        std::printf(
            "Embedding workload: tokens=%d hidden=%d vocabulary=%d active_vocabulary=%d\n",
            token_count,
            hidden_size,
            vocabulary_size,
            active_vocabulary);
        CUDA_CHECK(cudaProfilerStart());
        dscuda::embedding_forward_cuda(
            gpu_output,
            gpu_token_ids,
            gpu_weight,
            token_count,
            hidden_size);
        dscuda::embedding_backward_cuda(
            gpu_weight_gradient,
            gpu_output_gradient,
            gpu_token_ids,
            token_count,
            hidden_size);
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(gpu_weight_gradient);
        dscuda::device_free(gpu_output);
        dscuda::device_free(gpu_output_gradient);
        dscuda::device_free(gpu_weight);
        dscuda::device_free(gpu_token_ids);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Embedding benchmark failed: %s\n", error.what());
        return 1;
    }
}

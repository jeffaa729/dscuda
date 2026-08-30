// Runs FlashMLA-inspired split-KV decode on one compressed BF16 cache for Nsight Compute.
// Only the split and combine kernels are profiled so context-length scaling, cache traffic, occupancy, registers, and spills remain directly interpretable.

#include "cuda_common.h"
#include "mla.h"

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 2;
        const int context_length = argc > 2 ? std::atoi(argv[2]) : 1024;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 16;
        const int kv_rank = argc > 4 ? std::atoi(argv[4]) : 512;
        const int rope_size = argc > 5 ? std::atoi(argv[5]) : 64;
        const int splits = argc > 6 ? std::atoi(argv[6]) : 8;
        const float scale =
            1.0F / std::sqrt(static_cast<float>(kv_rank + rope_size));
        const std::size_t query_elements =
            static_cast<std::size_t>(batch_size) * heads * kv_rank;
        const std::size_t query_rope_elements =
            static_cast<std::size_t>(batch_size) * heads * rope_size;
        const std::size_t kv_elements =
            static_cast<std::size_t>(batch_size) * context_length * kv_rank;
        const std::size_t key_rope_elements =
            static_cast<std::size_t>(batch_size) * context_length * rope_size;
        const std::size_t output_elements = query_elements;
        const std::size_t rows =
            static_cast<std::size_t>(batch_size) * heads;
        const std::size_t workspace_elements =
            dscuda::mla_decode_workspace_elements(
                batch_size, heads, splits, kv_rank);

        std::vector<__nv_bfloat16> host_query(
            query_elements, __float2bfloat16(0.125F));
        std::vector<__nv_bfloat16> host_query_rope(
            query_rope_elements, __float2bfloat16(-0.0625F));
        std::vector<__nv_bfloat16> host_kv(
            kv_elements, __float2bfloat16(0.25F));
        std::vector<__nv_bfloat16> host_key_rope(
            key_rope_elements, __float2bfloat16(0.03125F));
        std::vector<int> host_lengths(batch_size, context_length);

        auto* query = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(query_elements * sizeof(__nv_bfloat16)));
        auto* query_rope = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(
                query_rope_elements * sizeof(__nv_bfloat16)));
        auto* kv = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(kv_elements * sizeof(__nv_bfloat16)));
        auto* key_rope = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(
                key_rope_elements * sizeof(__nv_bfloat16)));
        auto* lengths = static_cast<int*>(
            dscuda::device_malloc(batch_size * sizeof(int)));
        auto* output = static_cast<float*>(
            dscuda::device_malloc(output_elements * sizeof(float)));
        auto* logsumexp = static_cast<float*>(
            dscuda::device_malloc(rows * sizeof(float)));
        auto* workspace = static_cast<float*>(
            dscuda::device_malloc(workspace_elements * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            query, host_query.data(), query_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            query_rope, host_query_rope.data(),
            query_rope_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            kv, host_kv.data(), kv_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            key_rope, host_key_rope.data(),
            key_rope_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            lengths, host_lengths.data(), batch_size * sizeof(int),
            cudaMemcpyHostToDevice));

        auto run = [&]() {
            dscuda::mla_decode_forward_cuda(
                output, logsumexp, query, query_rope, kv, key_rope, lengths,
                workspace, batch_size, context_length, heads, kv_rank,
                rope_size, splits, scale);
        };
        run();
        dscuda::synchronize();

        std::printf(
            "MLA decode workload: batch=%d context=%d heads=%d kv_rank=%d rope=%d splits=%d\n",
            batch_size,
            context_length,
            heads,
            kv_rank,
            rope_size,
            splits);
        CUDA_CHECK(cudaProfilerStart());
        run();
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(workspace);
        dscuda::device_free(logsumexp);
        dscuda::device_free(output);
        dscuda::device_free(lengths);
        dscuda::device_free(key_rope);
        dscuda::device_free(kv);
        dscuda::device_free(query_rope);
        dscuda::device_free(query);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "MLA decode benchmark failed: %s\n", error.what());
        return 1;
    }
}

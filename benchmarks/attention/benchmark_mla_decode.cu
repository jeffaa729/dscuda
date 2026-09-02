// Profiles split-KV decode with the same packed query and paged cache contract as FlashMLA.
// The split and combine kernels remain the only profiled operations.

#include "cuda_common.h"
#include "mla.h"

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 2;
        const int context_length = argc > 2 ? std::atoi(argv[2]) : 1024;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 16;
        const int kv_rank = argc > 4 ? std::atoi(argv[4]) : 512;
        const int rope_size = argc > 5 ? std::atoi(argv[5]) : 64;
        const int splits = argc > 6 ? std::atoi(argv[6]) : 8;
        constexpr int page_size = 64;
        const int pages_per_sequence =
            (context_length + page_size - 1) / page_size;
        const int packed_width = kv_rank + rope_size;
        const float scale = 1.0F / std::sqrt(static_cast<float>(packed_width));
        const std::size_t query_elements =
            static_cast<std::size_t>(batch_size) * heads * packed_width;
        const std::size_t cache_elements =
            static_cast<std::size_t>(batch_size) * pages_per_sequence *
            page_size * packed_width;
        const std::size_t output_elements =
            static_cast<std::size_t>(batch_size) * heads * kv_rank;
        const std::size_t rows = static_cast<std::size_t>(batch_size) * heads;
        const std::size_t workspace_elements =
            dscuda::mla_decode_workspace_elements(
                batch_size, heads, splits, kv_rank);

        std::vector<__nv_bfloat16> host_query(
            query_elements, __float2bfloat16(0.125F));
        std::vector<__nv_bfloat16> host_cache(
            cache_elements, __float2bfloat16(0.25F));
        std::vector<int> host_table(batch_size * pages_per_sequence);
        std::iota(host_table.begin(), host_table.end(), 0);
        std::vector<int> host_lengths(batch_size, context_length);

        auto* query = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(query_elements * sizeof(__nv_bfloat16)));
        auto* cache = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(cache_elements * sizeof(__nv_bfloat16)));
        auto* block_table = static_cast<int*>(dscuda::device_malloc(
            host_table.size() * sizeof(int)));
        auto* lengths = static_cast<int*>(
            dscuda::device_malloc(batch_size * sizeof(int)));
        auto* output = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(output_elements * sizeof(__nv_bfloat16)));
        auto* logsumexp = static_cast<float*>(
            dscuda::device_malloc(rows * sizeof(float)));
        auto* workspace = static_cast<float*>(
            dscuda::device_malloc(workspace_elements * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            query, host_query.data(), query_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            cache, host_cache.data(), cache_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            block_table, host_table.data(), host_table.size() * sizeof(int),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            lengths, host_lengths.data(), batch_size * sizeof(int),
            cudaMemcpyHostToDevice));

        auto run = [&]() {
            dscuda::mla_decode_forward_cuda(
                output, logsumexp, query, cache, block_table, lengths,
                workspace, batch_size, heads, kv_rank, rope_size, page_size,
                pages_per_sequence, splits, scale);
        };
        run();
        dscuda::synchronize();

        std::printf(
            "MLA decode workload: batch=%d context=%d heads=%d kv_rank=%d rope=%d page=%d splits=%d\n",
            batch_size, context_length, heads, kv_rank, rope_size, page_size,
            splits);
        CUDA_CHECK(cudaProfilerStart());
        run();
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(workspace);
        dscuda::device_free(logsumexp);
        dscuda::device_free(output);
        dscuda::device_free(lengths);
        dscuda::device_free(block_table);
        dscuda::device_free(cache);
        dscuda::device_free(query);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MLA decode benchmark failed: %s\n", error.what());
        return 1;
    }
}

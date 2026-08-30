// Runs the selected BF16 compressed-MLA forward and backward kernels for external Nsight Compute measurement.
// The profiled region contains causal Tensor Core forward plus both backward ownership kernels so the report exposes end-to-end time, traffic, occupancy, registers, and spills.

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
        const int sequence_length = argc > 2 ? std::atoi(argv[2]) : 256;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 4;
        const int kv_rank = argc > 4 ? std::atoi(argv[4]) : 64;
        const int rope_size = argc > 5 ? std::atoi(argv[5]) : 32;
        const float scale =
            1.0F / std::sqrt(static_cast<float>(kv_rank + rope_size));
        const std::size_t query_elements =
            static_cast<std::size_t>(batch_size) * sequence_length * heads
            * kv_rank;
        const std::size_t query_rope_elements =
            static_cast<std::size_t>(batch_size) * sequence_length * heads
            * rope_size;
        const std::size_t kv_elements =
            static_cast<std::size_t>(batch_size) * sequence_length * kv_rank;
        const std::size_t key_rope_elements =
            static_cast<std::size_t>(batch_size) * sequence_length * rope_size;
        const std::size_t rows =
            static_cast<std::size_t>(batch_size) * heads * sequence_length;

        std::vector<__nv_bfloat16> host_query(
            query_elements, __float2bfloat16(0.125F));
        std::vector<__nv_bfloat16> host_query_rope(
            query_rope_elements, __float2bfloat16(-0.0625F));
        std::vector<__nv_bfloat16> host_kv(
            kv_elements, __float2bfloat16(0.25F));
        std::vector<__nv_bfloat16> host_key_rope(
            key_rope_elements, __float2bfloat16(0.03125F));
        std::vector<float> host_output_gradient(query_elements, 0.015625F);

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
        auto* output = static_cast<float*>(
            dscuda::device_malloc(query_elements * sizeof(float)));
        auto* logsumexp = static_cast<float*>(
            dscuda::device_malloc(rows * sizeof(float)));
        auto* output_gradient = static_cast<float*>(
            dscuda::device_malloc(query_elements * sizeof(float)));
        auto* query_gradient = static_cast<float*>(
            dscuda::device_malloc(query_elements * sizeof(float)));
        auto* query_rope_gradient = static_cast<float*>(
            dscuda::device_malloc(query_rope_elements * sizeof(float)));
        auto* kv_gradient = static_cast<float*>(
            dscuda::device_malloc(kv_elements * sizeof(float)));
        auto* key_rope_gradient = static_cast<float*>(
            dscuda::device_malloc(key_rope_elements * sizeof(float)));

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
            output_gradient, host_output_gradient.data(),
            query_elements * sizeof(float), cudaMemcpyHostToDevice));

        auto zero_gradients = [&]() {
            CUDA_CHECK(cudaMemset(
                query_gradient, 0, query_elements * sizeof(float)));
            CUDA_CHECK(cudaMemset(
                query_rope_gradient, 0,
                query_rope_elements * sizeof(float)));
            CUDA_CHECK(cudaMemset(
                kv_gradient, 0, kv_elements * sizeof(float)));
            CUDA_CHECK(cudaMemset(
                key_rope_gradient, 0,
                key_rope_elements * sizeof(float)));
        };
        auto run = [&]() {
            dscuda::mla_compressed_attention_forward_cuda(
                output, logsumexp, query, query_rope, kv, key_rope,
                batch_size, sequence_length, heads, kv_rank, rope_size,
                scale);
            dscuda::mla_compressed_attention_backward_cuda(
                query_gradient, query_rope_gradient, kv_gradient,
                key_rope_gradient, output_gradient, output, logsumexp, query,
                query_rope, kv, key_rope, batch_size, sequence_length, heads,
                kv_rank, rope_size, scale);
        };

        zero_gradients();
        run();
        dscuda::synchronize();
        zero_gradients();
        dscuda::synchronize();

        std::printf(
            "MLA workload: batch=%d sequence=%d heads=%d kv_rank=%d rope=%d\n",
            batch_size,
            sequence_length,
            heads,
            kv_rank,
            rope_size);
        CUDA_CHECK(cudaProfilerStart());
        run();
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(key_rope_gradient);
        dscuda::device_free(kv_gradient);
        dscuda::device_free(query_rope_gradient);
        dscuda::device_free(query_gradient);
        dscuda::device_free(output_gradient);
        dscuda::device_free(logsumexp);
        dscuda::device_free(output);
        dscuda::device_free(key_rope);
        dscuda::device_free(kv);
        dscuda::device_free(query_rope);
        dscuda::device_free(query);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MLA benchmark failed: %s\n", error.what());
        return 1;
    }
}

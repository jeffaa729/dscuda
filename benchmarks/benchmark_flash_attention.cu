// Runs one fused causal attention forward and backward workload for Nsight Compute using either FP32 or BF16 Q/K/V.
// The executable brackets only the three attention kernels, while the profiling script extracts duration, bandwidth, occupancy, and cache behavior.

#include "cuda_common.h"
#include "flash_attention.h"

#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 2;
        const int sequence_length = argc > 2 ? std::atoi(argv[2]) : 256;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 8;
        const int head_size = argc > 4 ? std::atoi(argv[4]) : 64;
        const char* precision = argc > 5 ? argv[5] : "bf16";
        const bool bf16 = std::strcmp(precision, "bf16") == 0;
        if (!bf16 && std::strcmp(precision, "fp32") != 0) {
            throw std::runtime_error("precision must be fp32 or bf16");
        }
        const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
        const std::size_t activations =
            static_cast<std::size_t>(batch_size) * sequence_length * heads
            * head_size;
        const std::size_t rows =
            static_cast<std::size_t>(batch_size) * heads * sequence_length;

        std::vector<float> host_query(activations, 0.125F);
        std::vector<float> host_key(activations, -0.0625F);
        std::vector<float> host_value(activations, 0.25F);
        std::vector<float> host_output_gradient(activations, 0.03125F);
        std::vector<__nv_bfloat16> host_bf16_query(
            activations, __float2bfloat16(0.125F));
        std::vector<__nv_bfloat16> host_bf16_key(
            activations, __float2bfloat16(-0.0625F));
        std::vector<__nv_bfloat16> host_bf16_value(
            activations, __float2bfloat16(0.25F));

        auto* query = static_cast<float*>(
            dscuda::device_malloc(activations * sizeof(float)));
        auto* key = static_cast<float*>(
            dscuda::device_malloc(activations * sizeof(float)));
        auto* value = static_cast<float*>(
            dscuda::device_malloc(activations * sizeof(float)));
        auto* bf16_query = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* bf16_key = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* bf16_value = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* output_gradient = static_cast<float*>(
            dscuda::device_malloc(activations * sizeof(float)));
        auto* output = static_cast<float*>(
            dscuda::device_malloc(activations * sizeof(float)));
        auto* logsumexp = static_cast<float*>(
            dscuda::device_malloc(rows * sizeof(float)));
        auto* query_gradient = static_cast<float*>(
            dscuda::device_malloc(activations * sizeof(float)));
        auto* key_gradient = static_cast<float*>(
            dscuda::device_malloc(activations * sizeof(float)));
        auto* value_gradient = static_cast<float*>(
            dscuda::device_malloc(activations * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            query,
            host_query.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            key,
            host_key.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            value,
            host_value.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            bf16_query,
            host_bf16_query.data(),
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            bf16_key,
            host_bf16_key.data(),
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            bf16_value,
            host_bf16_value.data(),
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            output_gradient,
            host_output_gradient.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));

        auto run = [&]() {
            if (bf16) {
                dscuda::flash_attention_forward_bf16_cuda(
                    output,
                    logsumexp,
                    bf16_query,
                    bf16_key,
                    bf16_value,
                    batch_size,
                    sequence_length,
                    heads,
                    head_size,
                    scale);
                dscuda::flash_attention_backward_bf16_cuda(
                    query_gradient,
                    key_gradient,
                    value_gradient,
                    output_gradient,
                    output,
                    logsumexp,
                    bf16_query,
                    bf16_key,
                    bf16_value,
                    batch_size,
                    sequence_length,
                    heads,
                    head_size,
                    scale);
            } else {
                dscuda::flash_attention_forward_cuda(
                    output,
                    logsumexp,
                    query,
                    key,
                    value,
                    batch_size,
                    sequence_length,
                    heads,
                    head_size,
                    scale);
                dscuda::flash_attention_backward_cuda(
                    query_gradient,
                    key_gradient,
                    value_gradient,
                    output_gradient,
                    output,
                    logsumexp,
                    query,
                    key,
                    value,
                    batch_size,
                    sequence_length,
                    heads,
                    head_size,
                    scale);
            }
        };

        CUDA_CHECK(cudaMemset(query_gradient, 0, activations * sizeof(float)));
        CUDA_CHECK(cudaMemset(key_gradient, 0, activations * sizeof(float)));
        CUDA_CHECK(cudaMemset(value_gradient, 0, activations * sizeof(float)));
        run();
        dscuda::synchronize();
        CUDA_CHECK(cudaMemset(query_gradient, 0, activations * sizeof(float)));
        CUDA_CHECK(cudaMemset(key_gradient, 0, activations * sizeof(float)));
        CUDA_CHECK(cudaMemset(value_gradient, 0, activations * sizeof(float)));
        dscuda::synchronize();

        std::printf(
            "Flash attention workload: batch=%d sequence=%d heads=%d head_size=%d precision=%s\n",
            batch_size,
            sequence_length,
            heads,
            head_size,
            precision);
        CUDA_CHECK(cudaProfilerStart());
        run();
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(value_gradient);
        dscuda::device_free(key_gradient);
        dscuda::device_free(query_gradient);
        dscuda::device_free(logsumexp);
        dscuda::device_free(output);
        dscuda::device_free(output_gradient);
        dscuda::device_free(bf16_value);
        dscuda::device_free(bf16_key);
        dscuda::device_free(bf16_query);
        dscuda::device_free(value);
        dscuda::device_free(key);
        dscuda::device_free(query);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Flash attention benchmark failed: %s\n", error.what());
        return 1;
    }
}

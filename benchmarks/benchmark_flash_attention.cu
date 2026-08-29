// Runs one BF16 causal FlashAttention workload for a same-shape comparison with the official implementation.
// Forward and backward can be profiled independently, while an optional raw dump supports cross-process correctness checks.

#include "cuda_common.h"
#include "flash_attention.h"

#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool selected(const char* requested, const char* operation) {
    return std::strcmp(requested, "all") == 0
        || std::strcmp(requested, operation) == 0;
}

template <typename T>
void dump_values(std::ofstream& output, const T* device, std::size_t elements) {
    std::vector<T> host(elements);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device,
        elements * sizeof(T),
        cudaMemcpyDeviceToHost));
    output.write(
        reinterpret_cast<const char*>(host.data()),
        static_cast<std::streamsize>(elements * sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 2;
        const int sequence_length = argc > 2 ? std::atoi(argv[2]) : 256;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 8;
        const int head_size = argc > 4 ? std::atoi(argv[4]) : 64;
        const char* operation = argc > 5 ? argv[5] : "all";
        const char* dump_path = argc > 6 ? argv[6] : nullptr;
        if (!selected(operation, "forward")
            && !selected(operation, "backward")) {
            throw std::runtime_error(
                "operation must be forward, backward, or all");
        }

        const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
        const std::size_t activations =
            static_cast<std::size_t>(batch_size) * sequence_length * heads
            * head_size;
        const std::size_t rows =
            static_cast<std::size_t>(batch_size) * heads * sequence_length;

        std::vector<__nv_bfloat16> host_query(activations);
        std::vector<__nv_bfloat16> host_key(activations);
        std::vector<__nv_bfloat16> host_value(activations);
        std::vector<float> host_output_gradient(activations);
        for (std::size_t index = 0; index < activations; ++index) {
            host_query[index] = __float2bfloat16(
                static_cast<float>(
                    static_cast<int>((index * 17) % 101) - 50) / 64.0F);
            host_key[index] = __float2bfloat16(
                static_cast<float>(
                    static_cast<int>((index * 23) % 97) - 48) / 61.0F);
            host_value[index] = __float2bfloat16(
                static_cast<float>(
                    static_cast<int>((index * 31) % 89) - 44) / 59.0F);
            host_output_gradient[index] = __bfloat162float(
                __float2bfloat16(
                    static_cast<float>(
                        static_cast<int>((index * 37) % 83) - 41)
                    / 67.0F));
        }

        auto* query = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* key = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* value = static_cast<__nv_bfloat16*>(
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
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            key,
            host_key.data(),
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            value,
            host_value.data(),
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            output_gradient,
            host_output_gradient.data(),
            activations * sizeof(float),
            cudaMemcpyHostToDevice));

        auto zero_gradients = [&]() {
            CUDA_CHECK(cudaMemset(
                query_gradient, 0, activations * sizeof(float)));
            CUDA_CHECK(cudaMemset(
                key_gradient, 0, activations * sizeof(float)));
            CUDA_CHECK(cudaMemset(
                value_gradient, 0, activations * sizeof(float)));
        };
        auto forward = [&]() {
            dscuda::flash_attention_forward_bf16_cuda(
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
        };
        auto backward = [&]() {
            dscuda::flash_attention_backward_bf16_cuda(
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
        };
        auto run_selected = [&]() {
            if (selected(operation, "forward")) {
                forward();
            }
            if (selected(operation, "backward")) {
                backward();
            }
        };

        // Backward consumes output and log-sum-exp produced by forward, but
        // that setup launch stays outside backward-only profiler capture.
        forward();
        dscuda::synchronize();
        zero_gradients();
        run_selected();
        dscuda::synchronize();
        zero_gradients();
        dscuda::synchronize();

        std::printf(
            "FlashAttention workload: B=%d T=%d H=%d D=%d operation=%s\n",
            batch_size,
            sequence_length,
            heads,
            head_size,
            operation);
        CUDA_CHECK(cudaProfilerStart());
        run_selected();
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        if (dump_path != nullptr) {
            forward();
            zero_gradients();
            backward();
            dscuda::synchronize();
            std::ofstream dump(dump_path, std::ios::binary | std::ios::trunc);
            if (!dump) {
                throw std::runtime_error(
                    std::string("cannot open dump file: ") + dump_path);
            }
            dump_values(dump, output, activations);
            dump_values(dump, query_gradient, activations);
            dump_values(dump, key_gradient, activations);
            dump_values(dump, value_gradient, activations);
        }

        dscuda::device_free(value_gradient);
        dscuda::device_free(key_gradient);
        dscuda::device_free(query_gradient);
        dscuda::device_free(logsumexp);
        dscuda::device_free(output);
        dscuda::device_free(output_gradient);
        dscuda::device_free(value);
        dscuda::device_free(key);
        dscuda::device_free(query);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "FlashAttention benchmark failed: %s\n", error.what());
        return 1;
    }
}

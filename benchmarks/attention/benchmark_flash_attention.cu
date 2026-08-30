// Runs one D128 BF16 causal FlashAttention workload for a same-shape comparison with the official implementation.
// Forward and backward can be profiled or timed independently, while an optional raw dump supports cross-process correctness checks.

#include "cuda_common.h"
#include "flash_attention.h"

#include <cuda_profiler_api.h>

#include <algorithm>
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

struct TimingOptions {
    bool enabled = false;
    int warmup = 10;
    int iterations = 50;
    int trials = 5;
};

struct TimingResult {
    float median_ms;
    float minimum_ms;
    float maximum_ms;
};

bool selected(const char* requested, const char* operation) {
    return std::strcmp(requested, "all") == 0
        || std::strcmp(requested, operation) == 0;
}

void dump_values(std::ofstream& output, const __nv_bfloat16* device, std::size_t elements) {
    std::vector<__nv_bfloat16> host(elements);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device,
        elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));
    std::vector<float> decoded(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        decoded[index] = __bfloat162float(host[index]);
    }
    output.write(
        reinterpret_cast<const char*>(decoded.data()),
        static_cast<std::streamsize>(elements * sizeof(float)));
}

template <typename Operation>
TimingResult measure_gpu(
    Operation operation,
    int warmup,
    int iterations,
    int trials) {
    for (int iteration = 0; iteration < warmup; ++iteration) {
        operation();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> measurements(trials);
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            operation();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&measurements[trial], start, stop));
        measurements[trial] /= static_cast<float>(iterations);
    }

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    std::sort(measurements.begin(), measurements.end());
    const float median = trials % 2 == 0
        ? 0.5F * (measurements[trials / 2 - 1] + measurements[trials / 2])
        : measurements[trials / 2];
    return {median, measurements.front(), measurements.back()};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 2;
        const int sequence_length = argc > 2 ? std::atoi(argv[2]) : 256;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 8;
        const int head_size = argc > 4 ? std::atoi(argv[4]) : 128;
        const char* operation = argc > 5 ? argv[5] : "all";
        const char* dump_path = nullptr;
        TimingOptions timing;
        for (int argument = 6; argument < argc; ++argument) {
            if (std::strcmp(argv[argument], "--timing") == 0) {
                timing.enabled = true;
            } else if (std::strcmp(argv[argument], "--warmup") == 0) {
                timing.warmup = std::atoi(argv[++argument]);
            } else if (std::strcmp(argv[argument], "--iterations") == 0) {
                timing.iterations = std::atoi(argv[++argument]);
            } else if (std::strcmp(argv[argument], "--trials") == 0) {
                timing.trials = std::atoi(argv[++argument]);
            } else {
                dump_path = argv[argument];
            }
        }
        if (!selected(operation, "forward")
            && !selected(operation, "backward")) {
            throw std::runtime_error(
                "operation must be forward, backward, or all");
        }
        if (timing.enabled && std::strcmp(operation, "all") == 0) {
            throw std::runtime_error(
                "timing requires forward or backward operation");
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
        std::vector<__nv_bfloat16> host_output_gradient(activations);
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
            host_output_gradient[index] = __float2bfloat16(
                static_cast<float>(static_cast<int>((index * 37) % 83) - 41) / 67.0F);
        }

        auto* query = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* key = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* value = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* output_gradient = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* output = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* logsumexp = static_cast<float*>(
            dscuda::device_malloc(rows * sizeof(float)));
        auto* query_gradient = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* key_gradient = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* value_gradient = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(activations * sizeof(__nv_bfloat16)));

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
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));

        auto forward = [&]() {
            dscuda::flash_attention_forward_bf16_io_cuda(
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
            dscuda::flash_attention_backward_bf16_io_cuda(
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
        run_selected();
        dscuda::synchronize();

        std::printf(
            "FlashAttention workload: B=%d T=%d H=%d D=%d operation=%s\n",
            batch_size,
            sequence_length,
            heads,
            head_size,
            operation);
        if (timing.enabled) {
            const TimingResult result = measure_gpu(
                run_selected,
                timing.warmup,
                timing.iterations,
                timing.trials);
            std::printf(
                "DSCUDA_TIMING {\"backend\":\"custom\",\"mode\":\"native_eager\","
                "\"operation\":\"%s\",\"batch\":%d,\"sequence\":%d,"
                "\"heads\":%d,\"head_size\":%d,\"warmup\":%d,"
                "\"iterations\":%d,\"trials\":%d,"
                "\"median_ms\":%.9g,\"minimum_ms\":%.9g,"
                "\"maximum_ms\":%.9g}\n",
                operation,
                batch_size,
                sequence_length,
                heads,
                head_size,
                timing.warmup,
                timing.iterations,
                timing.trials,
                result.median_ms,
                result.minimum_ms,
                result.maximum_ms);
        } else {
            CUDA_CHECK(cudaProfilerStart());
            run_selected();
            dscuda::synchronize();
            CUDA_CHECK(cudaProfilerStop());
        }

        if (dump_path != nullptr) {
            forward();
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

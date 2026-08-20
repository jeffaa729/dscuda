// Measures RMSNorm forward and backward GPU runtime after warm-up using CUDA events rather than host wall-clock timing.
// It reports the reduction kernels' latency and effective memory bandwidth across several token counts.

#include "cuda_common.h"
#include "rmsnorm.h"

#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kHiddenSize = 512;
constexpr int kWarmupIterations = 10;
constexpr int kBenchmarkIterations = 100;
constexpr float kEpsilon = 1.0e-6F;

struct BenchmarkResult {
    float forward_microseconds;
    float backward_microseconds;
    float forward_bandwidth;
    float backward_bandwidth;
};

float time_forward(
    float* output,
    float* inverse_rms,
    const float* input,
    const float* weight,
    int rows) {
    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        dscuda::rmsnorm_forward_cuda(
            output, inverse_rms, input, weight, rows, kHiddenSize, kEpsilon);
    }
    dscuda::synchronize();

    dscuda::CudaEventTimer timer;
    timer.start();
    for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
        dscuda::rmsnorm_forward_cuda(
            output, inverse_rms, input, weight, rows, kHiddenSize, kEpsilon);
    }
    return timer.stop() * 1000.0F / kBenchmarkIterations;
}

float time_backward(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const float* inverse_rms,
    int rows) {
    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        dscuda::rmsnorm_backward_cuda(
            input_gradient,
            weight_gradient,
            output_gradient,
            input,
            weight,
            inverse_rms,
            rows,
            kHiddenSize);
    }
    dscuda::synchronize();

    dscuda::CudaEventTimer timer;
    timer.start();
    for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
        dscuda::rmsnorm_backward_cuda(
            input_gradient,
            weight_gradient,
            output_gradient,
            input,
            weight,
            inverse_rms,
            rows,
            kHiddenSize);
    }
    return timer.stop() * 1000.0F / kBenchmarkIterations;
}

BenchmarkResult benchmark(int rows) {
    const int elements = rows * kHiddenSize;
    std::vector<float> input(elements, 0.5F);
    std::vector<float> weight(kHiddenSize, 1.0F);
    std::vector<float> output_gradient(elements, 0.25F);

    auto* gpu_input = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
    auto* gpu_weight = static_cast<float*>(dscuda::device_malloc(kHiddenSize * sizeof(float)));
    auto* gpu_output_gradient = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
    auto* gpu_output = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
    auto* gpu_inverse_rms = static_cast<float*>(dscuda::device_malloc(rows * sizeof(float)));
    auto* gpu_input_gradient = static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
    auto* gpu_weight_gradient = static_cast<float*>(dscuda::device_malloc(kHiddenSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(gpu_input, input.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_weight, weight.data(), kHiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_output_gradient,
        output_gradient.data(),
        elements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpu_input_gradient, 0, elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_weight_gradient, 0, kHiddenSize * sizeof(float)));

    dscuda::rmsnorm_forward_cuda(
        gpu_output,
        gpu_inverse_rms,
        gpu_input,
        gpu_weight,
        rows,
        kHiddenSize,
        kEpsilon);

    const float forward_microseconds =
        time_forward(gpu_output, gpu_inverse_rms, gpu_input, gpu_weight, rows);
    const float backward_microseconds = time_backward(
        gpu_input_gradient,
        gpu_weight_gradient,
        gpu_output_gradient,
        gpu_input,
        gpu_weight,
        gpu_inverse_rms,
        rows);

    dscuda::device_free(gpu_weight_gradient);
    dscuda::device_free(gpu_input_gradient);
    dscuda::device_free(gpu_inverse_rms);
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_output_gradient);
    dscuda::device_free(gpu_weight);
    dscuda::device_free(gpu_input);

    const double forward_bytes = static_cast<double>(3LL * elements + rows) * sizeof(float);
    const double backward_bytes = static_cast<double>(7LL * elements + rows) * sizeof(float);
    return {
        forward_microseconds,
        backward_microseconds,
        static_cast<float>(forward_bytes / (forward_microseconds * 1000.0)),
        static_cast<float>(backward_bytes / (backward_microseconds * 1000.0)),
    };
}

void print_result(int rows, const BenchmarkResult& result) {
    std::printf(
        "%8d %10.2f %10.2f %10.2f %10.2f\n",
        rows,
        result.forward_microseconds,
        result.forward_bandwidth,
        result.backward_microseconds,
        result.backward_bandwidth);
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        std::printf("RMSNorm: hidden=%d, warmup=%d, measured iterations=%d\n\n",
                    kHiddenSize,
                    kWarmupIterations,
                    kBenchmarkIterations);
        std::printf("%8s %10s %10s %10s %10s\n",
                    "rows",
                    "fwd us",
                    "fwd GB/s",
                    "bwd us",
                    "bwd GB/s");

        for (const int rows : {512, 2048, 8192}) {
            print_result(rows, benchmark(rows));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RMSNorm benchmark failed: %s\n", error.what());
        return 1;
    }
}

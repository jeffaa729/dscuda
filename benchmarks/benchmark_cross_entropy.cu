// Runs fused vocabulary cross-entropy forward and backward workloads for external Nsight Compute measurement.
// Only per-row log-sum-exp values are saved, so the benchmark excludes a vocabulary-sized probability buffer.

#include "cross_entropy.h"
#include "cuda_common.h"

#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int rows = argc > 1 ? std::atoi(argv[1]) : 2048;
        const int vocabulary_size =
            argc > 2 ? std::atoi(argv[2]) : 32768;
        const std::size_t elements =
            static_cast<std::size_t>(rows) * vocabulary_size;
        const std::size_t bytes = elements * sizeof(float);
        std::vector<int> targets(rows);
        for (int row = 0; row < rows; ++row) {
            targets[row] = (row * 7919 + 17) % vocabulary_size;
        }

        auto* logits =
            static_cast<float*>(dscuda::device_malloc(bytes));
        auto* targets_device = static_cast<int*>(
            dscuda::device_malloc(rows * sizeof(int)));
        auto* mean_loss = static_cast<float*>(
            dscuda::device_malloc(sizeof(float)));
        auto* logsumexp = static_cast<float*>(
            dscuda::device_malloc(rows * sizeof(float)));
        auto* logits_gradient =
            static_cast<float*>(dscuda::device_malloc(bytes));
        CUDA_CHECK(cudaMemset(logits, 0x3f, bytes));
        CUDA_CHECK(cudaMemcpy(
            targets_device,
            targets.data(),
            rows * sizeof(int),
            cudaMemcpyHostToDevice));

        std::printf(
            "Cross-entropy workload: rows=%d vocabulary=%d\n",
            rows,
            vocabulary_size);
        CUDA_CHECK(cudaProfilerStart());
        dscuda::cross_entropy_forward_cuda(
            mean_loss,
            logsumexp,
            logits,
            targets_device,
            rows,
            vocabulary_size);
        dscuda::cross_entropy_backward_cuda(
            logits_gradient,
            logits,
            logsumexp,
            targets_device,
            rows,
            vocabulary_size);
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(logits_gradient);
        dscuda::device_free(logsumexp);
        dscuda::device_free(mean_loss);
        dscuda::device_free(targets_device);
        dscuda::device_free(logits);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "Cross-entropy benchmark failed: %s\n", error.what());
        return 1;
    }
}

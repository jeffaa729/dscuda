// Compares fused stable vocabulary cross-entropy and reconstructed softmax gradients with the scalar CPU reference.
// A non-power-of-two vocabulary, extreme logits, row-sum invariants, and a finite difference exercise the complete loss contract.

#include "cross_entropy.h"
#include "cross_entropy_cpu.h"
#include "cuda_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kRows = 37;
constexpr int kVocabularySize = 509;
constexpr int kElements = kRows * kVocabularySize;

float max_error(
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    float error = 0.0F;
    for (int index = 0; index < static_cast<int>(expected.size()); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance) {
    const float error = max_error(expected, actual);
    const bool passed = error < tolerance;
    std::printf(
        "  %-24s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

float cpu_loss(
    const std::vector<float>& logits,
    const std::vector<int>& targets) {
    float loss = 0.0F;
    std::vector<float> logsumexp(kRows);
    dscuda::cross_entropy_forward_cpu(
        &loss,
        logsumexp.data(),
        logits.data(),
        targets.data(),
        kRows,
        kVocabularySize);
    return loss;
}

bool run_test() {
    std::vector<float> logits(kElements);
    std::vector<int> targets(kRows);
    for (int index = 0; index < kElements; ++index) {
        logits[index] =
            static_cast<float>((index * 17) % 1009 - 504) / 73.0F;
    }
    logits[2 * kVocabularySize + 11] = 25.0F;
    logits[7 * kVocabularySize + 19] = -25.0F;
    for (int row = 0; row < kRows; ++row) {
        targets[row] = (row * 97 + 13) % kVocabularySize;
    }

    float cpu_mean_loss = 0.0F;
    std::vector<float> cpu_logsumexp(kRows);
    std::vector<float> cpu_gradient(kElements);
    dscuda::cross_entropy_forward_cpu(
        &cpu_mean_loss,
        cpu_logsumexp.data(),
        logits.data(),
        targets.data(),
        kRows,
        kVocabularySize);
    dscuda::cross_entropy_backward_cpu(
        cpu_gradient.data(),
        logits.data(),
        cpu_logsumexp.data(),
        targets.data(),
        kRows,
        kVocabularySize);

    const int probe = 5 * kVocabularySize + targets[5];
    constexpr float kFiniteStep = 0.01F;
    logits[probe] += kFiniteStep;
    const float loss_plus = cpu_loss(logits, targets);
    logits[probe] -= 2.0F * kFiniteStep;
    const float loss_minus = cpu_loss(logits, targets);
    logits[probe] += kFiniteStep;
    const float finite_gradient =
        (loss_plus - loss_minus) / (2.0F * kFiniteStep);
    const float finite_error =
        std::abs(finite_gradient - cpu_gradient[probe]);

    float maximum_row_sum = 0.0F;
    for (int row = 0; row < kRows; ++row) {
        float sum = 0.0F;
        for (int column = 0; column < kVocabularySize; ++column) {
            sum += cpu_gradient[row * kVocabularySize + column];
        }
        maximum_row_sum = std::max(maximum_row_sum, std::abs(sum));
    }

    auto* gpu_logits = static_cast<float*>(
        dscuda::device_malloc(kElements * sizeof(float)));
    auto* gpu_targets = static_cast<int*>(
        dscuda::device_malloc(kRows * sizeof(int)));
    auto* gpu_mean_loss = static_cast<float*>(
        dscuda::device_malloc(sizeof(float)));
    auto* gpu_logsumexp = static_cast<float*>(
        dscuda::device_malloc(kRows * sizeof(float)));
    auto* gpu_gradient = static_cast<float*>(
        dscuda::device_malloc(kElements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        gpu_logits,
        logits.data(),
        kElements * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        gpu_targets,
        targets.data(),
        kRows * sizeof(int),
        cudaMemcpyHostToDevice));

    dscuda::cross_entropy_forward_cuda(
        gpu_mean_loss,
        gpu_logsumexp,
        gpu_logits,
        gpu_targets,
        kRows,
        kVocabularySize);
    dscuda::cross_entropy_backward_cuda(
        gpu_gradient,
        gpu_logits,
        gpu_logsumexp,
        gpu_targets,
        kRows,
        kVocabularySize);
    dscuda::synchronize();

    float gpu_mean_loss_host = 0.0F;
    std::vector<float> gpu_logsumexp_host(kRows);
    std::vector<float> gpu_gradient_host(kElements);
    CUDA_CHECK(cudaMemcpy(
        &gpu_mean_loss_host,
        gpu_mean_loss,
        sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_logsumexp_host.data(),
        gpu_logsumexp,
        kRows * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_gradient_host.data(),
        gpu_gradient,
        kElements * sizeof(float),
        cudaMemcpyDeviceToHost));

    dscuda::device_free(gpu_gradient);
    dscuda::device_free(gpu_logsumexp);
    dscuda::device_free(gpu_mean_loss);
    dscuda::device_free(gpu_targets);
    dscuda::device_free(gpu_logits);

    const float loss_error = std::abs(cpu_mean_loss - gpu_mean_loss_host);
    bool passed = true;
    std::printf(
        "  %-24s error = %.3e  %s\n",
        "mean loss",
        loss_error,
        loss_error < 2.0e-5F ? "PASS" : "FAIL");
    passed &= loss_error < 2.0e-5F;
    passed &= check(
        "log-sum-exp",
        cpu_logsumexp,
        gpu_logsumexp_host,
        3.0e-6F);
    passed &= check(
        "logits gradient",
        cpu_gradient,
        gpu_gradient_host,
        2.0e-7F);
    std::printf(
        "  %-24s error = %.3e  %s\n",
        "finite difference",
        finite_error,
        finite_error < 3.0e-4F ? "PASS" : "FAIL");
    std::printf(
        "  %-24s maximum = %.3e  %s\n",
        "gradient row sum",
        maximum_row_sum,
        maximum_row_sum < 2.0e-7F ? "PASS" : "FAIL");
    passed &= finite_error < 3.0e-4F;
    passed &= maximum_row_sum < 2.0e-7F;
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf(
            "Cross-entropy test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "Cross-entropy test failed: %s\n", error.what());
        return 1;
    }
}

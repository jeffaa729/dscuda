// Compares the BF16 Tensor Core grouped expert GEMM with the scalar CPU grouped-linear reference.
// Uneven expert ranges, an empty expert, and non-tile-aligned output rows exercise the variable-M scheduler and edge masking.

#include "cuda_common.h"
#include "expert_dispatch.h"
#include "expert_dispatch_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int M = 64;
constexpr int E = 5;
constexpr int K = 32;
constexpr int N = 48;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements)
        : data_(static_cast<T*>(
              dscuda::device_malloc(elements * sizeof(T)))) {}
    ~DeviceBuffer() { dscuda::device_free(data_); }
    T* data() { return data_; }
    void upload(const std::vector<T>& values) {
        CUDA_CHECK(cudaMemcpy(
            data_,
            values.data(),
            values.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }
    std::vector<T> download(std::size_t elements) const {
        std::vector<T> values(elements);
        CUDA_CHECK(cudaMemcpy(
            values.data(),
            data_,
            elements * sizeof(T),
            cudaMemcpyDeviceToHost));
        return values;
    }

private:
    T* data_;
};

}  // namespace

int main() {
    const std::vector<int> offsets = {0, 17, 17, 31, 52, M};
    std::vector<int> slot_expert(M);
    for (int expert = 0; expert < E; ++expert) {
        std::fill(
            slot_expert.begin() + offsets[expert],
            slot_expert.begin() + offsets[expert + 1],
            expert);
    }

    std::vector<__nv_bfloat16> input(M * K);
    std::vector<__nv_bfloat16> weight(E * K * N);
    std::vector<float> rounded_input(M * K);
    std::vector<float> rounded_weight(E * K * N);
    for (int index = 0; index < M * K; ++index) {
        input[index] = __float2bfloat16(
            static_cast<float>((index * 17) % 101 - 50) / 64.0F);
        rounded_input[index] = __bfloat162float(input[index]);
    }
    for (int index = 0; index < E * K * N; ++index) {
        weight[index] = __float2bfloat16(
            static_cast<float>((index * 23) % 97 - 48) / 96.0F);
        rounded_weight[index] = __bfloat162float(weight[index]);
    }

    std::vector<float> expected(M * N);
    dscuda::grouped_linear_forward_cpu(
        expected.data(),
        rounded_input.data(),
        rounded_weight.data(),
        slot_expert.data(),
        M,
        N,
        K);

    DeviceBuffer<__nv_bfloat16> gpu_input(input.size());
    DeviceBuffer<__nv_bfloat16> gpu_weight(weight.size());
    DeviceBuffer<int> gpu_offsets(offsets.size());
    DeviceBuffer<float> gpu_output(expected.size());
    gpu_input.upload(input);
    gpu_weight.upload(weight);
    gpu_offsets.upload(offsets);
    dscuda::grouped_linear_bf16_forward_cuda(
        gpu_output.data(),
        gpu_input.data(),
        gpu_weight.data(),
        gpu_offsets.data(),
        M,
        E,
        N,
        K);
    dscuda::synchronize();

    const auto actual = gpu_output.download(expected.size());
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        maximum_error = std::max(
            maximum_error, std::abs(expected[index] - actual[index]));
    }
    const bool passed = maximum_error < 3.0e-3F;
    std::printf(
        "BF16 grouped GEMM max error = %.3e  %s\n",
        maximum_error,
        passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

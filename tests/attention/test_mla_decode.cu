// Validates the split-KV compressed MLA decoder against a scalar CPU implementation for unequal cache lengths.
// The test confirms that independently computed online-softmax states combine into the same output and log-sum-exp.

#include "cuda_common.h"
#include "mla.h"
#include "mla_cpu.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int B = 2;
constexpr int MAX_T = 23;
constexpr int H = 4;
constexpr int C = dscuda::MLA_KV_RANK;
constexpr int R = dscuda::MLA_ROPE_SIZE;
constexpr int SPLITS = 4;
constexpr int QUERY_ELEMENTS = B * H * C;
constexpr int QUERY_ROPE_ELEMENTS = B * H * R;
constexpr int KV_ELEMENTS = B * MAX_T * C;
constexpr int KEY_ROPE_ELEMENTS = B * MAX_T * R;
constexpr float SCALE = 1.0F / 24.0F;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements)
        : elements_(elements),
          data_(static_cast<T*>(dscuda::device_malloc(elements * sizeof(T)))) {}
    ~DeviceBuffer() { dscuda::device_free(data_); }
    T* data() { return data_; }
    void upload(const std::vector<T>& values) {
        CUDA_CHECK(cudaMemcpy(
            data_, values.data(), elements_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(elements_);
        CUDA_CHECK(cudaMemcpy(
            values.data(), data_, elements_ * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
private:
    std::size_t elements_;
    T* data_;
};

std::vector<float> make_values(int elements, float phase) {
    std::vector<float> values(elements);
    for (int index = 0; index < elements; ++index) {
        values[index] = 0.13F *
            (std::sin(0.043F * static_cast<float>(index) + phase) +
             0.4F * std::cos(0.079F * static_cast<float>(index) - phase));
    }
    return values;
}

std::vector<__nv_bfloat16> round_to_bf16(std::vector<float>& values) {
    std::vector<__nv_bfloat16> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = __float2bfloat16(values[index]);
        values[index] = __bfloat162float(result[index]);
    }
    return result;
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
            maximum_error = INFINITY;
            break;
        }
        maximum_error = std::max(
            maximum_error, std::abs(expected[index] - actual[index]));
    }
    const bool passed = maximum_error < 2.0e-5F;
    std::printf(
        "  %-18s max error = %.3e  %s\n",
        name,
        maximum_error,
        passed ? "PASS" : "FAIL");
    return passed;
}

}  // namespace

int main() {
    auto query_latent = make_values(QUERY_ELEMENTS, 0.1F);
    auto query_rope = make_values(QUERY_ROPE_ELEMENTS, 0.3F);
    auto kv_cache = make_values(KV_ELEMENTS, 0.5F);
    auto key_rope_cache = make_values(KEY_ROPE_ELEMENTS, 0.7F);
    const std::vector<int> cache_lengths = {23, 17};

    const auto bf16_query_latent = round_to_bf16(query_latent);
    const auto bf16_query_rope = round_to_bf16(query_rope);
    const auto bf16_kv_cache = round_to_bf16(kv_cache);
    const auto bf16_key_rope_cache = round_to_bf16(key_rope_cache);

    std::vector<float> expected_output(QUERY_ELEMENTS);
    std::vector<float> expected_lse(B * H);
    dscuda::mla_decode_forward_cpu(
        expected_output.data(),
        expected_lse.data(),
        query_latent.data(),
        query_rope.data(),
        kv_cache.data(),
        key_rope_cache.data(),
        cache_lengths.data(),
        B,
        MAX_T,
        H,
        C,
        R,
        SCALE);

    DeviceBuffer<__nv_bfloat16> gpu_query_latent(QUERY_ELEMENTS);
    DeviceBuffer<__nv_bfloat16> gpu_query_rope(QUERY_ROPE_ELEMENTS);
    DeviceBuffer<__nv_bfloat16> gpu_kv_cache(KV_ELEMENTS);
    DeviceBuffer<__nv_bfloat16> gpu_key_rope_cache(KEY_ROPE_ELEMENTS);
    DeviceBuffer<int> gpu_cache_lengths(B);
    DeviceBuffer<float> gpu_output(QUERY_ELEMENTS);
    DeviceBuffer<float> gpu_lse(B * H);
    DeviceBuffer<float> gpu_workspace(dscuda::mla_decode_workspace_elements(
        B, H, SPLITS, C));

    gpu_query_latent.upload(bf16_query_latent);
    gpu_query_rope.upload(bf16_query_rope);
    gpu_kv_cache.upload(bf16_kv_cache);
    gpu_key_rope_cache.upload(bf16_key_rope_cache);
    gpu_cache_lengths.upload(cache_lengths);
    dscuda::mla_decode_forward_cuda(
        gpu_output.data(),
        gpu_lse.data(),
        gpu_query_latent.data(),
        gpu_query_rope.data(),
        gpu_kv_cache.data(),
        gpu_key_rope_cache.data(),
        gpu_cache_lengths.data(),
        gpu_workspace.data(),
        B,
        MAX_T,
        H,
        C,
        R,
        SPLITS,
        SCALE);
    dscuda::synchronize();

    std::printf("MLA split-KV decode test\n");
    bool passed = true;
    passed &= check("output", expected_output, gpu_output.download());
    passed &= check("logsumexp", expected_lse, gpu_lse.download());
    return passed ? 0 : 1;
}

// Compares the fused BF16 compressed-latent MLA forward and backward paths with the scalar CPU equations.
// Fixed C512/R64 cases cover causal tile boundaries, sequence tails, and every activation gradient.

#include "cuda_common.h"
#include "mla.h"
#include "mla_cpu.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

constexpr int C = dscuda::MLA_KV_RANK;
constexpr int R = dscuda::MLA_ROPE_SIZE;
constexpr float SCALE = 1.0F / 24.0F;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements)
        : elements_(elements),
          data_(static_cast<T*>(dscuda::device_malloc(elements * sizeof(T)))) {}

    ~DeviceBuffer() {
        dscuda::device_free(data_);
    }

    T* data() {
        return data_;
    }

    void upload(const std::vector<T>& values) {
        CUDA_CHECK(cudaMemcpy(
            data_,
            values.data(),
            elements_ * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    std::vector<T> download() const {
        std::vector<T> values(elements_);
        CUDA_CHECK(cudaMemcpy(
            values.data(),
            data_,
            elements_ * sizeof(T),
            cudaMemcpyDeviceToHost));
        return values;
    }

    void zero() {
        CUDA_CHECK(cudaMemset(data_, 0, elements_ * sizeof(T)));
    }

private:
    std::size_t elements_;
    T* data_;
};

std::vector<float> make_values(int elements, float scale, float phase) {
    std::vector<float> values(elements);
    for (int index = 0; index < elements; ++index) {
        values[index] = scale *
            (std::sin(0.071F * static_cast<float>(index) + phase) +
             0.5F * std::cos(0.037F * static_cast<float>(index) - phase));
    }
    return values;
}

std::vector<__nv_bfloat16> round_to_bf16(std::vector<float>& values) {
    std::vector<__nv_bfloat16> rounded(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        rounded[index] = __float2bfloat16(values[index]);
        values[index] = __bfloat162float(rounded[index]);
    }
    return rounded;
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance) {
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
            maximum_error = INFINITY;
            break;
        }
        maximum_error = std::max(
            maximum_error, std::abs(expected[index] - actual[index]));
    }
    const bool passed = maximum_error < tolerance;
    std::printf(
        "  %-28s max error = %.3e  %s\n",
        name,
        maximum_error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool run_case(int B, int T, int H) {
    const int QUERY_ELEMENTS = B * T * H * C;
    const int QUERY_ROPE_ELEMENTS = B * T * H * R;
    const int KV_ELEMENTS = B * T * C;
    const int KEY_ROPE_ELEMENTS = B * T * R;
    const int ROWS = B * H * T;
    auto query_latent = make_values(QUERY_ELEMENTS, 0.15F, 0.1F);
    auto query_rope = make_values(QUERY_ROPE_ELEMENTS, 0.12F, 0.3F);
    auto kv_latent = make_values(KV_ELEMENTS, 0.14F, 0.5F);
    auto key_rope = make_values(KEY_ROPE_ELEMENTS, 0.11F, 0.7F);
    const auto output_gradient = make_values(QUERY_ELEMENTS, 0.08F, 0.9F);

    const auto bf16_query_latent = round_to_bf16(query_latent);
    const auto bf16_query_rope = round_to_bf16(query_rope);
    const auto bf16_kv_latent = round_to_bf16(kv_latent);
    const auto bf16_key_rope = round_to_bf16(key_rope);

    std::vector<float> expected_output(QUERY_ELEMENTS);
    std::vector<float> expected_lse(ROWS);
    std::vector<float> expected_query_latent_gradient(QUERY_ELEMENTS, 0.0F);
    std::vector<float> expected_query_rope_gradient(QUERY_ROPE_ELEMENTS, 0.0F);
    std::vector<float> expected_kv_latent_gradient(KV_ELEMENTS, 0.0F);
    std::vector<float> expected_key_rope_gradient(KEY_ROPE_ELEMENTS, 0.0F);

    dscuda::mla_compressed_attention_forward_cpu(
        expected_output.data(), expected_lse.data(), query_latent.data(),
        query_rope.data(), kv_latent.data(), key_rope.data(), B, T, H, C, R,
        SCALE);
    dscuda::mla_compressed_attention_backward_cpu(
        expected_query_latent_gradient.data(),
        expected_query_rope_gradient.data(),
        expected_kv_latent_gradient.data(),
        expected_key_rope_gradient.data(),
        output_gradient.data(),
        expected_output.data(),
        expected_lse.data(),
        query_latent.data(),
        query_rope.data(),
        kv_latent.data(),
        key_rope.data(),
        B,
        T,
        H,
        C,
        R,
        SCALE);

    DeviceBuffer<__nv_bfloat16> gpu_query_latent(QUERY_ELEMENTS);
    DeviceBuffer<__nv_bfloat16> gpu_query_rope(QUERY_ROPE_ELEMENTS);
    DeviceBuffer<__nv_bfloat16> gpu_kv_latent(KV_ELEMENTS);
    DeviceBuffer<__nv_bfloat16> gpu_key_rope(KEY_ROPE_ELEMENTS);
    DeviceBuffer<float> gpu_output_gradient(QUERY_ELEMENTS);
    DeviceBuffer<float> gpu_output(QUERY_ELEMENTS);
    DeviceBuffer<float> gpu_lse(ROWS);
    DeviceBuffer<float> gpu_query_latent_gradient(QUERY_ELEMENTS);
    DeviceBuffer<float> gpu_query_rope_gradient(QUERY_ROPE_ELEMENTS);
    DeviceBuffer<float> gpu_kv_latent_gradient(KV_ELEMENTS);
    DeviceBuffer<float> gpu_key_rope_gradient(KEY_ROPE_ELEMENTS);

    gpu_query_latent.upload(bf16_query_latent);
    gpu_query_rope.upload(bf16_query_rope);
    gpu_kv_latent.upload(bf16_kv_latent);
    gpu_key_rope.upload(bf16_key_rope);
    gpu_output_gradient.upload(output_gradient);
    gpu_query_latent_gradient.zero();
    gpu_query_rope_gradient.zero();
    gpu_kv_latent_gradient.zero();
    gpu_key_rope_gradient.zero();

    dscuda::mla_compressed_attention_forward_cuda(
        gpu_output.data(), gpu_lse.data(), gpu_query_latent.data(),
        gpu_query_rope.data(), gpu_kv_latent.data(), gpu_key_rope.data(), B, T,
        H, C, R, SCALE);
    dscuda::mla_compressed_attention_backward_cuda(
        gpu_query_latent_gradient.data(),
        gpu_query_rope_gradient.data(),
        gpu_kv_latent_gradient.data(),
        gpu_key_rope_gradient.data(),
        gpu_output_gradient.data(),
        gpu_output.data(),
        gpu_lse.data(),
        gpu_query_latent.data(),
        gpu_query_rope.data(),
        gpu_kv_latent.data(),
        gpu_key_rope.data(),
        B,
        T,
        H,
        C,
        R,
        SCALE);
    dscuda::synchronize();

    std::printf("MLA C512/R64: B=%d T=%d H=%d\n", B, T, H);
    bool passed = true;
    passed &= check("output", expected_output, gpu_output.download(), 2.0e-5F);
    passed &= check("logsumexp", expected_lse, gpu_lse.download(), 2.0e-5F);
    passed &= check(
        "query latent gradient", expected_query_latent_gradient,
        gpu_query_latent_gradient.download(), 3.0e-5F);
    passed &= check(
        "query RoPE gradient", expected_query_rope_gradient,
        gpu_query_rope_gradient.download(), 3.0e-5F);
    passed &= check(
        "shared KV gradient", expected_kv_latent_gradient,
        gpu_kv_latent_gradient.download(), 4.0e-5F);
    passed &= check(
        "shared key RoPE gradient", expected_key_rope_gradient,
        gpu_key_rope_gradient.download(), 4.0e-5F);

    return passed;
}

}  // namespace

int main() {
    bool passed = true;
    passed &= run_case(1, 1, 1);
    passed &= run_case(2, 11, 3);
    passed &= run_case(1, 16, 2);
    passed &= run_case(1, 17, 2);
    passed &= run_case(1, 65, 2);
    bool rejected = false;
    try {
        dscuda::mla_compressed_attention_forward_cuda(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            1, 64, 1, 64, 32, SCALE);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    passed &= rejected;
    return passed ? 0 : 1;
}

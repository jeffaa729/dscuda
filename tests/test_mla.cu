// Compares the fused BF16 compressed-latent MLA forward and backward paths with the scalar CPU equations.
// The test uses different compressed and RoPE widths and checks every activation gradient required by end-to-end training.

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
constexpr int T = 11;
constexpr int H = 3;
constexpr int C = 64;
constexpr int R = 16;
constexpr int QUERY_ELEMENTS = B * T * H * C;
constexpr int QUERY_ROPE_ELEMENTS = B * T * H * R;
constexpr int KV_ELEMENTS = B * T * C;
constexpr int KEY_ROPE_ELEMENTS = B * T * R;
constexpr int ROWS = B * H * T;
constexpr float SCALE = 0.111803399F;

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

bool test_tensor_core_path() {
    constexpr int tensor_batch = 1;
    constexpr int tensor_sequence = 64;
    constexpr int tensor_heads = 2;
    constexpr int tensor_kv_rank = 64;
    constexpr int tensor_rope_size = 32;
    constexpr int tensor_query_elements =
        tensor_batch * tensor_sequence * tensor_heads * tensor_kv_rank;
    constexpr int tensor_query_rope_elements =
        tensor_batch * tensor_sequence * tensor_heads * tensor_rope_size;
    constexpr int tensor_kv_elements =
        tensor_batch * tensor_sequence * tensor_kv_rank;
    constexpr int tensor_key_rope_elements =
        tensor_batch * tensor_sequence * tensor_rope_size;
    constexpr int tensor_rows =
        tensor_batch * tensor_heads * tensor_sequence;
    constexpr float tensor_scale = 0.102062073F;

    auto query = make_values(tensor_query_elements, 0.15F, 0.15F);
    auto query_rope =
        make_values(tensor_query_rope_elements, 0.12F, 0.35F);
    auto kv = make_values(tensor_kv_elements, 0.14F, 0.55F);
    auto key_rope =
        make_values(tensor_key_rope_elements, 0.11F, 0.75F);
    const auto output_gradient =
        make_values(tensor_query_elements, 0.08F, 0.95F);
    const auto bf16_query = round_to_bf16(query);
    const auto bf16_query_rope = round_to_bf16(query_rope);
    const auto bf16_kv = round_to_bf16(kv);
    const auto bf16_key_rope = round_to_bf16(key_rope);

    std::vector<float> expected_output(tensor_query_elements);
    std::vector<float> expected_lse(tensor_rows);
    std::vector<float> expected_query_gradient(tensor_query_elements, 0.0F);
    std::vector<float> expected_query_rope_gradient(
        tensor_query_rope_elements, 0.0F);
    std::vector<float> expected_kv_gradient(tensor_kv_elements, 0.0F);
    std::vector<float> expected_key_rope_gradient(
        tensor_key_rope_elements, 0.0F);
    dscuda::mla_compressed_attention_forward_cpu(
        expected_output.data(), expected_lse.data(), query.data(),
        query_rope.data(), kv.data(), key_rope.data(), tensor_batch,
        tensor_sequence, tensor_heads, tensor_kv_rank, tensor_rope_size,
        tensor_scale);
    dscuda::mla_compressed_attention_backward_cpu(
        expected_query_gradient.data(),
        expected_query_rope_gradient.data(),
        expected_kv_gradient.data(),
        expected_key_rope_gradient.data(),
        output_gradient.data(),
        expected_output.data(),
        expected_lse.data(),
        query.data(),
        query_rope.data(),
        kv.data(),
        key_rope.data(),
        tensor_batch,
        tensor_sequence,
        tensor_heads,
        tensor_kv_rank,
        tensor_rope_size,
        tensor_scale);

    DeviceBuffer<__nv_bfloat16> gpu_query(tensor_query_elements);
    DeviceBuffer<__nv_bfloat16> gpu_query_rope(tensor_query_rope_elements);
    DeviceBuffer<__nv_bfloat16> gpu_kv(tensor_kv_elements);
    DeviceBuffer<__nv_bfloat16> gpu_key_rope(tensor_key_rope_elements);
    DeviceBuffer<float> gpu_output_gradient(tensor_query_elements);
    DeviceBuffer<float> gpu_output(tensor_query_elements);
    DeviceBuffer<float> gpu_lse(tensor_rows);
    DeviceBuffer<float> gpu_query_gradient(tensor_query_elements);
    DeviceBuffer<float> gpu_query_rope_gradient(tensor_query_rope_elements);
    DeviceBuffer<float> gpu_kv_gradient(tensor_kv_elements);
    DeviceBuffer<float> gpu_key_rope_gradient(tensor_key_rope_elements);
    gpu_query.upload(bf16_query);
    gpu_query_rope.upload(bf16_query_rope);
    gpu_kv.upload(bf16_kv);
    gpu_key_rope.upload(bf16_key_rope);
    gpu_output_gradient.upload(output_gradient);
    gpu_query_gradient.zero();
    gpu_query_rope_gradient.zero();
    gpu_kv_gradient.zero();
    gpu_key_rope_gradient.zero();

    dscuda::mla_compressed_attention_forward_cuda(
        gpu_output.data(), gpu_lse.data(), gpu_query.data(),
        gpu_query_rope.data(), gpu_kv.data(), gpu_key_rope.data(),
        tensor_batch, tensor_sequence, tensor_heads, tensor_kv_rank,
        tensor_rope_size, tensor_scale);
    dscuda::mla_compressed_attention_backward_cuda(
        gpu_query_gradient.data(),
        gpu_query_rope_gradient.data(),
        gpu_kv_gradient.data(),
        gpu_key_rope_gradient.data(),
        gpu_output_gradient.data(),
        gpu_output.data(),
        gpu_lse.data(),
        gpu_query.data(),
        gpu_query_rope.data(),
        gpu_kv.data(),
        gpu_key_rope.data(),
        tensor_batch,
        tensor_sequence,
        tensor_heads,
        tensor_kv_rank,
        tensor_rope_size,
        tensor_scale);
    dscuda::synchronize();

    std::printf("MLA SM89 Tensor Core path\n");
    bool passed = true;
    passed &= check(
        "Tensor Core output", expected_output, gpu_output.download(), 8.0e-3F);
    passed &= check(
        "Tensor Core logsumexp", expected_lse, gpu_lse.download(), 2.0e-3F);
    passed &= check(
        "Tensor Core query gradient", expected_query_gradient,
        gpu_query_gradient.download(), 3.0e-3F);
    passed &= check(
        "Tensor Core query RoPE grad", expected_query_rope_gradient,
        gpu_query_rope_gradient.download(), 3.0e-3F);
    passed &= check(
        "Tensor Core shared KV grad", expected_kv_gradient,
        gpu_kv_gradient.download(), 4.0e-3F);
    passed &= check(
        "Tensor Core key RoPE grad", expected_key_rope_gradient,
        gpu_key_rope_gradient.download(), 4.0e-3F);
    return passed;
}

}  // namespace

int main() {
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

    std::printf("MLA compressed training test\n");
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

    passed &= test_tensor_core_path();
    return passed ? 0 : 1;
}

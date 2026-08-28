// Verifies that sequential compressed-cache MLA decoding matches the full causal MLA layer at every token position.
// The test covers query absorption, per-position RoPE, BF16 cache append, FlashMLA-style split/combine softmax, value up-projection, and final output projection.

#include "cuda_common.h"
#include "mla.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int B = 1;
constexpr int T = 7;
constexpr int D = 32;
constexpr int H = 2;
constexpr int Q = 32;
constexpr int C = 32;
constexpr int N = 16;
constexpr int R = 8;
constexpr int V = 16;
constexpr int SPLITS = 3;
constexpr int ROWS = B * T;
constexpr dscuda::MlaLayerConfig CONFIG{
    B, T, D, H, Q, C, N, R, V, 1.0e-5F, 0.204124145F};

struct Offsets {
    std::size_t query_down = 0;
    std::size_t query_norm = query_down + D * Q;
    std::size_t query_up = query_norm + Q;
    std::size_t kv_down = query_up + Q * H * (N + R);
    std::size_t kv_norm = kv_down + D * (C + R);
    std::size_t key_up = kv_norm + C;
    std::size_t value_up = key_up + H * N * C;
    std::size_t output = value_up + H * C * V;
    std::size_t elements = output + H * V * D;
};
constexpr Offsets OFFSETS{};

template <typename TElement>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements)
        : elements_(elements),
          data_(static_cast<TElement*>(
              dscuda::device_malloc(elements * sizeof(TElement)))) {}
    ~DeviceBuffer() { dscuda::device_free(data_); }
    TElement* data() { return data_; }
    void upload(const std::vector<TElement>& values) {
        CUDA_CHECK(cudaMemcpy(
            data_, values.data(), elements_ * sizeof(TElement),
            cudaMemcpyHostToDevice));
    }
    std::vector<TElement> download() const {
        std::vector<TElement> values(elements_);
        CUDA_CHECK(cudaMemcpy(
            values.data(), data_, elements_ * sizeof(TElement),
            cudaMemcpyDeviceToHost));
        return values;
    }
private:
    std::size_t elements_;
    TElement* data_;
};

std::vector<float> make_values(int elements, float scale, float phase) {
    std::vector<float> values(elements);
    for (int index = 0; index < elements; ++index) {
        values[index] = scale *
            (std::sin(0.053F * index + phase)
             + 0.35F * std::cos(0.097F * index - phase));
    }
    return values;
}

dscuda::MlaLayerParameters parameters(const float* values) {
    return {
        values + OFFSETS.query_down,
        values + OFFSETS.query_norm,
        values + OFFSETS.query_up,
        values + OFFSETS.kv_down,
        values + OFFSETS.kv_norm,
        values + OFFSETS.key_up,
        values + OFFSETS.value_up,
        values + OFFSETS.output,
    };
}

}  // namespace

int main() {
    const auto input = make_values(ROWS * D, 0.11F, 0.1F);
    auto parameter_values = make_values(
        static_cast<int>(OFFSETS.elements), 0.07F, 0.3F);
    std::fill_n(parameter_values.data() + OFFSETS.query_norm, Q, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.kv_norm, C, 1.0F);
    std::vector<float> cosine(T * R / 2);
    std::vector<float> sine(T * R / 2);
    for (int token = 0; token < T; ++token) {
        for (int pair = 0; pair < R / 2; ++pair) {
            const float angle = token * std::pow(10000.0F, -2.0F * pair / R);
            cosine[token * (R / 2) + pair] = std::cos(angle);
            sine[token * (R / 2) + pair] = std::sin(angle);
        }
    }

    DeviceBuffer<float> gpu_input(input.size());
    DeviceBuffer<float> gpu_parameters(parameter_values.size());
    DeviceBuffer<float> gpu_cosine(cosine.size());
    DeviceBuffer<float> gpu_sine(sine.size());
    DeviceBuffer<float> gpu_full_output(ROWS * D);
    DeviceBuffer<float> gpu_full_activations(
        dscuda::mla_layer_activation_elements(CONFIG));
    DeviceBuffer<__nv_bfloat16> gpu_full_bf16(
        dscuda::mla_layer_bf16_workspace_elements(CONFIG));
    DeviceBuffer<float> gpu_decode_output(D);
    DeviceBuffer<__nv_bfloat16> gpu_kv_cache(B * T * C);
    DeviceBuffer<__nv_bfloat16> gpu_rope_cache(B * T * R);
    DeviceBuffer<int> gpu_cache_lengths(B);
    DeviceBuffer<float> gpu_decode_workspace(
        dscuda::mla_layer_decode_workspace_elements(CONFIG, SPLITS));
    DeviceBuffer<__nv_bfloat16> gpu_decode_bf16(
        dscuda::mla_layer_decode_bf16_workspace_elements(CONFIG));
    gpu_input.upload(input);
    gpu_parameters.upload(parameter_values);
    gpu_cosine.upload(cosine);
    gpu_sine.upload(sine);

    dscuda::mla_layer_forward_cuda(
        gpu_full_output.data(),
        gpu_input.data(),
        parameters(gpu_parameters.data()),
        gpu_cosine.data(),
        gpu_sine.data(),
        gpu_full_activations.data(),
        gpu_full_bf16.data(),
        CONFIG);
    std::vector<float> decoded(ROWS * D);
    for (int position = 0; position < T; ++position) {
        dscuda::mla_layer_decode_forward_cuda(
            gpu_decode_output.data(),
            gpu_input.data() + position * D,
            parameters(gpu_parameters.data()),
            gpu_cosine.data(),
            gpu_sine.data(),
            position,
            gpu_kv_cache.data(),
            gpu_rope_cache.data(),
            gpu_cache_lengths.data(),
            gpu_decode_workspace.data(),
            gpu_decode_bf16.data(),
            CONFIG,
            SPLITS);
        CUDA_CHECK(cudaMemcpy(
            decoded.data() + position * D,
            gpu_decode_output.data(),
            D * sizeof(float),
            cudaMemcpyDeviceToHost));
    }
    dscuda::synchronize();
    const std::vector<float> full = gpu_full_output.download();
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < full.size(); ++index) {
        maximum_error = std::max(
            maximum_error, std::abs(full[index] - decoded[index]));
    }
    const bool passed = maximum_error < 5.0e-5F;
    std::printf("MLA full-layer compressed decode test\n");
    std::printf(
        "  full versus cached decode max error = %.3e  %s\n",
        maximum_error,
        passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

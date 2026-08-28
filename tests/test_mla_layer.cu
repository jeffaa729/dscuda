// Compares the complete low-rank MLA projection graph and all parameter gradients with the readable CPU reference.
// This catches head-major packing, RoPE branch, absorbed-key, shared-KV, value up-projection, and reverse-layout errors together.

#include "cuda_common.h"
#include "mla.h"
#include "mla_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int B = 1;
constexpr int T = 5;
constexpr int D = 32;
constexpr int H = 2;
constexpr int Q = 32;
constexpr int C = 32;
constexpr int N = 16;
constexpr int R = 8;
constexpr int V = 16;
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
            (std::sin(0.053F * static_cast<float>(index) + phase) +
             0.35F * std::cos(0.097F * static_cast<float>(index) - phase));
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

dscuda::MlaLayerGradients gradients(float* values) {
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
        "  %-20s max error = %.3e  %s\n",
        name,
        maximum_error,
        passed ? "PASS" : "FAIL");
    return passed;
}

}  // namespace

int main() {
    auto input = make_values(ROWS * D, 0.11F, 0.1F);
    auto parameter_values = make_values(
        static_cast<int>(OFFSETS.elements), 0.07F, 0.3F);
    std::fill_n(
        parameter_values.data() + OFFSETS.query_norm, Q, 1.0F);
    std::fill_n(
        parameter_values.data() + OFFSETS.kv_norm, C, 1.0F);
    const auto output_gradient = make_values(ROWS * D, 0.09F, 0.5F);
    std::vector<float> cosine(T * R / 2);
    std::vector<float> sine(T * R / 2);
    for (int token = 0; token < T; ++token) {
        for (int pair = 0; pair < R / 2; ++pair) {
            const float angle = token * std::pow(10000.0F, -2.0F * pair / R);
            cosine[token * (R / 2) + pair] = std::cos(angle);
            sine[token * (R / 2) + pair] = std::sin(angle);
        }
    }

    std::vector<float> expected_output(ROWS * D);
    std::vector<float> expected_input_gradient(ROWS * D, 0.0F);
    std::vector<float> expected_parameter_gradient(OFFSETS.elements, 0.0F);
    dscuda::mla_layer_forward_cpu(
        expected_output.data(), input.data(), parameters(parameter_values.data()),
        cosine.data(), sine.data(), CONFIG);
    dscuda::mla_layer_backward_cpu(
        expected_input_gradient.data(),
        gradients(expected_parameter_gradient.data()),
        output_gradient.data(),
        input.data(),
        parameters(parameter_values.data()),
        cosine.data(),
        sine.data(),
        CONFIG);

    DeviceBuffer<float> gpu_input(ROWS * D);
    DeviceBuffer<float> gpu_parameters(OFFSETS.elements);
    DeviceBuffer<float> gpu_cosine(cosine.size());
    DeviceBuffer<float> gpu_sine(sine.size());
    DeviceBuffer<float> gpu_output_gradient(ROWS * D);
    DeviceBuffer<float> gpu_output(ROWS * D);
    DeviceBuffer<float> gpu_input_gradient(ROWS * D);
    DeviceBuffer<float> gpu_parameter_gradient(OFFSETS.elements);
    DeviceBuffer<float> gpu_activations(
        dscuda::mla_layer_activation_elements(CONFIG));
    DeviceBuffer<float> gpu_workspace(
        dscuda::mla_layer_backward_workspace_elements(CONFIG));
    DeviceBuffer<__nv_bfloat16> gpu_bf16_workspace(
        dscuda::mla_layer_bf16_workspace_elements(CONFIG));
    gpu_input.upload(input);
    gpu_parameters.upload(parameter_values);
    gpu_cosine.upload(cosine);
    gpu_sine.upload(sine);
    gpu_output_gradient.upload(output_gradient);
    gpu_input_gradient.zero();
    gpu_parameter_gradient.zero();

    dscuda::mla_layer_forward_cuda(
        gpu_output.data(),
        gpu_input.data(),
        parameters(gpu_parameters.data()),
        gpu_cosine.data(),
        gpu_sine.data(),
        gpu_activations.data(),
        gpu_bf16_workspace.data(),
        CONFIG);
    dscuda::mla_layer_backward_cuda(
        gpu_input_gradient.data(),
        gradients(gpu_parameter_gradient.data()),
        gpu_output_gradient.data(),
        gpu_input.data(),
        parameters(gpu_parameters.data()),
        gpu_cosine.data(),
        gpu_sine.data(),
        gpu_activations.data(),
        gpu_workspace.data(),
        gpu_bf16_workspace.data(),
        CONFIG);
    dscuda::synchronize();

    std::printf("Complete MLA layer test\n");
    bool passed = true;
    passed &= check("output", expected_output, gpu_output.download(), 4.0e-5F);
    passed &= check(
        "input gradient", expected_input_gradient,
        gpu_input_gradient.download(), 8.0e-5F);
    passed &= check(
        "parameter gradient", expected_parameter_gradient,
        gpu_parameter_gradient.download(), 1.5e-4F);
    return passed ? 0 : 1;
}

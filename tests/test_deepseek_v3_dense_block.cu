// Compares the complete MLA-plus-dense-SwiGLU V3 block against its scalar CPU graph.
// This is the correctness gate for the initial dense FFN layers before later blocks switch to routed DeepSeekMoE experts.

#include "cuda_common.h"
#include "deepseek_v3_dense_block.h"
#include "deepseek_v3_dense_block_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int B = 1;
constexpr int T = 4;
constexpr int D = 32;
constexpr int H = 2;
constexpr int Q = 32;
constexpr int C = 32;
constexpr int N = 16;
constexpr int R = 8;
constexpr int V = 16;
constexpr int F = 48;
constexpr int ROWS = B * T;
constexpr dscuda::MlaLayerConfig MLA{
    B, T, D, H, Q, C, N, R, V, 1.0e-5F, 0.204124145F};
constexpr dscuda::DeepSeekV3DenseBlockConfig CONFIG{
    MLA, ROWS, D, F, 1.0e-5F};

struct Offsets {
    std::size_t attention_norm = 0;
    std::size_t query_down = attention_norm + D;
    std::size_t query_norm = query_down + D * Q;
    std::size_t query_up = query_norm + Q;
    std::size_t kv_down = query_up + Q * H * (N + R);
    std::size_t kv_norm = kv_down + D * (C + R);
    std::size_t key_up = kv_norm + C;
    std::size_t value_up = key_up + H * N * C;
    std::size_t mla_output = value_up + H * C * V;
    std::size_t ffn_norm = mla_output + H * V * D;
    std::size_t gate = ffn_norm + D;
    std::size_t up = gate + D * F;
    std::size_t down = up + D * F;
    std::size_t elements = down + F * D;
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
    void zero() {
        CUDA_CHECK(cudaMemset(data_, 0, elements_ * sizeof(TElement)));
    }
private:
    std::size_t elements_;
    TElement* data_;
};

std::vector<float> make_values(int elements, float scale, float phase) {
    std::vector<float> values(elements);
    for (int index = 0; index < elements; ++index) {
        values[index] = scale *
            (std::sin(0.047F * index + phase)
             + 0.3F * std::cos(0.083F * index - phase));
    }
    return values;
}

dscuda::DeepSeekV3DenseBlockParameters parameters(const float* p) {
    return {
        p + OFFSETS.attention_norm,
        {
            p + OFFSETS.query_down, p + OFFSETS.query_norm,
            p + OFFSETS.query_up, p + OFFSETS.kv_down,
            p + OFFSETS.kv_norm, p + OFFSETS.key_up,
            p + OFFSETS.value_up, p + OFFSETS.mla_output,
        },
        p + OFFSETS.ffn_norm,
        p + OFFSETS.gate,
        p + OFFSETS.up,
        p + OFFSETS.down,
    };
}

dscuda::DeepSeekV3DenseBlockGradients gradients(float* p) {
    return {
        p + OFFSETS.attention_norm,
        {
            p + OFFSETS.query_down, p + OFFSETS.query_norm,
            p + OFFSETS.query_up, p + OFFSETS.kv_down,
            p + OFFSETS.kv_norm, p + OFFSETS.key_up,
            p + OFFSETS.value_up, p + OFFSETS.mla_output,
        },
        p + OFFSETS.ffn_norm,
        p + OFFSETS.gate,
        p + OFFSETS.up,
        p + OFFSETS.down,
    };
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance) {
    float error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    const bool passed = error < tolerance;
    std::printf(
        "  %-20s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

}  // namespace

int main() {
    const auto input = make_values(ROWS * D, 0.12F, 0.1F);
    auto parameter_values = make_values(
        static_cast<int>(OFFSETS.elements), 0.06F, 0.3F);
    std::fill_n(parameter_values.data() + OFFSETS.attention_norm, D, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.query_norm, Q, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.kv_norm, C, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.ffn_norm, D, 1.0F);
    const auto output_gradient = make_values(ROWS * D, 0.08F, 0.5F);
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
    dscuda::deepseek_v3_dense_block_forward_cpu(
        expected_output.data(), input.data(), parameters(parameter_values.data()),
        cosine.data(), sine.data(), CONFIG);
    dscuda::deepseek_v3_dense_block_backward_cpu(
        expected_input_gradient.data(), gradients(expected_parameter_gradient.data()),
        output_gradient.data(), input.data(), parameters(parameter_values.data()),
        cosine.data(), sine.data(), CONFIG);

    DeviceBuffer<float> gpu_input(input.size());
    DeviceBuffer<float> gpu_parameters(parameter_values.size());
    DeviceBuffer<float> gpu_cosine(cosine.size());
    DeviceBuffer<float> gpu_sine(sine.size());
    DeviceBuffer<float> gpu_output_gradient(output_gradient.size());
    DeviceBuffer<float> gpu_output(expected_output.size());
    DeviceBuffer<float> gpu_input_gradient(expected_input_gradient.size());
    DeviceBuffer<float> gpu_parameter_gradient(expected_parameter_gradient.size());
    DeviceBuffer<float> gpu_activations(
        dscuda::deepseek_v3_dense_block_activation_elements(CONFIG));
    DeviceBuffer<float> gpu_workspace(
        dscuda::deepseek_v3_dense_block_backward_workspace_elements(CONFIG));
    DeviceBuffer<__nv_bfloat16> gpu_bf16_workspace(
        dscuda::deepseek_v3_dense_block_bf16_workspace_elements(CONFIG));
    gpu_input.upload(input);
    gpu_parameters.upload(parameter_values);
    gpu_cosine.upload(cosine);
    gpu_sine.upload(sine);
    gpu_output_gradient.upload(output_gradient);
    gpu_input_gradient.zero();
    gpu_parameter_gradient.zero();

    dscuda::deepseek_v3_dense_block_forward_cuda(
        gpu_output.data(), gpu_input.data(), parameters(gpu_parameters.data()),
        gpu_cosine.data(), gpu_sine.data(), gpu_activations.data(),
        gpu_bf16_workspace.data(), CONFIG);
    dscuda::deepseek_v3_dense_block_backward_cuda(
        gpu_input_gradient.data(), gradients(gpu_parameter_gradient.data()),
        gpu_output_gradient.data(), gpu_input.data(),
        parameters(gpu_parameters.data()), gpu_cosine.data(), gpu_sine.data(),
        gpu_activations.data(), gpu_workspace.data(),
        gpu_bf16_workspace.data(), CONFIG);
    dscuda::synchronize();

    std::printf("DeepSeek-V3 dense transformer block test\n");
    bool passed = true;
    passed &= check("output", expected_output, gpu_output.download(), 8.0e-5F);
    passed &= check(
        "input gradient", expected_input_gradient,
        gpu_input_gradient.download(), 2.0e-4F);
    passed &= check(
        "parameter gradient", expected_parameter_gradient,
        gpu_parameter_gradient.download(), 3.0e-4F);
    return passed ? 0 : 1;
}

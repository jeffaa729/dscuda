// Compares one complete sequential MTP module and its reverse graph against the scalar CPU oracle.
// The test covers future-token embedding gather, dual RMSNorm fusion, projection, a full V3 block, and gradient scatter into both the prior depth and shared embedding table.

#include "cuda_common.h"
#include "mtp.h"
#include "mtp_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int B = 1;
constexpr int T = 5;
constexpr int DEPTH = 1;
constexpr int MT = T - DEPTH;
constexpr int D = 32;
constexpr int H = 2;
constexpr int Q = 32;
constexpr int C = 32;
constexpr int N = 16;
constexpr int R = 8;
constexpr int V = 16;
constexpr int F = 16;
constexpr int E = 4;
constexpr int S = 1;
constexpr int K = 2;
constexpr int VOCAB = 19;
constexpr int ROWS = B * MT;
constexpr int SF = S * F;
constexpr dscuda::MlaLayerConfig MLA{
    B, MT, D, H, Q, C, N, R, V, 1.0e-5F, 0.204124145F};
constexpr dscuda::DeepSeekMoeConfig MOE{
    ROWS, D, F, E, S, K, 1.0F};
constexpr dscuda::DeepSeekV3BlockConfig BLOCK{MLA, MOE, 1.0e-5F};
constexpr dscuda::MtpConfig CONFIG{B, T, D, DEPTH, 1.0e-5F, BLOCK};

struct Offsets {
    std::size_t hidden_norm = 0;
    std::size_t embedding_norm = hidden_norm + D;
    std::size_t projection = embedding_norm + D;
    std::size_t attention_norm = projection + 2 * D * D;
    std::size_t query_down = attention_norm + D;
    std::size_t query_norm = query_down + D * Q;
    std::size_t query_up = query_norm + Q;
    std::size_t kv_down = query_up + Q * H * (N + R);
    std::size_t kv_norm = kv_down + D * (C + R);
    std::size_t key_up = kv_norm + C;
    std::size_t value_up = key_up + H * N * C;
    std::size_t mla_output = value_up + H * C * V;
    std::size_t ffn_norm = mla_output + H * V * D;
    std::size_t router = ffn_norm + D;
    std::size_t bias = router + D * E;
    std::size_t routed_gate = bias + E;
    std::size_t routed_up = routed_gate + E * D * F;
    std::size_t routed_down = routed_up + E * D * F;
    std::size_t shared_gate = routed_down + E * F * D;
    std::size_t shared_up = shared_gate + D * SF;
    std::size_t shared_down = shared_up + D * SF;
    std::size_t elements = shared_down + SF * D;
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
            (std::sin(0.043F * index + phase)
             + 0.3F * std::cos(0.079F * index - phase));
    }
    return values;
}

dscuda::DeepSeekV3BlockParameters block_parameters(const float* p) {
    return {
        p + OFFSETS.attention_norm,
        {
            p + OFFSETS.query_down, p + OFFSETS.query_norm,
            p + OFFSETS.query_up, p + OFFSETS.kv_down,
            p + OFFSETS.kv_norm, p + OFFSETS.key_up,
            p + OFFSETS.value_up, p + OFFSETS.mla_output,
        },
        p + OFFSETS.ffn_norm,
        {
            p + OFFSETS.router, p + OFFSETS.bias,
            p + OFFSETS.routed_gate, p + OFFSETS.routed_up,
            p + OFFSETS.routed_down, p + OFFSETS.shared_gate,
            p + OFFSETS.shared_up, p + OFFSETS.shared_down,
        },
    };
}

dscuda::DeepSeekV3BlockGradients block_gradients(float* p) {
    return {
        p + OFFSETS.attention_norm,
        {
            p + OFFSETS.query_down, p + OFFSETS.query_norm,
            p + OFFSETS.query_up, p + OFFSETS.kv_down,
            p + OFFSETS.kv_norm, p + OFFSETS.key_up,
            p + OFFSETS.value_up, p + OFFSETS.mla_output,
        },
        p + OFFSETS.ffn_norm,
        {
            p + OFFSETS.router, p + OFFSETS.routed_gate,
            p + OFFSETS.routed_up, p + OFFSETS.routed_down,
            p + OFFSETS.shared_gate, p + OFFSETS.shared_up,
            p + OFFSETS.shared_down,
        },
    };
}

dscuda::MtpParameters parameters(const float* p) {
    return {
        p + OFFSETS.hidden_norm,
        p + OFFSETS.embedding_norm,
        p + OFFSETS.projection,
        block_parameters(p),
    };
}

dscuda::MtpGradients gradients(float* p) {
    return {
        p + OFFSETS.hidden_norm,
        p + OFFSETS.embedding_norm,
        p + OFFSETS.projection,
        block_gradients(p),
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
        "  %-22s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

}  // namespace

int main() {
    const auto previous_hidden = make_values(B * T * D, 0.11F, 0.1F);
    const std::vector<int> tokens{1, 3, 3, 7, 11};
    const auto embedding = make_values(VOCAB * D, 0.09F, 0.2F);
    auto parameter_values = make_values(
        static_cast<int>(OFFSETS.elements), 0.055F, 0.3F);
    std::fill_n(parameter_values.data() + OFFSETS.hidden_norm, D, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.embedding_norm, D, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.attention_norm, D, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.query_norm, Q, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.kv_norm, C, 1.0F);
    std::fill_n(parameter_values.data() + OFFSETS.ffn_norm, D, 1.0F);
    const auto output_gradient = make_values(ROWS * D, 0.07F, 0.5F);
    std::vector<float> cosine(MT * R / 2);
    std::vector<float> sine(MT * R / 2);
    for (int token = 0; token < MT; ++token) {
        for (int pair = 0; pair < R / 2; ++pair) {
            const float angle = token * std::pow(10000.0F, -2.0F * pair / R);
            cosine[token * (R / 2) + pair] = std::cos(angle);
            sine[token * (R / 2) + pair] = std::sin(angle);
        }
    }

    std::vector<float> expected_output(ROWS * D);
    std::vector<float> expected_previous_gradient(B * T * D, 0.0F);
    std::vector<float> expected_embedding_gradient(VOCAB * D, 0.0F);
    std::vector<float> expected_parameter_gradient(OFFSETS.elements, 0.0F);
    dscuda::mtp_forward_cpu(
        expected_output.data(),
        previous_hidden.data(),
        tokens.data(),
        embedding.data(),
        parameters(parameter_values.data()),
        cosine.data(),
        sine.data(),
        CONFIG);
    dscuda::mtp_backward_cpu(
        expected_previous_gradient.data(),
        expected_embedding_gradient.data(),
        gradients(expected_parameter_gradient.data()),
        output_gradient.data(),
        previous_hidden.data(),
        tokens.data(),
        embedding.data(),
        parameters(parameter_values.data()),
        cosine.data(),
        sine.data(),
        CONFIG);

    DeviceBuffer<float> gpu_previous(previous_hidden.size());
    DeviceBuffer<int> gpu_tokens(tokens.size());
    DeviceBuffer<float> gpu_embedding(embedding.size());
    DeviceBuffer<float> gpu_parameters(parameter_values.size());
    DeviceBuffer<float> gpu_cosine(cosine.size());
    DeviceBuffer<float> gpu_sine(sine.size());
    DeviceBuffer<float> gpu_output_gradient(output_gradient.size());
    DeviceBuffer<float> gpu_output(expected_output.size());
    DeviceBuffer<float> gpu_previous_gradient(expected_previous_gradient.size());
    DeviceBuffer<float> gpu_embedding_gradient(expected_embedding_gradient.size());
    DeviceBuffer<float> gpu_parameter_gradient(expected_parameter_gradient.size());
    DeviceBuffer<float> gpu_activations(dscuda::mtp_activation_elements(CONFIG));
    DeviceBuffer<int> gpu_integer_activations(
        dscuda::mtp_integer_activation_elements(CONFIG));
    DeviceBuffer<float> gpu_workspace(
        dscuda::mtp_backward_workspace_elements(CONFIG));
    DeviceBuffer<__nv_bfloat16> gpu_bf16_workspace(
        dscuda::mtp_bf16_workspace_elements(CONFIG));
    gpu_previous.upload(previous_hidden);
    gpu_tokens.upload(tokens);
    gpu_embedding.upload(embedding);
    gpu_parameters.upload(parameter_values);
    gpu_cosine.upload(cosine);
    gpu_sine.upload(sine);
    gpu_output_gradient.upload(output_gradient);
    gpu_previous_gradient.zero();
    gpu_embedding_gradient.zero();
    gpu_parameter_gradient.zero();

    dscuda::mtp_forward_cuda(
        gpu_output.data(),
        gpu_previous.data(),
        gpu_tokens.data(),
        gpu_embedding.data(),
        parameters(gpu_parameters.data()),
        gpu_cosine.data(),
        gpu_sine.data(),
        gpu_activations.data(),
        gpu_integer_activations.data(),
        gpu_bf16_workspace.data(),
        CONFIG);
    dscuda::mtp_backward_cuda(
        gpu_previous_gradient.data(),
        gpu_embedding_gradient.data(),
        gradients(gpu_parameter_gradient.data()),
        gpu_output_gradient.data(),
        gpu_previous.data(),
        gpu_tokens.data(),
        gpu_embedding.data(),
        parameters(gpu_parameters.data()),
        gpu_cosine.data(),
        gpu_sine.data(),
        gpu_activations.data(),
        gpu_integer_activations.data(),
        gpu_workspace.data(),
        gpu_bf16_workspace.data(),
        CONFIG);
    dscuda::synchronize();

    std::printf("Sequential MTP module test\n");
    bool passed = true;
    passed &= check("output", expected_output, gpu_output.download(), 8.0e-5F);
    passed &= check(
        "previous hidden grad",
        expected_previous_gradient,
        gpu_previous_gradient.download(),
        3.0e-4F);
    passed &= check(
        "embedding gradient",
        expected_embedding_gradient,
        gpu_embedding_gradient.download(),
        3.0e-4F);
    passed &= check(
        "parameter gradient",
        expected_parameter_gradient,
        gpu_parameter_gradient.download(),
        4.0e-4F);
    return passed ? 0 : 1;
}

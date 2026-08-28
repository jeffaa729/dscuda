// Compares the complete routed-plus-shared DeepSeekMoE forward and backward graphs against the scalar CPU reference.
// The test covers router, three matrices per routed expert, three shared-expert matrices, no-drop dispatch, and every trainable gradient.

#include "cuda_common.h"
#include "deepseek_moe.h"
#include "deepseek_moe_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int ROWS = 5;
constexpr int D = 16;
constexpr int F = 12;
constexpr int E = 4;
constexpr int S = 2;
constexpr int K = 2;
constexpr int SF = S * F;
constexpr dscuda::DeepSeekMoeConfig CONFIG{
    ROWS, D, F, E, S, K, 1.1F, 1, ROWS, 0.03F};

struct Offsets {
    std::size_t router = 0;
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
    void zero() { CUDA_CHECK(cudaMemset(data_, 0, elements_ * sizeof(T))); }
private:
    std::size_t elements_;
    T* data_;
};

std::vector<float> make_values(int elements, float scale, float phase) {
    std::vector<float> values(elements);
    for (int index = 0; index < elements; ++index) {
        values[index] = scale *
            (std::sin(0.061F * index + phase) +
             0.4F * std::cos(0.109F * index - phase));
    }
    return values;
}

dscuda::DeepSeekMoeParameters parameters(const float* values) {
    return {
        values + OFFSETS.router,
        values + OFFSETS.bias,
        values + OFFSETS.routed_gate,
        values + OFFSETS.routed_up,
        values + OFFSETS.routed_down,
        values + OFFSETS.shared_gate,
        values + OFFSETS.shared_up,
        values + OFFSETS.shared_down,
    };
}

dscuda::DeepSeekMoeGradients gradients(float* values) {
    return {
        values + OFFSETS.router,
        values + OFFSETS.routed_gate,
        values + OFFSETS.routed_up,
        values + OFFSETS.routed_down,
        values + OFFSETS.shared_gate,
        values + OFFSETS.shared_up,
        values + OFFSETS.shared_down,
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
    const auto input = make_values(ROWS * D, 0.14F, 0.1F);
    auto parameter_values = make_values(
        static_cast<int>(OFFSETS.elements), 0.09F, 0.3F);
    parameter_values[OFFSETS.bias + 0] = 0.04F;
    parameter_values[OFFSETS.bias + 1] = -0.02F;
    parameter_values[OFFSETS.bias + 2] = 0.01F;
    parameter_values[OFFSETS.bias + 3] = -0.03F;
    const auto output_gradient = make_values(ROWS * D, 0.1F, 0.5F);

    std::vector<float> expected_output(ROWS * D);
    std::vector<float> expected_input_gradient(ROWS * D, 0.0F);
    std::vector<float> expected_parameter_gradient(OFFSETS.elements, 0.0F);
    dscuda::deepseek_moe_forward_cpu(
        expected_output.data(), input.data(), parameters(parameter_values.data()),
        CONFIG);
    dscuda::deepseek_moe_backward_cpu(
        expected_input_gradient.data(), gradients(expected_parameter_gradient.data()),
        output_gradient.data(), input.data(), parameters(parameter_values.data()),
        CONFIG);

    DeviceBuffer<float> gpu_input(input.size());
    DeviceBuffer<float> gpu_parameters(parameter_values.size());
    DeviceBuffer<float> gpu_output_gradient(output_gradient.size());
    DeviceBuffer<float> gpu_output(expected_output.size());
    DeviceBuffer<float> gpu_input_gradient(expected_input_gradient.size());
    DeviceBuffer<float> gpu_parameter_gradient(expected_parameter_gradient.size());
    DeviceBuffer<float> gpu_activations(
        dscuda::deepseek_moe_activation_elements(CONFIG));
    DeviceBuffer<int> gpu_integer_activations(
        dscuda::deepseek_moe_integer_activation_elements(CONFIG));
    DeviceBuffer<float> gpu_workspace(
        dscuda::deepseek_moe_backward_workspace_elements(CONFIG));
    DeviceBuffer<float> gpu_balance_loss(1);
    gpu_input.upload(input);
    gpu_parameters.upload(parameter_values);
    gpu_output_gradient.upload(output_gradient);
    gpu_input_gradient.zero();
    gpu_parameter_gradient.zero();
    gpu_balance_loss.zero();

    dscuda::deepseek_moe_forward_cuda(
        gpu_output.data(), gpu_input.data(), parameters(gpu_parameters.data()),
        gpu_activations.data(), gpu_integer_activations.data(), CONFIG);
    dscuda::deepseek_moe_add_balance_loss_cuda(
        gpu_balance_loss.data(), gpu_activations.data(), CONFIG);
    dscuda::deepseek_moe_backward_cuda(
        gpu_input_gradient.data(), gradients(gpu_parameter_gradient.data()),
        gpu_output_gradient.data(), gpu_input.data(),
        parameters(gpu_parameters.data()), gpu_activations.data(),
        gpu_integer_activations.data(), gpu_workspace.data(), CONFIG);
    dscuda::synchronize();

    std::printf("Complete DeepSeekMoE test\n");
    bool passed = true;
    const float expected_balance_loss = dscuda::deepseek_moe_balance_loss_cpu(
        input.data(), parameters(parameter_values.data()), CONFIG);
    passed &= check(
        "balance loss",
        {expected_balance_loss},
        gpu_balance_loss.download(),
        1.0e-6F);
    passed &= check("output", expected_output, gpu_output.download(), 5.0e-5F);
    passed &= check(
        "input gradient", expected_input_gradient,
        gpu_input_gradient.download(), 1.0e-4F);
    passed &= check(
        "parameter gradient", expected_parameter_gradient,
        gpu_parameter_gradient.download(), 1.5e-4F);
    return passed ? 0 : 1;
}

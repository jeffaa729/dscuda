// Compares FP32 and BF16 fused causal attention against the scalar materialized CPU reference in forward and backward.
// It also runs the complete dense GPT graph with composed and fused attention to verify that the training integration preserves loss and gradients.

#include "attention_cpu.h"
#include "cuda_common.h"
#include "flash_attention.h"
#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int B = 2;
constexpr int T = 48;
constexpr int H = 3;
constexpr int D = 32;
constexpr int ACTIVATIONS = B * T * H * D;
constexpr int PROBABILITIES = B * H * T * T;
constexpr int ROWS = B * H * T;
constexpr float SCALE = 0.176776695F;

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

private:
    std::size_t elements_;
    T* data_;
};

float maximum_error(
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    float error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance) {
    const float error = maximum_error(expected, actual);
    const bool passed = error < tolerance;
    std::printf(
        "  %-28s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

void cpu_attention(
    std::vector<float>& output,
    std::vector<float>& query_gradient,
    std::vector<float>& key_gradient,
    std::vector<float>& value_gradient,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& output_gradient) {
    std::vector<float> probabilities(PROBABILITIES);
    dscuda::dense_attention_forward_cpu(
        output.data(),
        probabilities.data(),
        query.data(),
        key.data(),
        value.data(),
        B,
        T,
        H,
        D,
        SCALE);
    dscuda::dense_attention_backward_cpu(
        query_gradient.data(),
        key_gradient.data(),
        value_gradient.data(),
        output_gradient.data(),
        probabilities.data(),
        query.data(),
        key.data(),
        value.data(),
        B,
        T,
        H,
        D,
        SCALE);
}

bool test_fp32(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& output_gradient) {
    std::vector<float> cpu_output(ACTIVATIONS);
    std::vector<float> cpu_query_gradient(ACTIVATIONS);
    std::vector<float> cpu_key_gradient(ACTIVATIONS);
    std::vector<float> cpu_value_gradient(ACTIVATIONS);
    cpu_attention(
        cpu_output,
        cpu_query_gradient,
        cpu_key_gradient,
        cpu_value_gradient,
        query,
        key,
        value,
        output_gradient);

    DeviceBuffer<float> gpu_query(ACTIVATIONS);
    DeviceBuffer<float> gpu_key(ACTIVATIONS);
    DeviceBuffer<float> gpu_value(ACTIVATIONS);
    DeviceBuffer<float> gpu_output_gradient(ACTIVATIONS);
    DeviceBuffer<float> gpu_output(ACTIVATIONS);
    DeviceBuffer<float> gpu_logsumexp(ROWS);
    DeviceBuffer<float> gpu_query_gradient(ACTIVATIONS);
    DeviceBuffer<float> gpu_key_gradient(ACTIVATIONS);
    DeviceBuffer<float> gpu_value_gradient(ACTIVATIONS);
    gpu_query.upload(query);
    gpu_key.upload(key);
    gpu_value.upload(value);
    gpu_output_gradient.upload(output_gradient);
    CUDA_CHECK(cudaMemset(gpu_query_gradient.data(), 0, ACTIVATIONS * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_key_gradient.data(), 0, ACTIVATIONS * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_value_gradient.data(), 0, ACTIVATIONS * sizeof(float)));

    dscuda::flash_attention_forward_cuda(
        gpu_output.data(),
        gpu_logsumexp.data(),
        gpu_query.data(),
        gpu_key.data(),
        gpu_value.data(),
        B,
        T,
        H,
        D,
        SCALE);
    dscuda::flash_attention_backward_cuda(
        gpu_query_gradient.data(),
        gpu_key_gradient.data(),
        gpu_value_gradient.data(),
        gpu_output_gradient.data(),
        gpu_output.data(),
        gpu_logsumexp.data(),
        gpu_query.data(),
        gpu_key.data(),
        gpu_value.data(),
        B,
        T,
        H,
        D,
        SCALE);
    dscuda::synchronize();

    bool passed = true;
    passed &= check("FP32 output", cpu_output, gpu_output.download(), 3.0e-4F);
    passed &= check(
        "FP32 query gradient",
        cpu_query_gradient,
        gpu_query_gradient.download(),
        7.0e-4F);
    passed &= check(
        "FP32 key gradient",
        cpu_key_gradient,
        gpu_key_gradient.download(),
        7.0e-4F);
    passed &= check(
        "FP32 value gradient",
        cpu_value_gradient,
        gpu_value_gradient.download(),
        7.0e-4F);
    return passed;
}

bool test_bf16(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& output_gradient) {
    std::vector<__nv_bfloat16> bf16_query(ACTIVATIONS);
    std::vector<__nv_bfloat16> bf16_key(ACTIVATIONS);
    std::vector<__nv_bfloat16> bf16_value(ACTIVATIONS);
    std::vector<float> rounded_query(ACTIVATIONS);
    std::vector<float> rounded_key(ACTIVATIONS);
    std::vector<float> rounded_value(ACTIVATIONS);
    for (int index = 0; index < ACTIVATIONS; ++index) {
        bf16_query[index] = __float2bfloat16(query[index]);
        bf16_key[index] = __float2bfloat16(key[index]);
        bf16_value[index] = __float2bfloat16(value[index]);
        rounded_query[index] = __bfloat162float(bf16_query[index]);
        rounded_key[index] = __bfloat162float(bf16_key[index]);
        rounded_value[index] = __bfloat162float(bf16_value[index]);
    }

    std::vector<float> cpu_output(ACTIVATIONS);
    std::vector<float> cpu_query_gradient(ACTIVATIONS);
    std::vector<float> cpu_key_gradient(ACTIVATIONS);
    std::vector<float> cpu_value_gradient(ACTIVATIONS);
    cpu_attention(
        cpu_output,
        cpu_query_gradient,
        cpu_key_gradient,
        cpu_value_gradient,
        rounded_query,
        rounded_key,
        rounded_value,
        output_gradient);

    DeviceBuffer<__nv_bfloat16> gpu_query(ACTIVATIONS);
    DeviceBuffer<__nv_bfloat16> gpu_key(ACTIVATIONS);
    DeviceBuffer<__nv_bfloat16> gpu_value(ACTIVATIONS);
    DeviceBuffer<float> gpu_output_gradient(ACTIVATIONS);
    DeviceBuffer<float> gpu_output(ACTIVATIONS);
    DeviceBuffer<float> gpu_logsumexp(ROWS);
    DeviceBuffer<float> gpu_query_gradient(ACTIVATIONS);
    DeviceBuffer<float> gpu_key_gradient(ACTIVATIONS);
    DeviceBuffer<float> gpu_value_gradient(ACTIVATIONS);
    gpu_query.upload(bf16_query);
    gpu_key.upload(bf16_key);
    gpu_value.upload(bf16_value);
    gpu_output_gradient.upload(output_gradient);
    CUDA_CHECK(cudaMemset(gpu_query_gradient.data(), 0, ACTIVATIONS * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_key_gradient.data(), 0, ACTIVATIONS * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu_value_gradient.data(), 0, ACTIVATIONS * sizeof(float)));

    dscuda::flash_attention_forward_bf16_cuda(
        gpu_output.data(),
        gpu_logsumexp.data(),
        gpu_query.data(),
        gpu_key.data(),
        gpu_value.data(),
        B,
        T,
        H,
        D,
        SCALE);
    dscuda::flash_attention_backward_bf16_cuda(
        gpu_query_gradient.data(),
        gpu_key_gradient.data(),
        gpu_value_gradient.data(),
        gpu_output_gradient.data(),
        gpu_output.data(),
        gpu_logsumexp.data(),
        gpu_query.data(),
        gpu_key.data(),
        gpu_value.data(),
        B,
        T,
        H,
        D,
        SCALE);
    dscuda::synchronize();

    bool passed = true;
    passed &= check("BF16 output", cpu_output, gpu_output.download(), 3.0e-4F);
    passed &= check(
        "BF16 query gradient",
        cpu_query_gradient,
        gpu_query_gradient.download(),
        7.0e-4F);
    passed &= check(
        "BF16 key gradient",
        cpu_key_gradient,
        gpu_key_gradient.download(),
        7.0e-4F);
    passed &= check(
        "BF16 value gradient",
        cpu_value_gradient,
        gpu_value_gradient.download(),
        7.0e-4F);
    return passed;
}

bool test_tensor_core_bf16() {
    constexpr int tensor_batch = 1;
    constexpr int tensor_sequence = 64;
    constexpr int tensor_heads = 2;
    constexpr int tensor_head_size = 64;
    constexpr int tensor_activations =
        tensor_batch * tensor_sequence * tensor_heads * tensor_head_size;
    constexpr int tensor_probabilities =
        tensor_batch * tensor_heads * tensor_sequence * tensor_sequence;
    constexpr int tensor_rows =
        tensor_batch * tensor_heads * tensor_sequence;
    constexpr float tensor_scale = 0.125F;

    std::vector<__nv_bfloat16> bf16_query(tensor_activations);
    std::vector<__nv_bfloat16> bf16_key(tensor_activations);
    std::vector<__nv_bfloat16> bf16_value(tensor_activations);
    std::vector<float> query(tensor_activations);
    std::vector<float> key(tensor_activations);
    std::vector<float> value(tensor_activations);
    std::vector<float> output_gradient(tensor_activations);
    for (int index = 0; index < tensor_activations; ++index) {
        bf16_query[index] = __float2bfloat16(
            static_cast<float>((index * 17) % 101 - 50) / 64.0F);
        bf16_key[index] = __float2bfloat16(
            static_cast<float>((index * 23) % 97 - 48) / 61.0F);
        bf16_value[index] = __float2bfloat16(
            static_cast<float>((index * 31) % 89 - 44) / 59.0F);
        query[index] = __bfloat162float(bf16_query[index]);
        key[index] = __bfloat162float(bf16_key[index]);
        value[index] = __bfloat162float(bf16_value[index]);
        output_gradient[index] =
            static_cast<float>((index * 37) % 83 - 41) / 67.0F;
    }

    std::vector<float> cpu_output(tensor_activations);
    std::vector<float> cpu_probabilities(tensor_probabilities);
    std::vector<float> cpu_query_gradient(tensor_activations);
    std::vector<float> cpu_key_gradient(tensor_activations);
    std::vector<float> cpu_value_gradient(tensor_activations);
    dscuda::dense_attention_forward_cpu(
        cpu_output.data(),
        cpu_probabilities.data(),
        query.data(),
        key.data(),
        value.data(),
        tensor_batch,
        tensor_sequence,
        tensor_heads,
        tensor_head_size,
        tensor_scale);
    dscuda::dense_attention_backward_cpu(
        cpu_query_gradient.data(),
        cpu_key_gradient.data(),
        cpu_value_gradient.data(),
        output_gradient.data(),
        cpu_probabilities.data(),
        query.data(),
        key.data(),
        value.data(),
        tensor_batch,
        tensor_sequence,
        tensor_heads,
        tensor_head_size,
        tensor_scale);

    DeviceBuffer<__nv_bfloat16> gpu_query(tensor_activations);
    DeviceBuffer<__nv_bfloat16> gpu_key(tensor_activations);
    DeviceBuffer<__nv_bfloat16> gpu_value(tensor_activations);
    DeviceBuffer<float> gpu_output_gradient(tensor_activations);
    DeviceBuffer<float> gpu_output(tensor_activations);
    DeviceBuffer<float> gpu_logsumexp(tensor_rows);
    DeviceBuffer<float> gpu_query_gradient(tensor_activations);
    DeviceBuffer<float> gpu_key_gradient(tensor_activations);
    DeviceBuffer<float> gpu_value_gradient(tensor_activations);
    gpu_query.upload(bf16_query);
    gpu_key.upload(bf16_key);
    gpu_value.upload(bf16_value);
    gpu_output_gradient.upload(output_gradient);
    CUDA_CHECK(cudaMemset(
        gpu_query_gradient.data(), 0, tensor_activations * sizeof(float)));
    CUDA_CHECK(cudaMemset(
        gpu_key_gradient.data(), 0, tensor_activations * sizeof(float)));
    CUDA_CHECK(cudaMemset(
        gpu_value_gradient.data(), 0, tensor_activations * sizeof(float)));

    dscuda::flash_attention_forward_bf16_cuda(
        gpu_output.data(),
        gpu_logsumexp.data(),
        gpu_query.data(),
        gpu_key.data(),
        gpu_value.data(),
        tensor_batch,
        tensor_sequence,
        tensor_heads,
        tensor_head_size,
        tensor_scale);
    dscuda::flash_attention_backward_bf16_cuda(
        gpu_query_gradient.data(),
        gpu_key_gradient.data(),
        gpu_value_gradient.data(),
        gpu_output_gradient.data(),
        gpu_output.data(),
        gpu_logsumexp.data(),
        gpu_query.data(),
        gpu_key.data(),
        gpu_value.data(),
        tensor_batch,
        tensor_sequence,
        tensor_heads,
        tensor_head_size,
        tensor_scale);
    dscuda::synchronize();

    bool passed = true;
    passed &= check(
        "Tensor Core output",
        cpu_output,
        gpu_output.download(),
        8.0e-3F);
    passed &= check(
        "Tensor Core query gradient",
        cpu_query_gradient,
        gpu_query_gradient.download(),
        2.0e-2F);
    passed &= check(
        "Tensor Core key gradient",
        cpu_key_gradient,
        gpu_key_gradient.download(),
        2.0e-2F);
    passed &= check(
        "Tensor Core value gradient",
        cpu_value_gradient,
        gpu_value_gradient.download(),
        2.0e-2F);
    return passed;
}

bool test_model_integration() {
    dscuda::ModelConfig composed_config{
        1, 16, 128, 1, 64, 4, 192, 16, 1.0e-5F,
        dscuda::AttentionImplementation::composed};
    dscuda::ModelConfig flash_config = composed_config;
    flash_config.attention = dscuda::AttentionImplementation::flash2;
    std::vector<int> inputs(16);
    std::vector<int> targets(16);
    for (int index = 0; index < 16; ++index) {
        inputs[index] = (index * 7 + 3) % 128;
        targets[index] = (index * 11 + 5) % 128;
    }

    dscuda::DenseGptModel composed(composed_config);
    dscuda::DenseGptModel flash(flash_config);
    composed.initialize(2026);
    flash.load_parameters(composed.parameters_to_host());
    const float composed_loss = composed.forward(inputs, targets);
    const float flash_loss = flash.forward(inputs, targets);
    const float logits_error = maximum_error(
        composed.logits_to_host(), flash.logits_to_host());
    composed.zero_gradients();
    flash.zero_gradients();
    composed.backward();
    flash.backward();
    const float gradients_error = maximum_error(
        composed.gradients_to_host(), flash.gradients_to_host());
    const bool passed = std::abs(composed_loss - flash_loss) < 2.0e-4F
        && logits_error < 4.0e-4F && gradients_error < 2.0e-3F
        && flash.memory_report().saved_activation_elements
            < composed.memory_report().saved_activation_elements;
    std::printf(
        "  model loss error %.3e, logits error %.3e, gradients error %.3e  %s\n",
        std::abs(composed_loss - flash_loss),
        logits_error,
        gradients_error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool test_bf16_model_integration() {
    dscuda::ModelConfig composed_config{
        1, 64, 128, 1, 256, 4, 384, 64, 1.0e-5F,
        dscuda::AttentionImplementation::composed};
    dscuda::ModelConfig flash_config = composed_config;
    flash_config.attention = dscuda::AttentionImplementation::flash2;
    std::vector<int> inputs(64);
    std::vector<int> targets(64);
    for (int index = 0; index < 64; ++index) {
        inputs[index] = (index * 7 + 3) % 128;
        targets[index] = (index * 11 + 5) % 128;
    }

    dscuda::DenseGptModel composed(
        composed_config, dscuda::ModelPrecision::bf16);
    dscuda::DenseGptModel flash(
        flash_config, dscuda::ModelPrecision::bf16);
    composed.initialize(2027);
    flash.load_parameters(composed.parameters_to_host());
    const float composed_loss = composed.forward(inputs, targets);
    const float flash_loss = flash.forward(inputs, targets);
    const float logits_error = maximum_error(
        composed.logits_to_host(), flash.logits_to_host());
    composed.zero_gradients();
    flash.zero_gradients();
    composed.backward();
    flash.backward();
    const float gradients_error = maximum_error(
        composed.gradients_to_host(), flash.gradients_to_host());
    const bool passed = std::abs(composed_loss - flash_loss) < 5.0e-3F
        && logits_error < 1.0e-2F && gradients_error < 2.0e-2F;
    std::printf(
        "  BF16 model loss error %.3e, logits error %.3e, gradients error %.3e  %s\n",
        std::abs(composed_loss - flash_loss),
        logits_error,
        gradients_error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool run_test() {
    std::vector<float> query(ACTIVATIONS);
    std::vector<float> key(ACTIVATIONS);
    std::vector<float> value(ACTIVATIONS);
    std::vector<float> output_gradient(ACTIVATIONS);
    for (int index = 0; index < ACTIVATIONS; ++index) {
        query[index] = static_cast<float>((index * 17) % 101 - 50) / 64.0F;
        key[index] = static_cast<float>((index * 23) % 97 - 48) / 61.0F;
        value[index] = static_cast<float>((index * 31) % 89 - 44) / 59.0F;
        output_gradient[index] =
            static_cast<float>((index * 37) % 83 - 41) / 67.0F;
    }
    bool passed = test_fp32(query, key, value, output_gradient);
    passed &= test_bf16(query, key, value, output_gradient);
    passed &= test_tensor_core_bf16();
    passed &= test_model_integration();
    passed &= test_bf16_model_integration();
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Flash attention test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Flash attention test failed: %s\n", error.what());
        return 1;
    }
}

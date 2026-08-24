// Verifies the complete transformer block against a recomputing scalar CPU reference in forward and backward.
// Selected finite differences additionally check the composed analytical graph rather than only CPU-to-GPU agreement.

#include "cuda_common.h"
#include "transformer_block.h"
#include "transformer_block_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr dscuda::TransformerBlockConfig kConfig{
    2, 12, 32, 4, 8, 48, 8, 1.0e-5F, 0.353553390593F};

void fill_pattern(std::vector<float>& values, int multiplier, float scale) {
    for (int index = 0; index < static_cast<int>(values.size()); ++index) {
        values[index] =
            static_cast<float>((index * multiplier) % 101 - 50) * scale;
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements)
        : elements_(elements),
          data_(static_cast<float*>(
              dscuda::device_malloc(elements * sizeof(float)))) {}

    ~DeviceBuffer() {
        dscuda::device_free(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() {
        return data_;
    }

    const float* data() const {
        return data_;
    }

    void upload(const std::vector<float>& values) {
        CUDA_CHECK(cudaMemcpy(
            data_,
            values.data(),
            elements_ * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    std::vector<float> download() const {
        std::vector<float> values(elements_);
        CUDA_CHECK(cudaMemcpy(
            values.data(),
            data_,
            elements_ * sizeof(float),
            cudaMemcpyDeviceToHost));
        return values;
    }

private:
    std::size_t elements_;
    float* data_;
};

struct HostParameters {
    std::vector<float> attention_norm;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> output;
    std::vector<float> ffn_norm;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;

    HostParameters()
        : attention_norm(kConfig.hidden_size),
          query(kConfig.hidden_size * kConfig.hidden_size),
          key(kConfig.hidden_size * kConfig.hidden_size),
          value(kConfig.hidden_size * kConfig.hidden_size),
          output(kConfig.hidden_size * kConfig.hidden_size),
          ffn_norm(kConfig.hidden_size),
          gate(kConfig.hidden_size * kConfig.ffn_size),
          up(kConfig.hidden_size * kConfig.ffn_size),
          down(kConfig.ffn_size * kConfig.hidden_size) {
        fill_pattern(attention_norm, 7, 0.002F);
        fill_pattern(query, 11, 0.0015F);
        fill_pattern(key, 13, 0.0014F);
        fill_pattern(value, 17, 0.0013F);
        fill_pattern(output, 19, 0.0012F);
        fill_pattern(ffn_norm, 23, 0.002F);
        fill_pattern(gate, 29, 0.0012F);
        fill_pattern(up, 31, 0.0011F);
        fill_pattern(down, 37, 0.0010F);
        for (int index = 0; index < kConfig.hidden_size; ++index) {
            attention_norm[index] += 1.0F;
            ffn_norm[index] += 1.0F;
        }
    }

    dscuda::TransformerBlockParameters view() const {
        return {
            attention_norm.data(),
            query.data(),
            key.data(),
            value.data(),
            output.data(),
            ffn_norm.data(),
            gate.data(),
            up.data(),
            down.data()};
    }
};

struct HostGradients {
    std::vector<float> attention_norm;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> output;
    std::vector<float> ffn_norm;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;

    HostGradients()
        : attention_norm(kConfig.hidden_size),
          query(kConfig.hidden_size * kConfig.hidden_size),
          key(kConfig.hidden_size * kConfig.hidden_size),
          value(kConfig.hidden_size * kConfig.hidden_size),
          output(kConfig.hidden_size * kConfig.hidden_size),
          ffn_norm(kConfig.hidden_size),
          gate(kConfig.hidden_size * kConfig.ffn_size),
          up(kConfig.hidden_size * kConfig.ffn_size),
          down(kConfig.ffn_size * kConfig.hidden_size) {}

    void initialize() {
        fill_pattern(attention_norm, 3, 0.0001F);
        fill_pattern(query, 5, 0.0001F);
        fill_pattern(key, 7, 0.0001F);
        fill_pattern(value, 11, 0.0001F);
        fill_pattern(output, 13, 0.0001F);
        fill_pattern(ffn_norm, 17, 0.0001F);
        fill_pattern(gate, 19, 0.0001F);
        fill_pattern(up, 23, 0.0001F);
        fill_pattern(down, 29, 0.0001F);
    }

    dscuda::TransformerBlockGradients view() {
        return {
            attention_norm.data(),
            query.data(),
            key.data(),
            value.data(),
            output.data(),
            ffn_norm.data(),
            gate.data(),
            up.data(),
            down.data()};
    }
};

struct DeviceParameters {
    DeviceBuffer attention_norm;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer output;
    DeviceBuffer ffn_norm;
    DeviceBuffer gate;
    DeviceBuffer up;
    DeviceBuffer down;

    explicit DeviceParameters(const HostParameters& host)
        : attention_norm(host.attention_norm.size()),
          query(host.query.size()),
          key(host.key.size()),
          value(host.value.size()),
          output(host.output.size()),
          ffn_norm(host.ffn_norm.size()),
          gate(host.gate.size()),
          up(host.up.size()),
          down(host.down.size()) {
        attention_norm.upload(host.attention_norm);
        query.upload(host.query);
        key.upload(host.key);
        value.upload(host.value);
        output.upload(host.output);
        ffn_norm.upload(host.ffn_norm);
        gate.upload(host.gate);
        up.upload(host.up);
        down.upload(host.down);
    }

    dscuda::TransformerBlockParameters view() const {
        return {
            attention_norm.data(),
            query.data(),
            key.data(),
            value.data(),
            output.data(),
            ffn_norm.data(),
            gate.data(),
            up.data(),
            down.data()};
    }
};

struct DeviceGradients {
    DeviceBuffer attention_norm;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer output;
    DeviceBuffer ffn_norm;
    DeviceBuffer gate;
    DeviceBuffer up;
    DeviceBuffer down;

    explicit DeviceGradients(const HostGradients& host)
        : attention_norm(host.attention_norm.size()),
          query(host.query.size()),
          key(host.key.size()),
          value(host.value.size()),
          output(host.output.size()),
          ffn_norm(host.ffn_norm.size()),
          gate(host.gate.size()),
          up(host.up.size()),
          down(host.down.size()) {
        attention_norm.upload(host.attention_norm);
        query.upload(host.query);
        key.upload(host.key);
        value.upload(host.value);
        output.upload(host.output);
        ffn_norm.upload(host.ffn_norm);
        gate.upload(host.gate);
        up.upload(host.up);
        down.upload(host.down);
    }

    dscuda::TransformerBlockGradients view() {
        return {
            attention_norm.data(),
            query.data(),
            key.data(),
            value.data(),
            output.data(),
            ffn_norm.data(),
            gate.data(),
            up.data(),
            down.data()};
    }
};

float max_error(
    const std::vector<float>& expected,
    const std::vector<float>& actual) {
    float error = 0.0F;
    for (int index = 0; index < static_cast<int>(expected.size()); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance) {
    const float error = max_error(expected, actual);
    const bool passed = error < tolerance;
    std::printf(
        "  %-28s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

float loss(
    const std::vector<float>& input,
    const HostParameters& parameters,
    const std::vector<float>& cosine,
    const std::vector<float>& sine,
    const std::vector<float>& output_gradient) {
    std::vector<float> output(input.size());
    dscuda::transformer_block_forward_cpu(
        output.data(),
        input.data(),
        parameters.view(),
        cosine.data(),
        sine.data(),
        kConfig);
    float result = 0.0F;
    for (int index = 0; index < static_cast<int>(output.size()); ++index) {
        result += output[index] * output_gradient[index];
    }
    return result;
}

bool run_test() {
    const int rows = kConfig.batch_size * kConfig.sequence_length;
    const int elements = rows * kConfig.hidden_size;
    const int frequencies = kConfig.sequence_length * kConfig.rotary_size / 2;
    std::vector<float> input(elements);
    std::vector<float> output_gradient(elements);
    std::vector<float> cosine(frequencies);
    std::vector<float> sine(frequencies);
    fill_pattern(input, 41, 0.0125F);
    fill_pattern(output_gradient, 43, 0.004F);
    for (int index = 0; index < frequencies; ++index) {
        const float angle = static_cast<float>(index) * 0.017F;
        cosine[index] = std::cos(angle);
        sine[index] = std::sin(angle);
    }

    HostParameters parameters;
    HostGradients cpu_gradients;
    cpu_gradients.initialize();
    const HostGradients initial_gradients = cpu_gradients;
    std::vector<float> initial_input_gradient(elements);
    fill_pattern(initial_input_gradient, 47, 0.0001F);
    std::vector<float> cpu_input_gradient = initial_input_gradient;
    std::vector<float> cpu_output(elements);
    dscuda::transformer_block_forward_cpu(
        cpu_output.data(),
        input.data(),
        parameters.view(),
        cosine.data(),
        sine.data(),
        kConfig);
    dscuda::transformer_block_backward_cpu(
        cpu_input_gradient.data(),
        cpu_gradients.view(),
        output_gradient.data(),
        input.data(),
        parameters.view(),
        cosine.data(),
        sine.data(),
        kConfig);

    const int input_probe = 17;
    const int weight_probe = 73;
    constexpr float finite_step = 0.001F;
    input[input_probe] += finite_step;
    const float input_loss_plus =
        loss(input, parameters, cosine, sine, output_gradient);
    input[input_probe] -= 2.0F * finite_step;
    const float input_loss_minus =
        loss(input, parameters, cosine, sine, output_gradient);
    input[input_probe] += finite_step;
    const float input_finite_difference =
        (input_loss_plus - input_loss_minus) / (2.0F * finite_step);
    const float input_analytical =
        cpu_input_gradient[input_probe] - initial_input_gradient[input_probe];

    parameters.query[weight_probe] += finite_step;
    const float weight_loss_plus =
        loss(input, parameters, cosine, sine, output_gradient);
    parameters.query[weight_probe] -= 2.0F * finite_step;
    const float weight_loss_minus =
        loss(input, parameters, cosine, sine, output_gradient);
    parameters.query[weight_probe] += finite_step;
    const float weight_finite_difference =
        (weight_loss_plus - weight_loss_minus) / (2.0F * finite_step);
    const float weight_analytical =
        cpu_gradients.query[weight_probe] - initial_gradients.query[weight_probe];

    DeviceParameters gpu_parameters(parameters);
    DeviceGradients gpu_gradients(initial_gradients);
    DeviceBuffer gpu_input(elements);
    DeviceBuffer gpu_output_gradient(elements);
    DeviceBuffer gpu_cosine(frequencies);
    DeviceBuffer gpu_sine(frequencies);
    DeviceBuffer gpu_output(elements);
    DeviceBuffer gpu_input_gradient(elements);
    DeviceBuffer gpu_activations(
        dscuda::transformer_block_activation_elements(kConfig));
    DeviceBuffer gpu_workspace(
        dscuda::transformer_block_backward_workspace_elements(kConfig));
    gpu_input.upload(input);
    gpu_output_gradient.upload(output_gradient);
    gpu_cosine.upload(cosine);
    gpu_sine.upload(sine);
    gpu_input_gradient.upload(initial_input_gradient);

    dscuda::transformer_block_forward_cuda(
        gpu_output.data(),
        gpu_input.data(),
        gpu_parameters.view(),
        gpu_cosine.data(),
        gpu_sine.data(),
        gpu_activations.data(),
        kConfig);
    dscuda::transformer_block_backward_cuda(
        gpu_input_gradient.data(),
        gpu_gradients.view(),
        gpu_output_gradient.data(),
        gpu_input.data(),
        gpu_parameters.view(),
        gpu_cosine.data(),
        gpu_sine.data(),
        gpu_activations.data(),
        gpu_workspace.data(),
        kConfig);
    dscuda::synchronize();

    bool passed = true;
    passed &= check("output", cpu_output, gpu_output.download(), 3.0e-4F);
    passed &= check(
        "input gradient",
        cpu_input_gradient,
        gpu_input_gradient.download(),
        8.0e-4F);
    passed &= check(
        "attention norm gradient",
        cpu_gradients.attention_norm,
        gpu_gradients.attention_norm.download(),
        8.0e-4F);
    passed &= check(
        "query weight gradient",
        cpu_gradients.query,
        gpu_gradients.query.download(),
        8.0e-4F);
    passed &= check(
        "key weight gradient",
        cpu_gradients.key,
        gpu_gradients.key.download(),
        8.0e-4F);
    passed &= check(
        "value weight gradient",
        cpu_gradients.value,
        gpu_gradients.value.download(),
        8.0e-4F);
    passed &= check(
        "output weight gradient",
        cpu_gradients.output,
        gpu_gradients.output.download(),
        8.0e-4F);
    passed &= check(
        "FFN norm gradient",
        cpu_gradients.ffn_norm,
        gpu_gradients.ffn_norm.download(),
        8.0e-4F);
    passed &= check(
        "gate weight gradient",
        cpu_gradients.gate,
        gpu_gradients.gate.download(),
        8.0e-4F);
    passed &= check(
        "up weight gradient",
        cpu_gradients.up,
        gpu_gradients.up.download(),
        8.0e-4F);
    passed &= check(
        "down weight gradient",
        cpu_gradients.down,
        gpu_gradients.down.download(),
        8.0e-4F);

    const float input_finite_error =
        std::abs(input_analytical - input_finite_difference);
    const float weight_finite_error =
        std::abs(weight_analytical - weight_finite_difference);
    std::printf(
        "  %-28s error = %.3e  %s\n",
        "input finite difference",
        input_finite_error,
        input_finite_error < 3.0e-3F ? "PASS" : "FAIL");
    std::printf(
        "  %-28s error = %.3e  %s\n",
        "query finite difference",
        weight_finite_error,
        weight_finite_error < 3.0e-3F ? "PASS" : "FAIL");
    passed &= input_finite_error < 3.0e-3F;
    passed &= weight_finite_error < 3.0e-3F;
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Transformer block test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Transformer block test failed: %s\n", error.what());
        return 1;
    }
}

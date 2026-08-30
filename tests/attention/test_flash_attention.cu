// Compares FP32 and BF16 fused causal attention against the scalar materialized CPU reference in forward and backward.
// It checks native BF16 outputs, gradient overwrite semantics, and the D128 specialization independently of any model.

#include "attention_cpu.h"
#include "cuda_common.h"
#include "flash_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int B = 2;
constexpr int T = 47;
constexpr int H = 3;
constexpr int D = 128;
constexpr int ACTIVATIONS = B * T * H * D;
constexpr int PROBABILITIES = B * H * T * T;
constexpr int ROWS = B * H * T;
constexpr float SCALE = 0.0883883476F;

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
        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
            return std::numeric_limits<float>::infinity();
        }
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

bool check(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance,
    float relative_tolerance = 0.0F) {
    const float error = maximum_error(expected, actual);
    bool passed = std::isfinite(error);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        passed &= std::abs(expected[index] - actual[index])
            <= tolerance + relative_tolerance * std::abs(expected[index]);
    }
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

bool test_tensor_core_bf16(int tensor_sequence, int tensor_batch) {
    constexpr int tensor_heads = 2;
    constexpr int tensor_head_size = 128;
    const int tensor_activations =
        tensor_batch * tensor_sequence * tensor_heads * tensor_head_size;
    const int tensor_probabilities =
        tensor_batch * tensor_heads * tensor_sequence * tensor_sequence;
    const int tensor_rows =
        tensor_batch * tensor_heads * tensor_sequence;
    constexpr float tensor_scale = SCALE;
    std::printf("D128 Tensor Core: B=%d T=%d H=%d\n",
                tensor_batch, tensor_sequence, tensor_heads);

    std::vector<__nv_bfloat16> bf16_query(tensor_activations);
    std::vector<__nv_bfloat16> bf16_key(tensor_activations);
    std::vector<__nv_bfloat16> bf16_value(tensor_activations);
    std::vector<float> query(tensor_activations);
    std::vector<float> key(tensor_activations);
    std::vector<float> value(tensor_activations);
    std::vector<float> output_gradient(tensor_activations);
    std::mt19937 random(2026 + tensor_sequence);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    for (int index = 0; index < tensor_activations; ++index) {
        bf16_query[index] = __float2bfloat16(uniform(random));
        bf16_key[index] = __float2bfloat16(uniform(random));
        bf16_value[index] = __float2bfloat16(uniform(random));
        query[index] = __bfloat162float(bf16_query[index]);
        key[index] = __bfloat162float(bf16_key[index]);
        value[index] = __bfloat162float(bf16_value[index]);
        // Match the BF16 upstream gradient used by the official benchmark;
        // the FP32-output API stores these values in an FP32 buffer.
        output_gradient[index] = __bfloat162float(__float2bfloat16(uniform(random)));
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
    auto backward = [&]() {
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
    };
    backward();
    dscuda::synchronize();

    // Recover the reference LSE from the first visible key: log(P0) = S0 - LSE.
    std::vector<float> cpu_logsumexp(tensor_rows);
    for (int batch = 0; batch < tensor_batch; ++batch) {
        for (int head = 0; head < tensor_heads; ++head) {
            const int key_offset = (batch * tensor_sequence * tensor_heads + head) * D;
            for (int token = 0; token < tensor_sequence; ++token) {
                const int query_offset = key_offset + token * tensor_heads * D;
                float dot = 0.0F;
                for (int column = 0; column < D; ++column) {
                    dot += query[query_offset + column] * key[key_offset + column];
                }
                const int row = (batch * tensor_heads + head) * tensor_sequence + token;
                cpu_logsumexp[row] = tensor_scale * dot
                    - std::log(cpu_probabilities[row * tensor_sequence]);
            }
        }
    }
    bool passed = check("Tensor Core logsumexp", cpu_logsumexp,
                        gpu_logsumexp.download(), 2.0e-5F);
    // P, dS and dO are rounded to BF16 inside the Tensor Core path. Use an
    // absolute-plus-relative bound against the unrounded FP32 CPU computation.
    passed &= check(
        "Tensor Core output",
        cpu_output,
        gpu_output.download(),
        2.0e-3F, 1.0e-2F);
    passed &= check(
        "Tensor Core query gradient",
        cpu_query_gradient,
        gpu_query_gradient.download(),
        2.0e-3F, 1.0e-2F);
    passed &= check(
        "Tensor Core key gradient",
        cpu_key_gradient,
        gpu_key_gradient.download(),
        2.0e-3F, 1.0e-2F);
    passed &= check(
        "Tensor Core value gradient",
        cpu_value_gradient,
        gpu_value_gradient.download(),
        2.0e-3F, 1.0e-2F);
    // A second backward must add to, rather than overwrite, every gradient.
    auto twice_query = gpu_query_gradient.download();
    auto twice_key = gpu_key_gradient.download();
    auto twice_value = gpu_value_gradient.download();
    for (int index = 0; index < tensor_activations; ++index) {
        twice_query[index] *= 2.0F;
        twice_key[index] *= 2.0F;
        twice_value[index] *= 2.0F;
    }
    backward();
    passed &= check("accumulated dQ", twice_query, gpu_query_gradient.download(), 1.0e-5F);
    passed &= check("accumulated dK", twice_key, gpu_key_gradient.download(), 1.0e-5F);
    passed &= check("accumulated dV", twice_value, gpu_value_gradient.download(), 1.0e-5F);

    // Exercise the exact BF16 IO / overwrite contract used in library comparisons.
    DeviceBuffer<__nv_bfloat16> native_output(tensor_activations);
    DeviceBuffer<__nv_bfloat16> native_dout(tensor_activations);
    DeviceBuffer<__nv_bfloat16> native_dq(tensor_activations);
    DeviceBuffer<__nv_bfloat16> native_dk(tensor_activations);
    DeviceBuffer<__nv_bfloat16> native_dv(tensor_activations);
    std::vector<__nv_bfloat16> native_host_dout(tensor_activations);
    std::vector<__nv_bfloat16> poison(tensor_activations, __float2bfloat16(NAN));
    for (int index = 0; index < tensor_activations; ++index) {
        native_host_dout[index] = __float2bfloat16(output_gradient[index]);
    }
    native_dout.upload(native_host_dout);
    native_dq.upload(poison);
    native_dk.upload(poison);
    native_dv.upload(poison);
    auto download_float = [](const DeviceBuffer<__nv_bfloat16>& buffer) {
        const auto values = buffer.download();
        std::vector<float> result(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            result[index] = __bfloat162float(values[index]);
        }
        return result;
    };
    dscuda::flash_attention_forward_bf16_io_cuda(
        native_output.data(), gpu_logsumexp.data(), gpu_query.data(),
        gpu_key.data(), gpu_value.data(), tensor_batch, tensor_sequence,
        tensor_heads, D, tensor_scale);
    auto native_backward = [&]() {
        dscuda::flash_attention_backward_bf16_io_cuda(
            native_dq.data(), native_dk.data(), native_dv.data(), native_dout.data(),
            native_output.data(), gpu_logsumexp.data(), gpu_query.data(),
            gpu_key.data(), gpu_value.data(), tensor_batch, tensor_sequence,
            tensor_heads, D, tensor_scale);
    };
    native_backward();
    const auto first_dq = download_float(native_dq);
    const auto first_dk = download_float(native_dk);
    const auto first_dv = download_float(native_dv);
    passed &= check("native BF16 output", cpu_output, download_float(native_output), 3.0e-3F, 1.0e-2F);
    passed &= check("native BF16 dQ", cpu_query_gradient, first_dq, 3.0e-3F, 1.0e-2F);
    passed &= check("native BF16 dK", cpu_key_gradient, first_dk, 3.0e-3F, 1.0e-2F);
    passed &= check("native BF16 dV", cpu_value_gradient, first_dv, 3.0e-3F, 1.0e-2F);
    native_backward();
    passed &= check("overwritten BF16 dQ", first_dq, download_float(native_dq), 0.0F);
    passed &= check("overwritten BF16 dK", first_dk, download_float(native_dk), 0.0F);
    passed &= check("overwritten BF16 dV", first_dv, download_float(native_dv), 0.0F);
    return passed;
}

bool test_reject_d64() {
    // The size check must run before any device pointer is dereferenced.
    try {
        dscuda::flash_attention_forward_bf16_cuda(
            nullptr, nullptr, nullptr, nullptr, nullptr, 1, 64, 1, 64, 0.125F);
    } catch (const std::runtime_error& error) {
        const bool passed = std::string(error.what()).find("128") != std::string::npos;
        std::printf("  rejects D64  %s\n", passed ? "PASS" : "FAIL");
        return passed;
    }
    return false;
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
    passed &= test_tensor_core_bf16(64, 1);
    passed &= test_tensor_core_bf16(128, 2);
    passed &= test_tensor_core_bf16(256, 1);
    passed &= test_reject_d64();
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

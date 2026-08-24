// Runs one complete transformer-block forward and backward workload for external Nsight Compute measurement.
// The executable owns all model, activation, gradient, and workspace buffers while leaving timing to the profiler.

#include "cuda_common.h"
#include "transformer_block.h"

#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

float* make_device_buffer(std::size_t elements, float value) {
    std::vector<float> host(elements, value);
    auto* device =
        static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        device,
        host.data(),
        elements * sizeof(float),
        cudaMemcpyHostToDevice));
    return device;
}

float* make_zero_buffer(std::size_t elements) {
    auto* device =
        static_cast<float*>(dscuda::device_malloc(elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(device, 0, elements * sizeof(float)));
    return device;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 2;
        const int sequence_length = argc > 2 ? std::atoi(argv[2]) : 128;
        const int hidden_size = argc > 3 ? std::atoi(argv[3]) : 512;
        const int heads = argc > 4 ? std::atoi(argv[4]) : 8;
        const int ffn_size = argc > 5 ? std::atoi(argv[5]) : 1536;
        const int head_size = hidden_size / heads;
        const dscuda::TransformerBlockConfig config{
            batch_size,
            sequence_length,
            hidden_size,
            heads,
            head_size,
            ffn_size,
            head_size,
            1.0e-5F,
            1.0F / std::sqrt(static_cast<float>(head_size))};

        const std::size_t rows =
            static_cast<std::size_t>(batch_size) * sequence_length;
        const std::size_t activation_elements = rows * hidden_size;
        const std::size_t square_weight_elements =
            static_cast<std::size_t>(hidden_size) * hidden_size;
        const std::size_t gate_weight_elements =
            static_cast<std::size_t>(hidden_size) * ffn_size;
        const std::size_t down_weight_elements =
            static_cast<std::size_t>(ffn_size) * hidden_size;
        const std::size_t frequency_elements =
            static_cast<std::size_t>(sequence_length) * head_size / 2;

        float* input = make_device_buffer(activation_elements, 0.125F);
        float* output_gradient =
            make_device_buffer(activation_elements, 0.03125F);
        float* output = make_zero_buffer(activation_elements);
        float* input_gradient = make_zero_buffer(activation_elements);
        float* cosine = make_device_buffer(frequency_elements, 0.968912F);
        float* sine = make_device_buffer(frequency_elements, 0.247404F);

        float* attention_norm = make_device_buffer(hidden_size, 1.0F);
        float* query = make_device_buffer(square_weight_elements, 0.001F);
        float* key = make_device_buffer(square_weight_elements, -0.001F);
        float* value = make_device_buffer(square_weight_elements, 0.001F);
        float* projection = make_device_buffer(square_weight_elements, 0.001F);
        float* ffn_norm = make_device_buffer(hidden_size, 1.0F);
        float* gate = make_device_buffer(gate_weight_elements, 0.001F);
        float* up = make_device_buffer(gate_weight_elements, -0.001F);
        float* down = make_device_buffer(down_weight_elements, 0.001F);
        const dscuda::TransformerBlockParameters parameters{
            attention_norm,
            query,
            key,
            value,
            projection,
            ffn_norm,
            gate,
            up,
            down};

        float* attention_norm_gradient = make_zero_buffer(hidden_size);
        float* query_gradient = make_zero_buffer(square_weight_elements);
        float* key_gradient = make_zero_buffer(square_weight_elements);
        float* value_gradient = make_zero_buffer(square_weight_elements);
        float* projection_gradient = make_zero_buffer(square_weight_elements);
        float* ffn_norm_gradient = make_zero_buffer(hidden_size);
        float* gate_gradient = make_zero_buffer(gate_weight_elements);
        float* up_gradient = make_zero_buffer(gate_weight_elements);
        float* down_gradient = make_zero_buffer(down_weight_elements);
        const dscuda::TransformerBlockGradients parameter_gradients{
            attention_norm_gradient,
            query_gradient,
            key_gradient,
            value_gradient,
            projection_gradient,
            ffn_norm_gradient,
            gate_gradient,
            up_gradient,
            down_gradient};

        float* activations = make_zero_buffer(
            dscuda::transformer_block_activation_elements(config));
        float* workspace = make_zero_buffer(
            dscuda::transformer_block_backward_workspace_elements(config));

        std::printf(
            "Transformer block workload: batch=%d sequence=%d hidden=%d heads=%d ffn=%d\n",
            batch_size,
            sequence_length,
            hidden_size,
            heads,
            ffn_size);
        CUDA_CHECK(cudaProfilerStart());
        dscuda::transformer_block_forward_cuda(
            output,
            input,
            parameters,
            cosine,
            sine,
            activations,
            config);
        dscuda::transformer_block_backward_cuda(
            input_gradient,
            parameter_gradients,
            output_gradient,
            input,
            parameters,
            cosine,
            sine,
            activations,
            workspace,
            config);
        dscuda::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        dscuda::device_free(workspace);
        dscuda::device_free(activations);
        dscuda::device_free(down_gradient);
        dscuda::device_free(up_gradient);
        dscuda::device_free(gate_gradient);
        dscuda::device_free(ffn_norm_gradient);
        dscuda::device_free(projection_gradient);
        dscuda::device_free(value_gradient);
        dscuda::device_free(key_gradient);
        dscuda::device_free(query_gradient);
        dscuda::device_free(attention_norm_gradient);
        dscuda::device_free(down);
        dscuda::device_free(up);
        dscuda::device_free(gate);
        dscuda::device_free(ffn_norm);
        dscuda::device_free(projection);
        dscuda::device_free(value);
        dscuda::device_free(key);
        dscuda::device_free(query);
        dscuda::device_free(attention_norm);
        dscuda::device_free(sine);
        dscuda::device_free(cosine);
        dscuda::device_free(input_gradient);
        dscuda::device_free(output);
        dscuda::device_free(output_gradient);
        dscuda::device_free(input);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "Transformer block benchmark failed: %s\n", error.what());
        return 1;
    }
}

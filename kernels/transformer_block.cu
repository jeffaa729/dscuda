// Composes the existing CUDA operators into a complete pre-norm dense transformer block with explicit saved activations.
// Its backward pass follows both residual branches and accumulates all input and parameter gradients without internal allocation.

#include "transformer_block.h"

#include "attention.h"
#include "cuda_common.h"
#include "matmul.h"
#include "residual.h"
#include "rmsnorm.h"
#include "rope.h"
#include "swiglu.h"

namespace dscuda {
namespace {

struct ActivationLayout {
    float* norm1;
    float* inverse_rms1;
    float* query;
    float* key;
    float* value;
    float* probabilities;
    float* attention_output;
    float* residual1;
    float* norm2;
    float* inverse_rms2;
    float* gate;
    float* up;
    float* hidden;
    float* attention_workspace;
};

struct ConstActivationLayout {
    const float* norm1;
    const float* inverse_rms1;
    const float* query;
    const float* key;
    const float* value;
    const float* probabilities;
    const float* attention_output;
    const float* residual1;
    const float* norm2;
    const float* inverse_rms2;
    const float* gate;
    const float* up;
    const float* hidden;
};

struct BackwardLayout {
    float* residual1_gradient;
    float* hidden_gradient;
    float* gate_gradient;
    float* up_gradient;
    float* norm2_gradient;
    float* attention_projection_gradient;
    float* attention_output_gradient;
    float* rotated_query_gradient;
    float* rotated_key_gradient;
    float* value_gradient;
    float* query_gradient;
    float* key_gradient;
    float* norm1_gradient;
    float* attention_workspace;
};

ActivationLayout activation_layout(
    float* buffer,
    const TransformerBlockConfig& config) {
    const std::size_t rows =
        static_cast<std::size_t>(config.batch_size) * config.sequence_length;
    const std::size_t activations = rows * config.hidden_size;
    const std::size_t ffn_activations = rows * config.ffn_size;
    const std::size_t probabilities =
        static_cast<std::size_t>(config.batch_size) * config.heads *
        config.sequence_length * config.sequence_length;

    ActivationLayout layout;
    layout.norm1 = buffer;
    layout.inverse_rms1 = layout.norm1 + activations;
    layout.query = layout.inverse_rms1 + rows;
    layout.key = layout.query + activations;
    layout.value = layout.key + activations;
    layout.probabilities = layout.value + activations;
    layout.attention_output = layout.probabilities + probabilities;
    layout.residual1 = layout.attention_output + activations;
    layout.norm2 = layout.residual1 + activations;
    layout.inverse_rms2 = layout.norm2 + activations;
    layout.gate = layout.inverse_rms2 + rows;
    layout.up = layout.gate + ffn_activations;
    layout.hidden = layout.up + ffn_activations;
    layout.attention_workspace = layout.hidden + ffn_activations;
    return layout;
}

ConstActivationLayout activation_layout(
    const float* buffer,
    const TransformerBlockConfig& config) {
    const std::size_t rows =
        static_cast<std::size_t>(config.batch_size) * config.sequence_length;
    const std::size_t activations = rows * config.hidden_size;
    const std::size_t ffn_activations = rows * config.ffn_size;
    const std::size_t probabilities =
        static_cast<std::size_t>(config.batch_size) * config.heads *
        config.sequence_length * config.sequence_length;

    ConstActivationLayout layout;
    layout.norm1 = buffer;
    layout.inverse_rms1 = layout.norm1 + activations;
    layout.query = layout.inverse_rms1 + rows;
    layout.key = layout.query + activations;
    layout.value = layout.key + activations;
    layout.probabilities = layout.value + activations;
    layout.attention_output = layout.probabilities + probabilities;
    layout.residual1 = layout.attention_output + activations;
    layout.norm2 = layout.residual1 + activations;
    layout.inverse_rms2 = layout.norm2 + activations;
    layout.gate = layout.inverse_rms2 + rows;
    layout.up = layout.gate + ffn_activations;
    layout.hidden = layout.up + ffn_activations;
    return layout;
}

BackwardLayout backward_layout(
    float* buffer,
    const TransformerBlockConfig& config) {
    const std::size_t rows =
        static_cast<std::size_t>(config.batch_size) * config.sequence_length;
    const std::size_t activations = rows * config.hidden_size;
    const std::size_t ffn_activations = rows * config.ffn_size;

    BackwardLayout layout;
    layout.residual1_gradient = buffer;
    layout.hidden_gradient = layout.residual1_gradient + activations;
    layout.gate_gradient = layout.hidden_gradient + ffn_activations;
    layout.up_gradient = layout.gate_gradient + ffn_activations;
    layout.norm2_gradient = layout.up_gradient + ffn_activations;
    layout.attention_projection_gradient =
        layout.norm2_gradient + activations;
    layout.attention_output_gradient =
        layout.attention_projection_gradient + activations;
    layout.rotated_query_gradient =
        layout.attention_output_gradient + activations;
    layout.rotated_key_gradient =
        layout.rotated_query_gradient + activations;
    layout.value_gradient = layout.rotated_key_gradient + activations;
    layout.query_gradient = layout.value_gradient + activations;
    layout.key_gradient = layout.query_gradient + activations;
    layout.norm1_gradient = layout.key_gradient + activations;
    layout.attention_workspace = layout.norm1_gradient + activations;
    return layout;
}

void linear_input_gradient(
    float* input_gradient,
    const float* output_gradient,
    const float* weight,
    int rows,
    int input_size,
    int output_size,
    bool accumulate,
    cudaStream_t stream) {
    matmul_fp32_strided_batched_cuda(
        input_gradient,
        output_gradient,
        weight,
        rows,
        input_size,
        output_size,
        1,
        0,
        0,
        0,
        false,
        true,
        accumulate,
        stream);
}

void linear_weight_gradient(
    float* weight_gradient,
    const float* input,
    const float* output_gradient,
    int rows,
    int input_size,
    int output_size,
    cudaStream_t stream) {
    matmul_fp32_strided_batched_cuda(
        weight_gradient,
        input,
        output_gradient,
        input_size,
        output_size,
        rows,
        1,
        0,
        0,
        0,
        true,
        false,
        true,
        stream);
}

}  // namespace

std::size_t transformer_block_activation_elements(
    const TransformerBlockConfig& config) {
    const std::size_t rows =
        static_cast<std::size_t>(config.batch_size) * config.sequence_length;
    const std::size_t activations = rows * config.hidden_size;
    const std::size_t ffn_activations = rows * config.ffn_size;
    const std::size_t probabilities =
        static_cast<std::size_t>(config.batch_size) * config.heads *
        config.sequence_length * config.sequence_length;
    return 7 * activations + 3 * ffn_activations + 2 * rows + probabilities +
        dense_attention_forward_workspace_elements(
               config.batch_size,
               config.sequence_length,
               config.heads,
               config.head_size);
}

std::size_t transformer_block_backward_workspace_elements(
    const TransformerBlockConfig& config) {
    const std::size_t rows =
        static_cast<std::size_t>(config.batch_size) * config.sequence_length;
    const std::size_t activations = rows * config.hidden_size;
    const std::size_t ffn_activations = rows * config.ffn_size;
    return 10 * activations + 3 * ffn_activations +
        dense_attention_backward_workspace_elements(
               config.batch_size,
               config.sequence_length,
               config.heads,
               config.head_size);
}

void transformer_block_forward_cuda(
    float* output,
    const float* input,
    const TransformerBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    const TransformerBlockConfig& config,
    cudaStream_t stream) {
    const int rows = config.batch_size * config.sequence_length;
    const int activation_elements = rows * config.hidden_size;
    const int ffn_elements = rows * config.ffn_size;
    ActivationLayout saved = activation_layout(activations, config);

    rmsnorm_forward_cuda(
        saved.norm1,
        saved.inverse_rms1,
        input,
        parameters.attention_norm_weight,
        rows,
        config.hidden_size,
        config.epsilon,
        stream);
    matmul_fp32_forward_cuda(
        saved.query,
        saved.norm1,
        parameters.query_weight,
        rows,
        config.hidden_size,
        config.hidden_size,
        stream);
    matmul_fp32_forward_cuda(
        saved.key,
        saved.norm1,
        parameters.key_weight,
        rows,
        config.hidden_size,
        config.hidden_size,
        stream);
    matmul_fp32_forward_cuda(
        saved.value,
        saved.norm1,
        parameters.value_weight,
        rows,
        config.hidden_size,
        config.hidden_size,
        stream);
    rope_forward_cuda(
        saved.query,
        saved.query,
        cosine,
        sine,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.rotary_size,
        stream);
    rope_forward_cuda(
        saved.key,
        saved.key,
        cosine,
        sine,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.rotary_size,
        stream);
    dense_attention_forward_cuda(
        saved.attention_output,
        saved.probabilities,
        saved.query,
        saved.key,
        saved.value,
        saved.attention_workspace,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.attention_scale,
        stream);
    matmul_fp32_forward_cuda(
        output,
        saved.attention_output,
        parameters.output_weight,
        rows,
        config.hidden_size,
        config.hidden_size,
        stream);
    residual_forward_cuda(
        saved.residual1, input, output, activation_elements, stream);
    rmsnorm_forward_cuda(
        saved.norm2,
        saved.inverse_rms2,
        saved.residual1,
        parameters.ffn_norm_weight,
        rows,
        config.hidden_size,
        config.epsilon,
        stream);
    matmul_fp32_forward_cuda(
        saved.gate,
        saved.norm2,
        parameters.gate_weight,
        rows,
        config.ffn_size,
        config.hidden_size,
        stream);
    matmul_fp32_forward_cuda(
        saved.up,
        saved.norm2,
        parameters.up_weight,
        rows,
        config.ffn_size,
        config.hidden_size,
        stream);
    swiglu_forward_cuda(
        saved.hidden, saved.gate, saved.up, ffn_elements, stream);
    matmul_fp32_forward_cuda(
        output,
        saved.hidden,
        parameters.down_weight,
        rows,
        config.hidden_size,
        config.ffn_size,
        stream);
    residual_forward_cuda(
        output, saved.residual1, output, activation_elements, stream);
}

void transformer_block_backward_cuda(
    float* input_gradient,
    const TransformerBlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const TransformerBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const float* activations,
    float* workspace,
    const TransformerBlockConfig& config,
    cudaStream_t stream) {
    const int rows = config.batch_size * config.sequence_length;
    const int activation_elements = rows * config.hidden_size;
    const int ffn_elements = rows * config.ffn_size;
    ConstActivationLayout saved = activation_layout(activations, config);
    BackwardLayout gradient = backward_layout(workspace, config);

    CUDA_CHECK(cudaMemcpyAsync(
        gradient.residual1_gradient,
        output_gradient,
        activation_elements * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));
    linear_input_gradient(
        gradient.hidden_gradient,
        output_gradient,
        parameters.down_weight,
        rows,
        config.ffn_size,
        config.hidden_size,
        false,
        stream);
    linear_weight_gradient(
        parameter_gradients.down_weight,
        saved.hidden,
        output_gradient,
        rows,
        config.ffn_size,
        config.hidden_size,
        stream);
    swiglu_backward_cuda(
        gradient.gate_gradient,
        gradient.up_gradient,
        gradient.hidden_gradient,
        saved.gate,
        saved.up,
        ffn_elements,
        stream);

    linear_input_gradient(
        gradient.norm2_gradient,
        gradient.gate_gradient,
        parameters.gate_weight,
        rows,
        config.hidden_size,
        config.ffn_size,
        false,
        stream);
    linear_input_gradient(
        gradient.norm2_gradient,
        gradient.up_gradient,
        parameters.up_weight,
        rows,
        config.hidden_size,
        config.ffn_size,
        true,
        stream);
    linear_weight_gradient(
        parameter_gradients.gate_weight,
        saved.norm2,
        gradient.gate_gradient,
        rows,
        config.hidden_size,
        config.ffn_size,
        stream);
    linear_weight_gradient(
        parameter_gradients.up_weight,
        saved.norm2,
        gradient.up_gradient,
        rows,
        config.hidden_size,
        config.ffn_size,
        stream);
    rmsnorm_backward_cuda(
        gradient.residual1_gradient,
        parameter_gradients.ffn_norm_weight,
        gradient.norm2_gradient,
        saved.residual1,
        parameters.ffn_norm_weight,
        saved.inverse_rms2,
        rows,
        config.hidden_size,
        stream);

    CUDA_CHECK(cudaMemsetAsync(
        gradient.attention_projection_gradient,
        0,
        activation_elements * sizeof(float),
        stream));
    residual_backward_cuda(
        input_gradient,
        gradient.attention_projection_gradient,
        gradient.residual1_gradient,
        activation_elements,
        stream);
    linear_input_gradient(
        gradient.attention_output_gradient,
        gradient.attention_projection_gradient,
        parameters.output_weight,
        rows,
        config.hidden_size,
        config.hidden_size,
        false,
        stream);
    linear_weight_gradient(
        parameter_gradients.output_weight,
        saved.attention_output,
        gradient.attention_projection_gradient,
        rows,
        config.hidden_size,
        config.hidden_size,
        stream);

    CUDA_CHECK(cudaMemsetAsync(
        gradient.rotated_query_gradient,
        0,
        3 * activation_elements * sizeof(float),
        stream));
    dense_attention_backward_cuda(
        gradient.rotated_query_gradient,
        gradient.rotated_key_gradient,
        gradient.value_gradient,
        gradient.attention_output_gradient,
        saved.probabilities,
        saved.query,
        saved.key,
        saved.value,
        gradient.attention_workspace,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.attention_scale,
        stream);
    CUDA_CHECK(cudaMemsetAsync(
        gradient.query_gradient,
        0,
        2 * activation_elements * sizeof(float),
        stream));
    rope_backward_cuda(
        gradient.query_gradient,
        gradient.rotated_query_gradient,
        cosine,
        sine,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.rotary_size,
        stream);
    rope_backward_cuda(
        gradient.key_gradient,
        gradient.rotated_key_gradient,
        cosine,
        sine,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.rotary_size,
        stream);

    linear_input_gradient(
        gradient.norm1_gradient,
        gradient.query_gradient,
        parameters.query_weight,
        rows,
        config.hidden_size,
        config.hidden_size,
        false,
        stream);
    linear_input_gradient(
        gradient.norm1_gradient,
        gradient.key_gradient,
        parameters.key_weight,
        rows,
        config.hidden_size,
        config.hidden_size,
        true,
        stream);
    linear_input_gradient(
        gradient.norm1_gradient,
        gradient.value_gradient,
        parameters.value_weight,
        rows,
        config.hidden_size,
        config.hidden_size,
        true,
        stream);
    linear_weight_gradient(
        parameter_gradients.query_weight,
        saved.norm1,
        gradient.query_gradient,
        rows,
        config.hidden_size,
        config.hidden_size,
        stream);
    linear_weight_gradient(
        parameter_gradients.key_weight,
        saved.norm1,
        gradient.key_gradient,
        rows,
        config.hidden_size,
        config.hidden_size,
        stream);
    linear_weight_gradient(
        parameter_gradients.value_weight,
        saved.norm1,
        gradient.value_gradient,
        rows,
        config.hidden_size,
        config.hidden_size,
        stream);
    rmsnorm_backward_cuda(
        input_gradient,
        parameter_gradients.attention_norm_weight,
        gradient.norm1_gradient,
        input,
        parameters.attention_norm_weight,
        saved.inverse_rms1,
        rows,
        config.hidden_size,
        stream);
}

}  // namespace dscuda

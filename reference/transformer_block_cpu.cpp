// Composes the scalar operator references into a complete pre-norm dense transformer block.
// Backward recomputes forward values to keep the reference API simple and independently check the CUDA activation layout.

#include "transformer_block_cpu.h"

#include "attention_cpu.h"
#include "matmul_cpu.h"
#include "residual_cpu.h"
#include "rmsnorm_cpu.h"
#include "rope_cpu.h"
#include "swiglu_cpu.h"

#include <vector>

namespace dscuda {
namespace {

struct CpuActivations {
    std::vector<float> norm1;
    std::vector<float> inverse_rms1;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> probabilities;
    std::vector<float> attention_output;
    std::vector<float> attention_projection;
    std::vector<float> residual1;
    std::vector<float> norm2;
    std::vector<float> inverse_rms2;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::vector<float> ffn_output;
};

CpuActivations forward(
    float* output,
    const float* input,
    const TransformerBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const TransformerBlockConfig& config) {
    const int rows = config.batch_size * config.sequence_length;
    const int activations = rows * config.hidden_size;
    const int ffn_activations = rows * config.ffn_size;
    const int probabilities =
        config.batch_size * config.heads * config.sequence_length *
        config.sequence_length;

    CpuActivations saved{
        std::vector<float>(activations),
        std::vector<float>(rows),
        std::vector<float>(activations),
        std::vector<float>(activations),
        std::vector<float>(activations),
        std::vector<float>(probabilities),
        std::vector<float>(activations),
        std::vector<float>(activations),
        std::vector<float>(activations),
        std::vector<float>(activations),
        std::vector<float>(rows),
        std::vector<float>(ffn_activations),
        std::vector<float>(ffn_activations),
        std::vector<float>(ffn_activations),
        std::vector<float>(activations)};

    rmsnorm_forward_cpu(
        saved.norm1.data(),
        saved.inverse_rms1.data(),
        input,
        parameters.attention_norm_weight,
        rows,
        config.hidden_size,
        config.epsilon);
    matmul_forward_cpu(
        saved.query.data(),
        saved.norm1.data(),
        parameters.query_weight,
        rows,
        config.hidden_size,
        config.hidden_size);
    matmul_forward_cpu(
        saved.key.data(),
        saved.norm1.data(),
        parameters.key_weight,
        rows,
        config.hidden_size,
        config.hidden_size);
    matmul_forward_cpu(
        saved.value.data(),
        saved.norm1.data(),
        parameters.value_weight,
        rows,
        config.hidden_size,
        config.hidden_size);
    rope_forward_cpu(
        saved.query.data(),
        saved.query.data(),
        cosine,
        sine,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.rotary_size);
    rope_forward_cpu(
        saved.key.data(),
        saved.key.data(),
        cosine,
        sine,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.rotary_size);
    dense_attention_forward_cpu(
        saved.attention_output.data(),
        saved.probabilities.data(),
        saved.query.data(),
        saved.key.data(),
        saved.value.data(),
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.attention_scale);
    matmul_forward_cpu(
        saved.attention_projection.data(),
        saved.attention_output.data(),
        parameters.output_weight,
        rows,
        config.hidden_size,
        config.hidden_size);
    residual_forward_cpu(
        saved.residual1.data(),
        input,
        saved.attention_projection.data(),
        activations);
    rmsnorm_forward_cpu(
        saved.norm2.data(),
        saved.inverse_rms2.data(),
        saved.residual1.data(),
        parameters.ffn_norm_weight,
        rows,
        config.hidden_size,
        config.epsilon);
    matmul_forward_cpu(
        saved.gate.data(),
        saved.norm2.data(),
        parameters.gate_weight,
        rows,
        config.ffn_size,
        config.hidden_size);
    matmul_forward_cpu(
        saved.up.data(),
        saved.norm2.data(),
        parameters.up_weight,
        rows,
        config.ffn_size,
        config.hidden_size);
    swiglu_forward_cpu(
        saved.hidden.data(),
        saved.gate.data(),
        saved.up.data(),
        ffn_activations);
    matmul_forward_cpu(
        saved.ffn_output.data(),
        saved.hidden.data(),
        parameters.down_weight,
        rows,
        config.hidden_size,
        config.ffn_size);
    residual_forward_cpu(
        output, saved.residual1.data(), saved.ffn_output.data(), activations);
    return saved;
}

}  // namespace

void transformer_block_forward_cpu(
    float* output,
    const float* input,
    const TransformerBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const TransformerBlockConfig& config) {
    forward(output, input, parameters, cosine, sine, config);
}

void transformer_block_backward_cpu(
    float* input_gradient,
    const TransformerBlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const TransformerBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const TransformerBlockConfig& config) {
    const int rows = config.batch_size * config.sequence_length;
    const int activations = rows * config.hidden_size;
    const int ffn_activations = rows * config.ffn_size;
    std::vector<float> output(activations);
    CpuActivations saved =
        forward(output.data(), input, parameters, cosine, sine, config);

    std::vector<float> residual1_gradient(
        output_gradient, output_gradient + activations);
    std::vector<float> hidden_gradient(ffn_activations);
    matmul_backward_cpu(
        hidden_gradient.data(),
        parameter_gradients.down_weight,
        output_gradient,
        saved.hidden.data(),
        parameters.down_weight,
        rows,
        config.hidden_size,
        config.ffn_size);

    std::vector<float> gate_gradient(ffn_activations);
    std::vector<float> up_gradient(ffn_activations);
    swiglu_backward_cpu(
        gate_gradient.data(),
        up_gradient.data(),
        hidden_gradient.data(),
        saved.gate.data(),
        saved.up.data(),
        ffn_activations);

    std::vector<float> norm2_gradient(activations);
    matmul_backward_cpu(
        norm2_gradient.data(),
        parameter_gradients.gate_weight,
        gate_gradient.data(),
        saved.norm2.data(),
        parameters.gate_weight,
        rows,
        config.ffn_size,
        config.hidden_size);
    matmul_backward_cpu(
        norm2_gradient.data(),
        parameter_gradients.up_weight,
        up_gradient.data(),
        saved.norm2.data(),
        parameters.up_weight,
        rows,
        config.ffn_size,
        config.hidden_size);
    rmsnorm_backward_cpu(
        residual1_gradient.data(),
        parameter_gradients.ffn_norm_weight,
        norm2_gradient.data(),
        saved.residual1.data(),
        parameters.ffn_norm_weight,
        saved.inverse_rms2.data(),
        rows,
        config.hidden_size);

    std::vector<float> attention_projection_gradient(activations);
    residual_backward_cpu(
        input_gradient,
        attention_projection_gradient.data(),
        residual1_gradient.data(),
        activations);
    std::vector<float> attention_output_gradient(activations);
    matmul_backward_cpu(
        attention_output_gradient.data(),
        parameter_gradients.output_weight,
        attention_projection_gradient.data(),
        saved.attention_output.data(),
        parameters.output_weight,
        rows,
        config.hidden_size,
        config.hidden_size);

    std::vector<float> query_rotated_gradient(activations);
    std::vector<float> key_rotated_gradient(activations);
    std::vector<float> value_gradient(activations);
    dense_attention_backward_cpu(
        query_rotated_gradient.data(),
        key_rotated_gradient.data(),
        value_gradient.data(),
        attention_output_gradient.data(),
        saved.probabilities.data(),
        saved.query.data(),
        saved.key.data(),
        saved.value.data(),
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.attention_scale);

    std::vector<float> query_gradient(activations);
    std::vector<float> key_gradient(activations);
    rope_backward_cpu(
        query_gradient.data(),
        query_rotated_gradient.data(),
        cosine,
        sine,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.rotary_size);
    rope_backward_cpu(
        key_gradient.data(),
        key_rotated_gradient.data(),
        cosine,
        sine,
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.head_size,
        config.rotary_size);

    std::vector<float> norm1_gradient(activations);
    matmul_backward_cpu(
        norm1_gradient.data(),
        parameter_gradients.query_weight,
        query_gradient.data(),
        saved.norm1.data(),
        parameters.query_weight,
        rows,
        config.hidden_size,
        config.hidden_size);
    matmul_backward_cpu(
        norm1_gradient.data(),
        parameter_gradients.key_weight,
        key_gradient.data(),
        saved.norm1.data(),
        parameters.key_weight,
        rows,
        config.hidden_size,
        config.hidden_size);
    matmul_backward_cpu(
        norm1_gradient.data(),
        parameter_gradients.value_weight,
        value_gradient.data(),
        saved.norm1.data(),
        parameters.value_weight,
        rows,
        config.hidden_size,
        config.hidden_size);
    rmsnorm_backward_cpu(
        input_gradient,
        parameter_gradients.attention_norm_weight,
        norm1_gradient.data(),
        input,
        parameters.attention_norm_weight,
        saved.inverse_rms1.data(),
        rows,
        config.hidden_size);
}

}  // namespace dscuda

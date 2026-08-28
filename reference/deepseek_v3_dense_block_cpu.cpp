// Provides a scalar reference for the MLA-plus-dense-SwiGLU blocks at the beginning of DeepSeek-V3.
// Backward recomputes forward intermediates and accumulates the attention and FFN residual paths exactly like the CUDA graph.

#include "deepseek_v3_dense_block_cpu.h"

#include "matmul_cpu.h"
#include "mla_cpu.h"
#include "residual_cpu.h"
#include "rmsnorm_cpu.h"
#include "swiglu_cpu.h"

#include <vector>

namespace dscuda {
namespace {

struct SavedDenseBlock {
    std::vector<float> norm1;
    std::vector<float> inverse_rms1;
    std::vector<float> attention_output;
    std::vector<float> residual1;
    std::vector<float> norm2;
    std::vector<float> inverse_rms2;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::vector<float> ffn_output;
};

SavedDenseBlock forward_saved(
    float* output,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3DenseBlockConfig& config) {
    const int hidden_elements = config.rows * config.hidden_size;
    const int ffn_elements = config.rows * config.ffn_size;
    SavedDenseBlock saved;
    saved.norm1.resize(hidden_elements);
    saved.inverse_rms1.resize(config.rows);
    saved.attention_output.resize(hidden_elements);
    saved.residual1.resize(hidden_elements);
    saved.norm2.resize(hidden_elements);
    saved.inverse_rms2.resize(config.rows);
    saved.gate.resize(ffn_elements);
    saved.up.resize(ffn_elements);
    saved.hidden.resize(ffn_elements);
    saved.ffn_output.resize(hidden_elements);
    rmsnorm_forward_cpu(
        saved.norm1.data(), saved.inverse_rms1.data(), input,
        parameters.attention_norm_weight, config.rows, config.hidden_size,
        config.epsilon);
    mla_layer_forward_cpu(
        saved.attention_output.data(), saved.norm1.data(), parameters.mla,
        cosine, sine, config.mla);
    residual_forward_cpu(
        saved.residual1.data(), input, saved.attention_output.data(),
        hidden_elements);
    rmsnorm_forward_cpu(
        saved.norm2.data(), saved.inverse_rms2.data(), saved.residual1.data(),
        parameters.ffn_norm_weight, config.rows, config.hidden_size,
        config.epsilon);
    matmul_forward_cpu(
        saved.gate.data(), saved.norm2.data(), parameters.gate_weight,
        config.rows, config.ffn_size, config.hidden_size);
    matmul_forward_cpu(
        saved.up.data(), saved.norm2.data(), parameters.up_weight,
        config.rows, config.ffn_size, config.hidden_size);
    swiglu_forward_cpu(
        saved.hidden.data(), saved.gate.data(), saved.up.data(), ffn_elements);
    matmul_forward_cpu(
        saved.ffn_output.data(), saved.hidden.data(), parameters.down_weight,
        config.rows, config.hidden_size, config.ffn_size);
    residual_forward_cpu(
        output, saved.residual1.data(), saved.ffn_output.data(), hidden_elements);
    return saved;
}

}  // namespace

void deepseek_v3_dense_block_forward_cpu(
    float* output,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3DenseBlockConfig& config) {
    forward_saved(output, input, parameters, cosine, sine, config);
}

void deepseek_v3_dense_block_backward_cpu(
    float* input_gradient,
    const DeepSeekV3DenseBlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3DenseBlockConfig& config) {
    const int hidden_elements = config.rows * config.hidden_size;
    const int ffn_elements = config.rows * config.ffn_size;
    std::vector<float> discarded(hidden_elements);
    const SavedDenseBlock saved = forward_saved(
        discarded.data(), input, parameters, cosine, sine, config);
    std::vector<float> residual1_gradient(hidden_elements, 0.0F);
    std::vector<float> ffn_output_gradient(hidden_elements, 0.0F);
    residual_backward_cpu(
        residual1_gradient.data(), ffn_output_gradient.data(), output_gradient,
        hidden_elements);
    std::vector<float> hidden_gradient(ffn_elements, 0.0F);
    matmul_backward_cpu(
        hidden_gradient.data(), parameter_gradients.down_weight,
        ffn_output_gradient.data(), saved.hidden.data(), parameters.down_weight,
        config.rows, config.hidden_size, config.ffn_size);
    std::vector<float> gate_gradient(ffn_elements);
    std::vector<float> up_gradient(ffn_elements);
    swiglu_backward_cpu(
        gate_gradient.data(), up_gradient.data(), hidden_gradient.data(),
        saved.gate.data(), saved.up.data(), ffn_elements);
    std::vector<float> norm2_gradient(hidden_elements, 0.0F);
    matmul_backward_cpu(
        norm2_gradient.data(), parameter_gradients.gate_weight,
        gate_gradient.data(), saved.norm2.data(), parameters.gate_weight,
        config.rows, config.ffn_size, config.hidden_size);
    matmul_backward_cpu(
        norm2_gradient.data(), parameter_gradients.up_weight,
        up_gradient.data(), saved.norm2.data(), parameters.up_weight,
        config.rows, config.ffn_size, config.hidden_size);
    rmsnorm_backward_cpu(
        residual1_gradient.data(), parameter_gradients.ffn_norm_weight,
        norm2_gradient.data(), saved.residual1.data(),
        parameters.ffn_norm_weight, saved.inverse_rms2.data(), config.rows,
        config.hidden_size);
    std::vector<float> attention_output_gradient(hidden_elements, 0.0F);
    residual_backward_cpu(
        input_gradient, attention_output_gradient.data(),
        residual1_gradient.data(), hidden_elements);
    std::vector<float> norm1_gradient(hidden_elements, 0.0F);
    mla_layer_backward_cpu(
        norm1_gradient.data(), parameter_gradients.mla,
        attention_output_gradient.data(), saved.norm1.data(), parameters.mla,
        cosine, sine, config.mla);
    rmsnorm_backward_cpu(
        input_gradient, parameter_gradients.attention_norm_weight,
        norm1_gradient.data(), input, parameters.attention_norm_weight,
        saved.inverse_rms1.data(), config.rows, config.hidden_size);
}

}  // namespace dscuda

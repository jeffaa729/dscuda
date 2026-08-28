// Implements a readable pre-norm DeepSeek-V3 block from the CPU MLA, DeepSeekMoE, RMSNorm, and residual references.
// Backward recomputes the saved forward values and explicitly accumulates both residual branches into the block input gradient.

#include "deepseek_v3_block_cpu.h"

#include "deepseek_moe_cpu.h"
#include "mla_cpu.h"
#include "residual_cpu.h"
#include "rmsnorm_cpu.h"

#include <vector>

namespace dscuda {
namespace {

struct SavedBlock {
    std::vector<float> norm1;
    std::vector<float> inverse_rms1;
    std::vector<float> attention_output;
    std::vector<float> residual1;
    std::vector<float> norm2;
    std::vector<float> inverse_rms2;
    std::vector<float> moe_output;
};

SavedBlock forward_saved(
    float* output,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3BlockConfig& config) {
    const int elements = config.moe.rows * config.moe.hidden_size;
    SavedBlock saved;
    saved.norm1.resize(elements);
    saved.inverse_rms1.resize(config.moe.rows);
    saved.attention_output.resize(elements);
    saved.residual1.resize(elements);
    saved.norm2.resize(elements);
    saved.inverse_rms2.resize(config.moe.rows);
    saved.moe_output.resize(elements);
    rmsnorm_forward_cpu(
        saved.norm1.data(), saved.inverse_rms1.data(), input,
        parameters.attention_norm_weight, config.moe.rows,
        config.moe.hidden_size, config.epsilon);
    mla_layer_forward_cpu(
        saved.attention_output.data(), saved.norm1.data(), parameters.mla,
        cosine, sine, config.mla);
    residual_forward_cpu(
        saved.residual1.data(), input, saved.attention_output.data(), elements);
    rmsnorm_forward_cpu(
        saved.norm2.data(), saved.inverse_rms2.data(), saved.residual1.data(),
        parameters.ffn_norm_weight, config.moe.rows, config.moe.hidden_size,
        config.epsilon);
    deepseek_moe_forward_cpu(
        saved.moe_output.data(), saved.norm2.data(), parameters.moe,
        config.moe);
    residual_forward_cpu(
        output, saved.residual1.data(), saved.moe_output.data(), elements);
    return saved;
}

}  // namespace

void deepseek_v3_block_forward_cpu(
    float* output,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3BlockConfig& config) {
    forward_saved(output, input, parameters, cosine, sine, config);
}

void deepseek_v3_block_backward_cpu(
    float* input_gradient,
    const DeepSeekV3BlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3BlockConfig& config) {
    const int elements = config.moe.rows * config.moe.hidden_size;
    std::vector<float> discarded(elements);
    const SavedBlock saved = forward_saved(
        discarded.data(), input, parameters, cosine, sine, config);
    std::vector<float> residual1_gradient(elements, 0.0F);
    std::vector<float> moe_output_gradient(elements, 0.0F);
    residual_backward_cpu(
        residual1_gradient.data(), moe_output_gradient.data(), output_gradient,
        elements);
    std::vector<float> norm2_gradient(elements, 0.0F);
    deepseek_moe_backward_cpu(
        norm2_gradient.data(), parameter_gradients.moe,
        moe_output_gradient.data(), saved.norm2.data(), parameters.moe,
        config.moe);
    rmsnorm_backward_cpu(
        residual1_gradient.data(), parameter_gradients.ffn_norm_weight,
        norm2_gradient.data(), saved.residual1.data(),
        parameters.ffn_norm_weight, saved.inverse_rms2.data(), config.moe.rows,
        config.moe.hidden_size);
    std::vector<float> attention_output_gradient(elements, 0.0F);
    residual_backward_cpu(
        input_gradient, attention_output_gradient.data(),
        residual1_gradient.data(), elements);
    std::vector<float> norm1_gradient(elements, 0.0F);
    mla_layer_backward_cpu(
        norm1_gradient.data(), parameter_gradients.mla,
        attention_output_gradient.data(), saved.norm1.data(), parameters.mla,
        cosine, sine, config.mla);
    rmsnorm_backward_cpu(
        input_gradient, parameter_gradients.attention_norm_weight,
        norm1_gradient.data(), input, parameters.attention_norm_weight,
        saved.inverse_rms1.data(), config.moe.rows, config.moe.hidden_size);
}

}  // namespace dscuda

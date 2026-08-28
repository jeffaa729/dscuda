// Assembles a pre-norm DeepSeek-V3 transformer block from the complete MLA layer and DeepSeekMoE layer with two residual branches.
// The backward graph explicitly follows both residual paths and reuses the child operators' saved activations and workspace contracts.

#include "deepseek_v3_block.h"

#include "cuda_common.h"
#include "residual.h"
#include "rmsnorm.h"

namespace dscuda {
namespace {

std::size_t align_four(std::size_t value) {
    return (value + 3) & ~std::size_t{3};
}

template <typename T>
struct ActivationLayout {
    T* norm1;
    T* inverse_rms1;
    T* mla;
    T* attention_output;
    T* residual1;
    T* norm2;
    T* inverse_rms2;
    T* moe;
    T* moe_output;
    std::size_t elements;
};

template <typename T>
ActivationLayout<T> make_activation_layout(
    T* buffer,
    const DeepSeekV3BlockConfig& config) {
    const std::size_t rows = config.moe.rows;
    const std::size_t activations = rows * config.moe.hidden_size;
    const std::size_t aligned_rows = align_four(rows);
    std::size_t offset = 0;
    auto take = [&](std::size_t elements) {
        offset = align_four(offset);
        T* result = buffer == nullptr ? nullptr : buffer + offset;
        offset += elements;
        return result;
    };
    ActivationLayout<T> layout;
    layout.norm1 = take(activations);
    layout.inverse_rms1 = take(aligned_rows);
    layout.mla = take(mla_layer_activation_elements(config.mla));
    layout.attention_output = take(activations);
    layout.residual1 = take(activations);
    layout.norm2 = take(activations);
    layout.inverse_rms2 = take(aligned_rows);
    layout.moe = take(deepseek_moe_activation_elements(config.moe));
    layout.moe_output = take(activations);
    layout.elements = align_four(offset);
    return layout;
}

template <typename T>
struct BackwardLayout {
    T* residual1_gradient;
    T* moe_output_gradient;
    T* norm2_gradient;
    T* moe;
    T* attention_output_gradient;
    T* norm1_gradient;
    T* mla;
    std::size_t elements;
};

template <typename T>
BackwardLayout<T> make_backward_layout(
    T* buffer,
    const DeepSeekV3BlockConfig& config) {
    const std::size_t activations =
        static_cast<std::size_t>(config.moe.rows) * config.moe.hidden_size;
    std::size_t offset = 0;
    auto take = [&](std::size_t elements) {
        offset = align_four(offset);
        T* result = buffer == nullptr ? nullptr : buffer + offset;
        offset += elements;
        return result;
    };
    BackwardLayout<T> layout;
    layout.residual1_gradient = take(activations);
    layout.moe_output_gradient = take(activations);
    layout.norm2_gradient = take(activations);
    layout.moe = take(deepseek_moe_backward_workspace_elements(config.moe));
    layout.attention_output_gradient = take(activations);
    layout.norm1_gradient = take(activations);
    layout.mla = take(mla_layer_backward_workspace_elements(config.mla));
    layout.elements = align_four(offset);
    return layout;
}

}  // namespace

std::size_t deepseek_v3_block_activation_elements(
    const DeepSeekV3BlockConfig& config) {
    return make_activation_layout(static_cast<float*>(nullptr), config).elements;
}

std::size_t deepseek_v3_block_integer_activation_elements(
    const DeepSeekV3BlockConfig& config) {
    return deepseek_moe_integer_activation_elements(config.moe);
}

std::size_t deepseek_v3_block_backward_workspace_elements(
    const DeepSeekV3BlockConfig& config) {
    return make_backward_layout(static_cast<float*>(nullptr), config).elements;
}

std::size_t deepseek_v3_block_bf16_workspace_elements(
    const DeepSeekV3BlockConfig& config) {
    return mla_layer_bf16_workspace_elements(config.mla);
}

void deepseek_v3_block_forward_cuda(
    float* output,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    int* integer_activations,
    __nv_bfloat16* bf16_workspace,
    const DeepSeekV3BlockConfig& config,
    cudaStream_t stream) {
    const int elements = config.moe.rows * config.moe.hidden_size;
    auto saved = make_activation_layout(activations, config);
    rmsnorm_forward_cuda(
        saved.norm1,
        saved.inverse_rms1,
        input,
        parameters.attention_norm_weight,
        config.moe.rows,
        config.moe.hidden_size,
        config.epsilon,
        stream);
    mla_layer_forward_cuda(
        saved.attention_output,
        saved.norm1,
        parameters.mla,
        cosine,
        sine,
        saved.mla,
        bf16_workspace,
        config.mla,
        stream);
    residual_forward_cuda(
        saved.residual1, input, saved.attention_output, elements, stream);
    rmsnorm_forward_cuda(
        saved.norm2,
        saved.inverse_rms2,
        saved.residual1,
        parameters.ffn_norm_weight,
        config.moe.rows,
        config.moe.hidden_size,
        config.epsilon,
        stream);
    deepseek_moe_forward_cuda(
        saved.moe_output,
        saved.norm2,
        parameters.moe,
        saved.moe,
        integer_activations,
        config.moe,
        stream);
    residual_forward_cuda(
        output, saved.residual1, saved.moe_output, elements, stream);
}

void deepseek_v3_block_backward_cuda(
    float* input_gradient,
    const DeepSeekV3BlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const float* activations,
    const int* integer_activations,
    float* workspace,
    __nv_bfloat16* bf16_workspace,
    const DeepSeekV3BlockConfig& config,
    cudaStream_t stream) {
    const int elements = config.moe.rows * config.moe.hidden_size;
    const auto saved = make_activation_layout(activations, config);
    auto gradient = make_backward_layout(workspace, config);
    CUDA_CHECK(cudaMemsetAsync(
        workspace,
        0,
        deepseek_v3_block_backward_workspace_elements(config) * sizeof(float),
        stream));
    residual_backward_cuda(
        gradient.residual1_gradient,
        gradient.moe_output_gradient,
        output_gradient,
        elements,
        stream);
    deepseek_moe_backward_cuda(
        gradient.norm2_gradient,
        parameter_gradients.moe,
        gradient.moe_output_gradient,
        saved.norm2,
        parameters.moe,
        saved.moe,
        integer_activations,
        gradient.moe,
        config.moe,
        stream);
    rmsnorm_backward_cuda(
        gradient.residual1_gradient,
        parameter_gradients.ffn_norm_weight,
        gradient.norm2_gradient,
        saved.residual1,
        parameters.ffn_norm_weight,
        saved.inverse_rms2,
        config.moe.rows,
        config.moe.hidden_size,
        stream);
    residual_backward_cuda(
        input_gradient,
        gradient.attention_output_gradient,
        gradient.residual1_gradient,
        elements,
        stream);
    mla_layer_backward_cuda(
        gradient.norm1_gradient,
        parameter_gradients.mla,
        gradient.attention_output_gradient,
        saved.norm1,
        parameters.mla,
        cosine,
        sine,
        saved.mla,
        gradient.mla,
        bf16_workspace,
        config.mla,
        stream);
    rmsnorm_backward_cuda(
        input_gradient,
        parameter_gradients.attention_norm_weight,
        gradient.norm1_gradient,
        input,
        parameters.attention_norm_weight,
        saved.inverse_rms1,
        config.moe.rows,
        config.moe.hidden_size,
        stream);
}

void deepseek_v3_block_add_balance_loss_cuda(
    float* total_loss,
    const float* activations,
    const DeepSeekV3BlockConfig& config,
    cudaStream_t stream) {
    const auto saved = make_activation_layout(activations, config);
    deepseek_moe_add_balance_loss_cuda(
        total_loss, saved.moe, config.moe, stream);
}

}  // namespace dscuda

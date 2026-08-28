// Implements the MLA-plus-dense-SwiGLU blocks used before DeepSeek-V3 begins replacing FFNs with routed experts.
// Forward and backward retain both pre-norm residual branches while reusing the custom MLA, GEMM, RMSNorm, SwiGLU, and residual kernels.

#include "deepseek_v3_dense_block.h"

#include "cuda_common.h"
#include "matmul.h"
#include "residual.h"
#include "rmsnorm.h"
#include "swiglu.h"

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
    T* gate;
    T* up;
    T* hidden;
    T* ffn_output;
    std::size_t elements;
};

template <typename T>
ActivationLayout<T> make_activation_layout(
    T* buffer,
    const DeepSeekV3DenseBlockConfig& config) {
    const std::size_t hidden =
        static_cast<std::size_t>(config.rows) * config.hidden_size;
    const std::size_t ffn =
        static_cast<std::size_t>(config.rows) * config.ffn_size;
    std::size_t offset = 0;
    auto take = [&](std::size_t elements) {
        offset = align_four(offset);
        T* result = buffer == nullptr ? nullptr : buffer + offset;
        offset += elements;
        return result;
    };
    ActivationLayout<T> layout;
    layout.norm1 = take(hidden);
    layout.inverse_rms1 = take(align_four(config.rows));
    layout.mla = take(mla_layer_activation_elements(config.mla));
    layout.attention_output = take(hidden);
    layout.residual1 = take(hidden);
    layout.norm2 = take(hidden);
    layout.inverse_rms2 = take(align_four(config.rows));
    layout.gate = take(ffn);
    layout.up = take(ffn);
    layout.hidden = take(ffn);
    layout.ffn_output = take(hidden);
    layout.elements = align_four(offset);
    return layout;
}

template <typename T>
struct BackwardLayout {
    T* residual1_gradient;
    T* ffn_output_gradient;
    T* hidden_gradient;
    T* gate_gradient;
    T* up_gradient;
    T* norm2_gradient;
    T* attention_output_gradient;
    T* norm1_gradient;
    T* mla;
    std::size_t elements;
};

template <typename T>
BackwardLayout<T> make_backward_layout(
    T* buffer,
    const DeepSeekV3DenseBlockConfig& config) {
    const std::size_t hidden =
        static_cast<std::size_t>(config.rows) * config.hidden_size;
    const std::size_t ffn =
        static_cast<std::size_t>(config.rows) * config.ffn_size;
    std::size_t offset = 0;
    auto take = [&](std::size_t elements) {
        offset = align_four(offset);
        T* result = buffer == nullptr ? nullptr : buffer + offset;
        offset += elements;
        return result;
    };
    BackwardLayout<T> layout;
    layout.residual1_gradient = take(hidden);
    layout.ffn_output_gradient = take(hidden);
    layout.hidden_gradient = take(ffn);
    layout.gate_gradient = take(ffn);
    layout.up_gradient = take(ffn);
    layout.norm2_gradient = take(hidden);
    layout.attention_output_gradient = take(hidden);
    layout.norm1_gradient = take(hidden);
    layout.mla = take(mla_layer_backward_workspace_elements(config.mla));
    layout.elements = align_four(offset);
    return layout;
}

}  // namespace

std::size_t deepseek_v3_dense_block_activation_elements(
    const DeepSeekV3DenseBlockConfig& config) {
    return make_activation_layout(static_cast<float*>(nullptr), config).elements;
}

std::size_t deepseek_v3_dense_block_backward_workspace_elements(
    const DeepSeekV3DenseBlockConfig& config) {
    return make_backward_layout(static_cast<float*>(nullptr), config).elements;
}

std::size_t deepseek_v3_dense_block_bf16_workspace_elements(
    const DeepSeekV3DenseBlockConfig& config) {
    return mla_layer_bf16_workspace_elements(config.mla);
}

void deepseek_v3_dense_block_forward_cuda(
    float* output,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    __nv_bfloat16* bf16_workspace,
    const DeepSeekV3DenseBlockConfig& config,
    cudaStream_t stream) {
    const int hidden_elements = config.rows * config.hidden_size;
    auto saved = make_activation_layout(activations, config);
    rmsnorm_forward_cuda(
        saved.norm1,
        saved.inverse_rms1,
        input,
        parameters.attention_norm_weight,
        config.rows,
        config.hidden_size,
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
        saved.residual1, input, saved.attention_output, hidden_elements, stream);
    rmsnorm_forward_cuda(
        saved.norm2,
        saved.inverse_rms2,
        saved.residual1,
        parameters.ffn_norm_weight,
        config.rows,
        config.hidden_size,
        config.epsilon,
        stream);
    matmul_fp32_forward_cuda(
        saved.gate,
        saved.norm2,
        parameters.gate_weight,
        config.rows,
        config.ffn_size,
        config.hidden_size,
        stream);
    matmul_fp32_forward_cuda(
        saved.up,
        saved.norm2,
        parameters.up_weight,
        config.rows,
        config.ffn_size,
        config.hidden_size,
        stream);
    swiglu_forward_cuda(
        saved.hidden,
        saved.gate,
        saved.up,
        config.rows * config.ffn_size,
        stream);
    matmul_fp32_forward_cuda(
        saved.ffn_output,
        saved.hidden,
        parameters.down_weight,
        config.rows,
        config.hidden_size,
        config.ffn_size,
        stream);
    residual_forward_cuda(
        output, saved.residual1, saved.ffn_output, hidden_elements, stream);
}

void deepseek_v3_dense_block_backward_cuda(
    float* input_gradient,
    const DeepSeekV3DenseBlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const float* activations,
    float* workspace,
    __nv_bfloat16* bf16_workspace,
    const DeepSeekV3DenseBlockConfig& config,
    cudaStream_t stream) {
    const int hidden_elements = config.rows * config.hidden_size;
    const auto saved = make_activation_layout(activations, config);
    auto gradient = make_backward_layout(workspace, config);
    CUDA_CHECK(cudaMemsetAsync(
        workspace,
        0,
        deepseek_v3_dense_block_backward_workspace_elements(config)
            * sizeof(float),
        stream));
    residual_backward_cuda(
        gradient.residual1_gradient,
        gradient.ffn_output_gradient,
        output_gradient,
        hidden_elements,
        stream);
    matmul_fp32_backward_cuda(
        gradient.hidden_gradient,
        parameter_gradients.down_weight,
        gradient.ffn_output_gradient,
        saved.hidden,
        parameters.down_weight,
        config.rows,
        config.hidden_size,
        config.ffn_size,
        stream);
    swiglu_backward_cuda(
        gradient.gate_gradient,
        gradient.up_gradient,
        gradient.hidden_gradient,
        saved.gate,
        saved.up,
        config.rows * config.ffn_size,
        stream);
    matmul_fp32_backward_cuda(
        gradient.norm2_gradient,
        parameter_gradients.gate_weight,
        gradient.gate_gradient,
        saved.norm2,
        parameters.gate_weight,
        config.rows,
        config.ffn_size,
        config.hidden_size,
        stream);
    matmul_fp32_backward_cuda(
        gradient.norm2_gradient,
        parameter_gradients.up_weight,
        gradient.up_gradient,
        saved.norm2,
        parameters.up_weight,
        config.rows,
        config.ffn_size,
        config.hidden_size,
        stream);
    rmsnorm_backward_cuda(
        gradient.residual1_gradient,
        parameter_gradients.ffn_norm_weight,
        gradient.norm2_gradient,
        saved.residual1,
        parameters.ffn_norm_weight,
        saved.inverse_rms2,
        config.rows,
        config.hidden_size,
        stream);
    residual_backward_cuda(
        input_gradient,
        gradient.attention_output_gradient,
        gradient.residual1_gradient,
        hidden_elements,
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
        config.rows,
        config.hidden_size,
        stream);
}

}  // namespace dscuda

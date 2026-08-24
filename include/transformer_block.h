#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

struct TransformerBlockConfig {
    int batch_size;
    int sequence_length;
    int hidden_size;
    int heads;
    int head_size;
    int ffn_size;
    int rotary_size;
    float epsilon;
    float attention_scale;
};

struct TransformerBlockParameters {
    const float* attention_norm_weight;
    const float* query_weight;
    const float* key_weight;
    const float* value_weight;
    const float* output_weight;
    const float* ffn_norm_weight;
    const float* gate_weight;
    const float* up_weight;
    const float* down_weight;
};

struct TransformerBlockGradients {
    float* attention_norm_weight;
    float* query_weight;
    float* key_weight;
    float* value_weight;
    float* output_weight;
    float* ffn_norm_weight;
    float* gate_weight;
    float* up_weight;
    float* down_weight;
};

// The activation buffer holds every value required by backward plus the
// temporary storage used by the composed attention forward pass.
std::size_t transformer_block_activation_elements(
    const TransformerBlockConfig& config);

std::size_t transformer_block_backward_workspace_elements(
    const TransformerBlockConfig& config);

// Computes a pre-norm dense transformer block and saves its backward context in
// activations. Parameters use row-major [input features, output features].
void transformer_block_forward_cuda(
    float* output,
    const float* input,
    const TransformerBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    const TransformerBlockConfig& config,
    cudaStream_t stream = nullptr);

// Accumulates into input_gradient and every parameter-gradient buffer. The
// forward activations must remain unchanged until this call has completed.
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
    cudaStream_t stream = nullptr);

}  // namespace dscuda

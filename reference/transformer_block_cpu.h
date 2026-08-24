#pragma once

#include "transformer_block.h"

namespace dscuda {

void transformer_block_forward_cpu(
    float* output,
    const float* input,
    const TransformerBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const TransformerBlockConfig& config);

// Recomputes the forward intermediates, then accumulates the analytical input
// and parameter gradients using the scalar CPU references.
void transformer_block_backward_cpu(
    float* input_gradient,
    const TransformerBlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const TransformerBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const TransformerBlockConfig& config);

}  // namespace dscuda

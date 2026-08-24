#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

// Q, K, V, output, and their gradients use [B, T, H, D]. Saved probabilities
// use [B, H, T, T], and D must be divisible by four for vectorized layout copies.
std::size_t dense_attention_forward_workspace_elements(
    int batch_size,
    int sequence_length,
    int heads,
    int head_size);

std::size_t dense_attention_backward_workspace_elements(
    int batch_size,
    int sequence_length,
    int heads,
    int head_size);

void dense_attention_forward_cuda(
    float* output,
    float* probabilities,
    const float* query,
    const float* key,
    const float* value,
    float* workspace,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream = nullptr);

// Adds the analytical gradients to query_gradient, key_gradient, and
// value_gradient so the operator composes with other graph branches.
void dense_attention_backward_cuda(
    float* query_gradient,
    float* key_gradient,
    float* value_gradient,
    const float* output_gradient,
    const float* probabilities,
    const float* query,
    const float* key,
    const float* value,
    float* workspace,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream = nullptr);

}  // namespace dscuda

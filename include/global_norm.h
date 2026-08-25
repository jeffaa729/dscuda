#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

// Returns the FP32 scratch elements required by the two-pass global reduction.
// The flattened gradient buffer must contain a multiple of four elements.
std::size_t global_norm_workspace_elements(int elements);

void global_norm_cuda(
    float* norm,
    const float* gradients,
    float* workspace,
    int elements,
    cudaStream_t stream = nullptr);

// Scales the flattened gradient buffer in place when norm exceeds max_norm.
void clip_gradients_cuda(
    float* gradients,
    const float* norm,
    int elements,
    float max_norm,
    cudaStream_t stream = nullptr);

}  // namespace dscuda

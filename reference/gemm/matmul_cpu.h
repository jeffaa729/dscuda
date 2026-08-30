#pragma once

namespace dscuda {

// Scalar reference for the same row-major transpose/accumulation contract as the CUDA GEMMs.
void gemm_cpu(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    bool transpose_left = false,
    bool transpose_right = false,
    bool accumulate = false);

}  // namespace dscuda

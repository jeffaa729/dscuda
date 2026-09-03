#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Computes one variable-row linear layer over expert-grouped input. Weights
// use [E,K,N], and slot_expert selects the matrix for each dispatched row.
void grouped_linear_forward_cuda(
    float* output,
    const float* input,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size,
    cudaStream_t stream = nullptr);

// Runs one variable-M BF16 Tensor Core GEMM per expert over the contiguous
// ranges described by expert_offsets. Inputs and weights use BF16, output and
// output use BF16, accumulation uses FP32, and weights have layout [E,K,N].
void grouped_linear_bf16_forward_cuda(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const int* expert_offsets,
    int dispatched_rows,
    int experts,
    int output_size,
    int input_size,
    cudaStream_t stream = nullptr);

void grouped_linear_backward_cuda(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const int* expert_offsets,
    const int* slot_expert,
    int dispatched_rows,
    int experts,
    int output_size,
    int input_size,
    bool accumulate_input,
    cudaStream_t stream = nullptr);

}  // namespace dscuda

#pragma once

#include "grouped_gemm.h"

namespace dscuda {

void grouped_linear_forward_sm89_cuda(
    float* output,
    const float* input,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size,
    cudaStream_t stream);

void grouped_linear_bf16_forward_sm89_cuda(
    float* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const int* expert_offsets,
    int dispatched_rows,
    int experts,
    int output_size,
    int input_size,
    cudaStream_t stream);

void grouped_linear_backward_sm89_cuda(
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
    cudaStream_t stream);

}  // namespace dscuda

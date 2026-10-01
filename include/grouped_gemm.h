#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Runs one variable-M BF16 Tensor Core GEMM per expert over the contiguous
// ranges described by expert_offsets. Inputs, [E,K,N] weights and output
// use BF16; accumulation uses FP32.
void grouped_linear_bf16_forward_cuda(__nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                      const int* expert_offsets, int dispatched_rows, int experts, int output_size,
                                      int input_size, cudaStream_t stream = nullptr);

}  // namespace dscuda

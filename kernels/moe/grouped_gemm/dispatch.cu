#include "common.cuh"

namespace dscuda {

void grouped_linear_bf16_forward_cuda(__nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                      const int* expert_offsets, int dispatched_rows, int experts, int output_size,
                                      int input_size, cudaStream_t stream) {
    grouped_linear_bf16_forward_sm89_cuda(output, input, weight, expert_offsets, dispatched_rows,
                                           experts, output_size, input_size, stream);
}

}  // namespace dscuda

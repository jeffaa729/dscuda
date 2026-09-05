#include "common.cuh"

namespace dscuda {

void grouped_linear_forward_cuda(
    float* output,
    const float* input,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size,
    cudaStream_t stream) {
    grouped_linear_forward_sm89_cuda(
        output,
        input,
        weight,
        slot_expert,
        dispatched_rows,
        output_size,
        input_size,
        stream);
}

void grouped_linear_bf16_forward_cuda(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const int* expert_offsets,
    int dispatched_rows,
    int experts,
    int output_size,
    int input_size,
    cudaStream_t stream) {
    grouped_linear_bf16_forward_sm89_cuda(
        output,
        input,
        weight,
        expert_offsets,
        dispatched_rows,
        experts,
        output_size,
        input_size,
        stream);
}

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
    cudaStream_t stream) {
    grouped_linear_backward_sm89_cuda(
        input_gradient,
        weight_gradient,
        output_gradient,
        input,
        weight,
        expert_offsets,
        slot_expert,
        dispatched_rows,
        experts,
        output_size,
        input_size,
        accumulate_input,
        stream);
}

}  // namespace dscuda

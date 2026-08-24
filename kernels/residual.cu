// Adds a transformer branch to its residual stream with vectorized FP32 memory operations.
// The backward kernel accumulates the shared upstream gradient into both graph branches.

#include "cuda_common.h"
#include "residual.h"

namespace dscuda {
namespace {

constexpr int kThreads = 256;

__global__ void residual_forward_kernel(
    float4* output,
    const float4* input,
    const float4* branch,
    int vectors) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vectors) {
        return;
    }

    const float4 left = input[index];
    const float4 right = branch[index];
    output[index] = make_float4(
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
        left.w + right.w);
}

__global__ void residual_backward_kernel(
    float4* input_gradient,
    float4* branch_gradient,
    const float4* output_gradient,
    int vectors) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vectors) {
        return;
    }

    const float4 gradient = output_gradient[index];
    float4 input_value = input_gradient[index];
    float4 branch_value = branch_gradient[index];
    input_value.x += gradient.x;
    input_value.y += gradient.y;
    input_value.z += gradient.z;
    input_value.w += gradient.w;
    branch_value.x += gradient.x;
    branch_value.y += gradient.y;
    branch_value.z += gradient.z;
    branch_value.w += gradient.w;
    input_gradient[index] = input_value;
    branch_gradient[index] = branch_value;
}

}  // namespace

void residual_forward_cuda(
    float* output,
    const float* input,
    const float* branch,
    int elements,
    cudaStream_t stream) {
    const int vectors = elements / 4;
    const int blocks = (vectors + kThreads - 1) / kThreads;
    residual_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4*>(output),
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(branch),
        vectors);
    CUDA_CHECK(cudaGetLastError());
}

void residual_backward_cuda(
    float* input_gradient,
    float* branch_gradient,
    const float* output_gradient,
    int elements,
    cudaStream_t stream) {
    const int vectors = elements / 4;
    const int blocks = (vectors + kThreads - 1) / kThreads;
    residual_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4*>(input_gradient),
        reinterpret_cast<float4*>(branch_gradient),
        reinterpret_cast<const float4*>(output_gradient),
        vectors);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

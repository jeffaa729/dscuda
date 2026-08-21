// Implements the fused SwiGLU activation by multiplying a SiLU-activated gate projection with an up projection.
// Its backward kernel computes the gate and up-projection gradients in one coalesced elementwise pass.

#include "cuda_common.h"
#include "swiglu.h"

namespace dscuda {
namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
}

__global__ void swiglu_forward_kernel(
    float* output,
    const float* gate,
    const float* up,
    int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }

    const float gate_value = gate[index];
    output[index] = gate_value * sigmoid(gate_value) * up[index];
}

__global__ void swiglu_backward_kernel(
    float* gate_gradient,
    float* up_gradient,
    const float* output_gradient,
    const float* gate,
    const float* up,
    int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }

    const float gate_value = gate[index];
    const float sigmoid_value = sigmoid(gate_value);
    const float silu_value = gate_value * sigmoid_value;
    const float silu_gradient =
        sigmoid_value * (1.0F + gate_value * (1.0F - sigmoid_value));
    const float gradient = output_gradient[index];

    gate_gradient[index] = gradient * up[index] * silu_gradient;
    up_gradient[index] = gradient * silu_value;
}

}  // namespace

void swiglu_forward_cuda(
    float* output,
    const float* gate,
    const float* up,
    int elements,
    cudaStream_t stream) {
    const int blocks = (elements + kBlockSize - 1) / kBlockSize;
    swiglu_forward_kernel<<<blocks, kBlockSize, 0, stream>>>(output, gate, up, elements);
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_backward_cuda(
    float* gate_gradient,
    float* up_gradient,
    const float* output_gradient,
    const float* gate,
    const float* up,
    int elements,
    cudaStream_t stream) {
    const int blocks = (elements + kBlockSize - 1) / kBlockSize;
    swiglu_backward_kernel<<<blocks, kBlockSize, 0, stream>>>(
        gate_gradient, up_gradient, output_gradient, gate, up, elements);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

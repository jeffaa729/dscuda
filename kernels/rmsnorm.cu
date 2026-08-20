// Implements RMSNorm by scaling activations with their inverse root-mean-square and a learned per-channel weight.
// Its backward path computes activation and scale gradients using reduction kernels with FP32 accumulation.

#include "cuda_common.h"
#include "rmsnorm.h"

namespace dscuda {
namespace {

constexpr int kBlockSize = 256;

__global__ void rmsnorm_forward_kernel(
    float* output,
    float* inverse_rms,
    const float* input,
    const float* weight,
    int hidden_size,
    float epsilon) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int offset = row * hidden_size;

    float local_sum = 0.0F;
    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const float value = input[offset + column];
        local_sum += value * value;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        shared[0] = rsqrtf(shared[0] / hidden_size + epsilon);
        inverse_rms[row] = shared[0];
    }
    __syncthreads();

    const float scale = shared[0];
    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        output[offset + column] = input[offset + column] * scale * weight[column];
    }
}

__global__ void rmsnorm_backward_kernel(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const float* inverse_rms,
    int hidden_size) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int offset = row * hidden_size;
    const float scale = inverse_rms[row];

    float local_projection = 0.0F;
    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const int index = offset + column;
        local_projection += output_gradient[index] * weight[column] * input[index];
    }
    shared[threadIdx.x] = local_projection;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float correction = shared[0] * scale * scale * scale / hidden_size;
    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const int index = offset + column;
        input_gradient[index] +=
            scale * output_gradient[index] * weight[column] - input[index] * correction;
        atomicAdd(&weight_gradient[column], output_gradient[index] * input[index] * scale);
    }
}

}  // namespace

void rmsnorm_forward_cuda(
    float* output,
    float* inverse_rms,
    const float* input,
    const float* weight,
    int rows,
    int hidden_size,
    float epsilon,
    cudaStream_t stream) {
    rmsnorm_forward_kernel<<<rows, kBlockSize, kBlockSize * sizeof(float), stream>>>(
        output, inverse_rms, input, weight, hidden_size, epsilon);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_backward_cuda(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const float* inverse_rms,
    int rows,
    int hidden_size,
    cudaStream_t stream) {
    rmsnorm_backward_kernel<<<rows, kBlockSize, kBlockSize * sizeof(float), stream>>>(
        input_gradient,
        weight_gradient,
        output_gradient,
        input,
        weight,
        inverse_rms,
        hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

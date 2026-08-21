// Implements RMSNorm by scaling activations with their inverse root-mean-square and a learned per-channel weight.
// Its backward path computes activation and scale gradients using reduction kernels with FP32 accumulation.

#include "cuda_common.h"
#include "rmsnorm.h"

namespace dscuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;

template <int kWidth = kWarpSize>
__device__ __forceinline__ float warp_reduce_sum_f32(float value) {
#pragma unroll
    for (int mask = kWidth / 2; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, mask);
    }
    return value;
}

template <int kNumThreads = kBlockSize>
__device__ __forceinline__ float block_reduce_sum_f32(float value) {
    constexpr int kNumWarps = kNumThreads / kWarpSize;
    __shared__ float warp_sums[kNumWarps];

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    value = warp_reduce_sum_f32(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < kNumWarps ? warp_sums[lane] : 0.0F;
        value = warp_reduce_sum_f32(value);
    }
    return value;
}

__global__ void rmsnorm_forward_kernel(
    float* output,
    float* inverse_rms,
    const float* input,
    const float* weight,
    int hidden_size,
    float epsilon) {
    __shared__ float shared_scale;
    const int row = blockIdx.x;
    const int offset = row * hidden_size;

    float local_sum = 0.0F;
    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const float value = input[offset + column];
        local_sum += value * value;
    }
    local_sum = block_reduce_sum_f32(local_sum);

    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(local_sum / hidden_size + epsilon);
        inverse_rms[row] = shared_scale;
    }
    __syncthreads();

    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        output[offset + column] = input[offset + column] * shared_scale * weight[column];
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
    __shared__ float shared_correction;
    const int row = blockIdx.x;
    const int offset = row * hidden_size;
    const float scale = inverse_rms[row];

    float local_projection = 0.0F;
    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const int index = offset + column;
        local_projection += output_gradient[index] * weight[column] * input[index];
    }
    local_projection = block_reduce_sum_f32(local_projection);

    if (threadIdx.x == 0) {
        shared_correction = local_projection * scale * scale * scale / hidden_size;
    }
    __syncthreads();

    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const int index = offset + column;
        input_gradient[index] +=
            scale * output_gradient[index] * weight[column] -
            input[index] * shared_correction;
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
    rmsnorm_forward_kernel<<<rows, kBlockSize, 0, stream>>>(
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
    rmsnorm_backward_kernel<<<rows, kBlockSize, 0, stream>>>(
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

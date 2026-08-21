// Implements vectorized FP32 RMSNorm by scaling activations with their inverse root-mean-square and a learned per-channel weight.
// Its backward path uses FP32 accumulation; vector access assumes aligned rows and hidden sizes divisible by four.

#include "cuda_common.h"
#include "rmsnorm.h"

namespace dscuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;
constexpr int kForwardVectorWidth = 4;
constexpr int kBackwardVectorWidth = 2;

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
    const int vector_size = hidden_size / kForwardVectorWidth;
    const auto* input_vectors = reinterpret_cast<const float4*>(input + offset);
    const auto* weight_vectors = reinterpret_cast<const float4*>(weight);
    auto* output_vectors = reinterpret_cast<float4*>(output + offset);

    float local_sum = 0.0F;
    for (int column = threadIdx.x; column < vector_size; column += blockDim.x) {
        const float4 value = input_vectors[column];
        local_sum += value.x * value.x + value.y * value.y +
                     value.z * value.z + value.w * value.w;
    }
    local_sum = block_reduce_sum_f32(local_sum);

    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(local_sum / hidden_size + epsilon);
        inverse_rms[row] = shared_scale;
    }
    __syncthreads();

    for (int column = threadIdx.x; column < vector_size; column += blockDim.x) {
        const float4 value = input_vectors[column];
        const float4 channel_weight = weight_vectors[column];
        output_vectors[column] = make_float4(
            value.x * shared_scale * channel_weight.x,
            value.y * shared_scale * channel_weight.y,
            value.z * shared_scale * channel_weight.z,
            value.w * shared_scale * channel_weight.w);
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
    const int vector_size = hidden_size / kBackwardVectorWidth;
    const float scale = inverse_rms[row];
    const auto* output_gradient_vectors =
        reinterpret_cast<const float2*>(output_gradient + offset);
    const auto* input_vectors = reinterpret_cast<const float2*>(input + offset);
    const auto* weight_vectors = reinterpret_cast<const float2*>(weight);

    float local_projection = 0.0F;
    for (int column = threadIdx.x; column < vector_size; column += blockDim.x) {
        const float2 output_gradient_value = output_gradient_vectors[column];
        const float2 input_value = input_vectors[column];
        const float2 channel_weight = weight_vectors[column];
        local_projection +=
            output_gradient_value.x * channel_weight.x * input_value.x +
            output_gradient_value.y * channel_weight.y * input_value.y;
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

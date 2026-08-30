// Computes a flattened FP32 gradient L2 norm with a grid reduction followed by one final block reduction.
// A separate vectorized kernel clips gradients in place using the device-resident norm without a host synchronization.

#include "cuda_common.h"
#include "global_norm.h"

#include <algorithm>

namespace dscuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kMaximumBlocks = 1024;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    constexpr int kWarps = kThreads / kWarpSize;
    __shared__ float warp_sums[kWarps];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_sums[lane] : 0.0F;
        value = warp_reduce_sum(value);
    }
    return value;
}

__global__ void global_norm_partial_kernel(
    float* partial_sums,
    const float4* gradients,
    int vectors) {
    float sum = 0.0F;
    for (int vector = blockIdx.x * blockDim.x + threadIdx.x;
         vector < vectors;
         vector += blockDim.x * gridDim.x) {
        const float4 gradient = gradients[vector];
        sum += gradient.x * gradient.x + gradient.y * gradient.y +
               gradient.z * gradient.z + gradient.w * gradient.w;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sum;
    }
}

__global__ void global_norm_finalize_kernel(
    float* norm,
    const float* partial_sums,
    int partial_count) {
    float sum = 0.0F;
    for (int index = threadIdx.x;
         index < partial_count;
         index += blockDim.x) {
        sum += partial_sums[index];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        *norm = sqrtf(sum);
    }
}

__global__ void clip_gradients_kernel(
    float4* gradients,
    const float* norm,
    int vectors,
    float max_norm) {
    const float scale = fminf(1.0F, max_norm / *norm);
    for (int vector = blockIdx.x * blockDim.x + threadIdx.x;
         vector < vectors;
         vector += blockDim.x * gridDim.x) {
        const float4 gradient = gradients[vector];
        gradients[vector] = make_float4(
            gradient.x * scale,
            gradient.y * scale,
            gradient.z * scale,
            gradient.w * scale);
    }
}

int reduction_blocks(int elements) {
    const int vectors = elements / 4;
    return std::min(
        (vectors + kThreads - 1) / kThreads,
        kMaximumBlocks);
}

}  // namespace

std::size_t global_norm_workspace_elements(int elements) {
    return reduction_blocks(elements);
}

void global_norm_cuda(
    float* norm,
    const float* gradients,
    float* workspace,
    int elements,
    cudaStream_t stream) {
    const int vectors = elements / 4;
    const int blocks = reduction_blocks(elements);
    global_norm_partial_kernel<<<blocks, kThreads, 0, stream>>>(
        workspace,
        reinterpret_cast<const float4*>(gradients),
        vectors);
    CUDA_CHECK(cudaGetLastError());
    global_norm_finalize_kernel<<<1, kThreads, 0, stream>>>(
        norm, workspace, blocks);
    CUDA_CHECK(cudaGetLastError());
}

void clip_gradients_cuda(
    float* gradients,
    const float* norm,
    int elements,
    float max_norm,
    cudaStream_t stream) {
    const int vectors = elements / 4;
    const int blocks = reduction_blocks(elements);
    clip_gradients_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4*>(gradients),
        norm,
        vectors,
        max_norm);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

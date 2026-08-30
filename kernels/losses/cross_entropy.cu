// Fuses stable vocabulary softmax with mean negative log-likelihood without writing a probability tensor.
// Forward saves one log-sum-exp per token, and backward reconstructs probabilities directly into the logits gradient.

#include "cross_entropy.h"
#include "cuda_common.h"

#include <cuda_runtime.h>

#include <cfloat>

namespace dscuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

template <bool kMaximum>
__device__ __forceinline__ float block_reduce(float value) {
    constexpr int kWarps = kThreads / kWarpSize;
    __shared__ float warp_values[kWarps];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = kMaximum ? warp_reduce_max(value) : warp_reduce_sum(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps
            ? warp_values[lane]
            : (kMaximum ? -FLT_MAX : 0.0F);
        value = kMaximum ? warp_reduce_max(value) : warp_reduce_sum(value);
    }
    return value;
}

__global__ void cross_entropy_forward_kernel(
    float* mean_loss,
    float* logsumexp,
    const float* logits,
    const int* targets,
    int rows,
    int vocabulary_size) {
    __shared__ float row_maximum;
    const int row = blockIdx.x;
    const int offset = row * vocabulary_size;

    float maximum = -FLT_MAX;
    for (int column = threadIdx.x;
         column < vocabulary_size;
         column += blockDim.x) {
        maximum = fmaxf(maximum, logits[offset + column]);
    }
    maximum = block_reduce<true>(maximum);
    if (threadIdx.x == 0) {
        row_maximum = maximum;
    }
    __syncthreads();

    float sum = 0.0F;
    for (int column = threadIdx.x;
         column < vocabulary_size;
         column += blockDim.x) {
        sum += expf(logits[offset + column] - row_maximum);
    }
    sum = block_reduce<false>(sum);
    if (threadIdx.x == 0) {
        const float row_logsumexp = row_maximum + logf(sum);
        logsumexp[row] = row_logsumexp;
        atomicAdd(
            mean_loss,
            (row_logsumexp - logits[offset + targets[row]]) / rows);
    }
}

__global__ void cross_entropy_backward_kernel(
    float* logits_gradient,
    const float* logits,
    const float* logsumexp,
    const int* targets,
    int rows,
    int vocabulary_size) {
    const int row = blockIdx.x;
    const int offset = row * vocabulary_size;
    const float scale = 1.0F / rows;
    const float row_logsumexp = logsumexp[row];
    const int target = targets[row];
    for (int column = threadIdx.x;
         column < vocabulary_size;
         column += blockDim.x) {
        const float probability =
            expf(logits[offset + column] - row_logsumexp);
        logits_gradient[offset + column] = scale *
            (probability - static_cast<float>(column == target));
    }
}

}  // namespace

void cross_entropy_forward_cuda(
    float* mean_loss,
    float* logsumexp,
    const float* logits,
    const int* targets,
    int rows,
    int vocabulary_size,
    cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(mean_loss, 0, sizeof(float), stream));
    cross_entropy_forward_kernel<<<rows, kThreads, 0, stream>>>(
        mean_loss,
        logsumexp,
        logits,
        targets,
        rows,
        vocabulary_size);
    CUDA_CHECK(cudaGetLastError());
}

void cross_entropy_backward_cuda(
    float* logits_gradient,
    const float* logits,
    const float* logsumexp,
    const int* targets,
    int rows,
    int vocabulary_size,
    cudaStream_t stream) {
    cross_entropy_backward_kernel<<<rows, kThreads, 0, stream>>>(
        logits_gradient,
        logits,
        logsumexp,
        targets,
        rows,
        vocabulary_size);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

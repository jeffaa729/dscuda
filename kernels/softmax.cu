// Computes numerically stable scaled causal softmax over square attention-score rows.
// Its backward path applies the softmax Jacobian only to visible keys and accumulates gradients into the logits.

#include "cuda_common.h"
#include "softmax.h"

#include <cfloat>

namespace dscuda {
namespace {

constexpr int kWarpSize = 32;

template <int kBlockSize>
__device__ float block_reduce_sum(float value, float* warp_values) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    constexpr int kWarps = kBlockSize / kWarpSize;
    if (warp == 0) {
        value = lane < kWarps ? warp_values[lane] : 0.0F;
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

template <int kBlockSize>
__device__ float block_reduce_max(float value, float* warp_values) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    constexpr int kWarps = kBlockSize / kWarpSize;
    if (warp == 0) {
        value = lane < kWarps ? warp_values[lane] : -FLT_MAX;
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

template <int kBlockSize>
__global__ void causal_softmax_forward_kernel(
    float* probabilities,
    const float* logits,
    int sequence_length,
    float scale) {
    __shared__ float max_warp_values[kBlockSize / kWarpSize];
    __shared__ float sum_warp_values[kBlockSize / kWarpSize];

    const int query = blockIdx.x % sequence_length;
    const int visible_keys = query + 1;
    const int row_offset = blockIdx.x * sequence_length;

    float local_max = -FLT_MAX;
    for (int key = threadIdx.x; key < visible_keys; key += kBlockSize) {
        local_max = fmaxf(local_max, scale * logits[row_offset + key]);
    }
    const float row_max = block_reduce_max<kBlockSize>(local_max, max_warp_values);

    float local_sum = 0.0F;
    for (int key = threadIdx.x; key < visible_keys; key += kBlockSize) {
        const float exponential = expf(scale * logits[row_offset + key] - row_max);
        probabilities[row_offset + key] = exponential;
        local_sum += exponential;
    }
    const float inverse_sum =
        1.0F / block_reduce_sum<kBlockSize>(local_sum, sum_warp_values);

    for (int key = threadIdx.x; key < visible_keys; key += kBlockSize) {
        probabilities[row_offset + key] *= inverse_sum;
    }
    for (int key = visible_keys + threadIdx.x; key < sequence_length; key += kBlockSize) {
        probabilities[row_offset + key] = 0.0F;
    }
}

template <int kBlockSize>
__global__ void causal_softmax_backward_kernel(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int sequence_length,
    float scale) {
    __shared__ float sum_warp_values[kBlockSize / kWarpSize];

    const int query = blockIdx.x % sequence_length;
    const int visible_keys = query + 1;
    const int row_offset = blockIdx.x * sequence_length;

    float local_projection = 0.0F;
    for (int key = threadIdx.x; key < visible_keys; key += kBlockSize) {
        local_projection +=
            probabilities_gradient[row_offset + key] * probabilities[row_offset + key];
    }
    const float projection =
        block_reduce_sum<kBlockSize>(local_projection, sum_warp_values);

    for (int key = threadIdx.x; key < visible_keys; key += kBlockSize) {
        const float probability = probabilities[row_offset + key];
        logits_gradient[row_offset + key] +=
            scale * probability *
            (probabilities_gradient[row_offset + key] - projection);
    }
}

template <int kBlockSize>
void launch_causal_softmax_forward(
    float* probabilities,
    const float* logits,
    int rows,
    int sequence_length,
    float scale,
    cudaStream_t stream) {
    causal_softmax_forward_kernel<kBlockSize><<<rows, kBlockSize, 0, stream>>>(
        probabilities, logits, sequence_length, scale);
}

template <int kBlockSize>
void launch_causal_softmax_backward(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int rows,
    int sequence_length,
    float scale,
    cudaStream_t stream) {
    causal_softmax_backward_kernel<kBlockSize><<<rows, kBlockSize, 0, stream>>>(
        logits_gradient,
        probabilities_gradient,
        probabilities,
        sequence_length,
        scale);
}

}  // namespace

void causal_softmax_forward_cuda(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale,
    cudaStream_t stream) {
    const int rows = batch_size * heads * sequence_length;
    if (sequence_length <= 32) {
        launch_causal_softmax_forward<32>(
            probabilities, logits, rows, sequence_length, scale, stream);
    } else if (sequence_length <= 64) {
        launch_causal_softmax_forward<64>(
            probabilities, logits, rows, sequence_length, scale, stream);
    } else if (sequence_length <= 128) {
        launch_causal_softmax_forward<128>(
            probabilities, logits, rows, sequence_length, scale, stream);
    } else {
        launch_causal_softmax_forward<256>(
            probabilities, logits, rows, sequence_length, scale, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void causal_softmax_backward_cuda(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale,
    cudaStream_t stream) {
    const int rows = batch_size * heads * sequence_length;
    if (sequence_length <= 32) {
        launch_causal_softmax_backward<32>(
            logits_gradient,
            probabilities_gradient,
            probabilities,
            rows,
            sequence_length,
            scale,
            stream);
    } else if (sequence_length <= 64) {
        launch_causal_softmax_backward<64>(
            logits_gradient,
            probabilities_gradient,
            probabilities,
            rows,
            sequence_length,
            scale,
            stream);
    } else if (sequence_length <= 128) {
        launch_causal_softmax_backward<128>(
            logits_gradient,
            probabilities_gradient,
            probabilities,
            rows,
            sequence_length,
            scale,
            stream);
    } else {
        launch_causal_softmax_backward<256>(
            logits_gradient,
            probabilities_gradient,
            probabilities,
            rows,
            sequence_length,
            scale,
            stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

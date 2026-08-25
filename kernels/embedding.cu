// Implements vectorized FP32 token-embedding lookup from a row-major vocabulary table.
// Its backward kernel atomically accumulates four channel gradients per thread so repeated token IDs remain correct.

#include "cuda_common.h"
#include "embedding.h"

namespace dscuda {
namespace {

constexpr int kThreads = 256;

__global__ void embedding_forward_kernel(
    float4* output,
    const int* token_ids,
    const float4* weight,
    int vectors_per_token,
    int total_vectors) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= total_vectors) {
        return;
    }

    const int token = vector / vectors_per_token;
    const int column = vector % vectors_per_token;
    output[vector] =
        weight[token_ids[token] * vectors_per_token + column];
}

__global__ void embedding_backward_kernel(
    float4* weight_gradient,
    const float4* output_gradient,
    const int* token_ids,
    int vectors_per_token,
    int total_vectors) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= total_vectors) {
        return;
    }

    const int token = vector / vectors_per_token;
    const int column = vector % vectors_per_token;
    const int destination =
        token_ids[token] * vectors_per_token + column;
    const float4 gradient = output_gradient[vector];
    atomicAdd(&weight_gradient[destination].x, gradient.x);
    atomicAdd(&weight_gradient[destination].y, gradient.y);
    atomicAdd(&weight_gradient[destination].z, gradient.z);
    atomicAdd(&weight_gradient[destination].w, gradient.w);
}

}  // namespace

void embedding_forward_cuda(
    float* output,
    const int* token_ids,
    const float* weight,
    int token_count,
    int hidden_size,
    cudaStream_t stream) {
    const int vectors_per_token = hidden_size / 4;
    const int total_vectors = token_count * vectors_per_token;
    const int blocks = (total_vectors + kThreads - 1) / kThreads;
    embedding_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4*>(output),
        token_ids,
        reinterpret_cast<const float4*>(weight),
        vectors_per_token,
        total_vectors);
    CUDA_CHECK(cudaGetLastError());
}

void embedding_backward_cuda(
    float* weight_gradient,
    const float* output_gradient,
    const int* token_ids,
    int token_count,
    int hidden_size,
    cudaStream_t stream) {
    const int vectors_per_token = hidden_size / 4;
    const int total_vectors = token_count * vectors_per_token;
    const int blocks = (total_vectors + kThreads - 1) / kThreads;
    embedding_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4*>(weight_gradient),
        reinterpret_cast<const float4*>(output_gradient),
        token_ids,
        vectors_per_token,
        total_vectors);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

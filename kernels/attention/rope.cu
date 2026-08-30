// Applies interleaved Rotary Positional Embeddings to a configurable prefix of each attention head.
// Its backward path applies the inverse rotation while leaving dimensions outside the rotary prefix unchanged.

#include "cuda_common.h"
#include "rope.h"

namespace dscuda {
namespace {

constexpr int kBlockSize = 256;

__global__ void rope_forward_kernel(
    float* output,
    const float* input,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_head = head_size / 2;
    const int total_pairs = batch_size * sequence_length * heads * pairs_per_head;
    if (pair_index >= total_pairs) {
        return;
    }

    const int pair_in_head = pair_index % pairs_per_head;
    const int head_index = pair_index / pairs_per_head;
    const int position = (head_index / heads) % sequence_length;

    const float2 value = reinterpret_cast<const float2*>(input)[pair_index];
    float2 result = value;
    if (pair_in_head < rotary_size / 2) {
        const int frequency_index = position * (rotary_size / 2) + pair_in_head;
        const float cos_value = cosine[frequency_index];
        const float sin_value = sine[frequency_index];
        result.x = value.x * cos_value - value.y * sin_value;
        result.y = value.x * sin_value + value.y * cos_value;
    }
    reinterpret_cast<float2*>(output)[pair_index] = result;
}

__global__ void rope_backward_kernel(
    float* input_gradient,
    const float* output_gradient,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_head = head_size / 2;
    const int total_pairs = batch_size * sequence_length * heads * pairs_per_head;
    if (pair_index >= total_pairs) {
        return;
    }

    const int pair_in_head = pair_index % pairs_per_head;
    const int head_index = pair_index / pairs_per_head;
    const int position = (head_index / heads) % sequence_length;

    const float2 gradient = reinterpret_cast<const float2*>(output_gradient)[pair_index];
    float2 rotated_gradient = gradient;
    if (pair_in_head < rotary_size / 2) {
        const int frequency_index = position * (rotary_size / 2) + pair_in_head;
        const float cos_value = cosine[frequency_index];
        const float sin_value = sine[frequency_index];
        rotated_gradient.x = gradient.x * cos_value + gradient.y * sin_value;
        rotated_gradient.y = -gradient.x * sin_value + gradient.y * cos_value;
    }

    float2 accumulated = reinterpret_cast<const float2*>(input_gradient)[pair_index];
    accumulated.x += rotated_gradient.x;
    accumulated.y += rotated_gradient.y;
    reinterpret_cast<float2*>(input_gradient)[pair_index] = accumulated;
}

}  // namespace

void rope_forward_cuda(
    float* output,
    const float* input,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size,
    cudaStream_t stream) {
    const int total_pairs = batch_size * sequence_length * heads * (head_size / 2);
    const int blocks = (total_pairs + kBlockSize - 1) / kBlockSize;
    rope_forward_kernel<<<blocks, kBlockSize, 0, stream>>>(
        output,
        input,
        cosine,
        sine,
        batch_size,
        sequence_length,
        heads,
        head_size,
        rotary_size);
    CUDA_CHECK(cudaGetLastError());
}

void rope_backward_cuda(
    float* input_gradient,
    const float* output_gradient,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size,
    cudaStream_t stream) {
    const int total_pairs = batch_size * sequence_length * heads * (head_size / 2);
    const int blocks = (total_pairs + kBlockSize - 1) / kBlockSize;
    rope_backward_kernel<<<blocks, kBlockSize, 0, stream>>>(
        input_gradient,
        output_gradient,
        cosine,
        sine,
        batch_size,
        sequence_length,
        heads,
        head_size,
        rotary_size);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

// Composes dense causal attention from the optimized FP32 GEMM and softmax kernels.
// Vectorized layout transforms connect model-friendly [B,T,H,D] tensors to batched head matrices for forward and backward.

#include "attention.h"
#include "cuda_common.h"
#include "matmul.h"
#include "softmax.h"

#include <algorithm>

namespace dscuda {
namespace {

constexpr int kThreads = 256;

__global__ void attention_pack_qkv_kernel(
    float* packed_query,
    float* packed_key,
    float* packed_value,
    const float* query,
    const float* key,
    const float* value,
    int sequence_length,
    int heads,
    int vectors_per_head,
    int total_vectors) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= total_vectors) {
        return;
    }

    const int column_vector = vector % vectors_per_head;
    const int head = (vector / vectors_per_head) % heads;
    const int token =
        (vector / (vectors_per_head * heads)) % sequence_length;
    const int batch = vector / (vectors_per_head * heads * sequence_length);
    const int input_vector =
        ((batch * sequence_length + token) * heads + head) *
            vectors_per_head +
        column_vector;
    const int packed_vector =
        ((batch * heads + head) * sequence_length + token) *
            vectors_per_head +
        column_vector;

    reinterpret_cast<float4*>(packed_query)[packed_vector] =
        reinterpret_cast<const float4*>(query)[input_vector];
    reinterpret_cast<float4*>(packed_key)[packed_vector] =
        reinterpret_cast<const float4*>(key)[input_vector];
    reinterpret_cast<float4*>(packed_value)[packed_vector] =
        reinterpret_cast<const float4*>(value)[input_vector];
}

__global__ void attention_unpack_output_kernel(
    float* output,
    const float* packed_output,
    int sequence_length,
    int heads,
    int vectors_per_head,
    int total_vectors) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= total_vectors) {
        return;
    }

    const int column_vector = vector % vectors_per_head;
    const int head = (vector / vectors_per_head) % heads;
    const int token =
        (vector / (vectors_per_head * heads)) % sequence_length;
    const int batch = vector / (vectors_per_head * heads * sequence_length);
    const int output_vector =
        ((batch * sequence_length + token) * heads + head) *
            vectors_per_head +
        column_vector;
    const int packed_vector =
        ((batch * heads + head) * sequence_length + token) *
            vectors_per_head +
        column_vector;

    reinterpret_cast<float4*>(output)[output_vector] =
        reinterpret_cast<const float4*>(packed_output)[packed_vector];
}

__global__ void attention_pack_backward_kernel(
    float* packed_query,
    float* packed_key,
    float* packed_value,
    float* packed_output_gradient,
    const float* query,
    const float* key,
    const float* value,
    const float* output_gradient,
    int sequence_length,
    int heads,
    int vectors_per_head,
    int total_vectors) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= total_vectors) {
        return;
    }

    const int column_vector = vector % vectors_per_head;
    const int head = (vector / vectors_per_head) % heads;
    const int token =
        (vector / (vectors_per_head * heads)) % sequence_length;
    const int batch = vector / (vectors_per_head * heads * sequence_length);
    const int input_vector =
        ((batch * sequence_length + token) * heads + head) *
            vectors_per_head +
        column_vector;
    const int packed_vector =
        ((batch * heads + head) * sequence_length + token) *
            vectors_per_head +
        column_vector;

    reinterpret_cast<float4*>(packed_query)[packed_vector] =
        reinterpret_cast<const float4*>(query)[input_vector];
    reinterpret_cast<float4*>(packed_key)[packed_vector] =
        reinterpret_cast<const float4*>(key)[input_vector];
    reinterpret_cast<float4*>(packed_value)[packed_vector] =
        reinterpret_cast<const float4*>(value)[input_vector];
    reinterpret_cast<float4*>(packed_output_gradient)[packed_vector] =
        reinterpret_cast<const float4*>(output_gradient)[input_vector];
}

__global__ void attention_unpack_gradients_kernel(
    float* query_gradient,
    float* key_gradient,
    float* value_gradient,
    const float* packed_query_gradient,
    const float* packed_key_gradient,
    const float* packed_value_gradient,
    int sequence_length,
    int heads,
    int vectors_per_head,
    int total_vectors) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= total_vectors) {
        return;
    }

    const int column_vector = vector % vectors_per_head;
    const int head = (vector / vectors_per_head) % heads;
    const int token =
        (vector / (vectors_per_head * heads)) % sequence_length;
    const int batch = vector / (vectors_per_head * heads * sequence_length);
    const int output_vector =
        ((batch * sequence_length + token) * heads + head) *
            vectors_per_head +
        column_vector;
    const int packed_vector =
        ((batch * heads + head) * sequence_length + token) *
            vectors_per_head +
        column_vector;

    float4 query_update =
        reinterpret_cast<const float4*>(packed_query_gradient)[packed_vector];
    float4 key_update =
        reinterpret_cast<const float4*>(packed_key_gradient)[packed_vector];
    float4 value_update =
        reinterpret_cast<const float4*>(packed_value_gradient)[packed_vector];
    float4 query_previous =
        reinterpret_cast<const float4*>(query_gradient)[output_vector];
    float4 key_previous =
        reinterpret_cast<const float4*>(key_gradient)[output_vector];
    float4 value_previous =
        reinterpret_cast<const float4*>(value_gradient)[output_vector];

    query_update.x += query_previous.x;
    query_update.y += query_previous.y;
    query_update.z += query_previous.z;
    query_update.w += query_previous.w;
    key_update.x += key_previous.x;
    key_update.y += key_previous.y;
    key_update.z += key_previous.z;
    key_update.w += key_previous.w;
    value_update.x += value_previous.x;
    value_update.y += value_previous.y;
    value_update.z += value_previous.z;
    value_update.w += value_previous.w;

    reinterpret_cast<float4*>(query_gradient)[output_vector] = query_update;
    reinterpret_cast<float4*>(key_gradient)[output_vector] = key_update;
    reinterpret_cast<float4*>(value_gradient)[output_vector] = value_update;
}

int activation_elements(
    int batch_size,
    int sequence_length,
    int heads,
    int head_size) {
    return batch_size * sequence_length * heads * head_size;
}

int score_elements(int batch_size, int sequence_length, int heads) {
    return batch_size * heads * sequence_length * sequence_length;
}

}  // namespace

std::size_t dense_attention_forward_workspace_elements(
    int batch_size,
    int sequence_length,
    int heads,
    int head_size) {
    const std::size_t activations =
        static_cast<std::size_t>(batch_size) * sequence_length * heads *
        head_size;
    const std::size_t scores =
        static_cast<std::size_t>(batch_size) * heads * sequence_length *
        sequence_length;
    return 3 * activations + std::max(activations, scores);
}

std::size_t dense_attention_backward_workspace_elements(
    int batch_size,
    int sequence_length,
    int heads,
    int head_size) {
    const std::size_t activations =
        static_cast<std::size_t>(batch_size) * sequence_length * heads *
        head_size;
    const std::size_t scores =
        static_cast<std::size_t>(batch_size) * heads * sequence_length *
        sequence_length;
    return 7 * activations + 2 * scores;
}

void dense_attention_forward_cuda(
    float* output,
    float* probabilities,
    const float* query,
    const float* key,
    const float* value,
    float* workspace,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream) {
    const int activations =
        activation_elements(batch_size, sequence_length, heads, head_size);
    const int scores = score_elements(batch_size, sequence_length, heads);
    const int matrices = batch_size * heads;
    const int matrix_activations = sequence_length * head_size;
    const int matrix_scores = sequence_length * sequence_length;

    float* packed_query = workspace;
    float* packed_key = packed_query + activations;
    float* packed_value = packed_key + activations;
    float* scratch = packed_value + activations;

    const int vectors_per_head = head_size / 4;
    const int total_vectors = activations / 4;
    const int blocks = (total_vectors + kThreads - 1) / kThreads;
    attention_pack_qkv_kernel<<<blocks, kThreads, 0, stream>>>(
        packed_query,
        packed_key,
        packed_value,
        query,
        key,
        value,
        sequence_length,
        heads,
        vectors_per_head,
        total_vectors);
    CUDA_CHECK(cudaGetLastError());

    matmul_fp32_strided_batched_cuda(
        scratch,
        packed_query,
        packed_key,
        sequence_length,
        sequence_length,
        head_size,
        matrices,
        matrix_activations,
        matrix_activations,
        matrix_scores,
        false,
        true,
        false,
        stream);
    causal_softmax_forward_cuda(
        probabilities,
        scratch,
        batch_size,
        heads,
        sequence_length,
        scale,
        stream);

    matmul_fp32_strided_batched_cuda(
        scratch,
        probabilities,
        packed_value,
        sequence_length,
        head_size,
        sequence_length,
        matrices,
        matrix_scores,
        matrix_activations,
        matrix_activations,
        false,
        false,
        false,
        stream);
    attention_unpack_output_kernel<<<blocks, kThreads, 0, stream>>>(
        output,
        scratch,
        sequence_length,
        heads,
        vectors_per_head,
        total_vectors);
    CUDA_CHECK(cudaGetLastError());
}

void dense_attention_backward_cuda(
    float* query_gradient,
    float* key_gradient,
    float* value_gradient,
    const float* output_gradient,
    const float* probabilities,
    const float* query,
    const float* key,
    const float* value,
    float* workspace,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream) {
    const int activations =
        activation_elements(batch_size, sequence_length, heads, head_size);
    const int scores = score_elements(batch_size, sequence_length, heads);
    const int matrices = batch_size * heads;
    const int matrix_activations = sequence_length * head_size;
    const int matrix_scores = sequence_length * sequence_length;

    float* packed_query = workspace;
    float* packed_key = packed_query + activations;
    float* packed_value = packed_key + activations;
    float* packed_output_gradient = packed_value + activations;
    float* packed_query_gradient = packed_output_gradient + activations;
    float* packed_key_gradient = packed_query_gradient + activations;
    float* packed_value_gradient = packed_key_gradient + activations;
    float* probability_gradient = packed_value_gradient + activations;
    float* score_gradient = probability_gradient + scores;

    const int vectors_per_head = head_size / 4;
    const int total_vectors = activations / 4;
    const int blocks = (total_vectors + kThreads - 1) / kThreads;
    attention_pack_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        packed_query,
        packed_key,
        packed_value,
        packed_output_gradient,
        query,
        key,
        value,
        output_gradient,
        sequence_length,
        heads,
        vectors_per_head,
        total_vectors);
    CUDA_CHECK(cudaGetLastError());

    matmul_fp32_strided_batched_cuda(
        probability_gradient,
        packed_output_gradient,
        packed_value,
        sequence_length,
        sequence_length,
        head_size,
        matrices,
        matrix_activations,
        matrix_activations,
        matrix_scores,
        false,
        true,
        false,
        stream);
    matmul_fp32_strided_batched_cuda(
        packed_value_gradient,
        probabilities,
        packed_output_gradient,
        sequence_length,
        head_size,
        sequence_length,
        matrices,
        matrix_scores,
        matrix_activations,
        matrix_activations,
        true,
        false,
        false,
        stream);

    CUDA_CHECK(cudaMemsetAsync(
        score_gradient,
        0,
        static_cast<std::size_t>(scores) * sizeof(float),
        stream));
    causal_softmax_backward_cuda(
        score_gradient,
        probability_gradient,
        probabilities,
        batch_size,
        heads,
        sequence_length,
        scale,
        stream);

    matmul_fp32_strided_batched_cuda(
        packed_query_gradient,
        score_gradient,
        packed_key,
        sequence_length,
        head_size,
        sequence_length,
        matrices,
        matrix_scores,
        matrix_activations,
        matrix_activations,
        false,
        false,
        false,
        stream);
    matmul_fp32_strided_batched_cuda(
        packed_key_gradient,
        score_gradient,
        packed_query,
        sequence_length,
        head_size,
        sequence_length,
        matrices,
        matrix_scores,
        matrix_activations,
        matrix_activations,
        true,
        false,
        false,
        stream);

    attention_unpack_gradients_kernel<<<blocks, kThreads, 0, stream>>>(
        query_gradient,
        key_gradient,
        value_gradient,
        packed_query_gradient,
        packed_key_gradient,
        packed_value_gradient,
        sequence_length,
        heads,
        vectors_per_head,
        total_vectors);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

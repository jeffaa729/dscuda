// Implements one sequential DeepSeek-V3 MTP module by joining the preceding hidden state with a shifted future-token embedding before a complete transformer block.
// Custom gather, concatenate, split, and scatter kernels preserve the causal depth chain while the reverse path accumulates into shared embeddings and the previous MTP depth.

#include "mtp.h"

#include "cuda_common.h"
#include "matmul.h"
#include "rmsnorm.h"

namespace dscuda {
namespace {

constexpr int BLOCK_SIZE = 256;

std::size_t align_four(std::size_t value) {
    return (value + 3) & ~std::size_t{3};
}

int valid_sequence_length(const MtpConfig& config) {
    return config.sequence_length - config.depth;
}

template <typename T>
struct ActivationLayout {
    T* hidden;
    T* embedding;
    T* hidden_norm;
    T* hidden_inverse_rms;
    T* embedding_norm;
    T* embedding_inverse_rms;
    T* concatenated;
    T* projected;
    T* block;
    std::size_t elements;
};

template <typename T>
ActivationLayout<T> make_activation_layout(
    T* buffer,
    const MtpConfig& config) {
    const std::size_t rows = static_cast<std::size_t>(config.batch_size)
        * valid_sequence_length(config);
    const std::size_t hidden = rows * config.hidden_size;
    std::size_t offset = 0;
    auto take = [&](std::size_t elements) {
        offset = align_four(offset);
        T* result = buffer == nullptr ? nullptr : buffer + offset;
        offset += elements;
        return result;
    };
    ActivationLayout<T> layout;
    layout.hidden = take(hidden);
    layout.embedding = take(hidden);
    layout.hidden_norm = take(hidden);
    layout.hidden_inverse_rms = take(align_four(rows));
    layout.embedding_norm = take(hidden);
    layout.embedding_inverse_rms = take(align_four(rows));
    layout.concatenated = take(2 * hidden);
    layout.projected = take(hidden);
    layout.block = take(deepseek_v3_block_activation_elements(config.block));
    layout.elements = align_four(offset);
    return layout;
}

template <typename T>
struct BackwardLayout {
    T* projected_gradient;
    T* concatenated_gradient;
    T* hidden_norm_gradient;
    T* embedding_norm_gradient;
    T* hidden_gradient;
    T* embedding_gradient;
    T* block;
    std::size_t elements;
};

template <typename T>
BackwardLayout<T> make_backward_layout(
    T* buffer,
    const MtpConfig& config) {
    const std::size_t rows = static_cast<std::size_t>(config.batch_size)
        * valid_sequence_length(config);
    const std::size_t hidden = rows * config.hidden_size;
    std::size_t offset = 0;
    auto take = [&](std::size_t elements) {
        offset = align_four(offset);
        T* result = buffer == nullptr ? nullptr : buffer + offset;
        offset += elements;
        return result;
    };
    BackwardLayout<T> layout;
    layout.projected_gradient = take(hidden);
    layout.concatenated_gradient = take(2 * hidden);
    layout.hidden_norm_gradient = take(hidden);
    layout.embedding_norm_gradient = take(hidden);
    layout.hidden_gradient = take(hidden);
    layout.embedding_gradient = take(hidden);
    layout.block = take(
        deepseek_v3_block_backward_workspace_elements(config.block));
    layout.elements = align_four(offset);
    return layout;
}

__global__ void gather_inputs_kernel(
    float* hidden,
    float* embedding,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    int sequence_length,
    int valid_length,
    int hidden_size,
    int depth,
    int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = index / hidden_size;
    const int column = index % hidden_size;
    const int batch = row / valid_length;
    const int position = row % valid_length;
    const int previous_length = valid_length + 1;
    hidden[index] = previous_hidden[
        (batch * previous_length + position) * hidden_size + column];
    const int token = tokens[batch * sequence_length + position + depth];
    embedding[index] = embedding_table[token * hidden_size + column];
}

__global__ void concatenate_kernel(
    float* output,
    const float* hidden,
    const float* embedding,
    int rows,
    int hidden_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * 2 * hidden_size;
    if (index >= elements) {
        return;
    }
    const int row = index / (2 * hidden_size);
    const int column = index % (2 * hidden_size);
    output[index] = column < hidden_size
        ? hidden[row * hidden_size + column]
        : embedding[row * hidden_size + column - hidden_size];
}

__global__ void split_kernel(
    float* hidden_gradient,
    float* embedding_gradient,
    const float* concatenated_gradient,
    int rows,
    int hidden_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * hidden_size;
    if (index >= elements) {
        return;
    }
    const int row = index / hidden_size;
    const int column = index % hidden_size;
    hidden_gradient[index] =
        concatenated_gradient[row * 2 * hidden_size + column];
    embedding_gradient[index] = concatenated_gradient[
        row * 2 * hidden_size + hidden_size + column];
}

__global__ void scatter_gradients_kernel(
    float* previous_hidden_gradient,
    float* embedding_table_gradient,
    const float* hidden_gradient,
    const float* embedding_gradient,
    const int* tokens,
    int sequence_length,
    int valid_length,
    int hidden_size,
    int depth,
    int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = index / hidden_size;
    const int column = index % hidden_size;
    const int batch = row / valid_length;
    const int position = row % valid_length;
    const int previous_length = valid_length + 1;
    previous_hidden_gradient[
        (batch * previous_length + position) * hidden_size + column]
        += hidden_gradient[index];
    const int token = tokens[batch * sequence_length + position + depth];
    atomicAdd(
        &embedding_table_gradient[token * hidden_size + column],
        embedding_gradient[index]);
}

__global__ void gather_targets_kernel(
    int* shifted_targets,
    const int* targets,
    int sequence_length,
    int valid_length,
    int depth,
    int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int batch = row / valid_length;
    const int position = row % valid_length;
    shifted_targets[row] =
        targets[batch * sequence_length + position + depth];
}

__global__ void add_scaled_loss_kernel(
    float* total_loss,
    const float* module_loss,
    float scale) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *total_loss += scale * *module_loss;
    }
}

__global__ void scale_gradients_kernel(
    float* gradients,
    int elements,
    float scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        gradients[index] *= scale;
    }
}

}  // namespace

std::size_t mtp_activation_elements(const MtpConfig& config) {
    return make_activation_layout(static_cast<float*>(nullptr), config).elements;
}

std::size_t mtp_integer_activation_elements(const MtpConfig& config) {
    return deepseek_v3_block_integer_activation_elements(config.block);
}

std::size_t mtp_backward_workspace_elements(const MtpConfig& config) {
    return make_backward_layout(static_cast<float*>(nullptr), config).elements;
}

std::size_t mtp_bf16_workspace_elements(const MtpConfig& config) {
    return deepseek_v3_block_bf16_workspace_elements(config.block);
}

void mtp_forward_cuda(
    float* output,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    const MtpParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    int* integer_activations,
    __nv_bfloat16* bf16_workspace,
    const MtpConfig& config,
    cudaStream_t stream) {
    const int valid_length = valid_sequence_length(config);
    const int rows = config.batch_size * valid_length;
    const int elements = rows * config.hidden_size;
    auto saved = make_activation_layout(activations, config);
    const dim3 gather_grid(
        (elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        1);
    gather_inputs_kernel<<<gather_grid, BLOCK_SIZE, 0, stream>>>(
        saved.hidden,
        saved.embedding,
        previous_hidden,
        tokens,
        embedding_table,
        config.sequence_length,
        valid_length,
        config.hidden_size,
        config.depth,
        elements);
    CUDA_CHECK(cudaGetLastError());
    rmsnorm_forward_cuda(
        saved.hidden_norm,
        saved.hidden_inverse_rms,
        saved.hidden,
        parameters.hidden_norm_weight,
        rows,
        config.hidden_size,
        config.epsilon,
        stream);
    rmsnorm_forward_cuda(
        saved.embedding_norm,
        saved.embedding_inverse_rms,
        saved.embedding,
        parameters.embedding_norm_weight,
        rows,
        config.hidden_size,
        config.epsilon,
        stream);
    concatenate_kernel<<<
        (2 * elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
            saved.concatenated,
            saved.hidden_norm,
            saved.embedding_norm,
            rows,
            config.hidden_size);
    CUDA_CHECK(cudaGetLastError());
    matmul_fp32_forward_cuda(
        saved.projected,
        saved.concatenated,
        parameters.projection_weight,
        rows,
        config.hidden_size,
        2 * config.hidden_size,
        stream);
    deepseek_v3_block_forward_cuda(
        output,
        saved.projected,
        parameters.block,
        cosine,
        sine,
        saved.block,
        integer_activations,
        bf16_workspace,
        config.block,
        stream);
}

void mtp_backward_cuda(
    float* previous_hidden_gradient,
    float* embedding_gradient,
    const MtpGradients& parameter_gradients,
    const float* output_gradient,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    const MtpParameters& parameters,
    const float* cosine,
    const float* sine,
    const float* activations,
    const int* integer_activations,
    float* workspace,
    __nv_bfloat16* bf16_workspace,
    const MtpConfig& config,
    cudaStream_t stream) {
    (void) previous_hidden;
    (void) embedding_table;
    const int valid_length = valid_sequence_length(config);
    const int rows = config.batch_size * valid_length;
    const int elements = rows * config.hidden_size;
    const auto saved = make_activation_layout(activations, config);
    auto gradient = make_backward_layout(workspace, config);
    CUDA_CHECK(cudaMemsetAsync(
        workspace,
        0,
        mtp_backward_workspace_elements(config) * sizeof(float),
        stream));
    deepseek_v3_block_backward_cuda(
        gradient.projected_gradient,
        parameter_gradients.block,
        output_gradient,
        saved.projected,
        parameters.block,
        cosine,
        sine,
        saved.block,
        integer_activations,
        gradient.block,
        bf16_workspace,
        config.block,
        stream);
    matmul_fp32_backward_cuda(
        gradient.concatenated_gradient,
        parameter_gradients.projection_weight,
        gradient.projected_gradient,
        saved.concatenated,
        parameters.projection_weight,
        rows,
        config.hidden_size,
        2 * config.hidden_size,
        stream);
    split_kernel<<<
        (elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
            gradient.hidden_norm_gradient,
            gradient.embedding_norm_gradient,
            gradient.concatenated_gradient,
            rows,
            config.hidden_size);
    CUDA_CHECK(cudaGetLastError());
    rmsnorm_backward_cuda(
        gradient.hidden_gradient,
        parameter_gradients.hidden_norm_weight,
        gradient.hidden_norm_gradient,
        saved.hidden,
        parameters.hidden_norm_weight,
        saved.hidden_inverse_rms,
        rows,
        config.hidden_size,
        stream);
    rmsnorm_backward_cuda(
        gradient.embedding_gradient,
        parameter_gradients.embedding_norm_weight,
        gradient.embedding_norm_gradient,
        saved.embedding,
        parameters.embedding_norm_weight,
        saved.embedding_inverse_rms,
        rows,
        config.hidden_size,
        stream);
    scatter_gradients_kernel<<<
        (elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
            previous_hidden_gradient,
            embedding_gradient,
            gradient.hidden_gradient,
            gradient.embedding_gradient,
            tokens,
            config.sequence_length,
            valid_length,
            config.hidden_size,
            config.depth,
            elements);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_add_balance_loss_cuda(
    float* total_loss,
    const float* activations,
    const MtpConfig& config,
    cudaStream_t stream) {
    const auto saved = make_activation_layout(activations, config);
    deepseek_v3_block_add_balance_loss_cuda(
        total_loss, saved.block, config.block, stream);
}

void mtp_gather_targets_cuda(
    int* shifted_targets,
    const int* next_token_targets,
    const MtpConfig& config,
    cudaStream_t stream) {
    const int valid_length = valid_sequence_length(config);
    const int rows = config.batch_size * valid_length;
    gather_targets_kernel<<<
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
            shifted_targets,
            next_token_targets,
            config.sequence_length,
            valid_length,
            config.depth,
            rows);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_add_scaled_loss_cuda(
    float* total_loss,
    const float* module_loss,
    float scale,
    cudaStream_t stream) {
    add_scaled_loss_kernel<<<1, 1, 0, stream>>>(
        total_loss, module_loss, scale);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_scale_gradients_cuda(
    float* gradients,
    int elements,
    float scale,
    cudaStream_t stream) {
    scale_gradients_kernel<<<
        (elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(gradients, elements, scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

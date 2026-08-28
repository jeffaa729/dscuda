// Provides a scalar oracle for the sequential MTP fusion stem, complete V3 transformer block, and gradient scatter into shared embeddings.
// Forward intermediates are recomputed during backward so every CUDA result can be checked without mirroring device workspace layouts.

#include "mtp_cpu.h"

#include "deepseek_v3_block_cpu.h"
#include "matmul_cpu.h"
#include "rmsnorm_cpu.h"

#include <vector>

namespace dscuda {
namespace {

struct SavedMtp {
    std::vector<float> hidden;
    std::vector<float> embedding;
    std::vector<float> hidden_norm;
    std::vector<float> hidden_inverse_rms;
    std::vector<float> embedding_norm;
    std::vector<float> embedding_inverse_rms;
    std::vector<float> concatenated;
    std::vector<float> projected;
};

SavedMtp forward_saved(
    float* output,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    const MtpParameters& parameters,
    const float* cosine,
    const float* sine,
    const MtpConfig& config) {
    const int valid_length = config.sequence_length - config.depth;
    const int previous_length = valid_length + 1;
    const int rows = config.batch_size * valid_length;
    const int elements = rows * config.hidden_size;
    SavedMtp saved;
    saved.hidden.resize(elements);
    saved.embedding.resize(elements);
    saved.hidden_norm.resize(elements);
    saved.hidden_inverse_rms.resize(rows);
    saved.embedding_norm.resize(elements);
    saved.embedding_inverse_rms.resize(rows);
    saved.concatenated.resize(2 * elements);
    saved.projected.resize(elements);
    for (int batch = 0; batch < config.batch_size; ++batch) {
        for (int position = 0; position < valid_length; ++position) {
            const int row = batch * valid_length + position;
            const int previous_row = batch * previous_length + position;
            const int token =
                tokens[batch * config.sequence_length + position + config.depth];
            for (int column = 0; column < config.hidden_size; ++column) {
                saved.hidden[row * config.hidden_size + column] =
                    previous_hidden[previous_row * config.hidden_size + column];
                saved.embedding[row * config.hidden_size + column] =
                    embedding_table[token * config.hidden_size + column];
            }
        }
    }
    rmsnorm_forward_cpu(
        saved.hidden_norm.data(),
        saved.hidden_inverse_rms.data(),
        saved.hidden.data(),
        parameters.hidden_norm_weight,
        rows,
        config.hidden_size,
        config.epsilon);
    rmsnorm_forward_cpu(
        saved.embedding_norm.data(),
        saved.embedding_inverse_rms.data(),
        saved.embedding.data(),
        parameters.embedding_norm_weight,
        rows,
        config.hidden_size,
        config.epsilon);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < config.hidden_size; ++column) {
            saved.concatenated[row * 2 * config.hidden_size + column] =
                saved.hidden_norm[row * config.hidden_size + column];
            saved.concatenated[
                row * 2 * config.hidden_size + config.hidden_size + column] =
                saved.embedding_norm[row * config.hidden_size + column];
        }
    }
    matmul_forward_cpu(
        saved.projected.data(),
        saved.concatenated.data(),
        parameters.projection_weight,
        rows,
        config.hidden_size,
        2 * config.hidden_size);
    deepseek_v3_block_forward_cpu(
        output,
        saved.projected.data(),
        parameters.block,
        cosine,
        sine,
        config.block);
    return saved;
}

}  // namespace

void mtp_forward_cpu(
    float* output,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    const MtpParameters& parameters,
    const float* cosine,
    const float* sine,
    const MtpConfig& config) {
    forward_saved(
        output,
        previous_hidden,
        tokens,
        embedding_table,
        parameters,
        cosine,
        sine,
        config);
}

void mtp_backward_cpu(
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
    const MtpConfig& config) {
    const int valid_length = config.sequence_length - config.depth;
    const int previous_length = valid_length + 1;
    const int rows = config.batch_size * valid_length;
    const int elements = rows * config.hidden_size;
    std::vector<float> discarded(elements);
    const SavedMtp saved = forward_saved(
        discarded.data(),
        previous_hidden,
        tokens,
        embedding_table,
        parameters,
        cosine,
        sine,
        config);
    std::vector<float> projected_gradient(elements, 0.0F);
    deepseek_v3_block_backward_cpu(
        projected_gradient.data(),
        parameter_gradients.block,
        output_gradient,
        saved.projected.data(),
        parameters.block,
        cosine,
        sine,
        config.block);
    std::vector<float> concatenated_gradient(2 * elements, 0.0F);
    matmul_backward_cpu(
        concatenated_gradient.data(),
        parameter_gradients.projection_weight,
        projected_gradient.data(),
        saved.concatenated.data(),
        parameters.projection_weight,
        rows,
        config.hidden_size,
        2 * config.hidden_size);
    std::vector<float> hidden_norm_gradient(elements);
    std::vector<float> embedding_norm_gradient(elements);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < config.hidden_size; ++column) {
            hidden_norm_gradient[row * config.hidden_size + column] =
                concatenated_gradient[row * 2 * config.hidden_size + column];
            embedding_norm_gradient[row * config.hidden_size + column] =
                concatenated_gradient[
                    row * 2 * config.hidden_size
                    + config.hidden_size + column];
        }
    }
    std::vector<float> hidden_gradient(elements, 0.0F);
    std::vector<float> future_embedding_gradient(elements, 0.0F);
    rmsnorm_backward_cpu(
        hidden_gradient.data(),
        parameter_gradients.hidden_norm_weight,
        hidden_norm_gradient.data(),
        saved.hidden.data(),
        parameters.hidden_norm_weight,
        saved.hidden_inverse_rms.data(),
        rows,
        config.hidden_size);
    rmsnorm_backward_cpu(
        future_embedding_gradient.data(),
        parameter_gradients.embedding_norm_weight,
        embedding_norm_gradient.data(),
        saved.embedding.data(),
        parameters.embedding_norm_weight,
        saved.embedding_inverse_rms.data(),
        rows,
        config.hidden_size);
    for (int batch = 0; batch < config.batch_size; ++batch) {
        for (int position = 0; position < valid_length; ++position) {
            const int row = batch * valid_length + position;
            const int previous_row = batch * previous_length + position;
            const int token =
                tokens[batch * config.sequence_length + position + config.depth];
            for (int column = 0; column < config.hidden_size; ++column) {
                previous_hidden_gradient[
                    previous_row * config.hidden_size + column]
                    += hidden_gradient[row * config.hidden_size + column];
                embedding_gradient[token * config.hidden_size + column]
                    += future_embedding_gradient[
                        row * config.hidden_size + column];
            }
        }
    }
}

}  // namespace dscuda

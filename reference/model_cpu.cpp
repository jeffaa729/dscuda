// Recomputes a complete dense GPT graph with scalar CPU operators as the end-to-end correctness oracle.
// The tied embedding table receives gradients from both the vocabulary projection and the initial token lookup.

#include "model_cpu.h"

#include "cross_entropy_cpu.h"
#include "embedding_cpu.h"
#include "rmsnorm_cpu.h"
#include "transformer_block_cpu.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dscuda {
namespace {

TransformerBlockConfig block_config(const ModelConfig& config) {
    const int head_size = config.hidden_size / config.heads;
    return {
        config.batch_size,
        config.sequence_length,
        config.hidden_size,
        config.heads,
        head_size,
        config.ffn_size,
        config.rotary_size,
        config.rms_epsilon,
        1.0F / std::sqrt(static_cast<float>(head_size)),
    };
}

void tied_head_forward(
    float* logits,
    const float* hidden,
    const float* embedding,
    int rows,
    int vocabulary_size,
    int hidden_size) {
    for (int row = 0; row < rows; ++row) {
        for (int token = 0; token < vocabulary_size; ++token) {
            float value = 0.0F;
            for (int column = 0; column < hidden_size; ++column) {
                value += hidden[row * hidden_size + column]
                    * embedding[token * hidden_size + column];
            }
            logits[row * vocabulary_size + token] = value;
        }
    }
}

void tied_head_backward(
    float* hidden_gradient,
    float* embedding_gradient,
    const float* logits_gradient,
    const float* hidden,
    const float* embedding,
    int rows,
    int vocabulary_size,
    int hidden_size) {
    for (int row = 0; row < rows; ++row) {
        for (int token = 0; token < vocabulary_size; ++token) {
            const float gradient =
                logits_gradient[row * vocabulary_size + token];
            for (int column = 0; column < hidden_size; ++column) {
                hidden_gradient[row * hidden_size + column] +=
                    gradient * embedding[token * hidden_size + column];
                embedding_gradient[token * hidden_size + column] +=
                    gradient * hidden[row * hidden_size + column];
            }
        }
    }
}

}  // namespace

float dense_gpt_forward_cpu(
    DenseGptCpuCache& cache,
    const int* input_tokens,
    const int* target_tokens,
    const float* parameters,
    const ModelParameterLayout& layout,
    const std::vector<float>& cosine,
    const std::vector<float>& sine,
    const ModelConfig& config) {
    const int rows = config.batch_size * config.sequence_length;
    const int activation_elements = rows * config.hidden_size;
    const TransformerBlockConfig transformer_config = block_config(config);

    cache.hidden_states.assign(
        static_cast<std::size_t>(config.layers + 1) * activation_elements,
        0.0F);
    cache.final_norm.resize(activation_elements);
    cache.final_inverse_rms.resize(rows);
    cache.logits.resize(
        static_cast<std::size_t>(rows) * config.vocabulary_size);
    cache.logsumexp.resize(rows);

    embedding_forward_cpu(
        cache.hidden_states.data(),
        input_tokens,
        parameters + layout.token_embedding,
        rows,
        config.hidden_size);
    for (int layer = 0; layer < config.layers; ++layer) {
        const float* input = cache.hidden_states.data()
            + static_cast<std::size_t>(layer) * activation_elements;
        float* output = cache.hidden_states.data()
            + static_cast<std::size_t>(layer + 1) * activation_elements;
        transformer_block_forward_cpu(
            output,
            input,
            model_block_parameters(parameters, layout, layer),
            cosine.data(),
            sine.data(),
            transformer_config);
    }

    const float* final_input = cache.hidden_states.data()
        + static_cast<std::size_t>(config.layers) * activation_elements;
    rmsnorm_forward_cpu(
        cache.final_norm.data(),
        cache.final_inverse_rms.data(),
        final_input,
        parameters + layout.final_norm,
        rows,
        config.hidden_size,
        config.rms_epsilon);
    tied_head_forward(
        cache.logits.data(),
        cache.final_norm.data(),
        parameters + layout.token_embedding,
        rows,
        config.vocabulary_size,
        config.hidden_size);
    cross_entropy_forward_cpu(
        &cache.loss,
        cache.logsumexp.data(),
        cache.logits.data(),
        target_tokens,
        rows,
        config.vocabulary_size);
    return cache.loss;
}

void dense_gpt_backward_cpu(
    float* gradients,
    const DenseGptCpuCache& cache,
    const int* input_tokens,
    const int* target_tokens,
    const float* parameters,
    const ModelParameterLayout& layout,
    const std::vector<float>& cosine,
    const std::vector<float>& sine,
    const ModelConfig& config) {
    const int rows = config.batch_size * config.sequence_length;
    const int activation_elements = rows * config.hidden_size;
    const TransformerBlockConfig transformer_config = block_config(config);
    std::vector<float> logits_gradient(
        static_cast<std::size_t>(rows) * config.vocabulary_size);
    cross_entropy_backward_cpu(
        logits_gradient.data(),
        cache.logits.data(),
        cache.logsumexp.data(),
        target_tokens,
        rows,
        config.vocabulary_size);

    std::vector<float> final_norm_gradient(activation_elements, 0.0F);
    tied_head_backward(
        final_norm_gradient.data(),
        gradients + layout.token_embedding,
        logits_gradient.data(),
        cache.final_norm.data(),
        parameters + layout.token_embedding,
        rows,
        config.vocabulary_size,
        config.hidden_size);

    std::vector<float> output_gradient(activation_elements, 0.0F);
    const float* final_input = cache.hidden_states.data()
        + static_cast<std::size_t>(config.layers) * activation_elements;
    rmsnorm_backward_cpu(
        output_gradient.data(),
        gradients + layout.final_norm,
        final_norm_gradient.data(),
        final_input,
        parameters + layout.final_norm,
        cache.final_inverse_rms.data(),
        rows,
        config.hidden_size);

    for (int layer = config.layers - 1; layer >= 0; --layer) {
        std::vector<float> input_gradient(activation_elements, 0.0F);
        const float* input = cache.hidden_states.data()
            + static_cast<std::size_t>(layer) * activation_elements;
        transformer_block_backward_cpu(
            input_gradient.data(),
            model_block_gradients(gradients, layout, layer),
            output_gradient.data(),
            input,
            model_block_parameters(parameters, layout, layer),
            cosine.data(),
            sine.data(),
            transformer_config);
        output_gradient.swap(input_gradient);
    }

    embedding_backward_cpu(
        gradients + layout.token_embedding,
        output_gradient.data(),
        input_tokens,
        rows,
        config.hidden_size);
}

}  // namespace dscuda

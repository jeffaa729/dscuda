// Implements the absorbed-query MLA equations with explicit scalar loops and a materialized probability row.
// It is the correctness oracle for the fused BF16 CUDA path, not a competing runtime implementation.

#include "mla_cpu.h"

#include "matmul_cpu.h"
#include "rmsnorm_cpu.h"
#include "rope_cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>

namespace dscuda {
namespace {

struct CpuMlaActivations {
    std::vector<float> query_compressed;
    std::vector<float> query_inverse_rms;
    std::vector<float> query_normalized;
    std::vector<float> query_full;
    std::vector<float> query_nope_head_major;
    std::vector<float> query_rope;
    std::vector<float> query_absorbed_head_major;
    std::vector<float> query_absorbed;
    std::vector<float> kv_full;
    std::vector<float> kv_latent_raw;
    std::vector<float> kv_inverse_rms;
    std::vector<float> kv_latent;
    std::vector<float> key_rope;
    std::vector<float> attention_output;
    std::vector<float> logsumexp;
    std::vector<float> attention_output_head_major;
    std::vector<float> value_output_head_major;
    std::vector<float> value_output;
};

float round_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    bits &= 0xffff0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void round_vector_to_bf16(std::vector<float>& values) {
    for (float& value : values) {
        value = round_to_bf16(value);
    }
}

CpuMlaActivations make_cpu_activations(const MlaLayerConfig& config) {
    const int rows = config.batch_size * config.sequence_length;
    const int query_width =
        config.heads * (config.nope_size + config.rope_size);
    CpuMlaActivations saved;
    saved.query_compressed.resize(rows * config.query_rank);
    saved.query_inverse_rms.resize(rows);
    saved.query_normalized.resize(rows * config.query_rank);
    saved.query_full.resize(rows * query_width);
    saved.query_nope_head_major.resize(rows * config.heads * config.nope_size);
    saved.query_rope.resize(rows * config.heads * config.rope_size);
    saved.query_absorbed_head_major.resize(rows * config.heads * config.kv_rank);
    saved.query_absorbed.resize(rows * config.heads * config.kv_rank);
    saved.kv_full.resize(rows * (config.kv_rank + config.rope_size));
    saved.kv_latent_raw.resize(rows * config.kv_rank);
    saved.kv_inverse_rms.resize(rows);
    saved.kv_latent.resize(rows * config.kv_rank);
    saved.key_rope.resize(rows * config.rope_size);
    saved.attention_output.resize(rows * config.heads * config.kv_rank);
    saved.logsumexp.resize(rows * config.heads);
    saved.attention_output_head_major.resize(
        rows * config.heads * config.kv_rank);
    saved.value_output_head_major.resize(
        rows * config.heads * config.value_size);
    saved.value_output.resize(rows * config.heads * config.value_size);
    return saved;
}

void row_head_to_head_row_cpu(
    float* output,
    const float* input,
    int rows,
    int heads,
    int width) {
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            for (int column = 0; column < width; ++column) {
                output[(head * rows + row) * width + column] =
                    input[(row * heads + head) * width + column];
            }
        }
    }
}

void head_row_to_row_head_cpu(
    float* output,
    const float* input,
    int rows,
    int heads,
    int width) {
    for (int head = 0; head < heads; ++head) {
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < width; ++column) {
                output[(row * heads + head) * width + column] =
                    input[(head * rows + row) * width + column];
            }
        }
    }
}

CpuMlaActivations mla_layer_forward_saved_cpu(
    float* output,
    const float* input,
    const MlaLayerParameters& parameters,
    const float* cosine,
    const float* sine,
    const MlaLayerConfig& config) {
    const int rows = config.batch_size * config.sequence_length;
    const int query_head_size = config.nope_size + config.rope_size;
    const int query_width = config.heads * query_head_size;
    CpuMlaActivations saved = make_cpu_activations(config);

    matmul_forward_cpu(
        saved.query_compressed.data(), input, parameters.query_down_weight,
        rows, config.query_rank, config.hidden_size);
    rmsnorm_forward_cpu(
        saved.query_normalized.data(), saved.query_inverse_rms.data(),
        saved.query_compressed.data(), parameters.query_norm_weight, rows,
        config.query_rank, config.epsilon);
    matmul_forward_cpu(
        saved.query_full.data(), saved.query_normalized.data(),
        parameters.query_up_weight, rows, query_width, config.query_rank);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < config.heads; ++head) {
            const int source = (row * config.heads + head) * query_head_size;
            const int nope =
                (head * rows + row) * config.nope_size;
            const int rope =
                (row * config.heads + head) * config.rope_size;
            std::copy_n(
                saved.query_full.data() + source,
                config.nope_size,
                saved.query_nope_head_major.data() + nope);
            std::copy_n(
                saved.query_full.data() + source + config.nope_size,
                config.rope_size,
                saved.query_rope.data() + rope);
        }
    }
    rope_forward_cpu(
        saved.query_rope.data(), saved.query_rope.data(), cosine, sine,
        config.batch_size, config.sequence_length, config.heads,
        config.rope_size, config.rope_size);

    for (int head = 0; head < config.heads; ++head) {
        matmul_forward_cpu(
            saved.query_absorbed_head_major.data()
                + head * rows * config.kv_rank,
            saved.query_nope_head_major.data()
                + head * rows * config.nope_size,
            parameters.key_up_weight
                + head * config.nope_size * config.kv_rank,
            rows,
            config.kv_rank,
            config.nope_size);
    }
    head_row_to_row_head_cpu(
        saved.query_absorbed.data(),
        saved.query_absorbed_head_major.data(),
        rows,
        config.heads,
        config.kv_rank);

    matmul_forward_cpu(
        saved.kv_full.data(), input, parameters.kv_down_weight, rows,
        config.kv_rank + config.rope_size, config.hidden_size);
    for (int row = 0; row < rows; ++row) {
        std::copy_n(
            saved.kv_full.data() + row * (config.kv_rank + config.rope_size),
            config.kv_rank,
            saved.kv_latent_raw.data() + row * config.kv_rank);
        std::copy_n(
            saved.kv_full.data()
                + row * (config.kv_rank + config.rope_size) + config.kv_rank,
            config.rope_size,
            saved.key_rope.data() + row * config.rope_size);
    }
    rmsnorm_forward_cpu(
        saved.kv_latent.data(), saved.kv_inverse_rms.data(),
        saved.kv_latent_raw.data(), parameters.kv_norm_weight, rows,
        config.kv_rank, config.epsilon);
    rope_forward_cpu(
        saved.key_rope.data(), saved.key_rope.data(), cosine, sine,
        config.batch_size, config.sequence_length, 1, config.rope_size,
        config.rope_size);
    round_vector_to_bf16(saved.query_absorbed);
    round_vector_to_bf16(saved.query_rope);
    round_vector_to_bf16(saved.kv_latent);
    round_vector_to_bf16(saved.key_rope);
    mla_compressed_attention_forward_cpu(
        saved.attention_output.data(), saved.logsumexp.data(),
        saved.query_absorbed.data(), saved.query_rope.data(),
        saved.kv_latent.data(), saved.key_rope.data(), config.batch_size,
        config.sequence_length, config.heads, config.kv_rank,
        config.rope_size, config.attention_scale);

    row_head_to_head_row_cpu(
        saved.attention_output_head_major.data(),
        saved.attention_output.data(),
        rows,
        config.heads,
        config.kv_rank);
    for (int head = 0; head < config.heads; ++head) {
        matmul_forward_cpu(
            saved.value_output_head_major.data()
                + head * rows * config.value_size,
            saved.attention_output_head_major.data()
                + head * rows * config.kv_rank,
            parameters.value_up_weight
                + head * config.kv_rank * config.value_size,
            rows,
            config.value_size,
            config.kv_rank);
    }
    head_row_to_row_head_cpu(
        saved.value_output.data(), saved.value_output_head_major.data(), rows,
        config.heads, config.value_size);
    matmul_forward_cpu(
        output, saved.value_output.data(), parameters.output_weight, rows,
        config.hidden_size, config.heads * config.value_size);
    return saved;
}

int query_index(
    int batch,
    int token,
    int head,
    int column,
    int sequence_length,
    int heads,
    int width) {
    return ((batch * sequence_length + token) * heads + head) * width + column;
}

int shared_index(
    int batch,
    int token,
    int column,
    int sequence_length,
    int width) {
    return (batch * sequence_length + token) * width + column;
}

int row_index(
    int batch,
    int head,
    int token,
    int heads,
    int sequence_length) {
    return (batch * heads + head) * sequence_length + token;
}

float attention_score(
    const float* query_latent,
    const float* query_rope,
    const float* kv_latent,
    const float* key_rope,
    int batch,
    int query_token,
    int key_token,
    int head,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale) {
    float score = 0.0F;
    for (int column = 0; column < kv_rank; ++column) {
        score += query_latent[query_index(
                     batch,
                     query_token,
                     head,
                     column,
                     sequence_length,
                     heads,
                     kv_rank)] *
                 kv_latent[shared_index(
                     batch,
                     key_token,
                     column,
                     sequence_length,
                     kv_rank)];
    }
    for (int column = 0; column < rope_size; ++column) {
        score += query_rope[query_index(
                     batch,
                     query_token,
                     head,
                     column,
                     sequence_length,
                     heads,
                     rope_size)] *
                 key_rope[shared_index(
                     batch,
                     key_token,
                     column,
                     sequence_length,
                     rope_size)];
    }
    return score * scale;
}

}  // namespace

void mla_layer_forward_cpu(
    float* output,
    const float* input,
    const MlaLayerParameters& parameters,
    const float* cosine,
    const float* sine,
    const MlaLayerConfig& config) {
    mla_layer_forward_saved_cpu(
        output, input, parameters, cosine, sine, config);
}

void mla_layer_backward_cpu(
    float* input_gradient,
    const MlaLayerGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const MlaLayerParameters& parameters,
    const float* cosine,
    const float* sine,
    const MlaLayerConfig& config) {
    const int rows = config.batch_size * config.sequence_length;
    const int query_head_size = config.nope_size + config.rope_size;
    const int query_width = config.heads * query_head_size;
    const int attention_elements = rows * config.heads * config.kv_rank;
    const int query_rope_elements = rows * config.heads * config.rope_size;
    const int kv_elements = rows * config.kv_rank;
    const int key_rope_elements = rows * config.rope_size;
    const int value_elements = rows * config.heads * config.value_size;
    std::vector<float> discarded_output(rows * config.hidden_size);
    CpuMlaActivations saved = mla_layer_forward_saved_cpu(
        discarded_output.data(), input, parameters, cosine, sine, config);

    std::vector<float> value_output_gradient(value_elements, 0.0F);
    std::vector<float> value_head_gradient(value_elements, 0.0F);
    std::vector<float> attention_head_gradient(attention_elements, 0.0F);
    std::vector<float> attention_gradient(attention_elements, 0.0F);
    matmul_backward_cpu(
        value_output_gradient.data(),
        parameter_gradients.output_weight,
        output_gradient,
        saved.value_output.data(),
        parameters.output_weight,
        rows,
        config.hidden_size,
        config.heads * config.value_size);
    row_head_to_head_row_cpu(
        value_head_gradient.data(), value_output_gradient.data(), rows,
        config.heads, config.value_size);
    for (int head = 0; head < config.heads; ++head) {
        matmul_backward_cpu(
            attention_head_gradient.data()
                + head * rows * config.kv_rank,
            parameter_gradients.value_up_weight
                + head * config.kv_rank * config.value_size,
            value_head_gradient.data()
                + head * rows * config.value_size,
            saved.attention_output_head_major.data()
                + head * rows * config.kv_rank,
            parameters.value_up_weight
                + head * config.kv_rank * config.value_size,
            rows,
            config.value_size,
            config.kv_rank);
    }
    head_row_to_row_head_cpu(
        attention_gradient.data(), attention_head_gradient.data(), rows,
        config.heads, config.kv_rank);

    std::vector<float> query_absorbed_gradient(attention_elements, 0.0F);
    std::vector<float> query_rope_gradient(query_rope_elements, 0.0F);
    std::vector<float> kv_latent_gradient(kv_elements, 0.0F);
    std::vector<float> key_rope_gradient(key_rope_elements, 0.0F);
    mla_compressed_attention_backward_cpu(
        query_absorbed_gradient.data(),
        query_rope_gradient.data(),
        kv_latent_gradient.data(),
        key_rope_gradient.data(),
        attention_gradient.data(),
        saved.attention_output.data(),
        saved.logsumexp.data(),
        saved.query_absorbed.data(),
        saved.query_rope.data(),
        saved.kv_latent.data(),
        saved.key_rope.data(),
        config.batch_size,
        config.sequence_length,
        config.heads,
        config.kv_rank,
        config.rope_size,
        config.attention_scale);

    std::vector<float> query_absorbed_head_gradient(
        attention_elements, 0.0F);
    std::vector<float> query_nope_head_gradient(
        rows * config.heads * config.nope_size, 0.0F);
    row_head_to_head_row_cpu(
        query_absorbed_head_gradient.data(), query_absorbed_gradient.data(),
        rows, config.heads, config.kv_rank);
    for (int head = 0; head < config.heads; ++head) {
        matmul_backward_cpu(
            query_nope_head_gradient.data()
                + head * rows * config.nope_size,
            parameter_gradients.key_up_weight
                + head * config.nope_size * config.kv_rank,
            query_absorbed_head_gradient.data()
                + head * rows * config.kv_rank,
            saved.query_nope_head_major.data()
                + head * rows * config.nope_size,
            parameters.key_up_weight
                + head * config.nope_size * config.kv_rank,
            rows,
            config.kv_rank,
            config.nope_size);
    }

    std::vector<float> query_rope_raw_gradient(query_rope_elements, 0.0F);
    rope_backward_cpu(
        query_rope_raw_gradient.data(), query_rope_gradient.data(), cosine,
        sine, config.batch_size, config.sequence_length, config.heads,
        config.rope_size, config.rope_size);
    std::vector<float> query_full_gradient(rows * query_width, 0.0F);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < config.heads; ++head) {
            const int destination =
                (row * config.heads + head) * query_head_size;
            const int nope =
                (head * rows + row) * config.nope_size;
            const int rope =
                (row * config.heads + head) * config.rope_size;
            std::copy_n(
                query_nope_head_gradient.data() + nope,
                config.nope_size,
                query_full_gradient.data() + destination);
            std::copy_n(
                query_rope_raw_gradient.data() + rope,
                config.rope_size,
                query_full_gradient.data() + destination + config.nope_size);
        }
    }
    std::vector<float> query_normalized_gradient(
        rows * config.query_rank, 0.0F);
    std::vector<float> query_compressed_gradient(
        rows * config.query_rank, 0.0F);
    matmul_backward_cpu(
        query_normalized_gradient.data(),
        parameter_gradients.query_up_weight,
        query_full_gradient.data(),
        saved.query_normalized.data(),
        parameters.query_up_weight,
        rows,
        query_width,
        config.query_rank);
    rmsnorm_backward_cpu(
        query_compressed_gradient.data(),
        parameter_gradients.query_norm_weight,
        query_normalized_gradient.data(),
        saved.query_compressed.data(),
        parameters.query_norm_weight,
        saved.query_inverse_rms.data(),
        rows,
        config.query_rank);
    matmul_backward_cpu(
        input_gradient,
        parameter_gradients.query_down_weight,
        query_compressed_gradient.data(),
        input,
        parameters.query_down_weight,
        rows,
        config.query_rank,
        config.hidden_size);

    std::vector<float> kv_latent_raw_gradient(kv_elements, 0.0F);
    std::vector<float> key_rope_raw_gradient(key_rope_elements, 0.0F);
    rmsnorm_backward_cpu(
        kv_latent_raw_gradient.data(),
        parameter_gradients.kv_norm_weight,
        kv_latent_gradient.data(),
        saved.kv_latent_raw.data(),
        parameters.kv_norm_weight,
        saved.kv_inverse_rms.data(),
        rows,
        config.kv_rank);
    rope_backward_cpu(
        key_rope_raw_gradient.data(), key_rope_gradient.data(), cosine, sine,
        config.batch_size, config.sequence_length, 1, config.rope_size,
        config.rope_size);
    std::vector<float> kv_full_gradient(
        rows * (config.kv_rank + config.rope_size), 0.0F);
    for (int row = 0; row < rows; ++row) {
        std::copy_n(
            kv_latent_raw_gradient.data() + row * config.kv_rank,
            config.kv_rank,
            kv_full_gradient.data()
                + row * (config.kv_rank + config.rope_size));
        std::copy_n(
            key_rope_raw_gradient.data() + row * config.rope_size,
            config.rope_size,
            kv_full_gradient.data()
                + row * (config.kv_rank + config.rope_size) + config.kv_rank);
    }
    matmul_backward_cpu(
        input_gradient,
        parameter_gradients.kv_down_weight,
        kv_full_gradient.data(),
        input,
        parameters.kv_down_weight,
        rows,
        config.kv_rank + config.rope_size,
        config.hidden_size);
}

void mla_compressed_attention_forward_cpu(
    float* output,
    float* logsumexp,
    const float* query_latent,
    const float* query_rope,
    const float* kv_latent,
    const float* key_rope,
    int batch_size,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale) {
    std::vector<float> probabilities(sequence_length);

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            for (int query_token = 0; query_token < sequence_length; ++query_token) {
                float maximum = -std::numeric_limits<float>::infinity();
                for (int key_token = 0; key_token <= query_token; ++key_token) {
                    const float score = attention_score(
                        query_latent,
                        query_rope,
                        kv_latent,
                        key_rope,
                        batch,
                        query_token,
                        key_token,
                        head,
                        sequence_length,
                        heads,
                        kv_rank,
                        rope_size,
                        scale);
                    probabilities[key_token] = score;
                    maximum = std::fmax(maximum, score);
                }

                float normalizer = 0.0F;
                for (int key_token = 0; key_token <= query_token; ++key_token) {
                    probabilities[key_token] =
                        std::exp(probabilities[key_token] - maximum);
                    normalizer += probabilities[key_token];
                }
                logsumexp[row_index(
                    batch, head, query_token, heads, sequence_length)] =
                    maximum + std::log(normalizer);

                for (int column = 0; column < kv_rank; ++column) {
                    float result = 0.0F;
                    for (int key_token = 0; key_token <= query_token; ++key_token) {
                        result += probabilities[key_token] / normalizer *
                                  kv_latent[shared_index(
                                      batch,
                                      key_token,
                                      column,
                                      sequence_length,
                                      kv_rank)];
                    }
                    output[query_index(
                        batch,
                        query_token,
                        head,
                        column,
                        sequence_length,
                        heads,
                        kv_rank)] = result;
                }
            }
        }
    }
}

void mla_compressed_attention_backward_cpu(
    float* query_latent_gradient,
    float* query_rope_gradient,
    float* kv_latent_gradient,
    float* key_rope_gradient,
    const float* output_gradient,
    const float* output,
    const float* logsumexp,
    const float* query_latent,
    const float* query_rope,
    const float* kv_latent,
    const float* key_rope,
    int batch_size,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale) {
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            for (int query_token = 0; query_token < sequence_length; ++query_token) {
                float delta = 0.0F;
                for (int column = 0; column < kv_rank; ++column) {
                    const int index = query_index(
                        batch,
                        query_token,
                        head,
                        column,
                        sequence_length,
                        heads,
                        kv_rank);
                    delta += output_gradient[index] * output[index];
                }

                const float lse = logsumexp[row_index(
                    batch, head, query_token, heads, sequence_length)];
                for (int key_token = 0; key_token <= query_token; ++key_token) {
                    const float probability = std::exp(
                        attention_score(
                            query_latent,
                            query_rope,
                            kv_latent,
                            key_rope,
                            batch,
                            query_token,
                            key_token,
                            head,
                            sequence_length,
                            heads,
                            kv_rank,
                            rope_size,
                            scale) -
                        lse);

                    float probability_gradient = 0.0F;
                    for (int column = 0; column < kv_rank; ++column) {
                        probability_gradient += output_gradient[query_index(
                                                    batch,
                                                    query_token,
                                                    head,
                                                    column,
                                                    sequence_length,
                                                    heads,
                                                    kv_rank)] *
                                                kv_latent[shared_index(
                                                    batch,
                                                    key_token,
                                                    column,
                                                    sequence_length,
                                                    kv_rank)];
                    }
                    const float score_gradient =
                        probability * (probability_gradient - delta) * scale;

                    for (int column = 0; column < kv_rank; ++column) {
                        const int query_column = query_index(
                            batch,
                            query_token,
                            head,
                            column,
                            sequence_length,
                            heads,
                            kv_rank);
                        const int kv_column = shared_index(
                            batch,
                            key_token,
                            column,
                            sequence_length,
                            kv_rank);
                        query_latent_gradient[query_column] +=
                            score_gradient * kv_latent[kv_column];
                        kv_latent_gradient[kv_column] +=
                            probability * output_gradient[query_column] +
                            score_gradient * query_latent[query_column];
                    }
                    for (int column = 0; column < rope_size; ++column) {
                        const int query_column = query_index(
                            batch,
                            query_token,
                            head,
                            column,
                            sequence_length,
                            heads,
                            rope_size);
                        const int key_column = shared_index(
                            batch,
                            key_token,
                            column,
                            sequence_length,
                            rope_size);
                        query_rope_gradient[query_column] +=
                            score_gradient * key_rope[key_column];
                        key_rope_gradient[key_column] +=
                            score_gradient * query_rope[query_column];
                    }
                }
            }
        }
    }
}

void mla_decode_forward_cpu(
    float* output,
    float* logsumexp,
    const float* query_latent,
    const float* query_rope,
    const float* kv_cache,
    const float* key_rope_cache,
    const int* cache_lengths,
    int batch_size,
    int maximum_sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale) {
    std::vector<float> probabilities(maximum_sequence_length);

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            const int query_base = (batch * heads + head) * kv_rank;
            const int query_rope_base = (batch * heads + head) * rope_size;
            float maximum = -std::numeric_limits<float>::infinity();

            for (int token = 0; token < cache_lengths[batch]; ++token) {
                const int kv_base =
                    (batch * maximum_sequence_length + token) * kv_rank;
                const int key_rope_base =
                    (batch * maximum_sequence_length + token) * rope_size;
                float score = 0.0F;
                for (int column = 0; column < kv_rank; ++column) {
                    score += query_latent[query_base + column] *
                             kv_cache[kv_base + column];
                }
                for (int column = 0; column < rope_size; ++column) {
                    score += query_rope[query_rope_base + column] *
                             key_rope_cache[key_rope_base + column];
                }
                score *= scale;
                probabilities[token] = score;
                maximum = std::fmax(maximum, score);
            }

            float normalizer = 0.0F;
            for (int token = 0; token < cache_lengths[batch]; ++token) {
                probabilities[token] = std::exp(probabilities[token] - maximum);
                normalizer += probabilities[token];
            }
            logsumexp[batch * heads + head] = maximum + std::log(normalizer);

            for (int column = 0; column < kv_rank; ++column) {
                float result = 0.0F;
                for (int token = 0; token < cache_lengths[batch]; ++token) {
                    const int kv_base =
                        (batch * maximum_sequence_length + token) * kv_rank;
                    result += probabilities[token] / normalizer *
                              kv_cache[kv_base + column];
                }
                output[query_base + column] = result;
            }
        }
    }
}

}  // namespace dscuda

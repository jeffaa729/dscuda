// Implements the absorbed-query MLA equations with explicit scalar loops and a materialized probability row.
// It is the correctness oracle for the fused BF16 CUDA path, not a competing runtime implementation.

#include "mla_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dscuda {
namespace {

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

#pragma once

#include "mla.h"

namespace dscuda {

void mla_layer_forward_cpu(
    float* output,
    const float* input,
    const MlaLayerParameters& parameters,
    const float* cosine,
    const float* sine,
    const MlaLayerConfig& config);

void mla_layer_backward_cpu(
    float* input_gradient,
    const MlaLayerGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const MlaLayerParameters& parameters,
    const float* cosine,
    const float* sine,
    const MlaLayerConfig& config);

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
    float scale);

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
    float scale);

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
    float scale);

}  // namespace dscuda

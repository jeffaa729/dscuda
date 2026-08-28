#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

struct MlaLayerConfig {
    int batch_size;
    int sequence_length;
    int hidden_size;
    int heads;
    int query_rank;
    int kv_rank;
    int nope_size;
    int rope_size;
    int value_size;
    float epsilon;
    float attention_scale;
};

// Matrix weights use row-major input-feature/output-feature storage except
// key_up [H,NoPE,C] and value_up [H,C,V], which are packed per attention head.
struct MlaLayerParameters {
    const float* query_down_weight;
    const float* query_norm_weight;
    const float* query_up_weight;
    const float* kv_down_weight;
    const float* kv_norm_weight;
    const float* key_up_weight;
    const float* value_up_weight;
    const float* output_weight;
};

struct MlaLayerGradients {
    float* query_down_weight;
    float* query_norm_weight;
    float* query_up_weight;
    float* kv_down_weight;
    float* kv_norm_weight;
    float* key_up_weight;
    float* value_up_weight;
    float* output_weight;
};

std::size_t mla_layer_activation_elements(const MlaLayerConfig& config);
std::size_t mla_layer_backward_workspace_elements(const MlaLayerConfig& config);
std::size_t mla_layer_bf16_workspace_elements(const MlaLayerConfig& config);

// Computes the full DeepSeek-V3 MLA projection graph and saves the FP32 values
// needed by backward. The attention core itself always consumes BF16 operands.
void mla_layer_forward_cuda(
    float* output,
    const float* input,
    const MlaLayerParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    __nv_bfloat16* bf16_workspace,
    const MlaLayerConfig& config,
    cudaStream_t stream = nullptr);

void mla_layer_backward_cuda(
    float* input_gradient,
    const MlaLayerGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const MlaLayerParameters& parameters,
    const float* cosine,
    const float* sine,
    const float* activations,
    float* workspace,
    __nv_bfloat16* bf16_workspace,
    const MlaLayerConfig& config,
    cudaStream_t stream = nullptr);

// Implements the absorbed-query form of MLA. Query latents use [B,T,H,C],
// query RoPE uses [B,T,H,R], shared KV latents use [B,T,C], shared key RoPE
// uses [B,T,R], output uses [B,T,H,C], and logsumexp uses [B,H,T].
void mla_compressed_attention_forward_cuda(
    float* output,
    float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int batch_size,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale,
    cudaStream_t stream = nullptr);

// Recomputes causal probabilities from the saved logsumexp. Every gradient
// element has one writer, and all four FP32 gradient buffers are accumulated.
void mla_compressed_attention_backward_cuda(
    float* query_latent_gradient,
    float* query_rope_gradient,
    float* kv_latent_gradient,
    float* key_rope_gradient,
    const float* output_gradient,
    const float* output,
    const float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int batch_size,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale,
    cudaStream_t stream = nullptr);

// Returns the FP32 workspace for split maxima, normalizers, and unnormalized
// latent outputs stored as [B,H,S,2+C].
std::size_t mla_decode_workspace_elements(
    int batch_size,
    int heads,
    int splits,
    int kv_rank);

// Decodes one query per batch from a BF16 compressed KV cache. Independent
// split CTAs produce online-softmax states, then one combine CTA merges them.
void mla_decode_forward_cuda(
    float* output,
    float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_cache,
    const __nv_bfloat16* key_rope_cache,
    const int* cache_lengths,
    float* workspace,
    int batch_size,
    int maximum_sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    int splits,
    float scale,
    cudaStream_t stream = nullptr);

// Projects one new hidden state into absorbed queries and compressed KV,
// appends the KV pair to cache, runs split-KV decode, and applies value/output
// projections. The cache stores only [C] latent and [R] RoPE entries.
std::size_t mla_layer_decode_workspace_elements(
    const MlaLayerConfig& config,
    int splits);
std::size_t mla_layer_decode_bf16_workspace_elements(
    const MlaLayerConfig& config);

void mla_layer_decode_forward_cuda(
    float* output,
    const float* input,
    const MlaLayerParameters& parameters,
    const float* cosine,
    const float* sine,
    int position,
    __nv_bfloat16* kv_cache,
    __nv_bfloat16* key_rope_cache,
    int* cache_lengths,
    float* workspace,
    __nv_bfloat16* bf16_workspace,
    const MlaLayerConfig& config,
    int splits,
    cudaStream_t stream = nullptr);

}  // namespace dscuda

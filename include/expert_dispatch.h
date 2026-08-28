#pragma once

#include <cuda_runtime.h>

namespace dscuda {

// Converts router logits to sigmoid affinities, selects top-k with the routing
// bias, normalizes the original affinities, and counts assignments per expert.
void expert_route_forward_cuda(
    float* scores,
    int* expert_indices,
    float* route_weights,
    int* expert_counts,
    const float* router_logits,
    const float* routing_bias,
    int rows,
    int experts,
    int top_k,
    float route_scale,
    cudaStream_t stream = nullptr);

// Builds deterministic expert-grouped slots and copies [rows,D] tokens into
// the no-drop dispatched layout [rows*top_k,D].
void expert_dispatch_forward_cuda(
    float* dispatched_input,
    int* expert_offsets,
    int* route_to_slot,
    int* slot_to_route,
    int* slot_expert,
    const float* input,
    const int* expert_indices,
    const int* expert_counts,
    int rows,
    int hidden_size,
    int experts,
    int top_k,
    cudaStream_t stream = nullptr);

// Computes one variable-row linear layer over expert-grouped input. Weights
// use [E,K,N], and slot_expert selects the matrix for each dispatched row.
void grouped_linear_forward_cuda(
    float* output,
    const float* input,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size,
    cudaStream_t stream = nullptr);

void grouped_linear_backward_cuda(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const int* expert_offsets,
    const int* slot_expert,
    int dispatched_rows,
    int experts,
    int output_size,
    int input_size,
    bool accumulate_input,
    cudaStream_t stream = nullptr);

// Adds the shared-expert output to the weighted routed outputs. Backward
// produces routed-output gradients, gating-weight gradients, and shared output.
void expert_combine_forward_cuda(
    float* output,
    const float* shared_output,
    const float* dispatched_output,
    const float* route_weights,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream = nullptr);

void expert_combine_backward_cuda(
    float* dispatched_output_gradient,
    float* route_weight_gradient,
    float* shared_output_gradient,
    const float* output_gradient,
    const float* dispatched_output,
    const float* route_weights,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream = nullptr);

void expert_unroute_backward_cuda(
    float* input_gradient,
    const float* dispatched_input_gradient,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream = nullptr);

// Differentiates normalized selected sigmoid affinities while treating top-k
// indices and the load-balancing bias as non-differentiable routing decisions.
void expert_route_backward_cuda(
    float* router_logit_gradient,
    const float* route_weight_gradient,
    const float* scores,
    const int* expert_indices,
    int rows,
    int experts,
    int top_k,
    float route_scale,
    cudaStream_t stream = nullptr);

void update_routing_bias_cuda(
    float* routing_bias,
    const int* expert_counts,
    int experts,
    int total_routes,
    float update_speed,
    cudaStream_t stream = nullptr);

}  // namespace dscuda

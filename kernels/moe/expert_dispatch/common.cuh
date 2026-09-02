#pragma once

#include "expert_dispatch.h"

namespace dscuda {

void expert_route_forward_sm89_cuda(
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
    cudaStream_t stream);

void expert_dispatch_forward_sm89_cuda(
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
    cudaStream_t stream);

void expert_combine_forward_sm89_cuda(
    float* output,
    const float* shared_output,
    const float* dispatched_output,
    const float* route_weights,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream);

void expert_combine_backward_sm89_cuda(
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
    cudaStream_t stream);

void expert_unroute_backward_sm89_cuda(
    float* input_gradient,
    const float* dispatched_input_gradient,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream);

void expert_route_backward_sm89_cuda(
    float* router_logit_gradient,
    const float* route_weight_gradient,
    const float* scores,
    const int* expert_indices,
    int rows,
    int experts,
    int top_k,
    float route_scale,
    cudaStream_t stream);

void update_routing_bias_sm89_cuda(
    float* routing_bias,
    const int* expert_counts,
    int experts,
    int total_routes,
    float update_speed,
    cudaStream_t stream);

}  // namespace dscuda

#include "common.cuh"

namespace dscuda {

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
    cudaStream_t stream) {
    expert_route_forward_sm89_cuda(
        scores,
        expert_indices,
        route_weights,
        expert_counts,
        router_logits,
        routing_bias,
        rows,
        experts,
        top_k,
        route_scale,
        stream);
}

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
    cudaStream_t stream) {
    expert_dispatch_forward_sm89_cuda(
        dispatched_input,
        expert_offsets,
        route_to_slot,
        slot_to_route,
        slot_expert,
        input,
        expert_indices,
        expert_counts,
        rows,
        hidden_size,
        experts,
        top_k,
        stream);
}

void expert_combine_forward_cuda(
    float* output,
    const float* shared_output,
    const float* dispatched_output,
    const float* route_weights,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream) {
    expert_combine_forward_sm89_cuda(
        output,
        shared_output,
        dispatched_output,
        route_weights,
        route_to_slot,
        rows,
        hidden_size,
        top_k,
        stream);
}

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
    cudaStream_t stream) {
    expert_combine_backward_sm89_cuda(
        dispatched_output_gradient,
        route_weight_gradient,
        shared_output_gradient,
        output_gradient,
        dispatched_output,
        route_weights,
        route_to_slot,
        rows,
        hidden_size,
        top_k,
        stream);
}

void expert_unroute_backward_cuda(
    float* input_gradient,
    const float* dispatched_input_gradient,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream) {
    expert_unroute_backward_sm89_cuda(
        input_gradient,
        dispatched_input_gradient,
        route_to_slot,
        rows,
        hidden_size,
        top_k,
        stream);
}

void expert_route_backward_cuda(
    float* router_logit_gradient,
    const float* route_weight_gradient,
    const float* scores,
    const int* expert_indices,
    int rows,
    int experts,
    int top_k,
    float route_scale,
    cudaStream_t stream) {
    expert_route_backward_sm89_cuda(
        router_logit_gradient,
        route_weight_gradient,
        scores,
        expert_indices,
        rows,
        experts,
        top_k,
        route_scale,
        stream);
}

void update_routing_bias_cuda(
    float* routing_bias,
    const int* expert_counts,
    int experts,
    int total_routes,
    float update_speed,
    cudaStream_t stream) {
    update_routing_bias_sm89_cuda(
        routing_bias,
        expert_counts,
        experts,
        total_routes,
        update_speed,
        stream);
}

}  // namespace dscuda

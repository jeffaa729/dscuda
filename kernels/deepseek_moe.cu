// Composes sigmoid top-k routing, no-drop dispatch, grouped routed SwiGLU experts, and dense shared SwiGLU experts into DeepSeekMoE.
// Backward differentiates every expert and router weight while the routing bias follows its separate load-count update rule.

#include "deepseek_moe.h"

#include "cuda_common.h"
#include "expert_dispatch.h"
#include "matmul.h"
#include "swiglu.h"

namespace dscuda {
namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_TOP_K = 8;

__device__ void select_top_k(
    int* selected,
    const float* scores,
    int experts,
    int top_k) {
    for (int rank = 0; rank < top_k; ++rank) {
        int best_expert = -1;
        float best_score = -__int_as_float(0x7f800000);
        for (int expert = 0; expert < experts; ++expert) {
            bool used = false;
            for (int previous = 0; previous < rank; ++previous) {
                used |= selected[previous] == expert;
            }
            if (!used && scores[expert] > best_score) {
                best_score = scores[expert];
                best_expert = expert;
            }
        }
        selected[rank] = best_expert;
    }
}

__global__ void sequence_balance_loss_kernel(
    float* total_loss,
    const float* scores,
    int sequence_length,
    int experts,
    int top_k,
    float weight) {
    extern __shared__ int expert_counts[];
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        expert_counts[expert] = 0;
    }
    __syncthreads();

    const int row_offset = blockIdx.x * sequence_length;
    for (int token = threadIdx.x;
         token < sequence_length;
         token += blockDim.x) {
        int selected[MAX_TOP_K];
        select_top_k(
            selected,
            scores + static_cast<std::size_t>(row_offset + token) * experts,
            experts,
            top_k);
        for (int rank = 0; rank < top_k; ++rank) {
            atomicAdd(&expert_counts[selected[rank]], 1);
        }
    }
    __syncthreads();

    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        float probability_mean = 0.0F;
        for (int token = 0; token < sequence_length; ++token) {
            const float* row = scores
                + static_cast<std::size_t>(row_offset + token) * experts;
            float normalizer = 0.0F;
            for (int other = 0; other < experts; ++other) {
                normalizer += row[other];
            }
            probability_mean += row[expert] / normalizer;
        }
        probability_mean /= sequence_length;
        const float load = static_cast<float>(experts * expert_counts[expert])
            / (top_k * sequence_length);
        atomicAdd(
            total_loss,
            weight / gridDim.x * load * probability_mean);
    }
}

__global__ void sequence_balance_backward_kernel(
    float* router_logit_gradient,
    const float* scores,
    int sequence_length,
    int experts,
    int top_k,
    float weight) {
    extern __shared__ int expert_counts[];
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        expert_counts[expert] = 0;
    }
    __syncthreads();

    const int row_offset = blockIdx.x * sequence_length;
    for (int token = threadIdx.x;
         token < sequence_length;
         token += blockDim.x) {
        int selected[MAX_TOP_K];
        select_top_k(
            selected,
            scores + static_cast<std::size_t>(row_offset + token) * experts,
            experts,
            top_k);
        for (int rank = 0; rank < top_k; ++rank) {
            atomicAdd(&expert_counts[selected[rank]], 1);
        }
    }
    __syncthreads();

    for (int token = threadIdx.x;
         token < sequence_length;
         token += blockDim.x) {
        const int row_index = row_offset + token;
        const float* row = scores
            + static_cast<std::size_t>(row_index) * experts;
        float normalizer = 0.0F;
        float load_weighted_scores = 0.0F;
        for (int expert = 0; expert < experts; ++expert) {
            const float load =
                static_cast<float>(experts * expert_counts[expert])
                / (top_k * sequence_length);
            normalizer += row[expert];
            load_weighted_scores += load * row[expert];
        }
        const float scale = weight
            / (gridDim.x * sequence_length * normalizer * normalizer);
        for (int expert = 0; expert < experts; ++expert) {
            const float load =
                static_cast<float>(experts * expert_counts[expert])
                / (top_k * sequence_length);
            const float score = row[expert];
            const float score_gradient = scale
                * (load * normalizer - load_weighted_scores);
            router_logit_gradient[
                static_cast<std::size_t>(row_index) * experts + expert]
                += score_gradient * score * (1.0F - score);
        }
    }
}

std::size_t align_four(std::size_t value) {
    return (value + 3) & ~std::size_t{3};
}

template <typename T>
struct FloatLayout {
    T* router_logits;
    T* scores;
    T* route_weights;
    T* dispatched_input;
    T* routed_gate;
    T* routed_up;
    T* routed_hidden;
    T* routed_output;
    T* shared_gate;
    T* shared_up;
    T* shared_hidden;
    T* shared_output;
    std::size_t elements;
};

template <typename T>
FloatLayout<T> make_float_layout(T* buffer, const DeepSeekMoeConfig& config) {
    const std::size_t routes =
        static_cast<std::size_t>(config.rows) * config.top_k;
    const std::size_t routed_hidden = routes * config.expert_hidden_size;
    const std::size_t shared_width =
        static_cast<std::size_t>(config.shared_experts) *
        config.expert_hidden_size;
    std::size_t offset = 0;
    auto take = [&](std::size_t elements) {
        offset = align_four(offset);
        T* result = buffer == nullptr ? nullptr : buffer + offset;
        offset += elements;
        return result;
    };

    FloatLayout<T> layout;
    layout.router_logits = take(
        static_cast<std::size_t>(config.rows) * config.routed_experts);
    layout.scores = take(
        static_cast<std::size_t>(config.rows) * config.routed_experts);
    layout.route_weights = take(routes);
    layout.dispatched_input = take(routes * config.hidden_size);
    layout.routed_gate = take(routed_hidden);
    layout.routed_up = take(routed_hidden);
    layout.routed_hidden = take(routed_hidden);
    layout.routed_output = take(routes * config.hidden_size);
    layout.shared_gate = take(config.rows * shared_width);
    layout.shared_up = take(config.rows * shared_width);
    layout.shared_hidden = take(config.rows * shared_width);
    layout.shared_output = take(
        static_cast<std::size_t>(config.rows) * config.hidden_size);
    layout.elements = align_four(offset);
    return layout;
}

template <typename T>
struct IntegerLayout {
    T* expert_indices;
    T* expert_counts;
    T* expert_offsets;
    T* route_to_slot;
    T* slot_to_route;
    T* slot_expert;
    std::size_t elements;
};

template <typename T>
IntegerLayout<T> make_integer_layout(
    T* buffer,
    const DeepSeekMoeConfig& config) {
    const std::size_t routes =
        static_cast<std::size_t>(config.rows) * config.top_k;
    IntegerLayout<T> layout;
    layout.expert_indices = buffer;
    layout.expert_counts = buffer == nullptr ? nullptr : buffer + routes;
    layout.expert_offsets = buffer == nullptr
        ? nullptr
        : layout.expert_counts + config.routed_experts;
    layout.route_to_slot = buffer == nullptr
        ? nullptr
        : layout.expert_offsets + config.routed_experts + 1;
    layout.slot_to_route =
        buffer == nullptr ? nullptr : layout.route_to_slot + routes;
    layout.slot_expert =
        buffer == nullptr ? nullptr : layout.slot_to_route + routes;
    layout.elements =
        4 * routes + 2 * config.routed_experts + 1;
    return layout;
}

template <typename T>
struct BackwardLayout {
    T* router_logit_gradient;
    T* route_weight_gradient;
    T* routed_output_gradient;
    T* routed_hidden_gradient;
    T* routed_gate_gradient;
    T* routed_up_gradient;
    T* dispatched_input_gradient;
    T* shared_output_gradient;
    T* shared_hidden_gradient;
    T* shared_gate_gradient;
    T* shared_up_gradient;
    std::size_t elements;
};

template <typename T>
BackwardLayout<T> make_backward_layout(
    T* buffer,
    const DeepSeekMoeConfig& config) {
    const std::size_t routes =
        static_cast<std::size_t>(config.rows) * config.top_k;
    const std::size_t routed_hidden = routes * config.expert_hidden_size;
    const std::size_t shared_width =
        static_cast<std::size_t>(config.shared_experts) *
        config.expert_hidden_size;
    std::size_t offset = 0;
    auto take = [&](std::size_t elements) {
        offset = align_four(offset);
        T* result = buffer == nullptr ? nullptr : buffer + offset;
        offset += elements;
        return result;
    };

    BackwardLayout<T> layout;
    layout.router_logit_gradient = take(
        static_cast<std::size_t>(config.rows) * config.routed_experts);
    layout.route_weight_gradient = take(routes);
    layout.routed_output_gradient = take(routes * config.hidden_size);
    layout.routed_hidden_gradient = take(routed_hidden);
    layout.routed_gate_gradient = take(routed_hidden);
    layout.routed_up_gradient = take(routed_hidden);
    layout.dispatched_input_gradient = take(routes * config.hidden_size);
    layout.shared_output_gradient = take(
        static_cast<std::size_t>(config.rows) * config.hidden_size);
    layout.shared_hidden_gradient = take(config.rows * shared_width);
    layout.shared_gate_gradient = take(config.rows * shared_width);
    layout.shared_up_gradient = take(config.rows * shared_width);
    layout.elements = align_four(offset);
    return layout;
}

}  // namespace

std::size_t deepseek_moe_activation_elements(
    const DeepSeekMoeConfig& config) {
    return make_float_layout(static_cast<float*>(nullptr), config).elements;
}

std::size_t deepseek_moe_integer_activation_elements(
    const DeepSeekMoeConfig& config) {
    return make_integer_layout(static_cast<int*>(nullptr), config).elements;
}

std::size_t deepseek_moe_backward_workspace_elements(
    const DeepSeekMoeConfig& config) {
    return make_backward_layout(static_cast<float*>(nullptr), config).elements;
}

void deepseek_moe_forward_cuda(
    float* output,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    float* activations,
    int* integer_activations,
    const DeepSeekMoeConfig& config,
    cudaStream_t stream) {
    const int routes = config.rows * config.top_k;
    const int shared_width =
        config.shared_experts * config.expert_hidden_size;
    auto saved = make_float_layout(activations, config);
    auto indices = make_integer_layout(integer_activations, config);

    matmul_fp32_forward_cuda(
        saved.router_logits,
        input,
        parameters.router_weight,
        config.rows,
        config.routed_experts,
        config.hidden_size,
        stream);
    expert_route_forward_cuda(
        saved.scores,
        indices.expert_indices,
        saved.route_weights,
        indices.expert_counts,
        saved.router_logits,
        parameters.routing_bias,
        config.rows,
        config.routed_experts,
        config.top_k,
        config.route_scale,
        stream);
    expert_dispatch_forward_cuda(
        saved.dispatched_input,
        indices.expert_offsets,
        indices.route_to_slot,
        indices.slot_to_route,
        indices.slot_expert,
        input,
        indices.expert_indices,
        indices.expert_counts,
        config.rows,
        config.hidden_size,
        config.routed_experts,
        config.top_k,
        stream);
    grouped_linear_forward_cuda(
        saved.routed_gate,
        saved.dispatched_input,
        parameters.routed_gate_weight,
        indices.slot_expert,
        routes,
        config.expert_hidden_size,
        config.hidden_size,
        stream);
    grouped_linear_forward_cuda(
        saved.routed_up,
        saved.dispatched_input,
        parameters.routed_up_weight,
        indices.slot_expert,
        routes,
        config.expert_hidden_size,
        config.hidden_size,
        stream);
    swiglu_forward_cuda(
        saved.routed_hidden,
        saved.routed_gate,
        saved.routed_up,
        routes * config.expert_hidden_size,
        stream);
    grouped_linear_forward_cuda(
        saved.routed_output,
        saved.routed_hidden,
        parameters.routed_down_weight,
        indices.slot_expert,
        routes,
        config.hidden_size,
        config.expert_hidden_size,
        stream);

    matmul_fp32_forward_cuda(
        saved.shared_gate,
        input,
        parameters.shared_gate_weight,
        config.rows,
        shared_width,
        config.hidden_size,
        stream);
    matmul_fp32_forward_cuda(
        saved.shared_up,
        input,
        parameters.shared_up_weight,
        config.rows,
        shared_width,
        config.hidden_size,
        stream);
    swiglu_forward_cuda(
        saved.shared_hidden,
        saved.shared_gate,
        saved.shared_up,
        config.rows * shared_width,
        stream);
    matmul_fp32_forward_cuda(
        saved.shared_output,
        saved.shared_hidden,
        parameters.shared_down_weight,
        config.rows,
        config.hidden_size,
        shared_width,
        stream);
    expert_combine_forward_cuda(
        output,
        saved.shared_output,
        saved.routed_output,
        saved.route_weights,
        indices.route_to_slot,
        config.rows,
        config.hidden_size,
        config.top_k,
        stream);
}

void deepseek_moe_backward_cuda(
    float* input_gradient,
    const DeepSeekMoeGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const float* activations,
    const int* integer_activations,
    float* workspace,
    const DeepSeekMoeConfig& config,
    cudaStream_t stream) {
    const int routes = config.rows * config.top_k;
    const int shared_width =
        config.shared_experts * config.expert_hidden_size;
    const auto saved = make_float_layout(activations, config);
    const auto indices = make_integer_layout(integer_activations, config);
    auto gradient = make_backward_layout(workspace, config);
    CUDA_CHECK(cudaMemsetAsync(
        workspace,
        0,
        deepseek_moe_backward_workspace_elements(config) * sizeof(float),
        stream));

    expert_combine_backward_cuda(
        gradient.routed_output_gradient,
        gradient.route_weight_gradient,
        gradient.shared_output_gradient,
        output_gradient,
        saved.routed_output,
        saved.route_weights,
        indices.route_to_slot,
        config.rows,
        config.hidden_size,
        config.top_k,
        stream);
    grouped_linear_backward_cuda(
        gradient.routed_hidden_gradient,
        parameter_gradients.routed_down_weight,
        gradient.routed_output_gradient,
        saved.routed_hidden,
        parameters.routed_down_weight,
        indices.expert_offsets,
        indices.slot_expert,
        routes,
        config.routed_experts,
        config.hidden_size,
        config.expert_hidden_size,
        false,
        stream);
    swiglu_backward_cuda(
        gradient.routed_gate_gradient,
        gradient.routed_up_gradient,
        gradient.routed_hidden_gradient,
        saved.routed_gate,
        saved.routed_up,
        routes * config.expert_hidden_size,
        stream);
    grouped_linear_backward_cuda(
        gradient.dispatched_input_gradient,
        parameter_gradients.routed_gate_weight,
        gradient.routed_gate_gradient,
        saved.dispatched_input,
        parameters.routed_gate_weight,
        indices.expert_offsets,
        indices.slot_expert,
        routes,
        config.routed_experts,
        config.expert_hidden_size,
        config.hidden_size,
        false,
        stream);
    grouped_linear_backward_cuda(
        gradient.dispatched_input_gradient,
        parameter_gradients.routed_up_weight,
        gradient.routed_up_gradient,
        saved.dispatched_input,
        parameters.routed_up_weight,
        indices.expert_offsets,
        indices.slot_expert,
        routes,
        config.routed_experts,
        config.expert_hidden_size,
        config.hidden_size,
        true,
        stream);
    expert_unroute_backward_cuda(
        input_gradient,
        gradient.dispatched_input_gradient,
        indices.route_to_slot,
        config.rows,
        config.hidden_size,
        config.top_k,
        stream);
    expert_route_backward_cuda(
        gradient.router_logit_gradient,
        gradient.route_weight_gradient,
        saved.scores,
        indices.expert_indices,
        config.rows,
        config.routed_experts,
        config.top_k,
        config.route_scale,
        stream);
    if (config.balance_loss_weight != 0.0F) {
        sequence_balance_backward_kernel<<<
            config.batch_size,
            BLOCK_SIZE,
            config.routed_experts * sizeof(int),
            stream>>>(
                gradient.router_logit_gradient,
                saved.scores,
                config.sequence_length,
                config.routed_experts,
                config.top_k,
                config.balance_loss_weight);
        CUDA_CHECK(cudaGetLastError());
    }
    matmul_fp32_backward_cuda(
        input_gradient,
        parameter_gradients.router_weight,
        gradient.router_logit_gradient,
        input,
        parameters.router_weight,
        config.rows,
        config.routed_experts,
        config.hidden_size,
        stream);

    matmul_fp32_backward_cuda(
        gradient.shared_hidden_gradient,
        parameter_gradients.shared_down_weight,
        gradient.shared_output_gradient,
        saved.shared_hidden,
        parameters.shared_down_weight,
        config.rows,
        config.hidden_size,
        shared_width,
        stream);
    swiglu_backward_cuda(
        gradient.shared_gate_gradient,
        gradient.shared_up_gradient,
        gradient.shared_hidden_gradient,
        saved.shared_gate,
        saved.shared_up,
        config.rows * shared_width,
        stream);
    matmul_fp32_backward_cuda(
        input_gradient,
        parameter_gradients.shared_gate_weight,
        gradient.shared_gate_gradient,
        input,
        parameters.shared_gate_weight,
        config.rows,
        shared_width,
        config.hidden_size,
        stream);
    matmul_fp32_backward_cuda(
        input_gradient,
        parameter_gradients.shared_up_weight,
        gradient.shared_up_gradient,
        input,
        parameters.shared_up_weight,
        config.rows,
        shared_width,
        config.hidden_size,
        stream);
}

void deepseek_moe_update_bias_cuda(
    float* routing_bias,
    const int* integer_activations,
    const DeepSeekMoeConfig& config,
    float update_speed,
    cudaStream_t stream) {
    const auto indices = make_integer_layout(integer_activations, config);
    update_routing_bias_cuda(
        routing_bias,
        indices.expert_counts,
        config.routed_experts,
        config.rows * config.top_k,
        update_speed,
        stream);
}

void deepseek_moe_add_balance_loss_cuda(
    float* total_loss,
    const float* activations,
    const DeepSeekMoeConfig& config,
    cudaStream_t stream) {
    if (config.balance_loss_weight == 0.0F) {
        return;
    }
    const auto saved = make_float_layout(activations, config);
    sequence_balance_loss_kernel<<<
        config.batch_size,
        BLOCK_SIZE,
        config.routed_experts * sizeof(int),
        stream>>>(
            total_loss,
            saved.scores,
            config.sequence_length,
            config.routed_experts,
            config.top_k,
            config.balance_loss_weight);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

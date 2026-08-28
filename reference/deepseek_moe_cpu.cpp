// Composes the scalar router, dispatch, grouped expert, shared expert, and combine references into a complete DeepSeekMoE layer.
// Forward values are recomputed for backward so the code remains a compact and readable oracle for the CUDA graph.

#include "deepseek_moe_cpu.h"

#include "expert_dispatch_cpu.h"
#include "matmul_cpu.h"
#include "swiglu_cpu.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dscuda {
namespace {

struct CpuMoeSaved {
    std::vector<float> router_logits;
    std::vector<float> scores;
    std::vector<int> expert_indices;
    std::vector<float> route_weights;
    std::vector<int> expert_counts;
    std::vector<int> expert_offsets;
    std::vector<int> route_to_slot;
    std::vector<int> slot_to_route;
    std::vector<int> slot_expert;
    std::vector<float> dispatched_input;
    std::vector<float> routed_gate;
    std::vector<float> routed_up;
    std::vector<float> routed_hidden;
    std::vector<float> routed_output;
    std::vector<float> shared_gate;
    std::vector<float> shared_up;
    std::vector<float> shared_hidden;
    std::vector<float> shared_output;
};

CpuMoeSaved forward_saved(
    float* output,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const DeepSeekMoeConfig& config) {
    const int routes = config.rows * config.top_k;
    const int shared_width =
        config.shared_experts * config.expert_hidden_size;
    CpuMoeSaved saved;
    saved.router_logits.resize(config.rows * config.routed_experts);
    saved.scores.resize(config.rows * config.routed_experts);
    saved.expert_indices.resize(routes);
    saved.route_weights.resize(routes);
    saved.expert_counts.resize(config.routed_experts);
    saved.expert_offsets.resize(config.routed_experts + 1);
    saved.route_to_slot.resize(routes);
    saved.slot_to_route.resize(routes);
    saved.slot_expert.resize(routes);
    saved.dispatched_input.resize(routes * config.hidden_size);
    saved.routed_gate.resize(routes * config.expert_hidden_size);
    saved.routed_up.resize(routes * config.expert_hidden_size);
    saved.routed_hidden.resize(routes * config.expert_hidden_size);
    saved.routed_output.resize(routes * config.hidden_size);
    saved.shared_gate.resize(config.rows * shared_width);
    saved.shared_up.resize(config.rows * shared_width);
    saved.shared_hidden.resize(config.rows * shared_width);
    saved.shared_output.resize(config.rows * config.hidden_size);

    matmul_forward_cpu(
        saved.router_logits.data(), input, parameters.router_weight,
        config.rows, config.routed_experts, config.hidden_size);
    expert_route_forward_cpu(
        saved.scores.data(), saved.expert_indices.data(),
        saved.route_weights.data(), saved.expert_counts.data(),
        saved.router_logits.data(), parameters.routing_bias, config.rows,
        config.routed_experts, config.top_k, config.route_scale);
    expert_dispatch_forward_cpu(
        saved.dispatched_input.data(), saved.expert_offsets.data(),
        saved.route_to_slot.data(), saved.slot_to_route.data(),
        saved.slot_expert.data(), input, saved.expert_indices.data(),
        saved.expert_counts.data(), config.rows, config.hidden_size,
        config.routed_experts, config.top_k);
    grouped_linear_forward_cpu(
        saved.routed_gate.data(), saved.dispatched_input.data(),
        parameters.routed_gate_weight, saved.slot_expert.data(), routes,
        config.expert_hidden_size, config.hidden_size);
    grouped_linear_forward_cpu(
        saved.routed_up.data(), saved.dispatched_input.data(),
        parameters.routed_up_weight, saved.slot_expert.data(), routes,
        config.expert_hidden_size, config.hidden_size);
    swiglu_forward_cpu(
        saved.routed_hidden.data(), saved.routed_gate.data(),
        saved.routed_up.data(), routes * config.expert_hidden_size);
    grouped_linear_forward_cpu(
        saved.routed_output.data(), saved.routed_hidden.data(),
        parameters.routed_down_weight, saved.slot_expert.data(), routes,
        config.hidden_size, config.expert_hidden_size);

    matmul_forward_cpu(
        saved.shared_gate.data(), input, parameters.shared_gate_weight,
        config.rows, shared_width, config.hidden_size);
    matmul_forward_cpu(
        saved.shared_up.data(), input, parameters.shared_up_weight,
        config.rows, shared_width, config.hidden_size);
    swiglu_forward_cpu(
        saved.shared_hidden.data(), saved.shared_gate.data(),
        saved.shared_up.data(), config.rows * shared_width);
    matmul_forward_cpu(
        saved.shared_output.data(), saved.shared_hidden.data(),
        parameters.shared_down_weight, config.rows, config.hidden_size,
        shared_width);
    expert_combine_forward_cpu(
        output, saved.shared_output.data(), saved.routed_output.data(),
        saved.route_weights.data(), saved.route_to_slot.data(), config.rows,
        config.hidden_size, config.top_k);
    return saved;
}

std::vector<int> sequence_expert_counts(
    const float* scores,
    int sequence_length,
    int experts,
    int top_k) {
    std::vector<int> counts(experts, 0);
    std::vector<int> indices(experts);
    for (int token = 0; token < sequence_length; ++token) {
        for (int expert = 0; expert < experts; ++expert) {
            indices[expert] = expert;
        }
        const float* row = scores + token * experts;
        std::partial_sort(
            indices.begin(),
            indices.begin() + top_k,
            indices.end(),
            [&](int left, int right) {
                return row[left] == row[right]
                    ? left < right
                    : row[left] > row[right];
            });
        for (int rank = 0; rank < top_k; ++rank) {
            ++counts[indices[rank]];
        }
    }
    return counts;
}

float balance_loss_from_scores(
    const float* scores,
    const DeepSeekMoeConfig& config) {
    float loss = 0.0F;
    for (int batch = 0; batch < config.batch_size; ++batch) {
        const float* sequence = scores
            + static_cast<std::size_t>(batch) * config.sequence_length
                * config.routed_experts;
        const std::vector<int> counts = sequence_expert_counts(
            sequence,
            config.sequence_length,
            config.routed_experts,
            config.top_k);
        for (int expert = 0; expert < config.routed_experts; ++expert) {
            float probability_mean = 0.0F;
            for (int token = 0; token < config.sequence_length; ++token) {
                const float* row = sequence
                    + token * config.routed_experts;
                float normalizer = 0.0F;
                for (int other = 0;
                     other < config.routed_experts;
                     ++other) {
                    normalizer += row[other];
                }
                probability_mean += row[expert] / normalizer;
            }
            probability_mean /= config.sequence_length;
            const float load = static_cast<float>(
                config.routed_experts * counts[expert])
                / (config.top_k * config.sequence_length);
            loss += config.balance_loss_weight / config.batch_size
                * load * probability_mean;
        }
    }
    return loss;
}

void add_balance_gradient(
    float* router_logit_gradient,
    const float* scores,
    const DeepSeekMoeConfig& config) {
    for (int batch = 0; batch < config.batch_size; ++batch) {
        const int row_offset = batch * config.sequence_length;
        const float* sequence = scores
            + static_cast<std::size_t>(row_offset)
                * config.routed_experts;
        const std::vector<int> counts = sequence_expert_counts(
            sequence,
            config.sequence_length,
            config.routed_experts,
            config.top_k);
        for (int token = 0; token < config.sequence_length; ++token) {
            const int row_index = row_offset + token;
            const float* row =
                scores + row_index * config.routed_experts;
            float normalizer = 0.0F;
            float load_weighted_scores = 0.0F;
            for (int expert = 0;
                 expert < config.routed_experts;
                 ++expert) {
                const float load = static_cast<float>(
                    config.routed_experts * counts[expert])
                    / (config.top_k * config.sequence_length);
                normalizer += row[expert];
                load_weighted_scores += load * row[expert];
            }
            const float scale = config.balance_loss_weight
                / (config.batch_size * config.sequence_length
                   * normalizer * normalizer);
            for (int expert = 0;
                 expert < config.routed_experts;
                 ++expert) {
                const float load = static_cast<float>(
                    config.routed_experts * counts[expert])
                    / (config.top_k * config.sequence_length);
                const float score = row[expert];
                router_logit_gradient[
                    row_index * config.routed_experts + expert]
                    += scale * (load * normalizer - load_weighted_scores)
                        * score * (1.0F - score);
            }
        }
    }
}

}  // namespace

void deepseek_moe_forward_cpu(
    float* output,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const DeepSeekMoeConfig& config) {
    forward_saved(output, input, parameters, config);
}

void deepseek_moe_backward_cpu(
    float* input_gradient,
    const DeepSeekMoeGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const DeepSeekMoeConfig& config) {
    const int routes = config.rows * config.top_k;
    const int shared_width =
        config.shared_experts * config.expert_hidden_size;
    std::vector<float> discarded(config.rows * config.hidden_size);
    const CpuMoeSaved saved =
        forward_saved(discarded.data(), input, parameters, config);

    std::vector<float> routed_output_gradient(
        routes * config.hidden_size);
    std::vector<float> route_weight_gradient(routes);
    std::vector<float> shared_output_gradient(
        config.rows * config.hidden_size);
    expert_combine_backward_cpu(
        routed_output_gradient.data(), route_weight_gradient.data(),
        shared_output_gradient.data(), output_gradient,
        saved.routed_output.data(), saved.route_weights.data(),
        saved.route_to_slot.data(), config.rows, config.hidden_size,
        config.top_k);

    std::vector<float> routed_hidden_gradient(
        routes * config.expert_hidden_size, 0.0F);
    grouped_linear_backward_cpu(
        routed_hidden_gradient.data(),
        parameter_gradients.routed_down_weight,
        routed_output_gradient.data(),
        saved.routed_hidden.data(),
        parameters.routed_down_weight,
        saved.expert_offsets.data(),
        saved.slot_expert.data(),
        routes,
        config.routed_experts,
        config.hidden_size,
        config.expert_hidden_size,
        false);
    std::vector<float> routed_gate_gradient(
        routes * config.expert_hidden_size);
    std::vector<float> routed_up_gradient(
        routes * config.expert_hidden_size);
    swiglu_backward_cpu(
        routed_gate_gradient.data(), routed_up_gradient.data(),
        routed_hidden_gradient.data(), saved.routed_gate.data(),
        saved.routed_up.data(), routes * config.expert_hidden_size);
    std::vector<float> dispatched_input_gradient(
        routes * config.hidden_size, 0.0F);
    grouped_linear_backward_cpu(
        dispatched_input_gradient.data(),
        parameter_gradients.routed_gate_weight,
        routed_gate_gradient.data(),
        saved.dispatched_input.data(),
        parameters.routed_gate_weight,
        saved.expert_offsets.data(),
        saved.slot_expert.data(),
        routes,
        config.routed_experts,
        config.expert_hidden_size,
        config.hidden_size,
        false);
    grouped_linear_backward_cpu(
        dispatched_input_gradient.data(),
        parameter_gradients.routed_up_weight,
        routed_up_gradient.data(),
        saved.dispatched_input.data(),
        parameters.routed_up_weight,
        saved.expert_offsets.data(),
        saved.slot_expert.data(),
        routes,
        config.routed_experts,
        config.expert_hidden_size,
        config.hidden_size,
        true);
    expert_unroute_backward_cpu(
        input_gradient, dispatched_input_gradient.data(),
        saved.route_to_slot.data(), config.rows, config.hidden_size,
        config.top_k);

    std::vector<float> router_logit_gradient(
        config.rows * config.routed_experts);
    expert_route_backward_cpu(
        router_logit_gradient.data(), route_weight_gradient.data(),
        saved.scores.data(), saved.expert_indices.data(), config.rows,
        config.routed_experts, config.top_k, config.route_scale);
    if (config.balance_loss_weight != 0.0F) {
        add_balance_gradient(
            router_logit_gradient.data(), saved.scores.data(), config);
    }
    matmul_backward_cpu(
        input_gradient, parameter_gradients.router_weight,
        router_logit_gradient.data(), input, parameters.router_weight,
        config.rows, config.routed_experts, config.hidden_size);

    std::vector<float> shared_hidden_gradient(
        config.rows * shared_width, 0.0F);
    matmul_backward_cpu(
        shared_hidden_gradient.data(),
        parameter_gradients.shared_down_weight,
        shared_output_gradient.data(),
        saved.shared_hidden.data(),
        parameters.shared_down_weight,
        config.rows,
        config.hidden_size,
        shared_width);
    std::vector<float> shared_gate_gradient(config.rows * shared_width);
    std::vector<float> shared_up_gradient(config.rows * shared_width);
    swiglu_backward_cpu(
        shared_gate_gradient.data(), shared_up_gradient.data(),
        shared_hidden_gradient.data(), saved.shared_gate.data(),
        saved.shared_up.data(), config.rows * shared_width);
    matmul_backward_cpu(
        input_gradient, parameter_gradients.shared_gate_weight,
        shared_gate_gradient.data(), input, parameters.shared_gate_weight,
        config.rows, shared_width, config.hidden_size);
    matmul_backward_cpu(
        input_gradient, parameter_gradients.shared_up_weight,
        shared_up_gradient.data(), input, parameters.shared_up_weight,
        config.rows, shared_width, config.hidden_size);
}

float deepseek_moe_balance_loss_cpu(
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const DeepSeekMoeConfig& config) {
    std::vector<float> logits(config.rows * config.routed_experts);
    std::vector<float> scores(logits.size());
    matmul_forward_cpu(
        logits.data(),
        input,
        parameters.router_weight,
        config.rows,
        config.routed_experts,
        config.hidden_size);
    for (std::size_t index = 0; index < logits.size(); ++index) {
        scores[index] = 1.0F / (1.0F + std::exp(-logits[index]));
    }
    return balance_loss_from_scores(scores.data(), config);
}

}  // namespace dscuda

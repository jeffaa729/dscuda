// Implements deterministic scalar references for DeepSeek routing, no-drop dispatch, grouped linears, and combination.
// These loops intentionally mirror the CUDA buffer layouts so routing and gradient maps can be checked element by element.

#include "expert_dispatch_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dscuda {

void expert_route_forward_cpu(
    float* scores,
    int* expert_indices,
    float* route_weights,
    int* expert_counts,
    const float* router_logits,
    const float* routing_bias,
    int rows,
    int experts,
    int top_k,
    float route_scale) {
    std::fill_n(expert_counts, experts, 0);
    for (int row = 0; row < rows; ++row) {
        for (int expert = 0; expert < experts; ++expert) {
            const float logit = router_logits[row * experts + expert];
            scores[row * experts + expert] = 1.0F / (1.0F + std::exp(-logit));
        }
        float normalizer = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_expert = -1;
            float best_score = -std::numeric_limits<float>::infinity();
            for (int expert = 0; expert < experts; ++expert) {
                bool selected = false;
                for (int previous = 0; previous < rank; ++previous) {
                    selected |=
                        expert_indices[row * top_k + previous] == expert;
                }
                const float biased =
                    scores[row * experts + expert] + routing_bias[expert];
                if (!selected && biased > best_score) {
                    best_score = biased;
                    best_expert = expert;
                }
            }
            expert_indices[row * top_k + rank] = best_expert;
            normalizer += scores[row * experts + best_expert];
            ++expert_counts[best_expert];
        }
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = expert_indices[row * top_k + rank];
            route_weights[row * top_k + rank] =
                route_scale * scores[row * experts + expert] / normalizer;
        }
    }
}

void expert_dispatch_forward_cpu(
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
    int top_k) {
    expert_offsets[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
        expert_offsets[expert + 1] =
            expert_offsets[expert] + expert_counts[expert];
    }
    const int routes = rows * top_k;
    for (int expert = 0; expert < experts; ++expert) {
        int slot = expert_offsets[expert];
        for (int route = 0; route < routes; ++route) {
            if (expert_indices[route] == expert) {
                route_to_slot[route] = slot;
                slot_to_route[slot] = route;
                slot_expert[slot] = expert;
                ++slot;
            }
        }
    }
    for (int slot = 0; slot < routes; ++slot) {
        const int token = slot_to_route[slot] / top_k;
        std::copy_n(
            input + token * hidden_size,
            hidden_size,
            dispatched_input + slot * hidden_size);
    }
}

void grouped_linear_forward_cpu(
    float* output,
    const float* input,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size) {
    for (int row = 0; row < dispatched_rows; ++row) {
        const int expert = slot_expert[row];
        for (int column = 0; column < output_size; ++column) {
            float sum = 0.0F;
            for (int inner = 0; inner < input_size; ++inner) {
                sum += input[row * input_size + inner] *
                       weight[(expert * input_size + inner) * output_size
                              + column];
            }
            output[row * output_size + column] = sum;
        }
    }
}

void grouped_linear_backward_cpu(
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
    bool accumulate_input) {
    if (!accumulate_input) {
        std::fill_n(input_gradient, dispatched_rows * input_size, 0.0F);
    }
    for (int row = 0; row < dispatched_rows; ++row) {
        const int expert = slot_expert[row];
        for (int inner = 0; inner < input_size; ++inner) {
            for (int column = 0; column < output_size; ++column) {
                input_gradient[row * input_size + inner] +=
                    output_gradient[row * output_size + column] *
                    weight[(expert * input_size + inner) * output_size
                           + column];
            }
        }
    }
    for (int expert = 0; expert < experts; ++expert) {
        for (int inner = 0; inner < input_size; ++inner) {
            for (int column = 0; column < output_size; ++column) {
                float sum = 0.0F;
                for (int row = expert_offsets[expert];
                     row < expert_offsets[expert + 1];
                     ++row) {
                    sum += input[row * input_size + inner] *
                           output_gradient[row * output_size + column];
                }
                weight_gradient[
                    (expert * input_size + inner) * output_size + column] +=
                    sum;
            }
        }
    }
}

void expert_combine_forward_cpu(
    float* output,
    const float* shared_output,
    const float* dispatched_output,
    const float* route_weights,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k) {
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < hidden_size; ++column) {
            float result = shared_output[row * hidden_size + column];
            for (int rank = 0; rank < top_k; ++rank) {
                const int route = row * top_k + rank;
                result += route_weights[route] * dispatched_output[
                    route_to_slot[route] * hidden_size + column];
            }
            output[row * hidden_size + column] = result;
        }
    }
}

void expert_combine_backward_cpu(
    float* dispatched_output_gradient,
    float* route_weight_gradient,
    float* shared_output_gradient,
    const float* output_gradient,
    const float* dispatched_output,
    const float* route_weights,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k) {
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < hidden_size; ++column) {
            const float gradient = output_gradient[row * hidden_size + column];
            shared_output_gradient[row * hidden_size + column] = gradient;
            for (int rank = 0; rank < top_k; ++rank) {
                const int route = row * top_k + rank;
                dispatched_output_gradient[
                    route_to_slot[route] * hidden_size + column] =
                    route_weights[route] * gradient;
            }
        }
        for (int rank = 0; rank < top_k; ++rank) {
            const int route = row * top_k + rank;
            const int slot = route_to_slot[route];
            float sum = 0.0F;
            for (int column = 0; column < hidden_size; ++column) {
                sum += output_gradient[row * hidden_size + column] *
                       dispatched_output[slot * hidden_size + column];
            }
            route_weight_gradient[route] = sum;
        }
    }
}

void expert_unroute_backward_cpu(
    float* input_gradient,
    const float* dispatched_input_gradient,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k) {
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < hidden_size; ++column) {
            for (int rank = 0; rank < top_k; ++rank) {
                const int slot = route_to_slot[row * top_k + rank];
                input_gradient[row * hidden_size + column] +=
                    dispatched_input_gradient[slot * hidden_size + column];
            }
        }
    }
}

void expert_route_backward_cpu(
    float* router_logit_gradient,
    const float* route_weight_gradient,
    const float* scores,
    const int* expert_indices,
    int rows,
    int experts,
    int top_k,
    float route_scale) {
    std::fill_n(router_logit_gradient, rows * experts, 0.0F);
    for (int row = 0; row < rows; ++row) {
        float normalizer = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) {
            normalizer += scores[
                row * experts + expert_indices[row * top_k + rank]];
        }
        float projected_gradient = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = expert_indices[row * top_k + rank];
            projected_gradient += route_weight_gradient[row * top_k + rank] *
                scores[row * experts + expert] / normalizer;
        }
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = expert_indices[row * top_k + rank];
            const float score = scores[row * experts + expert];
            router_logit_gradient[row * experts + expert] =
                route_scale / normalizer *
                (route_weight_gradient[row * top_k + rank]
                 - projected_gradient) *
                score * (1.0F - score);
        }
    }
}

void update_routing_bias_cpu(
    float* routing_bias,
    const int* expert_counts,
    int experts,
    int total_routes,
    float update_speed) {
    const float target = static_cast<float>(total_routes) / experts;
    for (int expert = 0; expert < experts; ++expert) {
        routing_bias[expert] += expert_counts[expert] < target
            ? update_speed
            : (expert_counts[expert] > target ? -update_speed : 0.0F);
    }
}

}  // namespace dscuda

// Implements DeepSeek-V3 sigmoid top-k routing, parallel no-drop token dispatch, and weighted combination.
// Its reverse kernels preserve the discrete routing decision while differentiating expert outputs and normalized selected affinities.

#include "cuda_common.h"
#include "common.cuh"

#include <cmath>

namespace dscuda {
namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_TOP_K = 8;

__global__ void route_forward_kernel(
    float* scores,
    int* expert_indices,
    float* route_weights,
    int* expert_counts,
    const float* router_logits,
    const float* routing_bias,
    int experts,
    int top_k,
    float route_scale) {
    const int row = blockIdx.x;
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float logit = router_logits[row * experts + expert];
        scores[row * experts + expert] = 1.0F / (1.0F + expf(-logit));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int selected[MAX_TOP_K];
        float normalizer = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) {
            int best_expert = -1;
            float best_score = -__int_as_float(0x7f800000);
            for (int expert = 0; expert < experts; ++expert) {
                bool already_selected = false;
                for (int previous = 0; previous < rank; ++previous) {
                    already_selected |= selected[previous] == expert;
                }
                const float biased =
                    scores[row * experts + expert] + routing_bias[expert];
                if (!already_selected && biased > best_score) {
                    best_score = biased;
                    best_expert = expert;
                }
            }
            selected[rank] = best_expert;
            expert_indices[row * top_k + rank] = best_expert;
            normalizer += scores[row * experts + best_expert];
            atomicAdd(&expert_counts[best_expert], 1);
        }
        for (int rank = 0; rank < top_k; ++rank) {
            route_weights[row * top_k + rank] =
                route_scale *
                scores[row * experts + selected[rank]] / normalizer;
        }
    }
}

__global__ void build_dispatch_map_kernel(
    int* expert_offsets,
    int* route_to_slot,
    int* slot_to_route,
    int* slot_expert,
    const int* expert_indices,
    const int* expert_counts,
    int routes,
    int experts) {
    extern __shared__ int shared[];
    int* inclusive_counts = shared;
    int* next_slot = shared + experts;

    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        inclusive_counts[expert] = expert_counts[expert];
    }
    __syncthreads();
    for (int stride = 1; stride < experts; stride <<= 1) {
        int previous = 0;
        if (threadIdx.x < experts && threadIdx.x >= stride) {
            previous = inclusive_counts[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x < experts) {
            inclusive_counts[threadIdx.x] += previous;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        expert_offsets[0] = 0;
    }
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        expert_offsets[expert + 1] = inclusive_counts[expert];
    }
    __syncthreads();

    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        next_slot[expert] = expert_offsets[expert];
    }
    __syncthreads();

    for (int route = threadIdx.x; route < routes; route += blockDim.x) {
        const int expert = expert_indices[route];
        const int slot = atomicAdd(next_slot + expert, 1);
        route_to_slot[route] = slot;
        slot_to_route[slot] = route;
        slot_expert[slot] = expert;
    }
}

__global__ void dispatch_copy_kernel(
    float* dispatched_input,
    const float* input,
    const int* slot_to_route,
    int routes,
    int hidden_size,
    int top_k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = routes * hidden_size;
    if (index >= elements) {
        return;
    }
    const int slot = index / hidden_size;
    const int column = index % hidden_size;
    const int token = slot_to_route[slot] / top_k;
    dispatched_input[index] = input[token * hidden_size + column];
}

__global__ void combine_forward_kernel(
    float* output,
    const float* shared_output,
    const float* dispatched_output,
    const float* route_weights,
    const int* route_to_slot,
    int elements,
    int hidden_size,
    int top_k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = index / hidden_size;
    const int column = index % hidden_size;
    float result = shared_output[index];
    for (int rank = 0; rank < top_k; ++rank) {
        const int route = row * top_k + rank;
        const int slot = route_to_slot[route];
        result += route_weights[route] *
                  dispatched_output[slot * hidden_size + column];
    }
    output[index] = result;
}

__global__ void combine_output_backward_kernel(
    float* dispatched_output_gradient,
    float* shared_output_gradient,
    const float* output_gradient,
    const float* route_weights,
    const int* route_to_slot,
    int elements,
    int hidden_size,
    int top_k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = index / hidden_size;
    const int column = index % hidden_size;
    const float gradient = output_gradient[index];
    shared_output_gradient[index] = gradient;
    for (int rank = 0; rank < top_k; ++rank) {
        const int route = row * top_k + rank;
        const int slot = route_to_slot[route];
        dispatched_output_gradient[slot * hidden_size + column] =
            route_weights[route] * gradient;
    }
}

template <int WIDTH = 32>
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = WIDTH / 2; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, mask);
    }
    return value;
}

__global__ void route_weight_backward_kernel(
    float* route_weight_gradient,
    const float* output_gradient,
    const float* dispatched_output,
    const int* route_to_slot,
    int hidden_size,
    int top_k) {
    const int route = blockIdx.x;
    const int token = route / top_k;
    const int slot = route_to_slot[route];
    float sum = 0.0F;
    for (int column = threadIdx.x; column < hidden_size; column += 32) {
        sum += output_gradient[token * hidden_size + column] *
               dispatched_output[slot * hidden_size + column];
    }
    sum = warp_sum(sum);
    if (threadIdx.x == 0) {
        route_weight_gradient[route] = sum;
    }
}

__global__ void unroute_backward_kernel(
    float* input_gradient,
    const float* dispatched_input_gradient,
    const int* route_to_slot,
    int elements,
    int hidden_size,
    int top_k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = index / hidden_size;
    const int column = index % hidden_size;
    float sum = 0.0F;
    for (int rank = 0; rank < top_k; ++rank) {
        const int slot = route_to_slot[row * top_k + rank];
        sum += dispatched_input_gradient[slot * hidden_size + column];
    }
    input_gradient[index] += sum;
}

__global__ void route_backward_kernel(
    float* router_logit_gradient,
    const float* route_weight_gradient,
    const float* scores,
    const int* expert_indices,
    int experts,
    int top_k,
    float route_scale) {
    const int row = blockIdx.x;
    if (threadIdx.x == 0) {
        float normalizer = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) {
            normalizer += scores[
                row * experts + expert_indices[row * top_k + rank]];
        }
        float projected_gradient = 0.0F;
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = expert_indices[row * top_k + rank];
            const float probability =
                scores[row * experts + expert] / normalizer;
            projected_gradient +=
                route_weight_gradient[row * top_k + rank] * probability;
        }
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = expert_indices[row * top_k + rank];
            const float score = scores[row * experts + expert];
            const float score_gradient = route_scale / normalizer *
                (route_weight_gradient[row * top_k + rank]
                 - projected_gradient);
            router_logit_gradient[row * experts + expert] =
                score_gradient * score * (1.0F - score);
        }
    }
}

__global__ void update_routing_bias_kernel(
    float* routing_bias,
    const int* expert_counts,
    int experts,
    int total_routes,
    float update_speed) {
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= experts) {
        return;
    }
    const float target = static_cast<float>(total_routes) / experts;
    const float load = static_cast<float>(expert_counts[expert]);
    routing_bias[expert] +=
        load < target ? update_speed : (load > target ? -update_speed : 0.0F);
}

}  // namespace

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
    cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(
        expert_counts, 0, experts * sizeof(int), stream));
    route_forward_kernel<<<rows, BLOCK_SIZE, 0, stream>>>(
        scores,
        expert_indices,
        route_weights,
        expert_counts,
        router_logits,
        routing_bias,
        experts,
        top_k,
        route_scale);
    CUDA_CHECK(cudaGetLastError());
}

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
    cudaStream_t stream) {
    const int routes = rows * top_k;
    build_dispatch_map_kernel<<<
        1,
        BLOCK_SIZE,
        (2 * experts + 1) * sizeof(int),
        stream>>>(
        expert_offsets,
        route_to_slot,
        slot_to_route,
        slot_expert,
        expert_indices,
        expert_counts,
        routes,
        experts);
    CUDA_CHECK(cudaGetLastError());
    const int elements = routes * hidden_size;
    dispatch_copy_kernel<<<
        (elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
        dispatched_input, input, slot_to_route, routes, hidden_size, top_k);
    CUDA_CHECK(cudaGetLastError());
}

void expert_combine_forward_sm89_cuda(
    float* output,
    const float* shared_output,
    const float* dispatched_output,
    const float* route_weights,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream) {
    const int elements = rows * hidden_size;
    combine_forward_kernel<<<
        (elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
        output,
        shared_output,
        dispatched_output,
        route_weights,
        route_to_slot,
        elements,
        hidden_size,
        top_k);
    CUDA_CHECK(cudaGetLastError());
}

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
    cudaStream_t stream) {
    const int elements = rows * hidden_size;
    combine_output_backward_kernel<<<
        (elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
        dispatched_output_gradient,
        shared_output_gradient,
        output_gradient,
        route_weights,
        route_to_slot,
        elements,
        hidden_size,
        top_k);
    CUDA_CHECK(cudaGetLastError());
    route_weight_backward_kernel<<<rows * top_k, 32, 0, stream>>>(
        route_weight_gradient,
        output_gradient,
        dispatched_output,
        route_to_slot,
        hidden_size,
        top_k);
    CUDA_CHECK(cudaGetLastError());
}

void expert_unroute_backward_sm89_cuda(
    float* input_gradient,
    const float* dispatched_input_gradient,
    const int* route_to_slot,
    int rows,
    int hidden_size,
    int top_k,
    cudaStream_t stream) {
    const int elements = rows * hidden_size;
    unroute_backward_kernel<<<
        (elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
        input_gradient,
        dispatched_input_gradient,
        route_to_slot,
        elements,
        hidden_size,
        top_k);
    CUDA_CHECK(cudaGetLastError());
}

void expert_route_backward_sm89_cuda(
    float* router_logit_gradient,
    const float* route_weight_gradient,
    const float* scores,
    const int* expert_indices,
    int rows,
    int experts,
    int top_k,
    float route_scale,
    cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(
        router_logit_gradient,
        0,
        static_cast<std::size_t>(rows) * experts * sizeof(float),
        stream));
    route_backward_kernel<<<rows, 1, 0, stream>>>(
        router_logit_gradient,
        route_weight_gradient,
        scores,
        expert_indices,
        experts,
        top_k,
        route_scale);
    CUDA_CHECK(cudaGetLastError());
}

void update_routing_bias_sm89_cuda(
    float* routing_bias,
    const int* expert_counts,
    int experts,
    int total_routes,
    float update_speed,
    cudaStream_t stream) {
    const int blocks = (experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_routing_bias_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        routing_bias, expert_counts, experts, total_routes, update_speed);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

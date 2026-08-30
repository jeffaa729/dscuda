// Implements DeepSeek-V3 sigmoid top-k routing, parallel no-drop token dispatch, grouped expert linear algebra, and weighted combination.
// Its reverse kernels preserve the discrete routing decision while differentiating expert outputs and normalized selected affinities.

#include "cuda_common.h"
#include "expert_dispatch.h"

#include <cmath>
#include <mma.h>

namespace dscuda {
namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_TOP_K = 8;
constexpr int TILE = 16;
constexpr int WARP_SIZE = 32;
constexpr int GROUPED_BM = 64;
constexpr int GROUPED_BN = 64;
constexpr int GROUPED_WARPS = 8;

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

__global__ void grouped_linear_forward_kernel(
    float* output,
    const float* input,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size) {
    const int row = blockIdx.x * TILE + threadIdx.y;
    const int column = blockIdx.y * TILE + threadIdx.x;
    if (row >= dispatched_rows || column >= output_size) {
        return;
    }
    const int expert = slot_expert[row];
    const float* expert_weight =
        weight + static_cast<std::size_t>(expert) * input_size * output_size;
    float sum = 0.0F;
    for (int inner = 0; inner < input_size; ++inner) {
        sum += input[row * input_size + inner] *
               expert_weight[inner * output_size + column];
    }
    output[row * output_size + column] = sum;
}

__global__ void grouped_linear_bf16_tensor_core_kernel(
    float* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const int* expert_offsets,
    int packed_row_blocks,
    int experts,
    int output_size,
    int input_size) {
    const int packed_block = blockIdx.x;
    if (packed_block >= packed_row_blocks) {
        return;
    }

    int expert = -1;
    int expert_block = 0;
    int first_packed_block = 0;
    for (int current = 0; current < experts; ++current) {
        const int rows = expert_offsets[current + 1] - expert_offsets[current];
        const int blocks = (rows + GROUPED_BM - 1) / GROUPED_BM;
        if (packed_block < first_packed_block + blocks) {
            expert = current;
            expert_block = packed_block - first_packed_block;
            break;
        }
        first_packed_block += blocks;
    }
    if (expert < 0) {
        return;
    }

    __shared__ __align__(16) __nv_bfloat16
        shared_left[GROUPED_BM * TILE];
    __shared__ __align__(16) __nv_bfloat16
        shared_right[TILE * GROUPED_BN];
    __shared__ __align__(16) float
        shared_output[GROUPED_WARPS * 2 * TILE * TILE];

    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,
                   __nv_bfloat16, wmma::row_major> left_fragment_0;
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,
                   __nv_bfloat16, wmma::row_major> left_fragment_1;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                   __nv_bfloat16, wmma::row_major> right_fragment;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> accumulator_0;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> accumulator_1;
    wmma::fill_fragment(accumulator_0, 0.0F);
    wmma::fill_fragment(accumulator_1, 0.0F);

    const int expert_first_row = expert_offsets[expert];
    const int expert_rows = expert_offsets[expert + 1] - expert_first_row;
    const int first_row = expert_block * GROUPED_BM;
    const int first_column = blockIdx.y * GROUPED_BN;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warp_row = (warp / 4) * 2 * TILE;
    const int warp_column = (warp % 4) * TILE;

    for (int first_inner = 0; first_inner < input_size; first_inner += TILE) {
        for (int element = threadIdx.x; element < GROUPED_BM * TILE;
             element += BLOCK_SIZE) {
            const int row = element / TILE;
            const int column = element % TILE;
            shared_left[element] =
                first_row + row < expert_rows
                    && first_inner + column < input_size
                ? input[static_cast<std::size_t>(
                            expert_first_row + first_row + row)
                        * input_size
                    + first_inner + column]
                : __float2bfloat16(0.0F);
        }
        for (int element = threadIdx.x; element < TILE * GROUPED_BN;
             element += BLOCK_SIZE) {
            const int row = element / GROUPED_BN;
            const int column = element % GROUPED_BN;
            shared_right[element] =
                first_inner + row < input_size
                    && first_column + column < output_size
                ? weight[(static_cast<std::size_t>(expert) * input_size
                          + first_inner + row)
                         * output_size
                    + first_column + column]
                : __float2bfloat16(0.0F);
        }
        __syncthreads();
        wmma::load_matrix_sync(
            left_fragment_0,
            shared_left + warp_row * TILE,
            TILE);
        wmma::load_matrix_sync(
            left_fragment_1,
            shared_left + (warp_row + TILE) * TILE,
            TILE);
        wmma::load_matrix_sync(
            right_fragment,
            shared_right + warp_column,
            GROUPED_BN);
        wmma::mma_sync(
            accumulator_0,
            left_fragment_0,
            right_fragment,
            accumulator_0);
        wmma::mma_sync(
            accumulator_1,
            left_fragment_1,
            right_fragment,
            accumulator_1);
        __syncthreads();
    }

    float* warp_output =
        shared_output + warp * 2 * TILE * TILE;
    wmma::store_matrix_sync(
        warp_output,
        accumulator_0,
        TILE,
        wmma::mem_row_major);
    wmma::store_matrix_sync(
        warp_output + TILE * TILE,
        accumulator_1,
        TILE,
        wmma::mem_row_major);
    __syncwarp();
    const int lane = threadIdx.x % WARP_SIZE;
    for (int element = lane; element < 2 * TILE * TILE;
         element += WARP_SIZE) {
        const int fragment = element / (TILE * TILE);
        const int fragment_element = element % (TILE * TILE);
        const int row = warp_row + fragment * TILE
            + fragment_element / TILE;
        const int column = fragment_element % TILE;
        if (first_row + row < expert_rows
            && first_column + warp_column + column < output_size) {
            output[static_cast<std::size_t>(
                       expert_first_row + first_row + row)
                       * output_size
                   + first_column + warp_column + column] =
                warp_output[element];
        }
    }
}

__global__ void grouped_linear_input_backward_kernel(
    float* input_gradient,
    const float* output_gradient,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size,
    bool accumulate) {
    const int row = blockIdx.x * TILE + threadIdx.y;
    const int inner = blockIdx.y * TILE + threadIdx.x;
    if (row >= dispatched_rows || inner >= input_size) {
        return;
    }
    const int expert = slot_expert[row];
    const float* expert_weight =
        weight + static_cast<std::size_t>(expert) * input_size * output_size;
    float sum = 0.0F;
    for (int column = 0; column < output_size; ++column) {
        sum += output_gradient[row * output_size + column] *
               expert_weight[inner * output_size + column];
    }
    if (accumulate) {
        input_gradient[row * input_size + inner] += sum;
    } else {
        input_gradient[row * input_size + inner] = sum;
    }
}

__global__ void grouped_linear_weight_backward_kernel(
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const int* expert_offsets,
    int experts,
    int output_size,
    int input_size) {
    const int inner = blockIdx.x * TILE + threadIdx.y;
    const int column = blockIdx.y * TILE + threadIdx.x;
    const int expert = blockIdx.z;
    if (expert >= experts || inner >= input_size || column >= output_size) {
        return;
    }
    float sum = 0.0F;
    for (int row = expert_offsets[expert];
         row < expert_offsets[expert + 1];
         ++row) {
        sum += input[row * input_size + inner] *
               output_gradient[row * output_size + column];
    }
    weight_gradient[
        (static_cast<std::size_t>(expert) * input_size + inner) * output_size
        + column] += sum;
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

void grouped_linear_forward_cuda(
    float* output,
    const float* input,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size,
    cudaStream_t stream) {
    const dim3 block(TILE, TILE);
    const dim3 grid(
        (dispatched_rows + TILE - 1) / TILE,
        (output_size + TILE - 1) / TILE);
    grouped_linear_forward_kernel<<<grid, block, 0, stream>>>(
        output,
        input,
        weight,
        slot_expert,
        dispatched_rows,
        output_size,
        input_size);
    CUDA_CHECK(cudaGetLastError());
}

void grouped_linear_bf16_forward_cuda(
    float* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const int* expert_offsets,
    int dispatched_rows,
    int experts,
    int output_size,
    int input_size,
    cudaStream_t stream) {
    const int packed_row_blocks =
        (dispatched_rows + GROUPED_BM - 1) / GROUPED_BM + experts;
    const dim3 grid(
        packed_row_blocks,
        (output_size + GROUPED_BN - 1) / GROUPED_BN);
    grouped_linear_bf16_tensor_core_kernel<<<
        grid,
        GROUPED_WARPS * WARP_SIZE,
        0,
        stream>>>(
        output,
        input,
        weight,
        expert_offsets,
        packed_row_blocks,
        experts,
        output_size,
        input_size);
    CUDA_CHECK(cudaGetLastError());
}

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
    cudaStream_t stream) {
    const dim3 block(TILE, TILE);
    const dim3 input_grid(
        (dispatched_rows + TILE - 1) / TILE,
        (input_size + TILE - 1) / TILE);
    grouped_linear_input_backward_kernel<<<input_grid, block, 0, stream>>>(
        input_gradient,
        output_gradient,
        weight,
        slot_expert,
        dispatched_rows,
        output_size,
        input_size,
        accumulate_input);
    CUDA_CHECK(cudaGetLastError());
    const dim3 weight_grid(
        (input_size + TILE - 1) / TILE,
        (output_size + TILE - 1) / TILE,
        experts);
    grouped_linear_weight_backward_kernel<<<weight_grid, block, 0, stream>>>(
        weight_gradient,
        output_gradient,
        input,
        expert_offsets,
        experts,
        output_size,
        input_size);
    CUDA_CHECK(cudaGetLastError());
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

void expert_unroute_backward_cuda(
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

void update_routing_bias_cuda(
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

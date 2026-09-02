// Computes variable-row expert GEMMs with FP32 scalar and BF16 Tensor Core paths.
// The backward kernels compute input and weight gradients using the same expert-grouped layout.

#include "cuda_common.h"
#include "common.cuh"

#include <mma.h>

namespace dscuda {
namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int TILE = 16;
constexpr int WARP_SIZE = 32;
constexpr int GROUPED_BM = 64;
constexpr int GROUPED_BN = 64;
constexpr int GROUPED_WARPS = 8;

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

}  // namespace

void grouped_linear_forward_sm89_cuda(
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

void grouped_linear_bf16_forward_sm89_cuda(
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

void grouped_linear_backward_sm89_cuda(
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

}  // namespace dscuda

// Implements dense linear projections with native FP32 CUDA Core and BF16 Tensor Core matrix multiplication.
// Compile-time transpose modes reuse both kernels for the forward pass and the two analytical gradients.

#include "cuda_common.h"
#include "matmul.h"

#include <cuda_bf16.h>
#include <mma.h>

namespace dscuda {
namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int WM = 64;
constexpr int WN = 32;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int WARPS_PER_ROW = BN / WN;
constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * 32;
constexpr int VECTOR_WIDTH = 4;

namespace tc {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int WM = 64;
constexpr int WN = 32;
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;
constexpr int WARP_TILES_M = WM / MMA_M;
constexpr int WARP_TILES_N = WN / MMA_N;
constexpr int WARPS_N = BN / WN;
constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * 32;
constexpr int SKEW = 16;

}  // namespace tc

static_assert(WM == 8 * TM);
static_assert(WN == 4 * TN);
static_assert(BK % VECTOR_WIDTH == 0);
static_assert(BK % 2 == 0);

__device__ __forceinline__ float4 load_float4(
    const float* input,
    int index,
    bool major_valid,
    int start,
    int length) {
    int width = 0;
    if (major_valid && start < length) {
        const int remaining = length - start;
        width = remaining < VECTOR_WIDTH ? remaining : VECTOR_WIDTH;
    }
    if (width == VECTOR_WIDTH && index % VECTOR_WIDTH == 0) {
        return *reinterpret_cast<const float4*>(input + index);
    }

    float4 value = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (width > 0) {
        value.x = input[index];
    }
    if (width > 1) {
        value.y = input[index + 1];
    }
    if (width > 2) {
        value.z = input[index + 2];
    }
    return value;
}

template <bool kTransposeLeft, bool kTransposeRight, bool kAccumulate>
__global__ void matmul_kernel(
    float* output,
    const float* left,
    const float* right,
    int output_rows,
    int output_columns,
    int inner_size,
    int left_stride,
    int right_stride,
    int output_stride) {
    // Logical layouts are shared_left[stage][inner][row] and
    // shared_right[stage][inner][column].
    __shared__ float shared_left[2 * BK * BM];
    __shared__ float shared_right[2 * BK * BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_row = warp_id / WARPS_PER_ROW;
    const int warp_column = warp_id % WARPS_PER_ROW;
    const int lane_row = lane_id / 4;
    const int lane_column = lane_id % 4;
    const int local_row = warp_row * WM + lane_row * TM;
    const int local_column = warp_column * WN + lane_column * TN;
    const int block_row = blockIdx.y * BM;
    const int block_column = blockIdx.x * BN;

    float accumulator[TM][TN] = {};
    float left_fragment[2][TM];
    float right_fragment[2][TN];
    int write_stage = 0;

    // Each iteration first issues the global loads for the next tile, computes
    // the preceding tile, and then publishes the loaded values to shared memory.
    for (int tile_to_load = 0;; tile_to_load += BK) {
        const bool load_tile = tile_to_load < inner_size;
        const bool compute_tile = tile_to_load > 0;
        float4 loaded_left = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        float4 loaded_right = make_float4(0.0F, 0.0F, 0.0F, 0.0F);

        if (load_tile) {
            if constexpr (!kTransposeLeft) {
                const int vectors_per_row = BK / VECTOR_WIDTH;
                const int tile_row = tid / vectors_per_row;
                const int tile_inner = (tid % vectors_per_row) * VECTOR_WIDTH;
                const int global_row = block_row + tile_row;
                const int global_inner = tile_to_load + tile_inner;
                loaded_left = load_float4(
                    left,
                    global_row * left_stride + global_inner,
                    global_row < output_rows,
                    global_inner,
                    inner_size);
            } else {
                const int vectors_per_inner = BM / VECTOR_WIDTH;
                const int tile_inner = tid / vectors_per_inner;
                const int tile_row = (tid % vectors_per_inner) * VECTOR_WIDTH;
                const int global_inner = tile_to_load + tile_inner;
                const int global_row = block_row + tile_row;
                loaded_left = load_float4(
                    left,
                    global_inner * left_stride + global_row,
                    global_inner < inner_size,
                    global_row,
                    output_rows);
            }

            if constexpr (!kTransposeRight) {
                const int vectors_per_inner = BN / VECTOR_WIDTH;
                const int tile_inner = tid / vectors_per_inner;
                const int tile_column =
                    (tid % vectors_per_inner) * VECTOR_WIDTH;
                const int global_inner = tile_to_load + tile_inner;
                const int global_column = block_column + tile_column;
                loaded_right = load_float4(
                    right,
                    global_inner * right_stride + global_column,
                    global_inner < inner_size,
                    global_column,
                    output_columns);
            } else {
                const int vectors_per_column = BK / VECTOR_WIDTH;
                const int tile_column = tid / vectors_per_column;
                const int tile_inner =
                    (tid % vectors_per_column) * VECTOR_WIDTH;
                const int global_column = block_column + tile_column;
                const int global_inner = tile_to_load + tile_inner;
                loaded_right = load_float4(
                    right,
                    global_column * right_stride + global_inner,
                    global_column < output_columns,
                    global_inner,
                    inner_size);
            }
        }

        const int read_stage = write_stage ^ 1;
        if (compute_tile) {
#pragma unroll
            for (int inner = 0; inner < BK - 1; ++inner) {
                const int read_fragment = inner & 1;
                const int write_fragment = (inner + 1) & 1;

#pragma unroll
                for (int row = 0; row < TM; ++row) {
                    left_fragment[write_fragment][row] = shared_left[
                        (read_stage * BK + inner + 1) * BM +
                        local_row + row];
                }
#pragma unroll
                for (int column = 0; column < TN; ++column) {
                    right_fragment[write_fragment][column] = shared_right[
                        (read_stage * BK + inner + 1) * BN +
                        local_column + column];
                }

#pragma unroll
                for (int row = 0; row < TM; ++row) {
#pragma unroll
                    for (int column = 0; column < TN; ++column) {
                        accumulator[row][column] +=
                            left_fragment[read_fragment][row] *
                            right_fragment[read_fragment][column];
                    }
                }
            }
        }

        if (load_tile) {
            if constexpr (!kTransposeLeft) {
                const int vectors_per_row = BK / VECTOR_WIDTH;
                const int tile_row = tid / vectors_per_row;
                const int tile_inner = (tid % vectors_per_row) * VECTOR_WIDTH;
                shared_left[
                    (write_stage * BK + tile_inner + 0) * BM +
                    tile_row] = loaded_left.x;
                shared_left[
                    (write_stage * BK + tile_inner + 1) * BM +
                    tile_row] = loaded_left.y;
                shared_left[
                    (write_stage * BK + tile_inner + 2) * BM +
                    tile_row] = loaded_left.z;
                shared_left[
                    (write_stage * BK + tile_inner + 3) * BM +
                    tile_row] = loaded_left.w;
            } else {
                const int vectors_per_inner = BM / VECTOR_WIDTH;
                const int tile_inner = tid / vectors_per_inner;
                const int tile_row = (tid % vectors_per_inner) * VECTOR_WIDTH;
                const int index =
                    (write_stage * BK + tile_inner) * BM +
                    tile_row;
                shared_left[index + 0] = loaded_left.x;
                shared_left[index + 1] = loaded_left.y;
                shared_left[index + 2] = loaded_left.z;
                shared_left[index + 3] = loaded_left.w;
            }

            if constexpr (!kTransposeRight) {
                const int vectors_per_inner = BN / VECTOR_WIDTH;
                const int tile_inner = tid / vectors_per_inner;
                const int tile_column =
                    (tid % vectors_per_inner) * VECTOR_WIDTH;
                const int index =
                    (write_stage * BK + tile_inner) * BN +
                    tile_column;
                shared_right[index + 0] = loaded_right.x;
                shared_right[index + 1] = loaded_right.y;
                shared_right[index + 2] = loaded_right.z;
                shared_right[index + 3] = loaded_right.w;
            } else {
                const int vectors_per_column = BK / VECTOR_WIDTH;
                const int tile_column = tid / vectors_per_column;
                const int tile_inner =
                    (tid % vectors_per_column) * VECTOR_WIDTH;
                shared_right[
                    (write_stage * BK + tile_inner + 0) * BN +
                    tile_column] = loaded_right.x;
                shared_right[
                    (write_stage * BK + tile_inner + 1) * BN +
                    tile_column] = loaded_right.y;
                shared_right[
                    (write_stage * BK + tile_inner + 2) * BN +
                    tile_column] = loaded_right.z;
                shared_right[
                    (write_stage * BK + tile_inner + 3) * BN +
                    tile_column] = loaded_right.w;
            }

            __syncthreads();
            const int loaded_stage = write_stage;
            write_stage ^= 1;

#pragma unroll
            for (int row = 0; row < TM; ++row) {
                left_fragment[0][row] = shared_left[
                    loaded_stage * BK * BM + local_row + row];
            }
#pragma unroll
            for (int column = 0; column < TN; ++column) {
                right_fragment[0][column] = shared_right[
                    loaded_stage * BK * BN + local_column + column];
            }
        }

        if (compute_tile) {
#pragma unroll
            for (int row = 0; row < TM; ++row) {
#pragma unroll
                for (int column = 0; column < TN; ++column) {
                    accumulator[row][column] +=
                        left_fragment[1][row] * right_fragment[1][column];
                }
            }
        }

        if (!load_tile) {
            break;
        }
    }

#pragma unroll
    for (int row = 0; row < TM; ++row) {
        const int global_row = block_row + local_row + row;
#pragma unroll
        for (int vector = 0; vector < TN / VECTOR_WIDTH; ++vector) {
            const int thread_column = vector * VECTOR_WIDTH;
            const int global_column =
                block_column + local_column + thread_column;
            int width = 0;
            if (global_row < output_rows && global_column < output_columns) {
                const int remaining = output_columns - global_column;
                width = remaining < VECTOR_WIDTH ? remaining : VECTOR_WIDTH;
            }
            const int index = global_row * output_stride + global_column;
            float4 value = make_float4(
                accumulator[row][thread_column + 0],
                accumulator[row][thread_column + 1],
                accumulator[row][thread_column + 2],
                accumulator[row][thread_column + 3]);

            if (width == VECTOR_WIDTH && index % VECTOR_WIDTH == 0) {
                if constexpr (kAccumulate) {
                    const float4 previous =
                        *reinterpret_cast<const float4*>(output + index);
                    value.x += previous.x;
                    value.y += previous.y;
                    value.z += previous.z;
                    value.w += previous.w;
                }
                *reinterpret_cast<float4*>(output + index) = value;
            } else {
                if (width > 0) {
                    if constexpr (kAccumulate) {
                        output[index] += value.x;
                    } else {
                        output[index] = value.x;
                    }
                }
                if (width > 1) {
                    if constexpr (kAccumulate) {
                        output[index + 1] += value.y;
                    } else {
                        output[index + 1] = value.y;
                    }
                }
                if (width > 2) {
                    if constexpr (kAccumulate) {
                        output[index + 2] += value.z;
                    } else {
                        output[index + 2] = value.z;
                    }
                }
            }
        }
    }
}

template <bool kTransposeLeft, bool kTransposeRight, bool kAccumulate>
__global__ void matmul_tensor_core_kernel(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int output_rows,
    int output_columns,
    int inner_size,
    int left_stride,
    int right_stride,
    int output_stride) {
    __shared__ __nv_bfloat16 shared_left[tc::BK][tc::BM + tc::SKEW];
    __shared__ __nv_bfloat16 shared_right[tc::BK][tc::BN + tc::SKEW];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / tc::WARPS_N;
    const int warp_column = warp_id % tc::WARPS_N;
    const int block_row = blockIdx.y * tc::BM;
    const int block_column = blockIdx.x * tc::BN;

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        tc::MMA_M,
        tc::MMA_N,
        tc::MMA_K,
        __nv_bfloat16,
        nvcuda::wmma::col_major>
        left_fragments[tc::WARP_TILES_M];
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        tc::MMA_M,
        tc::MMA_N,
        tc::MMA_K,
        __nv_bfloat16,
        nvcuda::wmma::row_major>
        right_fragments[tc::WARP_TILES_N];
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        tc::MMA_M,
        tc::MMA_N,
        tc::MMA_K,
        float>
        accumulators[tc::WARP_TILES_M][tc::WARP_TILES_N];

#pragma unroll
    for (int tile_row = 0; tile_row < tc::WARP_TILES_M; ++tile_row) {
#pragma unroll
        for (int tile_column = 0; tile_column < tc::WARP_TILES_N; ++tile_column) {
            const int output_row =
                block_row + warp_row * tc::WM + tile_row * tc::MMA_M;
            const int output_column =
                block_column + warp_column * tc::WN + tile_column * tc::MMA_N;
            if constexpr (kAccumulate) {
                if (output_row + tc::MMA_M <= output_rows &&
                    output_column + tc::MMA_N <= output_columns) {
                    nvcuda::wmma::load_matrix_sync(
                        accumulators[tile_row][tile_column],
                        output + output_row * output_stride + output_column,
                        output_stride,
                        nvcuda::wmma::mem_row_major);
                } else {
                    nvcuda::wmma::fill_fragment(
                        accumulators[tile_row][tile_column], 0.0F);
                }
            } else {
                nvcuda::wmma::fill_fragment(
                    accumulators[tile_row][tile_column], 0.0F);
            }
        }
    }

    for (int tile_inner = 0; tile_inner < inner_size; tile_inner += tc::BK) {
        for (int index = tid; index < tc::BM * tc::BK; index += tc::NUM_THREADS) {
            const int local_row = index / tc::BK;
            const int local_inner = index % tc::BK;
            const int global_row = block_row + local_row;
            const int global_inner = tile_inner + local_inner;
            __nv_bfloat16 value = __float2bfloat16(0.0F);
            if (global_row < output_rows && global_inner < inner_size) {
                if constexpr (kTransposeLeft) {
                    value = left[global_inner * left_stride + global_row];
                } else {
                    value = left[global_row * left_stride + global_inner];
                }
            }
            shared_left[local_inner][local_row] = value;
        }

        for (int index = tid; index < tc::BK * tc::BN; index += tc::NUM_THREADS) {
            const int local_inner = index / tc::BN;
            const int local_column = index % tc::BN;
            const int global_inner = tile_inner + local_inner;
            const int global_column = block_column + local_column;
            __nv_bfloat16 value = __float2bfloat16(0.0F);
            if (global_inner < inner_size && global_column < output_columns) {
                if constexpr (kTransposeRight) {
                    value = right[global_column * right_stride + global_inner];
                } else {
                    value = right[global_inner * right_stride + global_column];
                }
            }
            shared_right[local_inner][local_column] = value;
        }
        __syncthreads();

#pragma unroll
        for (int tile_row = 0; tile_row < tc::WARP_TILES_M; ++tile_row) {
            const int shared_row =
                warp_row * tc::WM + tile_row * tc::MMA_M;
            nvcuda::wmma::load_matrix_sync(
                left_fragments[tile_row],
                &shared_left[0][shared_row],
                tc::BM + tc::SKEW);
        }
#pragma unroll
        for (int tile_column = 0; tile_column < tc::WARP_TILES_N; ++tile_column) {
            const int shared_column =
                warp_column * tc::WN + tile_column * tc::MMA_N;
            nvcuda::wmma::load_matrix_sync(
                right_fragments[tile_column],
                &shared_right[0][shared_column],
                tc::BN + tc::SKEW);
        }

#pragma unroll
        for (int tile_row = 0; tile_row < tc::WARP_TILES_M; ++tile_row) {
#pragma unroll
            for (int tile_column = 0; tile_column < tc::WARP_TILES_N;
                 ++tile_column) {
                nvcuda::wmma::mma_sync(
                    accumulators[tile_row][tile_column],
                    left_fragments[tile_row],
                    right_fragments[tile_column],
                    accumulators[tile_row][tile_column]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int tile_row = 0; tile_row < tc::WARP_TILES_M; ++tile_row) {
#pragma unroll
        for (int tile_column = 0; tile_column < tc::WARP_TILES_N; ++tile_column) {
            const int output_row =
                block_row + warp_row * tc::WM + tile_row * tc::MMA_M;
            const int output_column =
                block_column + warp_column * tc::WN + tile_column * tc::MMA_N;
            if (output_row + tc::MMA_M <= output_rows &&
                output_column + tc::MMA_N <= output_columns) {
                nvcuda::wmma::store_matrix_sync(
                    output + output_row * output_stride + output_column,
                    accumulators[tile_row][tile_column],
                    output_stride,
                    nvcuda::wmma::mem_row_major);
            }
        }
    }
}

template <bool kTransposeLeft, bool kTransposeRight, bool kAccumulate>
void launch_matmul(
    float* output,
    const float* left,
    const float* right,
    int output_rows,
    int output_columns,
    int inner_size,
    int left_stride,
    int right_stride,
    int output_stride,
    cudaStream_t stream) {
    const dim3 blocks(
        (output_columns + BN - 1) / BN,
        (output_rows + BM - 1) / BM);
    matmul_kernel<kTransposeLeft, kTransposeRight, kAccumulate>
        <<<blocks, NUM_THREADS, 0, stream>>>(
            output,
            left,
            right,
            output_rows,
            output_columns,
            inner_size,
            left_stride,
            right_stride,
            output_stride);
    CUDA_CHECK(cudaGetLastError());
}

template <bool kTransposeLeft, bool kTransposeRight, bool kAccumulate>
void launch_tensor_core_matmul(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int output_rows,
    int output_columns,
    int inner_size,
    int left_stride,
    int right_stride,
    int output_stride,
    cudaStream_t stream) {
    const dim3 blocks(
        (output_columns + tc::BN - 1) / tc::BN,
        (output_rows + tc::BM - 1) / tc::BM);
    matmul_tensor_core_kernel<kTransposeLeft, kTransposeRight, kAccumulate>
        <<<blocks, tc::NUM_THREADS, 0, stream>>>(
            output,
            left,
            right,
            output_rows,
            output_columns,
            inner_size,
            left_stride,
            right_stride,
            output_stride);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void matmul_fp32_forward_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_matmul<false, false, false>(
        output, left, right, M, N, K, K, N, N, stream);
}

void matmul_fp32_backward_cuda(
    float* left_gradient,
    float* right_gradient,
    const float* output_gradient,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_matmul<false, true, true>(
        left_gradient, output_gradient, right, M, K, N, N, N, K, stream);
    launch_matmul<true, false, true>(
        right_gradient, left, output_gradient, K, N, M, K, N, N, stream);
}

void matmul_bf16_forward_cuda(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_tensor_core_matmul<false, false, false>(
        output, left, right, M, N, K, K, N, N, stream);
}

void matmul_bf16_backward_cuda(
    float* left_gradient,
    float* right_gradient,
    const __nv_bfloat16* output_gradient,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_tensor_core_matmul<false, true, true>(
        left_gradient, output_gradient, right, M, K, N, N, N, K, stream);
    launch_tensor_core_matmul<true, false, true>(
        right_gradient, left, output_gradient, K, N, M, K, N, N, stream);
}

}  // namespace dscuda

// Implements row-major C[M,N] = A[M,K] * B[K,N] with FP32 CUDA Core and BF16 Tensor Core kernels.

#include "cuda_common.h"
#include "common.cuh"

#include <cuda_bf16.h>
#include <mma.h>

namespace dscuda {
namespace {

constexpr int BM = 128;
constexpr int BN = 128;
// Small GEMMs use more blocks and half-sized thread microtiles to expose
// enough warps while reducing accumulator register pressure.
constexpr int SMALL_BM = 64;
constexpr int SMALL_BN = 64;
constexpr int SMALL_WN = 16;
constexpr int SMALL_TN = 4;
constexpr int BK = 8;
constexpr int WM = 64;
constexpr int WN = 32;
constexpr int TM = 8;
constexpr int TN = 8;
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

namespace tc_mma {

constexpr int BK = 32;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int STAGES = 2;
constexpr int VECTOR_ELEMENTS = 8;

}  // namespace tc_mma

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
    if (width > 3) {
        value.w = input[index + 3];
    }
    return value;
}

template <
    int kBM,
    int kBN,
    int kWM,
    int kWN,
    int kTM,
    int kTN>
__global__ void matmul_kernel(
    float* __restrict__ output,
    const float* __restrict__ left,
    const float* __restrict__ right,
    int output_rows,
    int output_columns,
    int inner_size) {
    constexpr int kWarpsPerRow = kBN / kWN;
    constexpr int kNumThreads = (kBM / kWM) * (kBN / kWN) * 32;
    constexpr int kLeftLoads = (kBM * BK / VECTOR_WIDTH) / kNumThreads;
    constexpr int kRightLoads = (BK * kBN / VECTOR_WIDTH) / kNumThreads;
    static_assert(kBM % kWM == 0);
    static_assert(kBN % kWN == 0);
    static_assert(kWM == 8 * kTM);
    static_assert(kWN == 4 * kTN);
    static_assert(kTM % VECTOR_WIDTH == 0);
    static_assert(kTN % VECTOR_WIDTH == 0);
    static_assert((kBM * BK / VECTOR_WIDTH) % kNumThreads == 0);
    static_assert((BK * kBN / VECTOR_WIDTH) % kNumThreads == 0);

    // Logical layouts are shared_left[stage][inner][row] and
    // shared_right[stage][inner][column].
    __shared__ float shared_left[2 * BK * kBM];
    __shared__ float shared_right[2 * BK * kBN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_row = warp_id / kWarpsPerRow;
    const int warp_column = warp_id % kWarpsPerRow;
    const int lane_row = lane_id / 4;
    const int lane_column = lane_id % 4;
    const int local_row = warp_row * kWM + lane_row * kTM;
    const int local_column = warp_column * kWN + lane_column * kTN;
    const int block_row = blockIdx.y * kBM;
    const int block_column = blockIdx.x * kBN;
    // Aligned interior blocks bypass all scalar edge handling.
    const bool full_tile =
        block_row + kBM <= output_rows &&
        block_column + kBN <= output_columns &&
        inner_size % BK == 0 &&
        output_columns % VECTOR_WIDTH == 0;

    float accumulator[kTM][kTN] = {};
    float left_fragment[2][kTM];
    float right_fragment[2][kTN];
    int write_stage = 0;

    // Each iteration first issues the global loads for the next tile, computes
    // the preceding tile, and then publishes the loaded values to shared memory.
    for (int tile_to_load = 0;; tile_to_load += BK) {
        const bool load_tile = tile_to_load < inner_size;
        const bool compute_tile = tile_to_load > 0;
        float4 loaded_left[kLeftLoads];
        float4 loaded_right[kRightLoads];

        if (load_tile) {
#pragma unroll
            for (int load = 0; load < kLeftLoads; ++load) {
                const int vector_index = tid + load * kNumThreads;
                const int vectors_per_row = BK / VECTOR_WIDTH;
                const int tile_row = vector_index / vectors_per_row;
                const int tile_inner =
                    (vector_index % vectors_per_row) * VECTOR_WIDTH;
                const int global_row = block_row + tile_row;
                const int global_inner = tile_to_load + tile_inner;
                const int index = global_row * inner_size + global_inner;
                loaded_left[load] = full_tile
                    ? *reinterpret_cast<const float4*>(left + index)
                    : load_float4(
                          left,
                          index,
                          global_row < output_rows,
                          global_inner,
                          inner_size);
            }

#pragma unroll
            for (int load = 0; load < kRightLoads; ++load) {
                const int vector_index = tid + load * kNumThreads;
                const int vectors_per_inner = kBN / VECTOR_WIDTH;
                const int tile_inner = vector_index / vectors_per_inner;
                const int tile_column =
                    (vector_index % vectors_per_inner) * VECTOR_WIDTH;
                const int global_inner = tile_to_load + tile_inner;
                const int global_column = block_column + tile_column;
                const int index = global_inner * output_columns + global_column;
                loaded_right[load] = full_tile
                    ? *reinterpret_cast<const float4*>(right + index)
                    : load_float4(
                          right,
                          index,
                          global_inner < inner_size,
                          global_column,
                          output_columns);
            }
        }

        const int read_stage = write_stage ^ 1;
        if (compute_tile) {
#pragma unroll
            for (int inner = 0; inner < BK - 1; ++inner) {
                const int read_fragment = inner & 1;
                const int write_fragment = (inner + 1) & 1;

#pragma unroll
                for (int vector = 0; vector < kTM / VECTOR_WIDTH; ++vector) {
                    const float4 value = *reinterpret_cast<const float4*>(
                        &shared_left[
                            (read_stage * BK + inner + 1) * kBM +
                            local_row + vector * VECTOR_WIDTH]);
                    left_fragment[write_fragment][vector * VECTOR_WIDTH + 0] =
                        value.x;
                    left_fragment[write_fragment][vector * VECTOR_WIDTH + 1] =
                        value.y;
                    left_fragment[write_fragment][vector * VECTOR_WIDTH + 2] =
                        value.z;
                    left_fragment[write_fragment][vector * VECTOR_WIDTH + 3] =
                        value.w;
                }

#pragma unroll
                for (int vector = 0; vector < kTN / VECTOR_WIDTH; ++vector) {
                    const float4 value = *reinterpret_cast<const float4*>(
                        &shared_right[
                            (read_stage * BK + inner + 1) * kBN +
                            local_column + vector * VECTOR_WIDTH]);
                    right_fragment[write_fragment][vector * VECTOR_WIDTH + 0] =
                        value.x;
                    right_fragment[write_fragment][vector * VECTOR_WIDTH + 1] =
                        value.y;
                    right_fragment[write_fragment][vector * VECTOR_WIDTH + 2] =
                        value.z;
                    right_fragment[write_fragment][vector * VECTOR_WIDTH + 3] =
                        value.w;
                }

#pragma unroll
                for (int row = 0; row < kTM; ++row) {
#pragma unroll
                    for (int column = 0; column < kTN; ++column) {
                        accumulator[row][column] = __fmaf_rn(
                            left_fragment[read_fragment][row],
                            right_fragment[read_fragment][column],
                            accumulator[row][column]);
                    }
                }
            }
        }

        if (load_tile) {
#pragma unroll
            for (int load = 0; load < kLeftLoads; ++load) {
                const int vector_index = tid + load * kNumThreads;
                const int vectors_per_row = BK / VECTOR_WIDTH;
                const int tile_row = vector_index / vectors_per_row;
                const int tile_inner =
                    (vector_index % vectors_per_row) * VECTOR_WIDTH;
                shared_left[
                    (write_stage * BK + tile_inner + 0) * kBM +
                    tile_row] = loaded_left[load].x;
                shared_left[
                    (write_stage * BK + tile_inner + 1) * kBM +
                    tile_row] = loaded_left[load].y;
                shared_left[
                    (write_stage * BK + tile_inner + 2) * kBM +
                    tile_row] = loaded_left[load].z;
                shared_left[
                    (write_stage * BK + tile_inner + 3) * kBM +
                    tile_row] = loaded_left[load].w;
            }

#pragma unroll
            for (int load = 0; load < kRightLoads; ++load) {
                const int vector_index = tid + load * kNumThreads;
                const int vectors_per_inner = kBN / VECTOR_WIDTH;
                const int tile_inner = vector_index / vectors_per_inner;
                const int tile_column =
                    (vector_index % vectors_per_inner) * VECTOR_WIDTH;
                const int index =
                    (write_stage * BK + tile_inner) * kBN + tile_column;
                shared_right[index + 0] = loaded_right[load].x;
                shared_right[index + 1] = loaded_right[load].y;
                shared_right[index + 2] = loaded_right[load].z;
                shared_right[index + 3] = loaded_right[load].w;
            }

            __syncthreads();
            const int loaded_stage = write_stage;
            write_stage ^= 1;

#pragma unroll
            for (int vector = 0; vector < kTM / VECTOR_WIDTH; ++vector) {
                const float4 value = *reinterpret_cast<const float4*>(
                    &shared_left[
                        loaded_stage * BK * kBM + local_row +
                        vector * VECTOR_WIDTH]);
                left_fragment[0][vector * VECTOR_WIDTH + 0] = value.x;
                left_fragment[0][vector * VECTOR_WIDTH + 1] = value.y;
                left_fragment[0][vector * VECTOR_WIDTH + 2] = value.z;
                left_fragment[0][vector * VECTOR_WIDTH + 3] = value.w;
            }

#pragma unroll
            for (int vector = 0; vector < kTN / VECTOR_WIDTH; ++vector) {
                const float4 value = *reinterpret_cast<const float4*>(
                    &shared_right[
                        loaded_stage * BK * kBN + local_column +
                        vector * VECTOR_WIDTH]);
                right_fragment[0][vector * VECTOR_WIDTH + 0] = value.x;
                right_fragment[0][vector * VECTOR_WIDTH + 1] = value.y;
                right_fragment[0][vector * VECTOR_WIDTH + 2] = value.z;
                right_fragment[0][vector * VECTOR_WIDTH + 3] = value.w;
            }
        }

        if (compute_tile) {
#pragma unroll
            for (int row = 0; row < kTM; ++row) {
#pragma unroll
                for (int column = 0; column < kTN; ++column) {
                    accumulator[row][column] = __fmaf_rn(
                        left_fragment[1][row],
                        right_fragment[1][column],
                        accumulator[row][column]);
                }
            }
        }

        if (!load_tile) {
            break;
        }
    }

#pragma unroll
    for (int row = 0; row < kTM; ++row) {
        const int global_row = block_row + local_row + row;
#pragma unroll
        for (int vector = 0; vector < kTN / VECTOR_WIDTH; ++vector) {
            const int thread_column = vector * VECTOR_WIDTH;
            const int global_column =
                block_column + local_column + thread_column;
            int width = 0;
            if (global_row < output_rows && global_column < output_columns) {
                const int remaining = output_columns - global_column;
                width = remaining < VECTOR_WIDTH ? remaining : VECTOR_WIDTH;
            }
            const int index = global_row * output_columns + global_column;
            float4 value = make_float4(
                accumulator[row][thread_column + 0],
                accumulator[row][thread_column + 1],
                accumulator[row][thread_column + 2],
                accumulator[row][thread_column + 3]);

            if (width == VECTOR_WIDTH && index % VECTOR_WIDTH == 0) {
                *reinterpret_cast<float4*>(output + index) = value;
            } else {
                if (width > 0) {
                    output[index] = value.x;
                }
                if (width > 1) {
                    output[index + 1] = value.y;
                }
                if (width > 2) {
                    output[index + 2] = value.z;
                }
                if (width > 3) {
                    output[index + 3] = value.w;
                }
            }
        }
    }
}

__global__ void matmul_tensor_core_edge_kernel(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int output_rows,
    int output_columns,
    int inner_size) {
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
            nvcuda::wmma::fill_fragment(
                accumulators[tile_row][tile_column], 0.0F);
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
                value = left[global_row * inner_size + global_inner];
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
                value = right[global_inner * output_columns + global_column];
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
                    output + output_row * output_columns + output_column,
                    accumulators[tile_row][tile_column],
                    output_columns,
                    nvcuda::wmma::mem_row_major);
            }
        }
    }
}

__device__ __forceinline__ void cp_async_bf16x8(
    __nv_bfloat16* destination,
    const __nv_bfloat16* source) {
    const unsigned int shared_address =
        static_cast<unsigned int>(__cvta_generic_to_shared(destination));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
            "r"(shared_address),
            "l"(source));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

template <int kBits>
__device__ __forceinline__ int swizzle_bf16_offset(int offset) {
    constexpr int kMask = ((1 << kBits) - 1) << 6;
    return offset ^ ((offset & kMask) >> 3);
}

__device__ __forceinline__ unsigned int shared_address(const void* pointer) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void load_matrix_x4(
    unsigned int (&fragment)[4],
    unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment[0]),
          "=r"(fragment[1]),
          "=r"(fragment[2]),
          "=r"(fragment[3])
        : "r"(address));
}

// The PTX transpose arranges a row-major B tile into the column operand
// registers expected by mma.sync; the logical input remains B[K,N].
__device__ __forceinline__ void load_matrix_b_x2(
    unsigned int (&fragment)[2],
    unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address));
}

__device__ __forceinline__ void mma_bf16_m16n8k16(
    float (&accumulator)[4],
    const unsigned int (&left)[4],
    const unsigned int (&right)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(accumulator[0]),
          "+f"(accumulator[1]),
          "+f"(accumulator[2]),
          "+f"(accumulator[3])
        : "r"(left[0]),
          "r"(left[1]),
          "r"(left[2]),
          "r"(left[3]),
          "r"(right[0]),
          "r"(right[1]));
}

// The native MMA path uses CUTLASS-style XOR swizzles so cp.async writes and
// ldmatrix reads are both 128-bit aligned and free of shared-memory bank conflicts.
template <
    int kBM,
    int kBN,
    int kWarpTilesM,
    int kWarpTilesN>
__global__ __launch_bounds__(256, 2) void matmul_tensor_core_mma_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ left,
    const __nv_bfloat16* __restrict__ right,
    int output_columns,
    int inner_size) {
    constexpr int kWM = kWarpTilesM * tc_mma::MMA_M;
    constexpr int kWN = kWarpTilesN * tc_mma::MMA_N;
    constexpr int kWarpsN = kBN / kWN;
    constexpr int kNumThreads =
        (kBM / kWM) * kWarpsN * 32;
    constexpr int kLeftStageElements = kBM * tc_mma::BK;
    constexpr int kRightStageElements = tc_mma::BK * kBN;

    __shared__ __align__(16)
        __nv_bfloat16 shared_left[tc_mma::STAGES][kLeftStageElements];
    __shared__ __align__(16)
        __nv_bfloat16 shared_right[tc_mma::STAGES][kRightStageElements];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / kWarpsN;
    const int warp_column = warp_id % kWarpsN;
    const int block_row = blockIdx.y * kBM;
    const int block_column = blockIdx.x * kBN;

    float accumulators[kWarpTilesM][kWarpTilesN][4];
#pragma unroll
    for (int tile_row = 0; tile_row < kWarpTilesM; ++tile_row) {
#pragma unroll
        for (int tile_column = 0; tile_column < kWarpTilesN;
             ++tile_column) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulators[tile_row][tile_column][element] = 0.0F;
            }
        }
    }

    auto copy_stage = [&](int stage, int tile_inner) {
        constexpr int kLeftVectors =
            kLeftStageElements / tc_mma::VECTOR_ELEMENTS;
        constexpr int kRightVectors =
            kRightStageElements / tc_mma::VECTOR_ELEMENTS;
#pragma unroll
        for (int vector = tid; vector < kLeftVectors;
             vector += kNumThreads) {
            const int local_row =
                vector / (tc_mma::BK / tc_mma::VECTOR_ELEMENTS);
            const int local_inner =
                vector % (tc_mma::BK / tc_mma::VECTOR_ELEMENTS) *
                tc_mma::VECTOR_ELEMENTS;
            const int logical_offset =
                local_row * tc_mma::BK + local_inner;
            const int shared_offset =
                swizzle_bf16_offset<2>(logical_offset);
            const int global_offset =
                (block_row + local_row) * inner_size +
                tile_inner + local_inner;
            cp_async_bf16x8(
                shared_left[stage] + shared_offset,
                left + global_offset);
        }
#pragma unroll
        for (int vector = tid; vector < kRightVectors;
             vector += kNumThreads) {
            const int local_inner =
                vector / (kBN / tc_mma::VECTOR_ELEMENTS);
            const int local_column =
                vector % (kBN / tc_mma::VECTOR_ELEMENTS) *
                tc_mma::VECTOR_ELEMENTS;
            const int logical_offset =
                local_inner * kBN + local_column;
            const int shared_offset =
                swizzle_bf16_offset<3>(logical_offset);
            const int global_offset =
                (tile_inner + local_inner) * output_columns +
                block_column + local_column;
            cp_async_bf16x8(
                shared_right[stage] + shared_offset,
                right + global_offset);
        }
        cp_async_commit();
    };

    const int inner_tiles = inner_size / tc_mma::BK;
    copy_stage(0, 0);
    cp_async_wait();
    __syncthreads();

    for (int tile = 0; tile < inner_tiles; ++tile) {
        const int stage = tile % tc_mma::STAGES;
        const int next_tile = tile + 1;
        if (next_tile < inner_tiles) {
            copy_stage(next_tile % tc_mma::STAGES, next_tile * tc_mma::BK);
        }

#pragma unroll
        for (int tile_inner = 0; tile_inner < tc_mma::BK;
             tile_inner += tc_mma::MMA_K) {
            unsigned int left_fragments[kWarpTilesM][4];
            unsigned int right_fragments[kWarpTilesN][2];
#pragma unroll
            for (int tile_row = 0; tile_row < kWarpTilesM;
                 ++tile_row) {
                const int tile_row_base =
                    warp_row * kWM + tile_row * tc_mma::MMA_M;
                const int fragment_row = tile_row_base + lane % 16;
                const int fragment_inner = tile_inner + lane / 16 * 8;
                const int logical_offset =
                    fragment_row * tc_mma::BK + fragment_inner;
                const int swizzled_offset =
                    swizzle_bf16_offset<2>(logical_offset);
                load_matrix_x4(
                    left_fragments[tile_row],
                    shared_address(
                        shared_left[stage] + swizzled_offset));
            }
#pragma unroll
            for (int tile_column = 0;
                 tile_column < kWarpTilesN;
                 ++tile_column) {
                const int tile_column_base =
                    warp_column * kWN +
                    tile_column * tc_mma::MMA_N;
                const int fragment_inner = tile_inner + lane % 16;
                const int fragment_column = tile_column_base;
                const int logical_offset =
                    fragment_inner * kBN + fragment_column;
                const int swizzled_offset =
                    swizzle_bf16_offset<3>(logical_offset);
                load_matrix_b_x2(
                    right_fragments[tile_column],
                    shared_address(
                        shared_right[stage] + swizzled_offset));
            }
#pragma unroll
            for (int tile_row = 0; tile_row < kWarpTilesM;
                 ++tile_row) {
#pragma unroll
                for (int tile_column = 0;
                     tile_column < kWarpTilesN;
                     ++tile_column) {
                    mma_bf16_m16n8k16(
                        accumulators[tile_row][tile_column],
                        left_fragments[tile_row],
                        right_fragments[tile_column]);
                }
            }
        }

        if (next_tile < inner_tiles) {
            cp_async_wait();
            __syncthreads();
        }
    }

#pragma unroll
    for (int tile_row = 0; tile_row < kWarpTilesM; ++tile_row) {
#pragma unroll
        for (int tile_column = 0; tile_column < kWarpTilesN;
             ++tile_column) {
            const int output_row =
                block_row + warp_row * kWM +
                tile_row * tc_mma::MMA_M + lane / 4;
            const int output_column =
                block_column + warp_column * kWN +
                tile_column * tc_mma::MMA_N + (lane % 4) * 2;
            const float2 top = make_float2(
                accumulators[tile_row][tile_column][0],
                accumulators[tile_row][tile_column][1]);
            const float2 bottom = make_float2(
                accumulators[tile_row][tile_column][2],
                accumulators[tile_row][tile_column][3]);
            *reinterpret_cast<float2*>(
                output + output_row * output_columns + output_column) = top;
            *reinterpret_cast<float2*>(
                output + (output_row + 8) * output_columns + output_column) =
                bottom;
        }
    }
}

template <
    int kBM,
    int kBN,
    int kWarpTilesM,
    int kWarpTilesN>
void launch_tensor_core_mma_config(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int output_rows,
    int output_columns,
    int inner_size,
    cudaStream_t stream) {
    constexpr int kWM = kWarpTilesM * tc_mma::MMA_M;
    constexpr int kWN = kWarpTilesN * tc_mma::MMA_N;
    constexpr int kNumThreads =
        (kBM / kWM) * (kBN / kWN) * 32;
    const dim3 blocks(output_columns / kBN, output_rows / kBM);
    matmul_tensor_core_mma_kernel<
        kBM,
        kBN,
        kWarpTilesM,
        kWarpTilesN>
        <<<blocks, kNumThreads, 0, stream>>>(
            output,
            left,
            right,
            output_columns,
            inner_size);
}


template <
    int kBM,
    int kBN,
    int kWM,
    int kWN,
    int kTM,
    int kTN>
void launch_matmul_config(
    float* output,
    const float* left,
    const float* right,
    int output_rows,
    int output_columns,
    int inner_size,
    cudaStream_t stream) {
    constexpr int kNumThreads = (kBM / kWM) * (kBN / kWN) * 32;
    const dim3 blocks(
        (output_columns + kBN - 1) / kBN,
        (output_rows + kBM - 1) / kBM);
    matmul_kernel<
        kBM,
        kBN,
        kWM,
        kWN,
        kTM,
        kTN>
        <<<blocks, kNumThreads, 0, stream>>>(
            output,
            left,
            right,
            output_rows,
            output_columns,
            inner_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul(
    float* output,
    const float* left,
    const float* right,
    int output_rows,
    int output_columns,
    int inner_size,
    cudaStream_t stream) {
    // A 512x512 GEMM has only sixteen 128x128 blocks, so use the smaller
    // configuration to keep every SM supplied with multiple warps.
    if (output_rows <= 512 && output_columns <= 512) {
        launch_matmul_config<
            SMALL_BM,
            SMALL_BN,
            WM,
            SMALL_WN,
            TM,
            SMALL_TN>(
            output,
            left,
            right,
            output_rows,
            output_columns,
            inner_size,
            stream);
    } else {
        launch_matmul_config<
            BM,
            BN,
            WM,
            WN,
            TM,
            TN>(
            output,
            left,
            right,
            output_rows,
            output_columns,
            inner_size,
            stream);
    }
}

void launch_tensor_core_matmul(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int output_rows,
    int output_columns,
    int inner_size,
    cudaStream_t stream) {
    const bool vector_aligned =
        inner_size % tc_mma::BK == 0 &&
        output_columns % tc_mma::VECTOR_ELEMENTS == 0;
    const bool use_small_tile =
        output_rows <= 1024 &&
        output_rows % 64 == 0 &&
        output_columns % 128 == 0 &&
        vector_aligned;
    const bool use_large_tile =
        output_rows % 128 == 0 &&
        output_columns % 128 == 0 &&
        vector_aligned;
    if (use_small_tile) {
        launch_tensor_core_mma_config<64, 128, 4, 4>(
            output,
            left,
            right,
            output_rows,
            output_columns,
            inner_size,
            stream);
    } else if (use_large_tile) {
        launch_tensor_core_mma_config<128, 128, 4, 4>(
            output,
            left,
            right,
            output_rows,
            output_columns,
            inner_size,
            stream);
    } else {
        const dim3 blocks(
            (output_columns + tc::BN - 1) / tc::BN,
            (output_rows + tc::BM - 1) / tc::BM);
        matmul_tensor_core_edge_kernel<<<blocks, tc::NUM_THREADS, 0, stream>>>(
                output,
                left,
                right,
                output_rows,
                output_columns,
                inner_size);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void gemm_fp32_sm89_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_matmul(output, left, right, M, N, K, stream);
}

void gemm_bf16_sm89_cuda(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    launch_tensor_core_matmul(output, left, right, M, N, K, stream);
}

}  // namespace dscuda

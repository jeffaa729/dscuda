#include "common.cuh"
#include "cuda_common.h"

#include <cuda.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace dscuda {
namespace {

using bf16 = __nv_bfloat16;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int WGMMA_K = 16;
constexpr int NUM_THREADS = 128;

template <int TileRows, int TileColumns>
CUtensorMap make_tensor_map(const bf16* pointer, int rows, int columns) {
    CUtensorMap map;
    void* address = const_cast<bf16*>(pointer);
    const uint64_t global_shape[2] = {
        static_cast<uint64_t>(columns), static_cast<uint64_t>(rows)};
    const uint64_t global_stride[1] = {
        static_cast<uint64_t>(columns) * sizeof(bf16)};
    const uint32_t box_shape[2] = {TileColumns, TileRows};
    const uint32_t element_stride[2] = {1, 1};

    const CUresult result = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, address,
        global_shape, global_stride, box_shape, element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        throw std::runtime_error(
            std::string("cuTensorMapEncodeTiled failed: ") +
            (name ? name : "unknown driver error"));
    }
    return map;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

__device__ __forceinline__ uint64_t encode_descriptor(uint64_t value) {
    return (value & 0x3FFFFU) >> 4U;
}

// Optimization: use the 128-byte-swizzled shared-memory layout produced by TMA.
__device__ __forceinline__ uint64_t make_smem_descriptor(bf16* pointer) {
    const uint32_t address =
        static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    uint64_t descriptor = encode_descriptor(address);
    descriptor |= encode_descriptor(16) << 16U;
    descriptor |= encode_descriptor(1024) << 32U;
    descriptor |= 1ULL << 62U;
    return descriptor;
}

__device__ __forceinline__ void warpgroup_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

template <int ScaleD>
__device__ __forceinline__ void wgmma_m64n64k16(
    float (&accumulator)[4][8], bf16* left, bf16* right) {
    const uint64_t left_descriptor = make_smem_descriptor(left);
    const uint64_t right_descriptor = make_smem_descriptor(right);

    // Optimization: one warpgroup instruction computes a 64x64x16 Tensor Core tile.
    // TransB=1 adapts row-major B storage to WGMMA's B operand convention.
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, %34, 1, 1, 0, 1;\n"
        : "+f"(accumulator[0][0]), "+f"(accumulator[0][1]),
          "+f"(accumulator[0][2]), "+f"(accumulator[0][3]),
          "+f"(accumulator[0][4]), "+f"(accumulator[0][5]),
          "+f"(accumulator[0][6]), "+f"(accumulator[0][7]),
          "+f"(accumulator[1][0]), "+f"(accumulator[1][1]),
          "+f"(accumulator[1][2]), "+f"(accumulator[1][3]),
          "+f"(accumulator[1][4]), "+f"(accumulator[1][5]),
          "+f"(accumulator[1][6]), "+f"(accumulator[1][7]),
          "+f"(accumulator[2][0]), "+f"(accumulator[2][1]),
          "+f"(accumulator[2][2]), "+f"(accumulator[2][3]),
          "+f"(accumulator[2][4]), "+f"(accumulator[2][5]),
          "+f"(accumulator[2][6]), "+f"(accumulator[2][7]),
          "+f"(accumulator[3][0]), "+f"(accumulator[3][1]),
          "+f"(accumulator[3][2]), "+f"(accumulator[3][3]),
          "+f"(accumulator[3][4]), "+f"(accumulator[3][5]),
          "+f"(accumulator[3][6]), "+f"(accumulator[3][7])
        : "l"(left_descriptor), "l"(right_descriptor), "n"(ScaleD));
}

#endif

__global__ __launch_bounds__(NUM_THREADS) void gemm_bf16_kernel(
    bf16* __restrict__ output,
    const __grid_constant__ CUtensorMap left_map,
    const __grid_constant__ CUtensorMap right_map,
    int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __shared__ alignas(128) bf16 shared_left[BM * BK];
    __shared__ alignas(128) bf16 shared_right[BK * BN];
    __shared__ barrier left_barrier;
    __shared__ barrier right_barrier;

    float accumulator[4][8];
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;

    if (threadIdx.x == 0) {
        init(&left_barrier, blockDim.x);
        init(&right_barrier, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    for (int tile_k = 0; tile_k < K / BK; ++tile_k) {
        barrier::arrival_token left_token;
        barrier::arrival_token right_token;

        // Optimization: a single thread issues asynchronous 2D TMA tile loads.
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                shared_left, &left_map, tile_k * BK, tile_m * BM,
                left_barrier);
            left_token = cuda::device::barrier_arrive_tx(
                left_barrier, 1, sizeof(shared_left));
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                shared_right, &right_map, tile_n * BN, tile_k * BK,
                right_barrier);
            right_token = cuda::device::barrier_arrive_tx(
                right_barrier, 1, sizeof(shared_right));
        } else {
            left_token = left_barrier.arrive();
            right_token = right_barrier.arrive();
        }
        left_barrier.wait(std::move(left_token));
        right_barrier.wait(std::move(right_token));

        warpgroup_fence();
        if (tile_k == 0) {
            // Optimization: ScaleD=0 initializes accumulators in the first WGMMA.
            wgmma_m64n64k16<0>(accumulator, shared_left, shared_right);
        } else {
            wgmma_m64n64k16<1>(accumulator, shared_left, shared_right);
        }
        wgmma_m64n64k16<1>(
            accumulator, shared_left + WGMMA_K,
            shared_right + WGMMA_K * BN);
        wgmma_m64n64k16<1>(
            accumulator, shared_left + 2 * WGMMA_K,
            shared_right + 2 * WGMMA_K * BN);
        wgmma_m64n64k16<1>(
            accumulator, shared_left + 3 * WGMMA_K,
            shared_right + 3 * WGMMA_K * BN);
        warpgroup_commit();
        warpgroup_wait();
    }

    // WGMMA distributes the 64x64 FP32 accumulator over the 128-thread warpgroup.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = warp * 16 + lane / 4;
    bf16* tile_output = output + tile_m * BM * N + tile_n * BN;

#pragma unroll
    for (int group = 0; group < 4; ++group) {
        const int column = group * 16 + 2 * (threadIdx.x & 3);
#define STORE(Row, Column, Value) \
    tile_output[(Row) * N + (Column)] = __float2bfloat16(Value)
        STORE(row, column, accumulator[group][0]);
        STORE(row, column + 1, accumulator[group][1]);
        STORE(row + 8, column, accumulator[group][2]);
        STORE(row + 8, column + 1, accumulator[group][3]);
        STORE(row, column + 8, accumulator[group][4]);
        STORE(row, column + 9, accumulator[group][5]);
        STORE(row + 8, column + 8, accumulator[group][6]);
        STORE(row + 8, column + 9, accumulator[group][7]);
#undef STORE
    }
#endif
}

}  // namespace

void gemm_bf16_sm90_cuda(
    bf16* output, const bf16* left, const bf16* right,
    int M, int N, int K, cudaStream_t stream) {
    const CUtensorMap left_map = make_tensor_map<BM, BK>(left, M, K);
    const CUtensorMap right_map = make_tensor_map<BK, BN>(right, K, N);
    const dim3 grid(N / BN, M / BM);
    gemm_bf16_kernel<<<grid, NUM_THREADS, 0, stream>>>(
        output, left_map, right_map, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

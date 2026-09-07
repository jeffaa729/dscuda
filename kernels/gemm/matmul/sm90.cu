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

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;

constexpr int WGMMA_M = 64;
constexpr int WGMMA_N = BN;
constexpr int WGMMA_K = 16; // Number of K elements consumed by one WGMMA instruction

/*
C block: 128×128

rows   0–63: first  WGMMA M tile
rows 64–127: second WGMMA M tile
*/
constexpr int NUM_THREADS = 128; // threads in the block
constexpr int NUM_WARPGROUPS = NUM_THREADS / 128;
constexpr int WG_M = BM / NUM_WARPGROUPS;
constexpr int M_TILES = WG_M / WGMMA_M;
constexpr int B_PANEL_N = 64;
constexpr int B_PANELS = BN / B_PANEL_N;
constexpr unsigned int SMEM_ALIGNMENT = 1024;

static_assert(NUM_THREADS % 128 == 0);
static_assert(BM % NUM_WARPGROUPS == 0 && WG_M % WGMMA_M == 0);
static_assert(BK == 64 && BN == 128);
static_assert(BK % WGMMA_K == 0 && BN % B_PANEL_N == 0);

// Matmul3 reuses B across two 64-row WGMMA tiles in one warpgroup.
// Each B panel has a 128-byte row, as required by the TMA swizzle mode.
struct SharedStorage {
    alignas(SMEM_ALIGNMENT) bf16 A[BM * BK];
    alignas(SMEM_ALIGNMENT) bf16 B[B_PANELS][BK * B_PANEL_N];
};
constexpr size_t SMEM_BYTES = sizeof(SharedStorage) + SMEM_ALIGNMENT - 1;

template <int TileRows, int TileColumns>
CUtensorMap make_tensor_map(const bf16* pointer, int rows, int columns) {
    CUtensorMap map;
    void* address = const_cast<bf16*>(pointer);
    const uint64_t global_shape[2] = {static_cast<uint64_t>(columns), static_cast<uint64_t>(rows)};
    const uint64_t global_stride[1] = {static_cast<uint64_t>(columns) * sizeof(bf16)};
    const uint32_t box_shape[2] = {TileColumns, TileRows};
    const uint32_t element_stride[2] = {1, 1};

    const CUresult result =
        cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, address, global_shape, global_stride, box_shape, element_stride,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") + (name ? name : "unknown driver error"));
    }
    return map;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

__device__ __forceinline__ uint64_t encode_descriptor(uint64_t value) {
    return (value & 0x3FFFFU) >> 4U;
}

// Optimization: use the 128-byte-swizzled shared-memory layout produced by TMA.
template <int LeadingBytes>
__device__ __forceinline__ uint64_t make_smem_descriptor(bf16* pointer) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    uint64_t descriptor = encode_descriptor(address);
    descriptor |= encode_descriptor(LeadingBytes) << 16U;
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
__device__ __forceinline__ void wgmma_m64n128k16(float (&accumulator)[8][8], bf16* A, bf16* B) {
    const uint64_t A_descriptor = make_smem_descriptor<16>(A);
    // MN-major B advances by a complete panel when N crosses 64 columns.
    const uint64_t B_descriptor = make_smem_descriptor<BK * B_PANEL_N * sizeof(bf16)>(B);

    // Optimization: one warpgroup instruction computes a 64x128x16 Tensor Core tile.
    // TransB=1 adapts row-major B storage to WGMMA's B operand convention.
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, %66, 1, 1, 0, 1;\n"
        : "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]), "+f"(accumulator[0][4]), "+f"(accumulator[0][5]),
          "+f"(accumulator[0][6]), "+f"(accumulator[0][7]), "+f"(accumulator[1][0]), "+f"(accumulator[1][1]), "+f"(accumulator[1][2]), "+f"(accumulator[1][3]),
          "+f"(accumulator[1][4]), "+f"(accumulator[1][5]), "+f"(accumulator[1][6]), "+f"(accumulator[1][7]), "+f"(accumulator[2][0]), "+f"(accumulator[2][1]),
          "+f"(accumulator[2][2]), "+f"(accumulator[2][3]), "+f"(accumulator[2][4]), "+f"(accumulator[2][5]), "+f"(accumulator[2][6]), "+f"(accumulator[2][7]),
          "+f"(accumulator[3][0]), "+f"(accumulator[3][1]), "+f"(accumulator[3][2]), "+f"(accumulator[3][3]), "+f"(accumulator[3][4]), "+f"(accumulator[3][5]),
          "+f"(accumulator[3][6]), "+f"(accumulator[3][7]),
          "+f"(accumulator[4][0]), "+f"(accumulator[4][1]), "+f"(accumulator[4][2]), "+f"(accumulator[4][3]), "+f"(accumulator[4][4]), "+f"(accumulator[4][5]),
          "+f"(accumulator[4][6]), "+f"(accumulator[4][7]), "+f"(accumulator[5][0]), "+f"(accumulator[5][1]), "+f"(accumulator[5][2]), "+f"(accumulator[5][3]),
          "+f"(accumulator[5][4]), "+f"(accumulator[5][5]), "+f"(accumulator[5][6]), "+f"(accumulator[5][7]), "+f"(accumulator[6][0]), "+f"(accumulator[6][1]),
          "+f"(accumulator[6][2]), "+f"(accumulator[6][3]), "+f"(accumulator[6][4]), "+f"(accumulator[6][5]), "+f"(accumulator[6][6]), "+f"(accumulator[6][7]),
          "+f"(accumulator[7][0]), "+f"(accumulator[7][1]), "+f"(accumulator[7][2]), "+f"(accumulator[7][3]), "+f"(accumulator[7][4]), "+f"(accumulator[7][5]),
          "+f"(accumulator[7][6]), "+f"(accumulator[7][7])
        : "l"(A_descriptor), "l"(B_descriptor), "n"(ScaleD));
}

#endif

__global__ __launch_bounds__(NUM_THREADS) void gemm_bf16_kernel(
    bf16* __restrict__ C, const __grid_constant__ CUtensorMap A_map, const __grid_constant__ CUtensorMap B_map, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(16) unsigned char storage[];
    // Align the actual shared address: static barriers can shift the dynamic base.
    // The launch reserves padding so rounding up keeps both TMA tiles in bounds.
    const unsigned int storage_address = static_cast<unsigned int>(__cvta_generic_to_shared(storage));
    const unsigned int aligned_address = (storage_address + SMEM_ALIGNMENT - 1) & ~(SMEM_ALIGNMENT - 1);
    auto& shared = *reinterpret_cast<SharedStorage*>(__cvta_shared_to_generic(aligned_address));
    __shared__ barrier A_barrier;
    __shared__ barrier B_barrier;

    // [2][8][8]
    float accumulator[M_TILES][WGMMA_N / 16][8] = {};
    static_assert(sizeof(accumulator) * NUM_THREADS == BM * BN * sizeof(float));
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int wg_id = threadIdx.x / 128;

    if (threadIdx.x == 0) {
        init(&A_barrier, blockDim.x);
        init(&B_barrier, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    for (int tile_k = 0; tile_k < K / BK; ++tile_k) {
        barrier::arrival_token A_token;
        barrier::arrival_token B_token;

        // Optimization: a single thread issues asynchronous 2D TMA tile loads.
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(shared.A, &A_map, tile_k * BK, tile_m * BM, A_barrier);
            A_token = cuda::device::barrier_arrive_tx(A_barrier, 1, sizeof(shared.A));
#pragma unroll
            for (int panel = 0; panel < B_PANELS; ++panel) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    shared.B[panel], &B_map, tile_n * BN + panel * B_PANEL_N, tile_k * BK, B_barrier);
            }
            B_token = cuda::device::barrier_arrive_tx(B_barrier, 1, sizeof(shared.B));
        } else {
            A_token = A_barrier.arrive();
            B_token = B_barrier.arrive();
        }
        A_barrier.wait(std::move(A_token));
        B_barrier.wait(std::move(B_token));

        warpgroup_fence();
#pragma unroll
        for (int m_it = 0; m_it < M_TILES; ++m_it) {
            bf16* A_tile = shared.A + (wg_id * WG_M + m_it * WGMMA_M) * BK;
#pragma unroll
            for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
                wgmma_m64n128k16<1>(accumulator[m_it], A_tile + k_it * WGMMA_K, shared.B[0] + k_it * WGMMA_K * B_PANEL_N);
            }
        }
        warpgroup_commit();
        warpgroup_wait();
        // Every warpgroup must finish reading before thread 0 refills the buffers.
        __syncthreads();
    }

    // Two 64x128 fragments cover the row-major 128x128 output tile.
    const int wg_tid = threadIdx.x % 128;
    const int lane = wg_tid & 31;
    const int warp = wg_tid >> 5;
    bf16* tile_C = C + tile_m * BM * N + tile_n * BN;

#pragma unroll
    for (int m_it = 0; m_it < M_TILES; ++m_it) {
        const int row = wg_id * WG_M + m_it * WGMMA_M + warp * 16 + lane / 4;
#pragma unroll
        for (int group = 0; group < WGMMA_N / 16; ++group) {
            const int column = group * 16 + 2 * (lane & 3);
#define STORE(Row, Column, Value) tile_C[(Row) * N + (Column)] = __float2bfloat16(Value)
            STORE(row, column, accumulator[m_it][group][0]);
            STORE(row, column + 1, accumulator[m_it][group][1]);
            STORE(row + 8, column, accumulator[m_it][group][2]);
            STORE(row + 8, column + 1, accumulator[m_it][group][3]);
            STORE(row, column + 8, accumulator[m_it][group][4]);
            STORE(row, column + 9, accumulator[m_it][group][5]);
            STORE(row + 8, column + 8, accumulator[m_it][group][6]);
            STORE(row + 8, column + 9, accumulator[m_it][group][7]);
#undef STORE
        }
    }
#endif
}

}  // namespace

void gemm_bf16_sm90_cuda(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || M % BM != 0 || N % BN != 0 || K % BK != 0) {
        throw std::invalid_argument("SM90 BF16 GEMM requires M/N multiples of 128 and K a multiple of 64.");
    }
    const CUtensorMap A_map = make_tensor_map<BM, BK>(A, M, K);
    const CUtensorMap B_map = make_tensor_map<BK, B_PANEL_N>(B, K, N);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    const dim3 grid(N / BN, M / BM);
    gemm_bf16_kernel<<<grid, NUM_THREADS, SMEM_BYTES, stream>>>(C, A_map, B_map, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

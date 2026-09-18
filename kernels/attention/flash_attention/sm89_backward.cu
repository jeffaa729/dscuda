// Implements the SM89 D128 causal FlashAttention backward pipeline with BF16 Tensor Cores and FP32 dQ accumulation.
// A preprocessing kernel computes D, one K/V-tile-owned kernel computes dQ, dK and dV together, and an epilogue converts dQ to BF16.

#include "sm89_common.cuh"

namespace dscuda {
namespace flash_attention_sm89 {
namespace tensor_core {

constexpr int BACKWARD_THREADS = 256;
constexpr int HALF_OUTPUT_TILES = OUTPUT_N_TILES / 2;
constexpr int BACKWARD_SHARED_ELEMENTS = 4 * BM * D + 2 * BM * BN;
constexpr int BACKWARD_SHARED_BYTES = BACKWARD_SHARED_ELEMENTS * sizeof(__nv_bfloat16);

static_assert(BM == BN);
static_assert(OUTPUT_N_TILES % 2 == 0);

__device__ __forceinline__ float as_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <int FIRST_TILE>
__device__ __forceinline__ void matrix_product_right_half(float (&accumulators)[HALF_OUTPUT_TILES][4],
                                                          const unsigned int (&left)[TOKEN_K_TILES][4],
                                                          const __nv_bfloat16* right, int lane) {
#pragma unroll
    for (int tile_inner = 0; tile_inner < TOKEN_K_TILES; ++tile_inner) {
#pragma unroll
        for (int local_tile = 0; local_tile < HALF_OUTPUT_TILES; local_tile += 2) {
            unsigned int right_fragments[4];
            load_right_pair(right_fragments, right, tile_inner, FIRST_TILE + local_tile, lane);
            mma(accumulators[local_tile], left[tile_inner], right_fragments);
            mma(accumulators[local_tile + 1], left[tile_inner], right_fragments + 2);
        }
    }
}

__global__ __launch_bounds__(BACKWARD_THREADS, 1) void flash_attention_backward_preprocess_kernel(
    float* __restrict__ query_gradient_accumulator, float* __restrict__ row_delta,
    const __nv_bfloat16* __restrict__ output_gradient, const __nv_bfloat16* __restrict__ output,
    int sequence_length, int heads) {
    const int batch_head = blockIdx.y;
    const int batch = batch_head / heads;
    const int head = batch_head % heads;
    const int first_query = blockIdx.x * BM;

    // Clear the FP32 dQ tile before the sequence-parallel main kernel atomically
    // accumulates contributions from all visible K/V tiles.
    for (int offset = threadIdx.x; offset < BM * D; offset += BACKWARD_THREADS) {
        const int row = offset / D;
        const int column = offset % D;
        query_gradient_accumulator[tensor_index(batch, first_query + row, head, column, sequence_length, heads)] = 0.0F;
    }

    // Four adjacent lanes reduce one row of D = rowsum(dO * O). One block
    // handles exactly the same 64 query rows as the main backward tile.
    const int local_row = threadIdx.x / 4;
    const int subgroup_lane = threadIdx.x % 4;
    float sum = 0.0F;
#pragma unroll
    for (int column = subgroup_lane; column < D; column += 4) {
        const int index = tensor_index(batch, first_query + local_row, head, column, sequence_length, heads);
        sum += as_float(output_gradient[index]) * as_float(output[index]);
    }
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    if (subgroup_lane == 0) {
        row_delta[batch_head * sequence_length + first_query + local_row] = sum;
    }
}

__device__ __forceinline__ void make_probability_and_score_gradient(
    float (&probabilities)[SCORE_N_TILES][4], float (&score_gradients)[SCORE_N_TILES][4],
    const float (&scores)[SCORE_N_TILES][4], const float (&probability_gradients)[SCORE_N_TILES][4],
    float top_delta, float bottom_delta, float top_logsumexp, float bottom_logsumexp,
    int first_query, int first_key, int warp_row, int lane, float scale) {
    const int top_row = warp_row + lane / 4;
    const int bottom_row = top_row + 8;
    const bool triangular = first_query == first_key;
#pragma unroll
    for (int tile = 0; tile < SCORE_N_TILES; ++tile) {
        const int key_column = tile * MMA_N + (lane % 4) * 2;
        const bool top_first_visible = !triangular || top_row >= key_column;
        const bool top_second_visible = !triangular || top_row >= key_column + 1;
        const bool bottom_first_visible = !triangular || bottom_row >= key_column;
        const bool bottom_second_visible = !triangular || bottom_row >= key_column + 1;
        probabilities[tile][0] = top_first_visible ? expf(scale * scores[tile][0] - top_logsumexp) : 0.0F;
        probabilities[tile][1] = top_second_visible ? expf(scale * scores[tile][1] - top_logsumexp) : 0.0F;
        probabilities[tile][2] = bottom_first_visible ? expf(scale * scores[tile][2] - bottom_logsumexp) : 0.0F;
        probabilities[tile][3] = bottom_second_visible ? expf(scale * scores[tile][3] - bottom_logsumexp) : 0.0F;
        score_gradients[tile][0] = scale * probabilities[tile][0] * (probability_gradients[tile][0] - top_delta);
        score_gradients[tile][1] = scale * probabilities[tile][1] * (probability_gradients[tile][1] - top_delta);
        score_gradients[tile][2] = scale * probabilities[tile][2] * (probability_gradients[tile][2] - bottom_delta);
        score_gradients[tile][3] = scale * probabilities[tile][3] * (probability_gradients[tile][3] - bottom_delta);
    }
}

__device__ __forceinline__ void store_score_gradient(
    __nv_bfloat16* row_major, __nv_bfloat16* transposed,
    const float (&score_gradients)[SCORE_N_TILES][4], int warp_row, int lane) {
    const int top_row = warp_row + lane / 4;
    const int bottom_row = top_row + 8;
#pragma unroll
    for (int tile = 0; tile < SCORE_N_TILES; ++tile) {
        const int key_column = tile * MMA_N + (lane % 4) * 2;
        const int query_rows[4] = {top_row, top_row, bottom_row, bottom_row};
        const int key_rows[4] = {key_column, key_column + 1, key_column, key_column + 1};
#pragma unroll
        for (int element = 0; element < 4; ++element) {
            row_major[swizzle<BN>(query_rows[element] * BN + key_rows[element])] = __float2bfloat16(score_gradients[tile][element]);
            transposed[swizzle<BM>(key_rows[element] * BM + query_rows[element])] = __float2bfloat16(score_gradients[tile][element]);
        }
    }
}

__device__ __forceinline__ void store_probability_transposed(
    __nv_bfloat16* transposed, const float (&probabilities)[SCORE_N_TILES][4], int warp_row, int lane) {
    const int top_row = warp_row + lane / 4;
    const int bottom_row = top_row + 8;
#pragma unroll
    for (int tile = 0; tile < SCORE_N_TILES; ++tile) {
        const int key_column = tile * MMA_N + (lane % 4) * 2;
        const int query_rows[4] = {top_row, top_row, bottom_row, bottom_row};
        const int key_rows[4] = {key_column, key_column + 1, key_column, key_column + 1};
#pragma unroll
        for (int element = 0; element < 4; ++element) {
            transposed[swizzle<BM>(key_rows[element] * BM + query_rows[element])] = __float2bfloat16(probabilities[tile][element]);
        }
    }
}

template <int FIRST_TILE>
__device__ __forceinline__ void atomic_add_query_gradient(
    float* query_gradient_accumulator, const float (&accumulators)[HALF_OUTPUT_TILES][4],
    int batch, int first_query, int head, int sequence_length, int heads, int warp_row, int lane) {
    const int top_query = first_query + warp_row + lane / 4;
    const int bottom_query = top_query + 8;
#pragma unroll
    for (int local_tile = 0; local_tile < HALF_OUTPUT_TILES; ++local_tile) {
        const int column = (FIRST_TILE + local_tile) * MMA_N + (lane % 4) * 2;
        atomicAdd(query_gradient_accumulator + tensor_index(batch, top_query, head, column, sequence_length, heads),
                  accumulators[local_tile][0]);
        atomicAdd(query_gradient_accumulator + tensor_index(batch, top_query, head, column + 1, sequence_length, heads),
                  accumulators[local_tile][1]);
        atomicAdd(query_gradient_accumulator + tensor_index(batch, bottom_query, head, column, sequence_length, heads),
                  accumulators[local_tile][2]);
        atomicAdd(query_gradient_accumulator + tensor_index(batch, bottom_query, head, column + 1, sequence_length, heads),
                  accumulators[local_tile][3]);
    }
}

template <int FIRST_TILE>
__device__ __forceinline__ void store_key_value_gradient(
    __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
    const float (&key_accumulators)[HALF_OUTPUT_TILES][4], const float (&value_accumulators)[HALF_OUTPUT_TILES][4],
    int batch, int first_key, int head, int sequence_length, int heads, int warp_row, int lane) {
    const int top_key = first_key + warp_row + lane / 4;
    const int bottom_key = top_key + 8;
#pragma unroll
    for (int local_tile = 0; local_tile < HALF_OUTPUT_TILES; ++local_tile) {
        const int column = (FIRST_TILE + local_tile) * MMA_N + (lane % 4) * 2;
        store_pair(key_gradient + tensor_index(batch, top_key, head, column, sequence_length, heads),
                   make_float2(key_accumulators[local_tile][0], key_accumulators[local_tile][1]));
        store_pair(key_gradient + tensor_index(batch, bottom_key, head, column, sequence_length, heads),
                   make_float2(key_accumulators[local_tile][2], key_accumulators[local_tile][3]));
        store_pair(value_gradient + tensor_index(batch, top_key, head, column, sequence_length, heads),
                   make_float2(value_accumulators[local_tile][0], value_accumulators[local_tile][1]));
        store_pair(value_gradient + tensor_index(batch, bottom_key, head, column, sequence_length, heads),
                   make_float2(value_accumulators[local_tile][2], value_accumulators[local_tile][3]));
    }
}

__global__ __launch_bounds__(BACKWARD_THREADS, 1) void flash_attention_backward_fused_kernel(
    float* __restrict__ query_gradient_accumulator, __nv_bfloat16* __restrict__ key_gradient,
    __nv_bfloat16* __restrict__ value_gradient, const float* __restrict__ row_delta,
    const __nv_bfloat16* __restrict__ output_gradient, const float* __restrict__ logsumexp,
    const __nv_bfloat16* __restrict__ query, const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value, int sequence_length, int heads, float scale) {
    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* shared_key = shared;
    __nv_bfloat16* shared_value = shared_key + BN * D;
    __nv_bfloat16* shared_query = shared_value + BN * D;
    __nv_bfloat16* shared_output_gradient = shared_query + BM * D;
    __nv_bfloat16* shared_temporary = shared_output_gradient + BM * D;
    __nv_bfloat16* shared_score_gradient_transposed = shared_temporary + BM * BN;

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int local_warp = warp % 4;
    const int column_group = warp / 4;
    const int warp_row = local_warp * MMA_M;
    const int batch_head = blockIdx.y;
    const int batch = batch_head / heads;
    const int head = batch_head % heads;
    const int first_key = blockIdx.x * BN;

    if (threadIdx.x < THREADS) {
        copy_bf16_tile(shared_key, key, batch, first_key, head, sequence_length, heads);
        copy_bf16_tile(shared_value, value, batch, first_key, head, sequence_length, heads);
        cp_async_wait();
    }
    __syncthreads();

    float key_gradient_accumulators[HALF_OUTPUT_TILES][4] = {};
    float value_gradient_accumulators[HALF_OUTPUT_TILES][4] = {};

    for (int first_query = first_key; first_query < sequence_length; first_query += BM) {
        if (threadIdx.x < THREADS) {
            copy_bf16_tile(shared_query, query, batch, first_query, head, sequence_length, heads);
            copy_bf16_tile(shared_output_gradient, output_gradient, batch, first_query, head, sequence_length, heads);
            cp_async_wait();
        }
        __syncthreads();

        float probabilities[SCORE_N_TILES][4];
        float score_gradients[SCORE_N_TILES][4];
        if (warp < 4) {
            unsigned int query_fragments[HEAD_K_TILES][4];
            unsigned int output_gradient_fragments[HEAD_K_TILES][4];
            load_left_fragments(query_fragments, shared_query, warp_row, lane);
            load_left_fragments(output_gradient_fragments, shared_output_gradient, warp_row, lane);
            float scores[SCORE_N_TILES][4] = {};
            float probability_gradients[SCORE_N_TILES][4] = {};
            matrix_product_transposed_right(scores, query_fragments, shared_key, lane);
            matrix_product_transposed_right(probability_gradients, output_gradient_fragments, shared_value, lane);

            const int top_query = first_query + warp_row + lane / 4;
            const int bottom_query = top_query + 8;
            make_probability_and_score_gradient(
                probabilities, score_gradients, scores, probability_gradients,
                row_delta[batch_head * sequence_length + top_query],
                row_delta[batch_head * sequence_length + bottom_query],
                logsumexp[batch_head * sequence_length + top_query],
                logsumexp[batch_head * sequence_length + bottom_query],
                first_query, first_key, warp_row, lane, scale);
            store_score_gradient(shared_temporary, shared_score_gradient_transposed, score_gradients, warp_row, lane);
        }
        __syncthreads();

        unsigned int score_gradient_fragments[TOKEN_K_TILES][4];
        load_left_fragments<BM>(score_gradient_fragments, shared_temporary, warp_row, lane);
        float query_gradient_accumulators[HALF_OUTPUT_TILES][4] = {};
        if (column_group == 0) {
            matrix_product_right_half<0>(query_gradient_accumulators, score_gradient_fragments, shared_key, lane);
            atomic_add_query_gradient<0>(query_gradient_accumulator, query_gradient_accumulators, batch, first_query, head, sequence_length, heads,
                                         warp_row, lane);
        } else {
            matrix_product_right_half<HALF_OUTPUT_TILES>(query_gradient_accumulators, score_gradient_fragments, shared_key, lane);
            atomic_add_query_gradient<HALF_OUTPUT_TILES>(query_gradient_accumulator, query_gradient_accumulators, batch, first_query, head,
                                                         sequence_length, heads, warp_row, lane);
        }
        __syncthreads();

        if (warp < 4) {
            store_probability_transposed(shared_temporary, probabilities, warp_row, lane);
        }
        __syncthreads();

        unsigned int transposed_score_gradient_fragments[TOKEN_K_TILES][4];
        unsigned int probability_fragments[TOKEN_K_TILES][4];
        load_left_fragments<BM>(transposed_score_gradient_fragments, shared_score_gradient_transposed, warp_row, lane);
        load_left_fragments<BM>(probability_fragments, shared_temporary, warp_row, lane);
        if (column_group == 0) {
            matrix_product_right_half<0>(key_gradient_accumulators, transposed_score_gradient_fragments, shared_query, lane);
            matrix_product_right_half<0>(value_gradient_accumulators, probability_fragments, shared_output_gradient, lane);
        } else {
            matrix_product_right_half<HALF_OUTPUT_TILES>(key_gradient_accumulators, transposed_score_gradient_fragments, shared_query, lane);
            matrix_product_right_half<HALF_OUTPUT_TILES>(value_gradient_accumulators, probability_fragments, shared_output_gradient, lane);
        }
        __syncthreads();
    }

    if (column_group == 0) {
        store_key_value_gradient<0>(key_gradient, value_gradient, key_gradient_accumulators, value_gradient_accumulators, batch, first_key, head,
                                    sequence_length, heads, warp_row, lane);
    } else {
        store_key_value_gradient<HALF_OUTPUT_TILES>(key_gradient, value_gradient, key_gradient_accumulators, value_gradient_accumulators, batch, first_key,
                                                    head, sequence_length, heads, warp_row, lane);
    }
}

__global__ void flash_attention_backward_convert_query_kernel(
    __nv_bfloat16* query_gradient, const float* query_gradient_accumulator, int elements) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair * 2 < elements) {
        const float2 value = reinterpret_cast<const float2*>(query_gradient_accumulator)[pair];
        *reinterpret_cast<__nv_bfloat162*>(query_gradient + pair * 2) = __float22bfloat162_rn(value);
    }
}

void launch_backward(
    __nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
    float* query_gradient_accumulator, float* row_delta,
    const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, const float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch_size, int sequence_length, int heads, float scale, cudaStream_t stream) {
    static thread_local int configured_device = -1;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device != configured_device) {
        CUDA_CHECK(cudaFuncSetAttribute(flash_attention_backward_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BACKWARD_SHARED_BYTES));
        configured_device = device;
    }

    const dim3 tile_grid(sequence_length / BM, batch_size * heads);
    flash_attention_backward_preprocess_kernel<<<tile_grid, BACKWARD_THREADS, 0, stream>>>(
        query_gradient_accumulator, row_delta, output_gradient, output, sequence_length, heads);
    flash_attention_backward_fused_kernel<<<tile_grid, BACKWARD_THREADS, BACKWARD_SHARED_BYTES, stream>>>(
        query_gradient_accumulator, key_gradient, value_gradient, row_delta, output_gradient, logsumexp, query, key, value, sequence_length, heads, scale);

    const int elements = batch_size * sequence_length * heads * D;
    constexpr int CONVERT_THREADS = 256;
    const int pairs = elements / 2;
    flash_attention_backward_convert_query_kernel<<<(pairs + CONVERT_THREADS - 1) / CONVERT_THREADS, CONVERT_THREADS, 0, stream>>>(
        query_gradient, query_gradient_accumulator, elements);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace tensor_core
}  // namespace flash_attention_sm89

void flash_attention_backward_sm89_cuda(
    __nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
    float* query_gradient_accumulator, float* row_delta,
    const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, const float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch_size, int sequence_length, int heads, int head_size, float scale, cudaStream_t stream) {
    flash_attention_sm89::validate_head_size(head_size);
    if (sequence_length <= 0 || sequence_length % flash_attention_sm89::tensor_core::BM != 0) {
        throw std::runtime_error("flash attention requires T to be a positive multiple of 64");
    }
    flash_attention_sm89::tensor_core::launch_backward(
        query_gradient, key_gradient, value_gradient, query_gradient_accumulator, row_delta,
        output_gradient, output, logsumexp, query, key, value,
        batch_size, sequence_length, heads, scale, stream);
}

}  // namespace dscuda

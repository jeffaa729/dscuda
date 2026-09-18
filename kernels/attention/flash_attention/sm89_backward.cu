// Implements the SM89 D128 causal FlashAttention backward path with BF16 Tensor Cores and FP32 accumulation.
// It reconstructs probabilities from saved log-sum-exp values and computes dQ separately from dK/dV to avoid atomics and quadratic storage.

#include "sm89_common.cuh"

namespace dscuda {
namespace flash_attention_sm89 {
namespace tensor_core {

__device__ __forceinline__ void row_delta(float& top, float& bottom, const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, int batch,
                                          int first_query, int head, int sequence_length, int heads, int warp_row, int lane) {
    const int subgroup_lane = lane % 4;
    const int top_query = first_query + warp_row + lane / 4;
    const int bottom_query = top_query + 8;
    top = 0.0F;
    bottom = 0.0F;
    for (int column = subgroup_lane; column < D; column += 4) {
        const int top_index = tensor_index(batch, top_query, head, column, sequence_length, heads);
        const int bottom_index = tensor_index(batch, bottom_query, head, column, sequence_length, heads);
        top += as_float(output_gradient[top_index]) * as_float(output[top_index]);
        bottom += as_float(output_gradient[bottom_index]) * as_float(output[bottom_index]);
    }
    top += __shfl_xor_sync(0xffffffff, top, 1);
    top += __shfl_xor_sync(0xffffffff, top, 2);
    bottom += __shfl_xor_sync(0xffffffff, bottom, 1);
    bottom += __shfl_xor_sync(0xffffffff, bottom, 2);
}

__device__ __forceinline__ void store_gradient_pair(__nv_bfloat16* destination, float2 value) {
    store_pair(destination, value);
}

__global__ __launch_bounds__(THREADS, 1) void flash_attention_backward_query_tensor_core_kernel(
    __nv_bfloat16* __restrict__ query_gradient, const __nv_bfloat16* __restrict__ output_gradient, const __nv_bfloat16* __restrict__ output,
    const float* __restrict__ logsumexp, const __nv_bfloat16* __restrict__ query, const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value, int sequence_length, int heads, float scale) {
    // D128 needs 64 KiB here, exceeding the default 48 KiB static limit.
    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* shared_query = shared;
    __nv_bfloat16* shared_output_gradient = shared_query + BM * D;
    __nv_bfloat16* shared_key = shared_output_gradient + BM * D;
    __nv_bfloat16* shared_value = shared_key + BN * D;

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int batch_head = blockIdx.y;
    const int batch = batch_head / heads;
    const int head = batch_head % heads;
    const int first_query = blockIdx.x * BM;
    const int warp_row = warp * MMA_M;

    copy_bf16_tile(shared_query, query, batch, first_query, head, sequence_length, heads);
    copy_gradient_tile(shared_output_gradient, output_gradient, batch, first_query, head, sequence_length, heads);
    cp_async_wait();
    __syncthreads();

    unsigned int query_fragments[HEAD_K_TILES][4];
    unsigned int output_gradient_fragments[HEAD_K_TILES][4];
    load_left_fragments(query_fragments, shared_query, warp_row, lane);
    load_left_fragments(output_gradient_fragments, shared_output_gradient, warp_row, lane);
    float query_gradient_accumulators[OUTPUT_N_TILES][4] = {};
    float delta[2];
    row_delta(delta[0], delta[1], output_gradient, output, batch, first_query, head, sequence_length, heads, warp_row, lane);
    const int top_query = first_query + warp_row + lane / 4;
    const int bottom_query = top_query + 8;
    const float row_logsumexp[2] = {logsumexp[batch_head * sequence_length + top_query], logsumexp[batch_head * sequence_length + bottom_query]};

    for (int first_key = 0; first_key <= first_query; first_key += BN) {
        copy_bf16_tile(shared_key, key, batch, first_key, head, sequence_length, heads);
        copy_bf16_tile(shared_value, value, batch, first_key, head, sequence_length, heads);
        cp_async_wait();
        __syncthreads();

        float scores[SCORE_N_TILES][4] = {};
        float probability_gradients[SCORE_N_TILES][4] = {};
        matrix_product_transposed_right(scores, query_fragments, shared_key, lane);
        matrix_product_transposed_right(probability_gradients, output_gradient_fragments, shared_value, lane);

        const int local_top_row = warp_row + lane / 4;
        const int local_bottom_row = local_top_row + 8;
        float2 score_gradient_top[SCORE_N_TILES];
        float2 score_gradient_bottom[SCORE_N_TILES];
#pragma unroll
        for (int tile = 0; tile < SCORE_N_TILES; ++tile) {
            const int key_column = tile * MMA_N + (lane % 4) * 2;
            const bool top_visible = first_key != first_query || local_top_row >= key_column;
            const bool bottom_visible = first_key != first_query || local_bottom_row >= key_column;
            const float top_probability0 = top_visible ? expf(scale * scores[tile][0] - row_logsumexp[0]) : 0.0F;
            const float top_probability1 =
                first_key != first_query || local_top_row >= key_column + 1 ? expf(scale * scores[tile][1] - row_logsumexp[0]) : 0.0F;
            const float bottom_probability0 = bottom_visible ? expf(scale * scores[tile][2] - row_logsumexp[1]) : 0.0F;
            const float bottom_probability1 =
                first_key != first_query || local_bottom_row >= key_column + 1 ? expf(scale * scores[tile][3] - row_logsumexp[1]) : 0.0F;
            score_gradient_top[tile] = make_float2(scale * top_probability0 * (probability_gradients[tile][0] - delta[0]),
                                                   scale * top_probability1 * (probability_gradients[tile][1] - delta[0]));
            score_gradient_bottom[tile] = make_float2(scale * bottom_probability0 * (probability_gradients[tile][2] - delta[1]),
                                                      scale * bottom_probability1 * (probability_gradients[tile][3] - delta[1]));
        }

        unsigned int score_gradient_fragments[TOKEN_K_TILES][4];
        pack_row_matrix(score_gradient_fragments, score_gradient_top, score_gradient_bottom);
        matrix_product_right(query_gradient_accumulators, score_gradient_fragments, shared_key, lane);
        __syncthreads();
    }

#pragma unroll
    for (int tile = 0; tile < OUTPUT_N_TILES; ++tile) {
        const int column = tile * MMA_N + (lane % 4) * 2;
        __nv_bfloat16* top = query_gradient + tensor_index(batch, top_query, head, column, sequence_length, heads);
        __nv_bfloat16* bottom = query_gradient + tensor_index(batch, bottom_query, head, column, sequence_length, heads);
        store_gradient_pair(top, make_float2(query_gradient_accumulators[tile][0], query_gradient_accumulators[tile][1]));
        store_gradient_pair(bottom, make_float2(query_gradient_accumulators[tile][2], query_gradient_accumulators[tile][3]));
    }
}

__device__ __forceinline__ void store_transposed_rows(__nv_bfloat16* shared_score_gradient, __nv_bfloat16* shared_probability,
                                                      const float (&scores)[SCORE_N_TILES][4], const float (&probability_gradients)[SCORE_N_TILES][4],
                                                      const float (&delta)[2], const float (&row_logsumexp)[2], int first_query, int first_key, int warp_row,
                                                      int lane, float scale) {
    const int top_row = warp_row + lane / 4;
    const int bottom_row = top_row + 8;
#pragma unroll
    for (int tile = 0; tile < SCORE_N_TILES; ++tile) {
        const int key_column = tile * MMA_N + (lane % 4) * 2;
        const bool triangular = first_query == first_key;
        const float probabilities[4] = {!triangular || top_row >= key_column ? expf(scale * scores[tile][0] - row_logsumexp[0]) : 0.0F,
                                        !triangular || top_row >= key_column + 1 ? expf(scale * scores[tile][1] - row_logsumexp[0]) : 0.0F,
                                        !triangular || bottom_row >= key_column ? expf(scale * scores[tile][2] - row_logsumexp[1]) : 0.0F,
                                        !triangular || bottom_row >= key_column + 1 ? expf(scale * scores[tile][3] - row_logsumexp[1]) : 0.0F};
        const float score_gradients[4] = {
            scale * probabilities[0] * (probability_gradients[tile][0] - delta[0]), scale * probabilities[1] * (probability_gradients[tile][1] - delta[0]),
            scale * probabilities[2] * (probability_gradients[tile][2] - delta[1]), scale * probabilities[3] * (probability_gradients[tile][3] - delta[1])};
        const int query_rows[4] = {top_row, top_row, bottom_row, bottom_row};
        const int key_rows[4] = {key_column, key_column + 1, key_column, key_column + 1};
#pragma unroll
        for (int element = 0; element < 4; ++element) {
            // P^T and dS^T have BM columns, not D columns.
            const int transposed = key_rows[element] * BM + query_rows[element];
            shared_score_gradient[swizzle<BM>(transposed)] = __float2bfloat16(score_gradients[element]);
            shared_probability[swizzle<BM>(transposed)] = __float2bfloat16(probabilities[element]);
        }
    }
}

__global__ __launch_bounds__(THREADS, 1) void flash_attention_backward_key_value_tensor_core_kernel(
    __nv_bfloat16* __restrict__ key_gradient, __nv_bfloat16* __restrict__ value_gradient, const __nv_bfloat16* __restrict__ output_gradient,
    const __nv_bfloat16* __restrict__ output, const float* __restrict__ logsumexp, const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value, int sequence_length, int heads, float scale) {
    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* shared_left = shared;
    __nv_bfloat16* shared_right = shared_left + BN * D;
    __nv_bfloat16* shared_query = shared_right + BN * D;
    __nv_bfloat16* shared_output_gradient = shared_query + BM * D;

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int batch_head = blockIdx.y;
    const int batch = batch_head / heads;
    const int head = batch_head % heads;
    const int first_key = blockIdx.x * BN;
    const int warp_row = warp * MMA_M;
    float key_gradient_accumulators[OUTPUT_N_TILES][4] = {};
    float value_gradient_accumulators[OUTPUT_N_TILES][4] = {};

    for (int first_query = first_key; first_query < sequence_length; first_query += BM) {
        copy_bf16_tile(shared_left, key, batch, first_key, head, sequence_length, heads);
        copy_bf16_tile(shared_right, value, batch, first_key, head, sequence_length, heads);
        copy_bf16_tile(shared_query, query, batch, first_query, head, sequence_length, heads);
        copy_gradient_tile(shared_output_gradient, output_gradient, batch, first_query, head, sequence_length, heads);
        cp_async_wait();
        __syncthreads();

        unsigned int query_fragments[HEAD_K_TILES][4];
        unsigned int output_gradient_fragments[HEAD_K_TILES][4];
        load_left_fragments(query_fragments, shared_query, warp_row, lane);
        load_left_fragments(output_gradient_fragments, shared_output_gradient, warp_row, lane);
        float scores[SCORE_N_TILES][4] = {};
        float probability_gradients[SCORE_N_TILES][4] = {};
        matrix_product_transposed_right(scores, query_fragments, shared_left, lane);
        matrix_product_transposed_right(probability_gradients, output_gradient_fragments, shared_right, lane);
        float delta[2];
        row_delta(delta[0], delta[1], output_gradient, output, batch, first_query, head, sequence_length, heads, warp_row, lane);
        const int top_query = first_query + warp_row + lane / 4;
        const int bottom_query = top_query + 8;
        const float row_logsumexp[2] = {logsumexp[batch_head * sequence_length + top_query], logsumexp[batch_head * sequence_length + bottom_query]};

        __syncthreads();
        store_transposed_rows(shared_left, shared_right, scores, probability_gradients, delta, row_logsumexp, first_query, first_key, warp_row, lane, scale);
        __syncthreads();

        unsigned int score_gradient_fragments[TOKEN_K_TILES][4];
        unsigned int probability_fragments[TOKEN_K_TILES][4];
        load_left_fragments<BM>(score_gradient_fragments, shared_left, warp_row, lane);
        load_left_fragments<BM>(probability_fragments, shared_right, warp_row, lane);
        matrix_product_right(key_gradient_accumulators, score_gradient_fragments, shared_query, lane);
        matrix_product_right(value_gradient_accumulators, probability_fragments, shared_output_gradient, lane);
        __syncthreads();
    }

    const int top_key = first_key + warp_row + lane / 4;
    const int bottom_key = top_key + 8;
#pragma unroll
    for (int tile = 0; tile < OUTPUT_N_TILES; ++tile) {
        const int column = tile * MMA_N + (lane % 4) * 2;
        __nv_bfloat16* key_top = key_gradient + tensor_index(batch, top_key, head, column, sequence_length, heads);
        __nv_bfloat16* key_bottom = key_gradient + tensor_index(batch, bottom_key, head, column, sequence_length, heads);
        __nv_bfloat16* value_top = value_gradient + tensor_index(batch, top_key, head, column, sequence_length, heads);
        __nv_bfloat16* value_bottom = value_gradient + tensor_index(batch, bottom_key, head, column, sequence_length, heads);
        store_gradient_pair(key_top, make_float2(key_gradient_accumulators[tile][0], key_gradient_accumulators[tile][1]));
        store_gradient_pair(key_bottom, make_float2(key_gradient_accumulators[tile][2], key_gradient_accumulators[tile][3]));
        store_gradient_pair(value_top, make_float2(value_gradient_accumulators[tile][0], value_gradient_accumulators[tile][1]));
        store_gradient_pair(value_bottom, make_float2(value_gradient_accumulators[tile][2], value_gradient_accumulators[tile][3]));
    }
}

void launch_backward(__nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient, const __nv_bfloat16* output_gradient,
                     const __nv_bfloat16* output, const float* logsumexp, const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
                     int batch_size, int sequence_length, int heads, float scale, cudaStream_t stream) {
    // Configure before capture during warm-up, then avoid repeated host setup.
    // Like the rest of this runtime, this uses the device's primary CUDA context.
    static thread_local int configured_device = -1;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device != configured_device) {
        CUDA_CHECK(cudaFuncSetAttribute(flash_attention_backward_query_tensor_core_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BACKWARD_SHARED_BYTES));
        CUDA_CHECK(
            cudaFuncSetAttribute(flash_attention_backward_key_value_tensor_core_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BACKWARD_SHARED_BYTES));
        configured_device = device;
    }
    const dim3 grid(sequence_length / BM, batch_size * heads);
    flash_attention_backward_query_tensor_core_kernel<<<grid, THREADS, BACKWARD_SHARED_BYTES, stream>>>(query_gradient, output_gradient, output, logsumexp,
                                                                                                        query, key, value, sequence_length, heads, scale);
    flash_attention_backward_key_value_tensor_core_kernel<<<grid, THREADS, BACKWARD_SHARED_BYTES, stream>>>(
        key_gradient, value_gradient, output_gradient, output, logsumexp, query, key, value, sequence_length, heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace tensor_core
}  // namespace flash_attention_sm89

void flash_attention_backward_sm89_cuda(__nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
                                        const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, const float* logsumexp,
                                        const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value, int batch_size,
                                        int sequence_length, int heads, int head_size, float scale, cudaStream_t stream) {
    flash_attention_sm89::validate_head_size(head_size);
    if (sequence_length <= 0 || sequence_length % flash_attention_sm89::tensor_core::BM != 0) {
        throw std::runtime_error("flash attention requires T to be a positive multiple of 64");
    }
    flash_attention_sm89::tensor_core::launch_backward(query_gradient, key_gradient, value_gradient, output_gradient, output, logsumexp, query, key,
                                                       value, batch_size, sequence_length, heads, scale, stream);
}

}  // namespace dscuda

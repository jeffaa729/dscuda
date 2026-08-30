// Fuses causal online softmax with DeepSeek MLA's absorbed query and shared compressed KV representation in BF16.
// The backward pass recomputes probabilities from saved FP32 log-sum-exp values and assigns one CTA to every independently writable gradient row.

#include "cuda_common.h"
#include "mla.h"

#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

namespace dscuda {
namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_KV_RANK = 512;

template <int WIDTH = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int mask = WIDTH / 2; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, mask);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < BLOCK_SIZE / WARP_SIZE
        ? warp_sums[threadIdx.x]
        : 0.0F;
    if (warp == 0) {
        value = warp_reduce_sum<BLOCK_SIZE / WARP_SIZE>(value);
    }
    // All warps must finish reading scratch before the next reduction reuses it.
    __syncthreads();
    return value;
}

__device__ __forceinline__ int query_offset(
    int batch,
    int token,
    int head,
    int sequence_length,
    int heads,
    int width) {
    return ((batch * sequence_length + token) * heads + head) * width;
}

__device__ __forceinline__ int shared_offset(
    int batch,
    int token,
    int sequence_length,
    int width) {
    return (batch * sequence_length + token) * width;
}

__device__ __forceinline__ int lse_offset(
    int batch,
    int head,
    int token,
    int heads,
    int sequence_length) {
    return (batch * heads + head) * sequence_length + token;
}

__device__ float dot_query_key(
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int query_base,
    int query_rope_base,
    int kv_base,
    int key_rope_base,
    int kv_rank,
    int rope_size) {
    float partial = 0.0F;
    for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
        partial += __bfloat162float(query_latent[query_base + column]) *
                   __bfloat162float(kv_latent[kv_base + column]);
    }
    for (int column = threadIdx.x; column < rope_size; column += BLOCK_SIZE) {
        partial += __bfloat162float(query_rope[query_rope_base + column]) *
                   __bfloat162float(key_rope[key_rope_base + column]);
    }
    return block_reduce_sum(partial);
}

__global__ void mla_forward_kernel(
    float* output,
    float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale) {
    __shared__ float output_accumulator[MAX_KV_RANK];
    __shared__ float row_maximum;
    __shared__ float row_normalizer;
    __shared__ float previous_scale;
    __shared__ float probability_scale;

    const int query_token = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int query_base = query_offset(
        batch, query_token, head, sequence_length, heads, kv_rank);
    const int query_rope_base = query_offset(
        batch, query_token, head, sequence_length, heads, rope_size);

    for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
        output_accumulator[column] = 0.0F;
    }
    if (threadIdx.x == 0) {
        row_maximum = -__int_as_float(0x7f800000);
        row_normalizer = 0.0F;
    }
    __syncthreads();

    for (int key_token = 0; key_token <= query_token; ++key_token) {
        const int kv_base = shared_offset(
            batch, key_token, sequence_length, kv_rank);
        const int key_rope_base = shared_offset(
            batch, key_token, sequence_length, rope_size);
        const float score = dot_query_key(
            query_latent,
            query_rope,
            kv_latent,
            key_rope,
            query_base,
            query_rope_base,
            kv_base,
            key_rope_base,
            kv_rank,
            rope_size) *
            scale;

        if (threadIdx.x == 0) {
            const float next_maximum = fmaxf(row_maximum, score);
            previous_scale = expf(row_maximum - next_maximum);
            probability_scale = expf(score - next_maximum);
            row_normalizer =
                row_normalizer * previous_scale + probability_scale;
            row_maximum = next_maximum;
        }
        __syncthreads();

        for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
            output_accumulator[column] =
                output_accumulator[column] * previous_scale +
                probability_scale * __bfloat162float(kv_latent[kv_base + column]);
        }
        __syncthreads();
    }

    for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
        output[query_base + column] =
            output_accumulator[column] / row_normalizer;
    }
    if (threadIdx.x == 0) {
        logsumexp[lse_offset(
            batch, head, query_token, heads, sequence_length)] =
            row_maximum + logf(row_normalizer);
    }
}

namespace mla_tensor_core {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int C = 64;
constexpr int R = 32;
constexpr int K = C + R;
constexpr int THREADS = 128;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int SCORE_K_TILES = K / MMA_K;
constexpr int VALUE_K_TILES = BN / MMA_K;
constexpr int N_TILES = C / MMA_N;

__device__ __forceinline__ int swizzle(int offset) {
    return offset ^ ((offset & (7 << 6)) >> 3);
}

__device__ __forceinline__ unsigned int shared_address(
    const void* pointer) {
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

__device__ __forceinline__ void load_matrix_x2(
    unsigned int (&fragment)[2],
    unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address));
}

__device__ __forceinline__ void load_matrix_x2_transpose(
    unsigned int (&fragment)[2],
    unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address));
}

__device__ __forceinline__ void mma(
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

__device__ __forceinline__ unsigned int pack_bf16x2(float2 values) {
    union Packed {
        __nv_bfloat162 bf16;
        unsigned int bits;
    } packed;
    packed.bf16 = __float22bfloat162_rn(values);
    return packed.bits;
}

__device__ __forceinline__ void copy_query_tile(
    __nv_bfloat16* shared,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    int batch,
    int first_query,
    int head,
    int sequence_length,
    int heads) {
    for (int index = threadIdx.x; index < BM * K; index += THREADS) {
        const int row = index / K;
        const int column = index % K;
        shared[swizzle(index)] = column < C
            ? query_latent[query_offset(
                  batch,
                  first_query + row,
                  head,
                  sequence_length,
                  heads,
                  C) + column]
            : query_rope[query_offset(
                  batch,
                  first_query + row,
                  head,
                  sequence_length,
                  heads,
                  R) + column - C];
    }
}

__device__ __forceinline__ void copy_key_value_tile(
    __nv_bfloat16* shared_key,
    __nv_bfloat16* shared_value,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int batch,
    int first_key,
    int sequence_length) {
    for (int index = threadIdx.x; index < BN * K; index += THREADS) {
        const int row = index / K;
        const int column = index % K;
        shared_key[swizzle(index)] = column < C
            ? kv_latent[shared_offset(
                  batch, first_key + row, sequence_length, C) + column]
            : key_rope[shared_offset(
                  batch, first_key + row, sequence_length, R) + column - C];
    }
    for (int index = threadIdx.x; index < BN * C; index += THREADS) {
        const int row = index / C;
        const int column = index % C;
        shared_value[swizzle(index)] = kv_latent[shared_offset(
            batch, first_key + row, sequence_length, C) + column];
    }
}

__device__ __forceinline__ void load_query_fragments(
    unsigned int (&fragments)[SCORE_K_TILES][4],
    const __nv_bfloat16* shared,
    int warp_row,
    int lane) {
#pragma unroll
    for (int tile = 0; tile < SCORE_K_TILES; ++tile) {
        const int row = warp_row + lane % MMA_M;
        const int column = tile * MMA_K + lane / MMA_M * 8;
        load_matrix_x4(
            fragments[tile],
            shared_address(shared + swizzle(row * K + column)));
    }
}

__device__ __forceinline__ void score_matrix_product(
    float (&accumulators)[N_TILES][4],
    const unsigned int (&query)[SCORE_K_TILES][4],
    const __nv_bfloat16* key,
    int lane) {
#pragma unroll
    for (int tile_inner = 0; tile_inner < SCORE_K_TILES; ++tile_inner) {
#pragma unroll
        for (int tile_column = 0; tile_column < N_TILES; ++tile_column) {
            const int row = tile_column * MMA_N + lane % MMA_N;
            const int column =
                tile_inner * MMA_K + (lane % MMA_M) / 8 * 8;
            unsigned int key_fragment[2];
            load_matrix_x2(
                key_fragment,
                shared_address(key + swizzle(row * K + column)));
            mma(accumulators[tile_column], query[tile_inner], key_fragment);
        }
    }
}

__device__ __forceinline__ void pack_probability_fragments(
    unsigned int (&fragments)[VALUE_K_TILES][4],
    const float2 (&top)[N_TILES],
    const float2 (&bottom)[N_TILES]) {
#pragma unroll
    for (int tile = 0; tile < VALUE_K_TILES; ++tile) {
        fragments[tile][0] = pack_bf16x2(top[tile * 2]);
        fragments[tile][1] = pack_bf16x2(bottom[tile * 2]);
        fragments[tile][2] = pack_bf16x2(top[tile * 2 + 1]);
        fragments[tile][3] = pack_bf16x2(bottom[tile * 2 + 1]);
    }
}

__device__ __forceinline__ void value_matrix_product(
    float (&accumulators)[N_TILES][4],
    const unsigned int (&probability)[VALUE_K_TILES][4],
    const __nv_bfloat16* value,
    int lane) {
#pragma unroll
    for (int tile_inner = 0; tile_inner < VALUE_K_TILES; ++tile_inner) {
#pragma unroll
        for (int tile_column = 0; tile_column < N_TILES; ++tile_column) {
            const int row = tile_inner * MMA_K + lane % MMA_M;
            const int column = tile_column * MMA_N;
            unsigned int value_fragment[2];
            load_matrix_x2_transpose(
                value_fragment,
                shared_address(value + swizzle(row * C + column)));
            mma(
                accumulators[tile_column],
                probability[tile_inner],
                value_fragment);
        }
    }
}

__global__ __launch_bounds__(THREADS, 2)
void mla_forward_tensor_core_kernel(
    float* __restrict__ output,
    float* __restrict__ logsumexp,
    const __nv_bfloat16* __restrict__ query_latent,
    const __nv_bfloat16* __restrict__ query_rope,
    const __nv_bfloat16* __restrict__ kv_latent,
    const __nv_bfloat16* __restrict__ key_rope,
    int sequence_length,
    int heads,
    float scale) {
    __shared__ __align__(16) __nv_bfloat16 shared_query[BM * K];
    __shared__ __align__(16) __nv_bfloat16 shared_key[BN * K];
    __shared__ __align__(16) __nv_bfloat16 shared_value[BN * C];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int batch_head = blockIdx.y;
    const int batch = batch_head / heads;
    const int head = batch_head % heads;
    const int first_query = blockIdx.x * BM;
    const int warp_row = warp * MMA_M;

    copy_query_tile(
        shared_query,
        query_latent,
        query_rope,
        batch,
        first_query,
        head,
        sequence_length,
        heads);
    __syncthreads();

    unsigned int query_fragments[SCORE_K_TILES][4];
    load_query_fragments(query_fragments, shared_query, warp_row, lane);
    float output_accumulators[N_TILES][4] = {};
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0F, 0.0F};

    for (int first_key = 0; first_key <= first_query; first_key += BN) {
        copy_key_value_tile(
            shared_key,
            shared_value,
            kv_latent,
            key_rope,
            batch,
            first_key,
            sequence_length);
        __syncthreads();

        float scores[N_TILES][4] = {};
        score_matrix_product(scores, query_fragments, shared_key, lane);
        const int local_top_row = warp_row + lane / 4;
        const int local_bottom_row = local_top_row + 8;
        float current_max[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
        for (int tile = 0; tile < N_TILES; ++tile) {
            const int key_column = tile * MMA_N + (lane % 4) * 2;
            if (first_key == first_query) {
                scores[tile][0] = local_top_row >= key_column
                    ? scores[tile][0] * scale
                    : -FLT_MAX;
                scores[tile][1] = local_top_row >= key_column + 1
                    ? scores[tile][1] * scale
                    : -FLT_MAX;
                scores[tile][2] = local_bottom_row >= key_column
                    ? scores[tile][2] * scale
                    : -FLT_MAX;
                scores[tile][3] = local_bottom_row >= key_column + 1
                    ? scores[tile][3] * scale
                    : -FLT_MAX;
            } else {
                scores[tile][0] *= scale;
                scores[tile][1] *= scale;
                scores[tile][2] *= scale;
                scores[tile][3] *= scale;
            }
            current_max[0] = fmaxf(
                current_max[0], fmaxf(scores[tile][0], scores[tile][1]));
            current_max[1] = fmaxf(
                current_max[1], fmaxf(scores[tile][2], scores[tile][3]));
        }
        current_max[0] = fmaxf(
            current_max[0],
            __shfl_xor_sync(0xffffffffU, current_max[0], 1));
        current_max[0] = fmaxf(
            current_max[0],
            __shfl_xor_sync(0xffffffffU, current_max[0], 2));
        current_max[1] = fmaxf(
            current_max[1],
            __shfl_xor_sync(0xffffffffU, current_max[1], 1));
        current_max[1] = fmaxf(
            current_max[1],
            __shfl_xor_sync(0xffffffffU, current_max[1], 2));

        const float next_max[2] = {
            fmaxf(row_max[0], current_max[0]),
            fmaxf(row_max[1], current_max[1])};
        const float previous_scale[2] = {
            expf(row_max[0] - next_max[0]),
            expf(row_max[1] - next_max[1])};
#pragma unroll
        for (int tile = 0; tile < N_TILES; ++tile) {
            output_accumulators[tile][0] *= previous_scale[0];
            output_accumulators[tile][1] *= previous_scale[0];
            output_accumulators[tile][2] *= previous_scale[1];
            output_accumulators[tile][3] *= previous_scale[1];
        }
        row_sum[0] *= previous_scale[0];
        row_sum[1] *= previous_scale[1];

        float2 probability_top[N_TILES];
        float2 probability_bottom[N_TILES];
        float local_sum[2] = {};
#pragma unroll
        for (int tile = 0; tile < N_TILES; ++tile) {
            probability_top[tile] = make_float2(
                expf(scores[tile][0] - next_max[0]),
                expf(scores[tile][1] - next_max[0]));
            probability_bottom[tile] = make_float2(
                expf(scores[tile][2] - next_max[1]),
                expf(scores[tile][3] - next_max[1]));
            local_sum[0] +=
                probability_top[tile].x + probability_top[tile].y;
            local_sum[1] +=
                probability_bottom[tile].x + probability_bottom[tile].y;
        }
        local_sum[0] += __shfl_xor_sync(0xffffffffU, local_sum[0], 1);
        local_sum[0] += __shfl_xor_sync(0xffffffffU, local_sum[0], 2);
        local_sum[1] += __shfl_xor_sync(0xffffffffU, local_sum[1], 1);
        local_sum[1] += __shfl_xor_sync(0xffffffffU, local_sum[1], 2);
        row_sum[0] += local_sum[0];
        row_sum[1] += local_sum[1];
        row_max[0] = next_max[0];
        row_max[1] = next_max[1];

        unsigned int probability_fragments[VALUE_K_TILES][4];
        pack_probability_fragments(
            probability_fragments, probability_top, probability_bottom);
        value_matrix_product(
            output_accumulators,
            probability_fragments,
            shared_value,
            lane);
        __syncthreads();
    }

    const int top_query = first_query + warp_row + lane / 4;
    const int bottom_query = top_query + 8;
#pragma unroll
    for (int tile = 0; tile < N_TILES; ++tile) {
        const int column = tile * MMA_N + (lane % 4) * 2;
        *reinterpret_cast<float2*>(output + query_offset(
            batch, top_query, head, sequence_length, heads, C) + column) =
            make_float2(
                output_accumulators[tile][0] / row_sum[0],
                output_accumulators[tile][1] / row_sum[0]);
        *reinterpret_cast<float2*>(output + query_offset(
            batch, bottom_query, head, sequence_length, heads, C) + column) =
            make_float2(
                output_accumulators[tile][2] / row_sum[1],
                output_accumulators[tile][3] / row_sum[1]);
    }
    if (lane % 4 == 0) {
        logsumexp[lse_offset(
            batch, head, top_query, heads, sequence_length)] =
            row_max[0] + logf(row_sum[0]);
        logsumexp[lse_offset(
            batch, head, bottom_query, heads, sequence_length)] =
            row_max[1] + logf(row_sum[1]);
    }
}

}  // namespace mla_tensor_core

__global__ void mla_query_backward_kernel(
    float* query_latent_gradient,
    float* query_rope_gradient,
    const float* output_gradient,
    const float* output,
    const float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale,
    bool accumulate) {
    __shared__ float delta;
    __shared__ float score_gradient;

    const int query_token = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int query_base = query_offset(
        batch, query_token, head, sequence_length, heads, kv_rank);
    const int query_rope_base = query_offset(
        batch, query_token, head, sequence_length, heads, rope_size);

    float local_delta = 0.0F;
    for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
        local_delta +=
            output_gradient[query_base + column] * output[query_base + column];
    }
    local_delta = block_reduce_sum(local_delta);
    if (threadIdx.x == 0) {
        delta = local_delta;
    }
    __syncthreads();

    float latent_update[2] = {0.0F, 0.0F};
    float rope_update = 0.0F;
    const float lse = logsumexp[lse_offset(
        batch, head, query_token, heads, sequence_length)];

    for (int key_token = 0; key_token <= query_token; ++key_token) {
        const int kv_base = shared_offset(
            batch, key_token, sequence_length, kv_rank);
        const int key_rope_base = shared_offset(
            batch, key_token, sequence_length, rope_size);
        const float score = dot_query_key(
            query_latent,
            query_rope,
            kv_latent,
            key_rope,
            query_base,
            query_rope_base,
            kv_base,
            key_rope_base,
            kv_rank,
            rope_size) *
            scale;

        float probability_gradient = 0.0F;
        for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
            probability_gradient +=
                output_gradient[query_base + column] *
                __bfloat162float(kv_latent[kv_base + column]);
        }
        probability_gradient = block_reduce_sum(probability_gradient);
        if (threadIdx.x == 0) {
            const float probability = expf(score - lse);
            score_gradient =
                probability * (probability_gradient - delta) * scale;
        }
        __syncthreads();

        for (int column = threadIdx.x, slot = 0;
             column < kv_rank;
             column += BLOCK_SIZE, ++slot) {
            latent_update[slot] +=
                score_gradient * __bfloat162float(kv_latent[kv_base + column]);
        }
        if (threadIdx.x < rope_size) {
            rope_update += score_gradient *
                           __bfloat162float(key_rope[key_rope_base + threadIdx.x]);
        }
        __syncthreads();
    }

    for (int column = threadIdx.x, slot = 0;
         column < kv_rank;
         column += BLOCK_SIZE, ++slot) {
        query_latent_gradient[query_base + column] =
            (accumulate ? query_latent_gradient[query_base + column] : 0.0F) + latent_update[slot];
    }
    if (threadIdx.x < rope_size) {
        query_rope_gradient[query_rope_base + threadIdx.x] =
            (accumulate ? query_rope_gradient[query_rope_base + threadIdx.x] : 0.0F) + rope_update;
    }
}

__global__ void mla_kv_backward_kernel(
    float* kv_latent_gradient,
    float* key_rope_gradient,
    const float* output_gradient,
    const float* output,
    const float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale,
    bool accumulate) {
    __shared__ float delta;
    __shared__ float score_gradient;
    __shared__ float probability;

    const int key_token = blockIdx.x;
    const int batch = blockIdx.y;
    const int kv_base = shared_offset(
        batch, key_token, sequence_length, kv_rank);
    const int key_rope_base = shared_offset(
        batch, key_token, sequence_length, rope_size);
    float latent_update[2] = {0.0F, 0.0F};
    float rope_update = 0.0F;

    for (int head = 0; head < heads; ++head) {
        for (int query_token = key_token;
             query_token < sequence_length;
             ++query_token) {
            const int query_base = query_offset(
                batch, query_token, head, sequence_length, heads, kv_rank);
            const int query_rope_base = query_offset(
                batch, query_token, head, sequence_length, heads, rope_size);
            const float score = dot_query_key(
                query_latent,
                query_rope,
                kv_latent,
                key_rope,
                query_base,
                query_rope_base,
                kv_base,
                key_rope_base,
                kv_rank,
                rope_size) *
                scale;

            float local_probability_gradient = 0.0F;
            float local_delta = 0.0F;
            for (int column = threadIdx.x;
                 column < kv_rank;
                 column += BLOCK_SIZE) {
                const float gradient = output_gradient[query_base + column];
                local_probability_gradient +=
                    gradient * __bfloat162float(kv_latent[kv_base + column]);
                local_delta += gradient * output[query_base + column];
            }
            local_probability_gradient = block_reduce_sum(local_probability_gradient);
            local_delta = block_reduce_sum(local_delta);
            if (threadIdx.x == 0) {
                delta = local_delta;
                probability = expf(
                    score - logsumexp[lse_offset(
                                batch,
                                head,
                                query_token,
                                heads,
                                sequence_length)]);
                score_gradient =
                    probability * (local_probability_gradient - delta) * scale;
            }
            __syncthreads();

            for (int column = threadIdx.x, slot = 0;
                 column < kv_rank;
                 column += BLOCK_SIZE, ++slot) {
                latent_update[slot] +=
                    probability * output_gradient[query_base + column] +
                    score_gradient *
                        __bfloat162float(query_latent[query_base + column]);
            }
            if (threadIdx.x < rope_size) {
                rope_update += score_gradient * __bfloat162float(
                    query_rope[query_rope_base + threadIdx.x]);
            }
            __syncthreads();
        }
    }

    for (int column = threadIdx.x, slot = 0;
         column < kv_rank;
         column += BLOCK_SIZE, ++slot) {
        kv_latent_gradient[kv_base + column] =
            (accumulate ? kv_latent_gradient[kv_base + column] : 0.0F) + latent_update[slot];
    }
    if (threadIdx.x < rope_size) {
        key_rope_gradient[key_rope_base + threadIdx.x] =
            (accumulate ? key_rope_gradient[key_rope_base + threadIdx.x] : 0.0F) + rope_update;
    }
}

__global__ void mla_decode_split_kernel(
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_cache,
    const __nv_bfloat16* key_rope_cache,
    const int* cache_lengths,
    float* workspace,
    int maximum_sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    int splits,
    float scale) {
    __shared__ float output_accumulator[MAX_KV_RANK];
    __shared__ float row_maximum;
    __shared__ float row_normalizer;
    __shared__ float previous_scale;
    __shared__ float probability_scale;

    const int split = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int cache_length = cache_lengths[batch];
    const int tokens_per_split = (cache_length + splits - 1) / splits;
    const int first_token = split * tokens_per_split;
    const int last_token = min(first_token + tokens_per_split, cache_length);
    const int query_base = (batch * heads + head) * kv_rank;
    const int query_rope_base = (batch * heads + head) * rope_size;

    for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
        output_accumulator[column] = 0.0F;
    }
    if (threadIdx.x == 0) {
        row_maximum = -__int_as_float(0x7f800000);
        row_normalizer = 0.0F;
    }
    __syncthreads();

    for (int token = first_token; token < last_token; ++token) {
        const int kv_base =
            (batch * maximum_sequence_length + token) * kv_rank;
        const int key_rope_base =
            (batch * maximum_sequence_length + token) * rope_size;
        const float score = dot_query_key(
            query_latent,
            query_rope,
            kv_cache,
            key_rope_cache,
            query_base,
            query_rope_base,
            kv_base,
            key_rope_base,
            kv_rank,
            rope_size) *
            scale;

        if (threadIdx.x == 0) {
            const float next_maximum = fmaxf(row_maximum, score);
            previous_scale = expf(row_maximum - next_maximum);
            probability_scale = expf(score - next_maximum);
            row_normalizer =
                row_normalizer * previous_scale + probability_scale;
            row_maximum = next_maximum;
        }
        __syncthreads();
        for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
            output_accumulator[column] =
                output_accumulator[column] * previous_scale +
                probability_scale * __bfloat162float(kv_cache[kv_base + column]);
        }
        __syncthreads();
    }

    const int state_base =
        ((batch * heads + head) * splits + split) * (kv_rank + 2);
    if (threadIdx.x == 0) {
        workspace[state_base] = row_maximum;
        workspace[state_base + 1] = row_normalizer;
    }
    for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
        workspace[state_base + 2 + column] = output_accumulator[column];
    }
}

__global__ void mla_decode_combine_kernel(
    float* output,
    float* logsumexp,
    const float* workspace,
    int heads,
    int kv_rank,
    int splits) {
    __shared__ float row_maximum;
    __shared__ float row_normalizer;
    const int head = blockIdx.x;
    const int batch = blockIdx.y;
    const int row = batch * heads + head;

    if (threadIdx.x == 0) {
        float maximum = -__int_as_float(0x7f800000);
        for (int split = 0; split < splits; ++split) {
            const int state_base = (row * splits + split) * (kv_rank + 2);
            maximum = fmaxf(maximum, workspace[state_base]);
        }
        float normalizer = 0.0F;
        for (int split = 0; split < splits; ++split) {
            const int state_base = (row * splits + split) * (kv_rank + 2);
            normalizer += workspace[state_base + 1] *
                          expf(workspace[state_base] - maximum);
        }
        row_maximum = maximum;
        row_normalizer = normalizer;
        logsumexp[row] = maximum + logf(normalizer);
    }
    __syncthreads();

    for (int column = threadIdx.x; column < kv_rank; column += BLOCK_SIZE) {
        float numerator = 0.0F;
        for (int split = 0; split < splits; ++split) {
            const int state_base = (row * splits + split) * (kv_rank + 2);
            numerator += workspace[state_base + 2 + column] *
                         expf(workspace[state_base] - row_maximum);
        }
        output[row * kv_rank + column] = numerator / row_normalizer;
    }
}

}  // namespace

void mla_compressed_attention_forward_cuda(
    float* output,
    float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int batch_size,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale,
    cudaStream_t stream) {
    if (kv_rank == mla_tensor_core::C
        && rope_size == mla_tensor_core::R
        && sequence_length % mla_tensor_core::BM == 0) {
        const dim3 tensor_grid(
            sequence_length / mla_tensor_core::BM,
            batch_size * heads);
        mla_tensor_core::mla_forward_tensor_core_kernel<<<
            tensor_grid,
            mla_tensor_core::THREADS,
            0,
            stream>>>(
                output,
                logsumexp,
                query_latent,
                query_rope,
                kv_latent,
                key_rope,
                sequence_length,
                heads,
                scale);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 grid(sequence_length, heads, batch_size);
    mla_forward_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        output,
        logsumexp,
        query_latent,
        query_rope,
        kv_latent,
        key_rope,
        sequence_length,
        heads,
        kv_rank,
        rope_size,
        scale);
    CUDA_CHECK(cudaGetLastError());
}

void mla_compressed_attention_backward_cuda(
    float* query_latent_gradient,
    float* query_rope_gradient,
    float* kv_latent_gradient,
    float* key_rope_gradient,
    const float* output_gradient,
    const float* output,
    const float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int batch_size,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale,
    bool accumulate,
    cudaStream_t stream) {
    const dim3 query_grid(sequence_length, heads, batch_size);
    mla_query_backward_kernel<<<query_grid, BLOCK_SIZE, 0, stream>>>(
        query_latent_gradient,
        query_rope_gradient,
        output_gradient,
        output,
        logsumexp,
        query_latent,
        query_rope,
        kv_latent,
        key_rope,
        sequence_length,
        heads,
        kv_rank,
        rope_size,
        scale,
        accumulate);
    CUDA_CHECK(cudaGetLastError());

    const dim3 kv_grid(sequence_length, batch_size);
    mla_kv_backward_kernel<<<kv_grid, BLOCK_SIZE, 0, stream>>>(
        kv_latent_gradient,
        key_rope_gradient,
        output_gradient,
        output,
        logsumexp,
        query_latent,
        query_rope,
        kv_latent,
        key_rope,
        sequence_length,
        heads,
        kv_rank,
        rope_size,
        scale,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t mla_decode_workspace_elements(
    int batch_size,
    int heads,
    int splits,
    int kv_rank) {
    return static_cast<std::size_t>(batch_size) * heads * splits *
           (kv_rank + 2);
}

void mla_decode_forward_cuda(
    float* output,
    float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_cache,
    const __nv_bfloat16* key_rope_cache,
    const int* cache_lengths,
    float* workspace,
    int batch_size,
    int maximum_sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    int splits,
    float scale,
    cudaStream_t stream) {
    const dim3 split_grid(splits, heads, batch_size);
    mla_decode_split_kernel<<<split_grid, BLOCK_SIZE, 0, stream>>>(
        query_latent,
        query_rope,
        kv_cache,
        key_rope_cache,
        cache_lengths,
        workspace,
        maximum_sequence_length,
        heads,
        kv_rank,
        rope_size,
        splits,
        scale);
    CUDA_CHECK(cudaGetLastError());

    const dim3 combine_grid(heads, batch_size);
    mla_decode_combine_kernel<<<combine_grid, BLOCK_SIZE, 0, stream>>>(
        output, logsumexp, workspace, heads, kv_rank, splits);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

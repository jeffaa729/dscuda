// Fuses causal online softmax with DeepSeek MLA's absorbed query and shared compressed KV representation in BF16.
// The backward pass recomputes probabilities from saved FP32 log-sum-exp values and assigns one CTA to every independently writable gradient row.

#include "cuda_common.h"
#include "mla.h"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace dscuda {
namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_KV_RANK = MLA_KV_RANK;

void check_shape(int kv_rank, int rope_size) {
    if (kv_rank != MLA_KV_RANK || rope_size != MLA_ROPE_SIZE) {
        throw std::invalid_argument("MLA requires C=512 and RoPE=64");
    }
}

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

namespace mla_forward {

constexpr int BM = 16;
constexpr int BN = 16;
constexpr int C = MLA_KV_RANK;
constexpr int R = MLA_ROPE_SIZE;
constexpr int K = C + R;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * WARP_SIZE;
constexpr int SCORE_TILES = BN / 8;
constexpr int OUTPUT_TILES = C / (WARPS * 8);
constexpr int QUERY_TILES = K / (WARPS * 16);

// Rows are 576 BF16 elements wide; XOR within each 64-element segment keeps
// 16-byte copies aligned and distributes ldmatrix rows across shared banks.
__device__ __forceinline__ int swizzle(int offset) {
    return offset ^ (((offset / K) & 7) << 3);
}

__device__ __forceinline__ unsigned int shared_address(const void* pointer) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void load_x4(unsigned int (&fragment)[4], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(address));
}

__device__ __forceinline__ void load_x2(unsigned int (&fragment)[2], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1]) : "r"(address));
}

__device__ __forceinline__ void load_x2_transpose(
    unsigned int (&fragment)[2], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1]) : "r"(address));
}

__device__ __forceinline__ void mma(
    float (&accumulator)[4], const unsigned int (&left)[4], const unsigned int (&right)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(accumulator[0]), "+f"(accumulator[1]),
          "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]),
          "r"(right[0]), "r"(right[1]));
}

// The same tile loader handles strided queries and head-shared KV. Each thread
// copies eight BF16 elements, with zero padding for incomplete sequence tiles.
__device__ __forceinline__ void copy_tile(
    __nv_bfloat16* shared, const __nv_bfloat16* latent, const __nv_bfloat16* rope,
    int first, int sequence_length, int latent_stride, int rope_stride) {
    for (int vector = threadIdx.x; vector < BM * K / 8; vector += THREADS) {
        const int row = vector / (K / 8);
        const int column = vector % (K / 8) * 8;
        uint4 values = make_uint4(0, 0, 0, 0);
        if (first + row < sequence_length) {
            const __nv_bfloat16* source = column < C
                ? latent + (first + row) * latent_stride + column
                : rope + (first + row) * rope_stride + column - C;
            values = *reinterpret_cast<const uint4*>(source);
        }
        *reinterpret_cast<uint4*>(shared + swizzle(row * K + column)) = values;
    }
}

// P stays FP32 for softmax statistics. Two BF16 parts approximate each weight
// for PV, retaining the FP32-output accuracy contract without a SIMT fallback.
__device__ __forceinline__ void split_probability(
    float x, float y, unsigned int& high, unsigned int& low) {
    union Packed {
        __nv_bfloat162 value;
        unsigned int bits;
    } packed;
    packed.value = __floats2bfloat162_rn(x, y);
    high = packed.bits;
    const float2 rounded = __bfloat1622float2(packed.value);
    packed.value = __floats2bfloat162_rn(x - rounded.x, y - rounded.y);
    low = packed.bits;
}

// One CTA owns 16 query rows. Its four warps split the 576-wide QK reduction,
// then each warp owns 128 output columns; KV is shared across all query heads.
__global__ __launch_bounds__(THREADS, 2)
void mla_forward_kernel(
    float* __restrict__ output,
    float* __restrict__ logsumexp,
    const __nv_bfloat16* __restrict__ query_latent,
    const __nv_bfloat16* __restrict__ query_rope,
    const __nv_bfloat16* __restrict__ kv_latent,
    const __nv_bfloat16* __restrict__ key_rope,
    int sequence_length, int heads, float scale) {
    __shared__ __align__(16) __nv_bfloat16 shared_query[BM * K];
    __shared__ __align__(16) __nv_bfloat16 shared_kv[BN * K];
    __shared__ float partial[WARPS][SCORE_TILES][WARP_SIZE][4];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int batch = blockIdx.y / heads;
    const int head = blockIdx.y % heads;
    const int first_query = blockIdx.x * BM;
    const int top_query = first_query + lane / 4;
    const int bottom_query = top_query + 8;

    copy_tile(shared_query,
              query_latent + (batch * sequence_length * heads + head) * C,
              query_rope + (batch * sequence_length * heads + head) * R,
              first_query, sequence_length, heads * C, heads * R);
    __syncthreads();
    unsigned int query[QUERY_TILES][4];
#pragma unroll
    for (int tile = 0; tile < QUERY_TILES; ++tile) {
        const int column = (warp * QUERY_TILES + tile) * 16 + lane / 16 * 8;
        load_x4(query[tile], shared_address(shared_query + swizzle(lane % 16 * K + column)));
    }

    float numerator[OUTPUT_TILES][4] = {};
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {};
    for (int first_key = 0; first_key <= first_query; first_key += BN) {
        copy_tile(shared_kv, kv_latent + batch * sequence_length * C,
                  key_rope + batch * sequence_length * R,
                  first_key, sequence_length, C, R);
        __syncthreads();

        float scores[SCORE_TILES][4] = {};
#pragma unroll
        for (int inner = 0; inner < QUERY_TILES; ++inner) {
#pragma unroll
            for (int tile = 0; tile < SCORE_TILES; ++tile) {
                const int row = tile * 8 + lane % 8;
                const int column = (warp * QUERY_TILES + inner) * 16 + lane % 16 / 8 * 8;
                unsigned int key[2];
                load_x2(key, shared_address(shared_kv + swizzle(row * K + column)));
                mma(scores[tile], query[inner], key);
            }
        }
#pragma unroll
        for (int tile = 0; tile < SCORE_TILES; ++tile) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                partial[warp][tile][lane][i] = scores[tile][i];
            }
        }
        __syncthreads();

        float next_max[2] = {row_max[0], row_max[1]};
#pragma unroll
        for (int tile = 0; tile < SCORE_TILES; ++tile) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float value = 0.0F;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) {
                    value += partial[w][tile][lane][i];
                }
                const int q = i < 2 ? top_query : bottom_query;
                const int k = first_key + tile * 8 + lane % 4 * 2 + i % 2;
                scores[tile][i] = q < sequence_length && k < sequence_length && k <= q
                    ? value * scale : -CUDART_INF_F;
                next_max[i / 2] = fmaxf(next_max[i / 2], scores[tile][i]);
            }
        }
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            next_max[row] = fmaxf(next_max[row], __shfl_xor_sync(0xffffffffU, next_max[row], 1));
            next_max[row] = fmaxf(next_max[row], __shfl_xor_sync(0xffffffffU, next_max[row], 2));
        }
        const float alpha[2] = {expf(row_max[0] - next_max[0]), expf(row_max[1] - next_max[1])};
#pragma unroll
        for (int tile = 0; tile < OUTPUT_TILES; ++tile) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                numerator[tile][i] *= alpha[i / 2];
            }
        }

        unsigned int probability_high[4], probability_low[4];
        float local_sum[2] = {};
#pragma unroll
        for (int tile = 0; tile < SCORE_TILES; ++tile) {
#pragma unroll
            for (int row = 0; row < 2; ++row) {
                const float p0 = expf(scores[tile][2 * row] - next_max[row]);
                const float p1 = expf(scores[tile][2 * row + 1] - next_max[row]);
                local_sum[row] += p0 + p1;
                split_probability(p0, p1, probability_high[2 * tile + row],
                                  probability_low[2 * tile + row]);
            }
        }
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            local_sum[row] += __shfl_xor_sync(0xffffffffU, local_sum[row], 1);
            local_sum[row] += __shfl_xor_sync(0xffffffffU, local_sum[row], 2);
            row_sum[row] = row_sum[row] * alpha[row] + local_sum[row];
            row_max[row] = next_max[row];
        }

#pragma unroll
        for (int tile = 0; tile < OUTPUT_TILES; ++tile) {
            const int column = (warp * OUTPUT_TILES + tile) * 8;
            unsigned int value[2];
            load_x2_transpose(value, shared_address(shared_kv + swizzle(lane % 16 * K + column)));
            mma(numerator[tile], probability_high, value);
            mma(numerator[tile], probability_low, value);
        }
        __syncthreads();
    }

#pragma unroll
    for (int tile = 0; tile < OUTPUT_TILES; ++tile) {
        const int column = (warp * OUTPUT_TILES + tile) * 8 + lane % 4 * 2;
        if (top_query < sequence_length) {
            *reinterpret_cast<float2*>(output + query_offset(
                batch, top_query, head, sequence_length, heads, C) + column) =
                make_float2(numerator[tile][0] / row_sum[0], numerator[tile][1] / row_sum[0]);
        }
        if (bottom_query < sequence_length) {
            *reinterpret_cast<float2*>(output + query_offset(
                batch, bottom_query, head, sequence_length, heads, C) + column) =
                make_float2(numerator[tile][2] / row_sum[1], numerator[tile][3] / row_sum[1]);
        }
    }
    if (warp == 0 && lane % 4 == 0) {
        if (top_query < sequence_length) {
            logsumexp[lse_offset(batch, head, top_query, heads, sequence_length)] =
                row_max[0] + logf(row_sum[0]);
        }
        if (bottom_query < sequence_length) {
            logsumexp[lse_offset(batch, head, bottom_query, heads, sequence_length)] =
                row_max[1] + logf(row_sum[1]);
        }
    }
}

}  // namespace mla_forward

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
    check_shape(kv_rank, rope_size);
    const dim3 grid((sequence_length + mla_forward::BM - 1) / mla_forward::BM,
                    batch_size * heads);
    mla_forward::mla_forward_kernel<<<grid, mla_forward::THREADS, 0, stream>>>(
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
    check_shape(kv_rank, rope_size);
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
    check_shape(kv_rank, rope_size);
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

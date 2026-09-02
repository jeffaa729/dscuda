#include "common.cuh"

namespace dscuda {

void gemm_fp32_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    gemm_fp32_sm89_cuda(output, left, right, M, N, K, stream);
}

void gemm_bf16_cuda(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    gemm_bf16_sm89_cuda(output, left, right, M, N, K, stream);
}

}  // namespace dscuda

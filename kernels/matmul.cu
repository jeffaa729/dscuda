// Implements dense linear projections using interchangeable custom CUDA and cuBLAS-backed matrix-multiplication paths.
// Its backward path computes input, weight, and optional bias gradients for attention, feed-forward, router, and output layers.

#include "cuda_common.h"
#include "matmul.h"

namespace dscuda {
namespace {

__global__ void matmul_forward_kernel(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K) {
}

__global__ void matmul_left_backward_kernel(
    float* left_gradient,
    const float* output_gradient,
    const float* right,
    int M,
    int N,
    int K) {
}

__global__ void matmul_right_backward_kernel(
    float* right_gradient,
    const float* output_gradient,
    const float* left,
    int M,
    int N,
    int K) {
}

}  // namespace

void matmul_forward_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
}

void matmul_backward_cuda(
    float* left_gradient,
    float* right_gradient,
    const float* output_gradient,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
}

}  // namespace dscuda

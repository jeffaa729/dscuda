// Implements readable FP32 matrix multiplication and its two analytical gradients on the CPU.
// The scalar loops prioritize transparent accumulation order over performance so CUDA kernels have a trustworthy oracle.

#include "matmul_cpu.h"

namespace dscuda {

void matmul_forward_cpu(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K) {
    for (int row = 0; row < M; ++row) {
        for (int column = 0; column < N; ++column) {
            float sum = 0.0F;
            for (int inner = 0; inner < K; ++inner) {
                sum += left[row * K + inner] * right[inner * N + column];
            }
            output[row * N + column] = sum;
        }
    }
}

void matmul_backward_cpu(
    float* left_gradient,
    float* right_gradient,
    const float* output_gradient,
    const float* left,
    const float* right,
    int M,
    int N,
    int K) {
    for (int row = 0; row < M; ++row) {
        for (int column = 0; column < N; ++column) {
            const float gradient = output_gradient[row * N + column];
            for (int inner = 0; inner < K; ++inner) {
                left_gradient[row * K + inner] += gradient * right[inner * N + column];
                right_gradient[inner * N + column] += left[row * K + inner] * gradient;
            }
        }
    }
}

}  // namespace dscuda

// Implements a scalar row-major GEMM with optional operand transposes and output accumulation.
// Tests use this readable FP32 oracle with original FP32 or BF16-rounded inputs.

#include "matmul_cpu.h"

namespace dscuda {

void gemm_cpu(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    bool transpose_left,
    bool transpose_right,
    bool accumulate) {
    for (int row = 0; row < M; ++row) {
        for (int column = 0; column < N; ++column) {
            float sum = 0.0F;
            for (int inner = 0; inner < K; ++inner) {
                const float a = left[transpose_left ? inner * M + row : row * K + inner];
                const float b = right[transpose_right ? column * K + inner : inner * N + column];
                sum += a * b;
            }
            output[row * N + column] = sum + (accumulate ? output[row * N + column] : 0.0F);
        }
    }
}

}  // namespace dscuda

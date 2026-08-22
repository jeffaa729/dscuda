#pragma once

namespace dscuda {

void matmul_forward_cpu(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K);

void matmul_backward_cpu(
    float* left_gradient,
    float* right_gradient,
    const float* output_gradient,
    const float* left,
    const float* right,
    int M,
    int N,
    int K);

}  // namespace dscuda

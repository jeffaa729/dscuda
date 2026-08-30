// Applies a bias-corrected FP32 AdamW update to four contiguous parameters per CUDA thread.
// First and second moments are updated from the raw gradient while decoupled weight decay acts only on the parameter.

#include "cuda_common.h"
#include "optimizer.h"

#include <cmath>

namespace dscuda {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ void adamw_update_element(
    float& parameter,
    float& first_moment,
    float& second_moment,
    float gradient,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float first_correction,
    float second_correction) {
    first_moment =
        beta1 * first_moment + (1.0F - beta1) * gradient;
    second_moment =
        beta2 * second_moment + (1.0F - beta2) * gradient * gradient;
    parameter -= learning_rate * weight_decay * parameter;
    parameter -= learning_rate * (first_moment * first_correction) /
        (sqrtf(second_moment * second_correction) + epsilon);
}

__global__ void adamw_step_kernel(
    float4* parameters,
    float4* first_moments,
    float4* second_moments,
    const float4* gradients,
    int vectors,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float first_correction,
    float second_correction) {
    const int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= vectors) {
        return;
    }

    float4 parameter = parameters[vector];
    float4 first = first_moments[vector];
    float4 second = second_moments[vector];
    const float4 gradient = gradients[vector];
    adamw_update_element(
        parameter.x, first.x, second.x, gradient.x,
        learning_rate, beta1, beta2, epsilon, weight_decay,
        first_correction, second_correction);
    adamw_update_element(
        parameter.y, first.y, second.y, gradient.y,
        learning_rate, beta1, beta2, epsilon, weight_decay,
        first_correction, second_correction);
    adamw_update_element(
        parameter.z, first.z, second.z, gradient.z,
        learning_rate, beta1, beta2, epsilon, weight_decay,
        first_correction, second_correction);
    adamw_update_element(
        parameter.w, first.w, second.w, gradient.w,
        learning_rate, beta1, beta2, epsilon, weight_decay,
        first_correction, second_correction);
    parameters[vector] = parameter;
    first_moments[vector] = first;
    second_moments[vector] = second;
}

}  // namespace

void adamw_step_cuda(
    float* parameters,
    float* first_moment,
    float* second_moment,
    const float* gradients,
    int elements,
    int step,
    const AdamWConfig& config,
    cudaStream_t stream) {
    const int vectors = elements / 4;
    const int blocks = (vectors + kThreads - 1) / kThreads;
    const float first_correction =
        1.0F / (1.0F - std::pow(config.beta1, step));
    const float second_correction =
        1.0F / (1.0F - std::pow(config.beta2, step));
    adamw_step_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4*>(parameters),
        reinterpret_cast<float4*>(first_moment),
        reinterpret_cast<float4*>(second_moment),
        reinterpret_cast<const float4*>(gradients),
        vectors,
        config.learning_rate,
        config.beta1,
        config.beta2,
        config.epsilon,
        config.weight_decay,
        first_correction,
        second_correction);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

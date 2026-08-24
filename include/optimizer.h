#pragma once

#include <cuda_runtime.h>

namespace dscuda {

struct AdamWConfig {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
};

// Updates FP32 parameters and moment buffers in place. All buffers contain a
// multiple of four elements, and step starts at one.
void adamw_step_cuda(
    float* parameters,
    float* first_moment,
    float* second_moment,
    const float* gradients,
    int elements,
    int step,
    const AdamWConfig& config,
    cudaStream_t stream = nullptr);

}  // namespace dscuda

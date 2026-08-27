#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

void cuda_check(cudaError_t result, const char* expression, const char* file, int line);

#define CUDA_CHECK(expression) \
    ::dscuda::cuda_check((expression), #expression, __FILE__, __LINE__)

void* device_malloc(std::size_t bytes);
void device_free(void* pointer);
void synchronize(cudaStream_t stream = nullptr);
cudaDeviceProp device_properties(int device = 0);
void print_device_summary(int device = 0);

void convert_fp32_to_bf16_cuda(
    __nv_bfloat16* output,
    const float* input,
    std::size_t elements,
    cudaStream_t stream = nullptr);

}  // namespace dscuda

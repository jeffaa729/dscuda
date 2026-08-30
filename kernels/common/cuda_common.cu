// Provides shared CUDA error handling, device allocation, synchronization, and device inspection utilities.
// These helpers keep kernel launch code concise while preserving actionable failure messages and explicit runtime behavior.

#include "cuda_common.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace dscuda {
namespace {

constexpr int kConversionThreads = 256;

__global__ void convert_fp32_to_bf16_kernel(
    __nv_bfloat16* output,
    const float* input,
    std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2bfloat16(input[index]);
    }
}

}  // namespace

void cuda_check(cudaError_t result, const char* expression, const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA failure at " << file << ':' << line << " while evaluating "
            << expression << ": " << cudaGetErrorString(result);
    throw std::runtime_error(message.str());
}

void* device_malloc(std::size_t bytes) {
    void* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(&pointer, bytes));
    return pointer;
}

void device_free(void* pointer) {
    CUDA_CHECK(cudaFree(pointer));
}

void synchronize(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

cudaDeviceProp device_properties(int device) {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties;
}

void print_device_summary(int device) {
    const cudaDeviceProp properties = device_properties(device);
    std::printf(
        "CUDA device %d: %s, compute capability %d.%d, %.1f GiB global memory\n",
        device,
        properties.name,
        properties.major,
        properties.minor,
        static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
}

void convert_fp32_to_bf16_cuda(
    __nv_bfloat16* output,
    const float* input,
    std::size_t elements,
    cudaStream_t stream) {
    const int blocks = static_cast<int>(
        (elements + kConversionThreads - 1) / kConversionThreads);
    convert_fp32_to_bf16_kernel<<<blocks, kConversionThreads, 0, stream>>>(
        output, input, elements);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace dscuda

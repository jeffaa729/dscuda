// Provides shared CUDA error handling, device allocation, synchronization, device inspection, and event-based timing utilities.
// These helpers keep kernel launch code concise while preserving actionable failure messages and explicit runtime behavior.

#include "cuda_common.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace dscuda {

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

CudaEventTimer::CudaEventTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

CudaEventTimer::~CudaEventTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void CudaEventTimer::start(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_event_, stream));
}

float CudaEventTimer::stop(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(stop_event_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));
    return milliseconds;
}

}  // namespace dscuda

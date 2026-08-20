#pragma once

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

class CudaEventTimer {
public:
    CudaEventTimer();
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start(cudaStream_t stream = nullptr);
    float stop(cudaStream_t stream = nullptr);

private:
    cudaEvent_t start_event_{};
    cudaEvent_t stop_event_{};
};

}  // namespace dscuda

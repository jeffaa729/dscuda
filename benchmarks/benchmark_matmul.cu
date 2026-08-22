// Runs CUDA Core, Tensor Core, and matching cuBLAS matrix-multiplication workloads for Nsight Compute.
// Timing and hardware metrics are collected by the profiler rather than CUDA events in this executable.

#include "cuda_common.h"
#include "matmul.h"

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

void cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS call failed");
    }
}

void cublas_gemm(
    cublasHandle_t handle,
    bool bf16,
    cublasOperation_t transpose_left,
    cublasOperation_t transpose_right,
    int M,
    int N,
    int K,
    const void* left,
    int left_stride,
    const void* right,
    int right_stride,
    float beta,
    float* output,
    int output_stride) {
    const float alpha = 1.0F;
    if (bf16) {
        cublas_check(cublasGemmEx(
            handle,
            transpose_left,
            transpose_right,
            M,
            N,
            K,
            &alpha,
            left,
            CUDA_R_16BF,
            left_stride,
            right,
            CUDA_R_16BF,
            right_stride,
            &beta,
            output,
            CUDA_R_32F,
            output_stride,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        cublas_check(cublasSgemm(
            handle,
            transpose_left,
            transpose_right,
            M,
            N,
            K,
            &alpha,
            static_cast<const float*>(left),
            left_stride,
            static_cast<const float*>(right),
            right_stride,
            &beta,
            output,
            output_stride));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int M = argc > 1 ? std::atoi(argv[1]) : 1024;
        const int N = argc > 2 ? std::atoi(argv[2]) : 1024;
        const int K = argc > 3 ? std::atoi(argv[3]) : 1024;
        const char* backend = argc > 4 ? argv[4] : "fp32";
        const size_t left_elements = static_cast<size_t>(M) * K;
        const size_t right_elements = static_cast<size_t>(K) * N;
        const size_t output_elements = static_cast<size_t>(M) * N;

        std::vector<float> left(left_elements, 0.5F);
        std::vector<float> right(right_elements, 0.25F);
        std::vector<float> output_gradient(output_elements, 0.125F);
        std::vector<__nv_bfloat16> bf16_left(
            left_elements, __float2bfloat16(0.5F));
        std::vector<__nv_bfloat16> bf16_right(
            right_elements, __float2bfloat16(0.25F));
        std::vector<__nv_bfloat16> bf16_output_gradient(
            output_elements, __float2bfloat16(0.125F));

        auto* gpu_left = static_cast<float*>(dscuda::device_malloc(left_elements * sizeof(float)));
        auto* gpu_right =
            static_cast<float*>(dscuda::device_malloc(right_elements * sizeof(float)));
        auto* gpu_output_gradient =
            static_cast<float*>(dscuda::device_malloc(output_elements * sizeof(float)));
        auto* gpu_output =
            static_cast<float*>(dscuda::device_malloc(output_elements * sizeof(float)));
        auto* gpu_left_gradient =
            static_cast<float*>(dscuda::device_malloc(left_elements * sizeof(float)));
        auto* gpu_right_gradient =
            static_cast<float*>(dscuda::device_malloc(right_elements * sizeof(float)));
        auto* gpu_bf16_left = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(left_elements * sizeof(__nv_bfloat16)));
        auto* gpu_bf16_right = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(right_elements * sizeof(__nv_bfloat16)));
        auto* gpu_bf16_output_gradient = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(output_elements * sizeof(__nv_bfloat16)));

        CUDA_CHECK(cudaMemcpy(
            gpu_left, left.data(), left_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_right, right.data(), right_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_output_gradient,
            output_gradient.data(),
            output_elements * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_bf16_left,
            bf16_left.data(),
            left_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_bf16_right,
            bf16_right.data(),
            right_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_bf16_output_gradient,
            bf16_output_gradient.data(),
            output_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(gpu_left_gradient, 0, left_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpu_right_gradient, 0, right_elements * sizeof(float)));

        std::printf("Matmul workload: M=%d N=%d K=%d backend=%s\n", M, N, K, backend);
        if (std::strcmp(backend, "fp32") == 0) {
            dscuda::matmul_fp32_forward_cuda(
                gpu_output, gpu_left, gpu_right, M, N, K);
            dscuda::matmul_fp32_backward_cuda(
                gpu_left_gradient,
                gpu_right_gradient,
                gpu_output_gradient,
                gpu_left,
                gpu_right,
                M,
                N,
                K);
        } else if (std::strcmp(backend, "bf16") == 0) {
            dscuda::matmul_bf16_forward_cuda(
                gpu_output, gpu_bf16_left, gpu_bf16_right, M, N, K);
            dscuda::matmul_bf16_backward_cuda(
                gpu_left_gradient,
                gpu_right_gradient,
                gpu_bf16_output_gradient,
                gpu_bf16_left,
                gpu_bf16_right,
                M,
                N,
                K);
        } else if (
            std::strcmp(backend, "cublas_fp32") == 0 ||
            std::strcmp(backend, "cublas_bf16") == 0) {
            const bool bf16 = std::strcmp(backend, "cublas_bf16") == 0;
            cublasHandle_t handle;
            cublas_check(cublasCreate(&handle));
            cublas_check(cublasSetMathMode(
                handle,
                bf16 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
            const float overwrite = 0.0F;
            const float accumulate = 1.0F;

            // Row-major Y = L * R is column-major Y^T = R^T * L^T.
            cublas_gemm(
                handle,
                bf16,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                bf16 ? static_cast<const void*>(gpu_bf16_right)
                     : static_cast<const void*>(gpu_right),
                N,
                bf16 ? static_cast<const void*>(gpu_bf16_left)
                     : static_cast<const void*>(gpu_left),
                K,
                overwrite,
                gpu_output,
                N);

            // Row-major dL = dY * R^T becomes dL^T = R * dY^T.
            cublas_gemm(
                handle,
                bf16,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                K,
                M,
                N,
                bf16 ? static_cast<const void*>(gpu_bf16_right)
                     : static_cast<const void*>(gpu_right),
                N,
                bf16 ? static_cast<const void*>(gpu_bf16_output_gradient)
                     : static_cast<const void*>(gpu_output_gradient),
                N,
                accumulate,
                gpu_left_gradient,
                K);

            // Row-major dR = L^T * dY becomes dR^T = dY^T * L.
            cublas_gemm(
                handle,
                bf16,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                N,
                K,
                M,
                bf16 ? static_cast<const void*>(gpu_bf16_output_gradient)
                     : static_cast<const void*>(gpu_output_gradient),
                N,
                bf16 ? static_cast<const void*>(gpu_bf16_left)
                     : static_cast<const void*>(gpu_left),
                K,
                accumulate,
                gpu_right_gradient,
                N);
            cublas_check(cublasDestroy(handle));
        } else {
            throw std::runtime_error(
                "backend must be fp32, bf16, cublas_fp32, or cublas_bf16");
        }
        dscuda::synchronize();

        dscuda::device_free(gpu_bf16_output_gradient);
        dscuda::device_free(gpu_bf16_right);
        dscuda::device_free(gpu_bf16_left);
        dscuda::device_free(gpu_right_gradient);
        dscuda::device_free(gpu_left_gradient);
        dscuda::device_free(gpu_output);
        dscuda::device_free(gpu_output_gradient);
        dscuda::device_free(gpu_right);
        dscuda::device_free(gpu_left);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Matmul benchmark failed: %s\n", error.what());
        return 1;
    }
}

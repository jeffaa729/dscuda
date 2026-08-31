// Exposes GEMM to Python-owned CUDA buffers without a PyTorch extension build.
// cuBLAS uses the same stream and operand/output precision as the custom GEMM.
#include "matmul.h"
#include "grouped_gemm.h"

#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;
thread_local cublasHandle_t handle = nullptr;

void cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS status " + std::to_string(status));
    }
}
}  // namespace

extern "C" const char* dscuda_operator_last_error() {
    return last_error.c_str();
}

extern "C" int dscuda_cublas_init() {
    try {
        if (!handle) cublas_check(cublasCreate(&handle));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" void dscuda_cublas_destroy() {
    if (handle) cublasDestroy(handle);
    handle = nullptr;
}

extern "C" int dscuda_cublas_version() {
    int version = 0;
    return cublasGetVersion(handle, &version) == CUBLAS_STATUS_SUCCESS ? version : 0;
}

extern "C" int dscuda_gemm(
    float* output, const void* left, const void* right,
    int M, int N, int K, int bf16,
    int transpose_left, int transpose_right, int accumulate,
    int reference, cudaStream_t stream) {
    try {
        if (!reference) {
            if (bf16) {
                dscuda::gemm_bf16_cuda(
                    output, static_cast<const __nv_bfloat16*>(left),
                    static_cast<const __nv_bfloat16*>(right),
                    M, N, K, transpose_left, transpose_right, accumulate, stream);
            } else {
                dscuda::gemm_fp32_cuda(
                    output, static_cast<const float*>(left), static_cast<const float*>(right),
                    M, N, K, transpose_left, transpose_right, accumulate, stream);
            }
        } else {
            cublas_check(cublasSetStream(handle, stream));
            cublas_check(cublasSetMathMode(
                handle, bf16 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
            const float alpha = 1.0F;
            const float beta = accumulate ? 1.0F : 0.0F;
            const cudaDataType_t type = bf16 ? CUDA_R_16BF : CUDA_R_32F;
            cublas_check(cublasGemmEx(
                handle,
                transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
                transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
                N, M, K, &alpha,
                right, type, transpose_right ? K : N,
                left, type, transpose_left ? M : K,
                &beta, output, CUDA_R_32F, N,
                bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC,
                bf16 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

// Uses the same packed expert rows for custom CUDA and the cuBLAS-per-expert reference.
// Host offsets are prepared before graph capture, so no device-to-host copy is timed.
extern "C" int dscuda_grouped_gemm(
    float* output, const void* input, const void* weights,
    const int* device_offsets, const int* slot_expert, const int* host_offsets,
    int rows, int experts, int N, int K, int bf16, int reference, cudaStream_t stream) {
    try {
        if (!reference) {
            if (bf16) {
                dscuda::grouped_linear_bf16_forward_cuda(output,
                    static_cast<const __nv_bfloat16*>(input),
                    static_cast<const __nv_bfloat16*>(weights), device_offsets,
                    rows, experts, N, K, stream);
            } else {
                dscuda::grouped_linear_forward_cuda(output,
                    static_cast<const float*>(input), static_cast<const float*>(weights),
                    slot_expert, rows, N, K, stream);
            }
        } else {
            const int bytes = bf16 ? 2 : 4;
            for (int e = 0; e < experts; ++e) {
                const int begin = host_offsets[e], count = host_offsets[e + 1] - begin;
                if (!count) continue;
                const auto* x = static_cast<const char*>(input) + static_cast<size_t>(begin) * K * bytes;
                const auto* w = static_cast<const char*>(weights) + static_cast<size_t>(e) * K * N * bytes;
                if (dscuda_gemm(output + static_cast<size_t>(begin) * N, x, w,
                               count, N, K, bf16, 0, 0, 0, 1, stream)) return 1;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

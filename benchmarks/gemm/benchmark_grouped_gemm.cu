// Profiles the BF16 grouped expert GEMM with uniform, hot, or empty-expert row distributions.
// The custom Tensor Core result is checked against the same per-expert cuBLAS operations before profiling starts.

#include "cuda_common.h"
#include "grouped_gemm.h"

#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

void cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS call failed");
    }
}

std::vector<int> make_offsets(
    int rows,
    int experts,
    const char* distribution) {
    std::vector<int> counts(experts);
    if (std::strcmp(distribution, "uniform") == 0) {
        for (int expert = 0; expert < experts; ++expert) {
            counts[expert] = rows / experts + (expert < rows % experts);
        }
    } else if (std::strcmp(distribution, "empty") == 0) {
        if (experts < 4) {
            throw std::runtime_error("empty distribution requires at least four experts");
        }
        counts[1] = rows / 3;
        const int remaining = rows - counts[1];
        for (int expert = 3; expert < experts; ++expert) {
            counts[expert] = remaining / (experts - 3);
        }
        counts.back() += remaining % (experts - 3);
    } else if (std::strcmp(distribution, "hot") == 0) {
        counts[0] = rows * 4 / 5;
        const int remaining = rows - counts[0];
        for (int expert = 1; expert < experts; ++expert) {
            counts[expert] = remaining / (experts - 1)
                + (expert - 1 < remaining % (experts - 1));
        }
    } else {
        throw std::runtime_error(
            "distribution must be uniform, hot, or empty");
    }

    std::vector<int> offsets(experts + 1);
    for (int expert = 0; expert < experts; ++expert) {
        offsets[expert + 1] = offsets[expert] + counts[expert];
    }
    return offsets;
}

void cublas_grouped_linear(
    cublasHandle_t handle,
    float* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const std::vector<int>& offsets,
    int experts,
    int output_size,
    int input_size) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (int expert = 0; expert < experts; ++expert) {
        const int first_row = offsets[expert];
        const int expert_rows = offsets[expert + 1] - first_row;
        if (expert_rows == 0) {
            continue;
        }
        cublas_check(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            output_size,
            expert_rows,
            input_size,
            &alpha,
            weight + static_cast<std::size_t>(expert) * input_size * output_size,
            CUDA_R_16BF,
            output_size,
            input + static_cast<std::size_t>(first_row) * input_size,
            CUDA_R_16BF,
            input_size,
            &beta,
            output + static_cast<std::size_t>(first_row) * output_size,
            CUDA_R_32F,
            output_size,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
        const int experts = argc > 2 ? std::atoi(argv[2]) : 8;
        const int input_size = argc > 3 ? std::atoi(argv[3]) : 512;
        const int output_size = argc > 4 ? std::atoi(argv[4]) : 1536;
        const char* distribution = argc > 5 ? argv[5] : "uniform";
        const char* backend = argc > 6 ? argv[6] : "custom";
        if (std::strcmp(backend, "custom") != 0
            && std::strcmp(backend, "cublas") != 0) {
            throw std::runtime_error("backend must be custom or cublas");
        }

        const std::vector<int> offsets =
            make_offsets(rows, experts, distribution);
        const std::size_t input_elements =
            static_cast<std::size_t>(rows) * input_size;
        const std::size_t weight_elements =
            static_cast<std::size_t>(experts) * input_size * output_size;
        const std::size_t output_elements =
            static_cast<std::size_t>(rows) * output_size;

        std::vector<__nv_bfloat16> input(input_elements);
        std::vector<__nv_bfloat16> weight(weight_elements);
        for (std::size_t index = 0; index < input_elements; ++index) {
            input[index] = __float2bfloat16(
                (static_cast<int>(index % 29) - 14) / 32.0F);
        }
        for (std::size_t index = 0; index < weight_elements; ++index) {
            weight[index] = __float2bfloat16(
                (static_cast<int>(index % 31) - 15) / 64.0F);
        }

        auto* gpu_input = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(input_elements * sizeof(__nv_bfloat16)));
        auto* gpu_weight = static_cast<__nv_bfloat16*>(
            dscuda::device_malloc(weight_elements * sizeof(__nv_bfloat16)));
        auto* gpu_offsets = static_cast<int*>(
            dscuda::device_malloc(offsets.size() * sizeof(int)));
        auto* gpu_custom = static_cast<float*>(
            dscuda::device_malloc(output_elements * sizeof(float)));
        auto* gpu_reference = static_cast<float*>(
            dscuda::device_malloc(output_elements * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(
            gpu_input,
            input.data(),
            input_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_weight,
            weight.data(),
            weight_elements * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            gpu_offsets,
            offsets.data(),
            offsets.size() * sizeof(int),
            cudaMemcpyHostToDevice));

        cublasHandle_t handle = nullptr;
        cublas_check(cublasCreate(&handle));
        cublas_check(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

        dscuda::grouped_linear_bf16_forward_cuda(
            gpu_custom,
            gpu_input,
            gpu_weight,
            gpu_offsets,
            rows,
            experts,
            output_size,
            input_size);
        cublas_grouped_linear(
            handle,
            gpu_reference,
            gpu_input,
            gpu_weight,
            offsets,
            experts,
            output_size,
            input_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> custom(output_elements);
        std::vector<float> reference(output_elements);
        CUDA_CHECK(cudaMemcpy(
            custom.data(),
            gpu_custom,
            output_elements * sizeof(float),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            reference.data(),
            gpu_reference,
            output_elements * sizeof(float),
            cudaMemcpyDeviceToHost));
        float max_error = 0.0F;
        for (std::size_t index = 0; index < output_elements; ++index) {
            max_error = std::max(
                max_error, std::abs(custom[index] - reference[index]));
        }
        if (max_error > 3.0e-3F) {
            throw std::runtime_error("custom/cuBLAS correctness check failed");
        }

        auto run_backend = [&]() {
            if (std::strcmp(backend, "custom") == 0) {
                dscuda::grouped_linear_bf16_forward_cuda(
                    gpu_custom,
                    gpu_input,
                    gpu_weight,
                    gpu_offsets,
                    rows,
                    experts,
                    output_size,
                    input_size);
            } else {
                cublas_grouped_linear(
                    handle,
                    gpu_reference,
                    gpu_input,
                    gpu_weight,
                    offsets,
                    experts,
                    output_size,
                    input_size);
            }
        };
        run_backend();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStart());
        run_backend();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());

        std::printf(
            "MoE grouped GEMM: rows=%d experts=%d K=%d N=%d distribution=%s "
            "backend=%s max_error=%.6g\n",
            rows,
            experts,
            input_size,
            output_size,
            distribution,
            backend,
            max_error);

        cublas_check(cublasDestroy(handle));
        dscuda::device_free(gpu_reference);
        dscuda::device_free(gpu_custom);
        dscuda::device_free(gpu_offsets);
        dscuda::device_free(gpu_weight);
        dscuda::device_free(gpu_input);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "benchmark_grouped_gemm: %s\n", error.what());
        return 1;
    }
}

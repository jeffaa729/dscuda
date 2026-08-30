// Compares generic FP32 and BF16 GEMMs against a scalar CPU oracle across all transpose modes.
// Rectangular/tail shapes, nonzero initial outputs, and repeated calls check overwrite and accumulation independently.

#include "cuda_common.h"
#include "matmul.h"
#include "matmul_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

float max_error(const std::vector<float>& expected, const std::vector<float>& actual) {
    float error = 0.0F;
    for (size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
            return std::numeric_limits<float>::infinity();
        }
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

// Pack the same logical matrix into normal or physically transposed row-major storage.
template <typename T>
std::vector<T> pack(const std::vector<float>& input, int rows, int columns, bool transpose) {
    std::vector<T> output(input.size());
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            output[transpose ? column * rows + row : row * columns + column] =
                static_cast<T>(input[row * columns + column]);
        }
    }
    return output;
}

template <typename T>
bool run_case(int M, int N, int K, cudaStream_t stream) {
    constexpr bool bf16 = std::is_same_v<T, __nv_bfloat16>;
    std::vector<float> left(M * K), right(K * N), product(M * N);
    for (int index = 0; index < M * K; ++index) {
        left[index] = static_cast<float>((index * 17) % 101 - 50) / 63.0F;
        if constexpr (bf16) left[index] = __bfloat162float(__float2bfloat16(left[index]));
    }
    for (int index = 0; index < K * N; ++index) {
        right[index] = static_cast<float>((index * 23) % 97 - 48) / 61.0F;
        if constexpr (bf16) right[index] = __bfloat162float(__float2bfloat16(right[index]));
    }
    dscuda::gemm_cpu(product.data(), left.data(), right.data(), M, N, K);

    auto* gpu_left = static_cast<T*>(dscuda::device_malloc(left.size() * sizeof(T)));
    auto* gpu_right = static_cast<T*>(dscuda::device_malloc(right.size() * sizeof(T)));
    auto* gpu_output = static_cast<float*>(dscuda::device_malloc(product.size() * sizeof(float)));
    std::vector<float> actual(product.size()), expected(product.size());
    const std::vector<float> initial(product.size(), 0.25F);
    bool passed = true;

    for (const char* operation : {"NN", "NT", "TN", "TT"}) {
        const bool transpose_left = operation[0] == 'T';
        const bool transpose_right = operation[1] == 'T';
        const auto packed_left = pack<T>(left, M, K, transpose_left);
        const auto packed_right = pack<T>(right, K, N, transpose_right);
        const auto cpu_left = pack<float>(left, M, K, transpose_left);
        const auto cpu_right = pack<float>(right, K, N, transpose_right);
        CUDA_CHECK(cudaMemcpyAsync(gpu_left, packed_left.data(), left.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_right, packed_right.data(), right.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));

        for (bool accumulate : {false, true}) {
            auto cpu_output = initial;
            CUDA_CHECK(cudaMemcpyAsync(gpu_output, initial.data(), initial.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
            for (int repeat = 0; repeat < 2; ++repeat) {
                dscuda::gemm_cpu(cpu_output.data(), cpu_left.data(), cpu_right.data(),
                                 M, N, K, transpose_left, transpose_right, accumulate);
                if constexpr (bf16) {
                    dscuda::gemm_bf16_cuda(gpu_output, gpu_left, gpu_right, M, N, K,
                                           transpose_left, transpose_right, accumulate, stream);
                } else {
                    dscuda::gemm_fp32_cuda(gpu_output, gpu_left, gpu_right, M, N, K,
                                           transpose_left, transpose_right, accumulate, stream);
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(actual.data(), gpu_output, actual.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (size_t index = 0; index < product.size(); ++index) {
                expected[index] = accumulate ? initial[index] + 2.0F * product[index] : product[index];
            }
            const float error = std::max(max_error(expected, actual), max_error(expected, cpu_output));
            const bool ok = error < 5.0e-4F;
            std::printf("  %-4s %4dx%4dx%3d %s beta=%d max error=%.3e %s\n",
                        bf16 ? "BF16" : "FP32", M, N, K, operation,
                        accumulate ? 1 : 0, error, ok ? "PASS" : "FAIL");
            passed &= ok;
        }
    }
    dscuda::device_free(gpu_output);
    dscuda::device_free(gpu_right);
    dscuda::device_free(gpu_left);
    return passed;
}

}  // namespace

int main() {
    try {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        bool passed = true;
        // Exercise small/large tile dispatch and the edge fallback in both precisions.
        for (const auto& shape : {std::vector<int>{128, 256, 64}, {640, 128, 96}, {1152, 128, 32}}) {
            passed &= run_case<float>(shape[0], shape[1], shape[2], stream);
            passed &= run_case<__nv_bfloat16>(shape[0], shape[1], shape[2], stream);
        }
        passed &= run_case<float>(37, 53, 29, stream);
        passed &= run_case<__nv_bfloat16>(48, 80, 48, stream);
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::printf("GEMM test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GEMM test failed: %s\n", error.what());
        return 1;
    }
}

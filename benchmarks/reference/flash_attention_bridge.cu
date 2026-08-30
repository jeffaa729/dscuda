// Exposes the native BF16 attention operator to the shared Python comparison harness without depending on PyTorch C++ headers.
// PyTorch owns the buffers and stream, while exceptions are returned as error strings instead of crossing the C ABI.

#include "flash_attention.h"

#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" const char* dscuda_flash_last_error() {
    return last_error.c_str();
}

extern "C" int dscuda_flash_forward(
    __nv_bfloat16* output, float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch, int sequence, int heads, int dimension, float scale, cudaStream_t stream) {
    try {
        dscuda::flash_attention_forward_bf16_io_cuda(
            output, logsumexp, query, key, value,
            batch, sequence, heads, dimension, scale, stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" int dscuda_flash_backward(
    __nv_bfloat16* dq, __nv_bfloat16* dk, __nv_bfloat16* dv,
    const __nv_bfloat16* dout, const __nv_bfloat16* output, const float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch, int sequence, int heads, int dimension, float scale, cudaStream_t stream) {
    try {
        dscuda::flash_attention_backward_bf16_io_cuda(
            dq, dk, dv, dout, output, logsumexp, query, key, value,
            batch, sequence, heads, dimension, scale, stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

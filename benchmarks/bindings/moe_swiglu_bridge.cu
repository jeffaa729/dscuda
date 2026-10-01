#include "moe_swiglu.h"

#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" const char* dscuda_moe_last_error() { return last_error.c_str(); }

extern "C" int dscuda_moe_swiglu(__nv_bfloat16* output, const __nv_bfloat16* gate_up, const float* weights,
                                  const int* valid_rows, int rows, int intermediate, float clamp, cudaStream_t stream) {
    try {
        dscuda::moe_swiglu_bf16_cuda(output, gate_up, weights, valid_rows, rows, intermediate, clamp, stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

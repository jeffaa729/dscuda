// Exposes MoE routing and data movement to PyTorch-owned CUDA buffers.
// These wrappers only forward arguments and report host exceptions through the C ABI.
#include "expert_dispatch.h"
#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
template<class Function> int run(Function function) {
    try { function(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return 1; }
}
}
extern "C" const char* dscuda_expert_last_error() { return last_error.c_str(); }

extern "C" int dscuda_route(float* scores, int* ids, float* weights, int* counts,
    const float* logits, const float* bias, int rows, int experts, int topk, float scale, cudaStream_t stream) {
    return run([&] { dscuda::expert_route_forward_cuda(scores, ids, weights, counts,
        logits, bias, rows, experts, topk, scale, stream); });
}

extern "C" int dscuda_dispatch(float* packed, int* offsets, int* route_to_slot,
    int* slot_to_route, int* slot_expert, const float* input, const int* ids, const int* counts,
    int rows, int width, int experts, int topk, cudaStream_t stream) {
    return run([&] { dscuda::expert_dispatch_forward_cuda(packed, offsets, route_to_slot,
        slot_to_route, slot_expert, input, ids, counts, rows, width, experts, topk, stream); });
}

extern "C" int dscuda_combine(float* output, const float* shared, const float* packed,
    const float* weights, const int* slots, int rows, int width, int topk, cudaStream_t stream) {
    return run([&] { dscuda::expert_combine_forward_cuda(output, shared, packed, weights,
        slots, rows, width, topk, stream); });
}

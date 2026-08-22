// Computes numerically stable scaled causal softmax over square attention-score rows.
// Its backward path applies the softmax Jacobian only to visible keys and accumulates gradients into the logits.

#include "cuda_common.h"
#include "softmax.h"

namespace dscuda {
namespace {

__global__ void causal_softmax_forward_kernel(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale) {
}

__global__ void causal_softmax_backward_kernel(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale) {
}

}  // namespace

void causal_softmax_forward_cuda(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale,
    cudaStream_t stream) {
}

void causal_softmax_backward_cuda(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale,
    cudaStream_t stream) {
}

}  // namespace dscuda

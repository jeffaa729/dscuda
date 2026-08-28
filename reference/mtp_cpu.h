#pragma once

#include "mtp.h"

namespace dscuda {

void mtp_forward_cpu(
    float* output,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    const MtpParameters& parameters,
    const float* cosine,
    const float* sine,
    const MtpConfig& config);

void mtp_backward_cpu(
    float* previous_hidden_gradient,
    float* embedding_gradient,
    const MtpGradients& parameter_gradients,
    const float* output_gradient,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    const MtpParameters& parameters,
    const float* cosine,
    const float* sine,
    const MtpConfig& config);

}  // namespace dscuda

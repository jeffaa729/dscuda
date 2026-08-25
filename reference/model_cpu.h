#pragma once

#include "model.h"

#include <vector>

namespace dscuda {

struct DenseGptCpuCache {
    std::vector<float> hidden_states;
    std::vector<float> final_norm;
    std::vector<float> final_inverse_rms;
    std::vector<float> logits;
    std::vector<float> logsumexp;
    float loss = 0.0F;
};

float dense_gpt_forward_cpu(
    DenseGptCpuCache& cache,
    const int* input_tokens,
    const int* target_tokens,
    const float* parameters,
    const ModelParameterLayout& layout,
    const std::vector<float>& cosine,
    const std::vector<float>& sine,
    const ModelConfig& config);

void dense_gpt_backward_cpu(
    float* gradients,
    const DenseGptCpuCache& cache,
    const int* input_tokens,
    const int* target_tokens,
    const float* parameters,
    const ModelParameterLayout& layout,
    const std::vector<float>& cosine,
    const std::vector<float>& sine,
    const ModelConfig& config);

}  // namespace dscuda

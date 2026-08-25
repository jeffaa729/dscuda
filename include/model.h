#pragma once

#include "generation.h"
#include "optimizer.h"
#include "transformer_block.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dscuda {

struct ModelConfig {
    int batch_size;
    int sequence_length;
    int vocabulary_size;
    int layers;
    int hidden_size;
    int heads;
    int ffn_size;
    int rotary_size;
    float rms_epsilon;
};

struct ModelBlockOffsets {
    std::size_t attention_norm;
    std::size_t query;
    std::size_t key;
    std::size_t value;
    std::size_t output;
    std::size_t ffn_norm;
    std::size_t gate;
    std::size_t up;
    std::size_t down;
};

struct ModelParameterLayout {
    std::size_t token_embedding;
    std::vector<ModelBlockOffsets> blocks;
    std::size_t final_norm;
    std::size_t elements;
};

struct ModelMemoryReport {
    std::size_t parameter_elements;
    std::size_t saved_activation_elements;
    std::size_t workspace_elements;
    std::size_t total_bytes;
};

struct TrainStepResult {
    float loss;
    float gradient_norm;
};

struct ModelTrainingState {
    std::vector<float> parameters;
    std::vector<float> first_moment;
    std::vector<float> second_moment;
};

ModelParameterLayout make_model_parameter_layout(const ModelConfig& config);

TransformerBlockParameters model_block_parameters(
    const float* parameters,
    const ModelParameterLayout& layout,
    int layer);

TransformerBlockGradients model_block_gradients(
    float* gradients,
    const ModelParameterLayout& layout,
    int layer);

// Owns all persistent FP32 parameters, optimizer state, saved activations, and
// reusable backward workspace for a dense pre-norm causal language model.
class DenseGptModel : public AutoregressiveModel {
public:
    explicit DenseGptModel(const ModelConfig& config);
    ~DenseGptModel();

    DenseGptModel(const DenseGptModel&) = delete;
    DenseGptModel& operator=(const DenseGptModel&) = delete;

    void initialize(std::uint64_t seed = 1337);
    void load_parameters(const std::vector<float>& parameters);
    ModelTrainingState training_state_to_host() const;
    void load_training_state(const ModelTrainingState& state);

    float forward(
        const std::vector<int>& input_tokens,
        const std::vector<int>& target_tokens);
    void zero_gradients();
    void backward();
    TrainStepResult train_step(
        const std::vector<int>& input_tokens,
        const std::vector<int>& target_tokens,
        int step,
        const AdamWConfig& optimizer,
        float maximum_gradient_norm);

    std::vector<float> parameters_to_host() const;
    std::vector<float> gradients_to_host() const;
    std::vector<float> logits_to_host() const;

    int vocabulary_size() const override;
    int maximum_context_length() const override;
    std::vector<float> forward_last_logits(
        const std::vector<int>& tokens) override;

    const ModelConfig& config() const;
    const ModelParameterLayout& parameter_layout() const;
    ModelMemoryReport memory_report() const;

private:
    void run_model_forward(const std::vector<int>& input_tokens);

    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace dscuda

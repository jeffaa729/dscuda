#pragma once

#include "deepseek_v3_block.h"
#include "deepseek_v3_dense_block.h"
#include "generation.h"
#include "model.h"
#include "mtp.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dscuda {

struct DeepSeekV3Config {
    int batch_size;
    int sequence_length;
    int vocabulary_size;
    int layers;
    int hidden_size;
    int heads;
    int query_rank;
    int kv_rank;
    int nope_size;
    int rope_size;
    int value_size;
    int expert_hidden_size;
    int routed_experts;
    int shared_experts;
    int top_k;
    float rms_epsilon;
    float route_scale;
    float routing_bias_update_speed;
    float balance_loss_weight = 0.0F;
    int mtp_depth = 0;
    float mtp_loss_weight = 0.0F;
    int dense_layers = 0;
    int dense_ffn_size = 0;
};

struct DeepSeekV3BlockOffsets {
    std::size_t attention_norm;
    std::size_t query_down;
    std::size_t query_norm;
    std::size_t query_up;
    std::size_t kv_down;
    std::size_t kv_norm;
    std::size_t key_up;
    std::size_t value_up;
    std::size_t attention_output;
    std::size_t ffn_norm;
    std::size_t router;
    std::size_t routed_gate;
    std::size_t routed_up;
    std::size_t routed_down;
    std::size_t shared_gate;
    std::size_t shared_up;
    std::size_t shared_down;
};

struct DeepSeekV3ParameterLayout {
    std::size_t token_embedding;
    std::vector<DeepSeekV3BlockOffsets> blocks;
    std::size_t final_norm;
    struct MtpOffsets {
        std::size_t hidden_norm;
        std::size_t embedding_norm;
        std::size_t projection;
        DeepSeekV3BlockOffsets block;
    };
    std::vector<MtpOffsets> mtp_modules;
    std::size_t elements;
};

struct DeepSeekV3TrainingState {
    ModelTrainingState optimizer;
    std::vector<float> routing_bias;
};

DeepSeekV3ParameterLayout make_deepseek_v3_parameter_layout(
    const DeepSeekV3Config& config);

class DeepSeekV3Model : public AutoregressiveModel {
public:
    explicit DeepSeekV3Model(const DeepSeekV3Config& config);
    ~DeepSeekV3Model();

    DeepSeekV3Model(const DeepSeekV3Model&) = delete;
    DeepSeekV3Model& operator=(const DeepSeekV3Model&) = delete;

    void initialize(std::uint64_t seed = 1337);
    void load_parameters(const std::vector<float>& parameters);
    DeepSeekV3TrainingState training_state_to_host() const;
    void load_training_state(const DeepSeekV3TrainingState& state);

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
    std::vector<float> routing_bias_to_host() const;

    int vocabulary_size() const override;
    int maximum_context_length() const override;
    std::size_t kv_cache_bytes_per_token() const;
    std::vector<float> forward_last_logits(
        const std::vector<int>& tokens) override;

    const DeepSeekV3Config& config() const;
    const DeepSeekV3ParameterLayout& parameter_layout() const;
    ModelMemoryReport memory_report() const;

private:
    void run_model_forward(const std::vector<int>& input_tokens);

    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace dscuda

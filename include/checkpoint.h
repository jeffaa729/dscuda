#pragma once

#include "deepseek_v3_model.h"
#include "model.h"

#include <cstdint>
#include <string>

namespace dscuda {

enum class CheckpointArchitecture : std::uint32_t {
    dense_gpt = 1,
    deepseek_v3 = 2,
    deepseek_v4 = 3,
};

struct CheckpointMetadata {
    std::uint32_t format_version;
    CheckpointArchitecture architecture;
    ModelConfig config;
    DeepSeekV3Config deepseek_v3_config;
    std::uint64_t step;
    std::uint64_t parameter_elements;
    std::uint64_t routing_bias_elements;
};

std::string checkpoint_step_directory(
    const std::string& output_directory,
    std::uint64_t step);

void save_dense_gpt_checkpoint(
    const std::string& checkpoint_directory,
    const DenseGptModel& model,
    std::uint64_t step);

CheckpointMetadata read_checkpoint_metadata(
    const std::string& checkpoint_directory);

CheckpointMetadata load_dense_gpt_checkpoint(
    const std::string& checkpoint_directory,
    DenseGptModel& model);

void save_deepseek_v3_checkpoint(
    const std::string& checkpoint_directory,
    const DeepSeekV3Model& model,
    std::uint64_t step);

CheckpointMetadata load_deepseek_v3_checkpoint(
    const std::string& checkpoint_directory,
    DeepSeekV3Model& model);

}  // namespace dscuda

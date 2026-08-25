#pragma once

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
    std::uint64_t step;
    std::uint64_t parameter_elements;
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

}  // namespace dscuda

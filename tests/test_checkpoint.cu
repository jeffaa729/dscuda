// Verifies checkpoint metadata, parameters, and AdamW moments by resuming the same model into an identical second update.
// The DONE marker and versioned architecture header ensure only complete, compatible checkpoints are accepted.

#include "checkpoint.h"
#include "cuda_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <vector>

namespace {

constexpr dscuda::ModelConfig kConfig{
    1, 8, 32, 1, 16, 2, 24, 8, 1.0e-5F};

float maximum_error(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    float error = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        error = std::max(error, std::abs(left[index] - right[index]));
    }
    return error;
}

bool run_test() {
    std::vector<int> inputs(8);
    std::vector<int> targets(8);
    for (int index = 0; index < 8; ++index) {
        inputs[index] = (index * 3 + 1) % 32;
        targets[index] = (index * 5 + 2) % 32;
    }
    const dscuda::AdamWConfig optimizer{
        1.0e-3F, 0.9F, 0.95F, 1.0e-8F, 0.01F};

    dscuda::DenseGptModel original(kConfig);
    original.initialize(2026);
    original.train_step(inputs, targets, 1, optimizer, 1.0F);

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "dscuda_checkpoint_test";
    std::filesystem::remove_all(directory);
    dscuda::save_dense_gpt_checkpoint(directory.string(), original, 1);

    const dscuda::CheckpointMetadata metadata =
        dscuda::read_checkpoint_metadata(directory.string());
    dscuda::DenseGptModel restored(kConfig);
    const dscuda::CheckpointMetadata loaded =
        dscuda::load_dense_gpt_checkpoint(directory.string(), restored);

    bool passed = metadata.step == 1 && loaded.step == 1;
    passed &= metadata.architecture
        == dscuda::CheckpointArchitecture::dense_gpt;
    const dscuda::ModelTrainingState original_state =
        original.training_state_to_host();
    const dscuda::ModelTrainingState restored_state =
        restored.training_state_to_host();
    passed &= maximum_error(
        original_state.parameters, restored_state.parameters) == 0.0F;
    passed &= maximum_error(
        original_state.first_moment, restored_state.first_moment) == 0.0F;
    passed &= maximum_error(
        original_state.second_moment, restored_state.second_moment) == 0.0F;

    original.train_step(inputs, targets, 2, optimizer, 1.0F);
    restored.train_step(inputs, targets, 2, optimizer, 1.0F);
    const float resumed_error = maximum_error(
        original.parameters_to_host(), restored.parameters_to_host());
    passed &= resumed_error == 0.0F;
    std::printf("  resumed update max error = %.3e\n", resumed_error);

    std::filesystem::remove_all(directory);
    return passed;
}

}  // namespace

int main() {
    try {
        dscuda::print_device_summary();
        const bool passed = run_test();
        std::printf("Checkpoint test: %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Checkpoint test failed: %s\n", error.what());
        return 1;
    }
}

// Verifies that a uint16 token stream becomes contiguous input/target pairs
// with the one-token shift required by causal language-model training.

#include "dataset.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    const auto path = std::filesystem::temp_directory_path()
        / "dscuda_test_dataset.bin";
    const std::vector<std::uint16_t> tokens = {
        10, 11, 12, 13, 14, 15, 16, 17, 18,
    };
    {
        std::ofstream file(path, std::ios::binary);
        file.write(
            reinterpret_cast<const char*>(tokens.data()),
            tokens.size() * sizeof(std::uint16_t));
    }

    bool passed = true;
    try {
        const dscuda::TokenDataset dataset(path.string());
        std::vector<int> inputs;
        std::vector<int> targets;
        dataset.get_batch(1, 2, 3, inputs, targets);
        passed &= inputs == std::vector<int>({11, 12, 13, 14, 15, 16});
        passed &= targets == std::vector<int>({12, 13, 14, 15, 16, 17});
        passed &= dataset.token_count() == tokens.size();
        passed &= dataset.batch_count(2, 3) == 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Dataset test failed: %s\n", error.what());
        passed = false;
    }

    if (argc == 2) {
        try {
            const dscuda::TokenDataset dataset(argv[1]);
            std::vector<int> inputs;
            std::vector<int> targets;
            dataset.get_batch(0, 4, 256, inputs, targets);
            for (std::size_t index = 0; index + 1 < inputs.size(); ++index) {
                passed &= inputs[index + 1] == targets[index];
            }
            std::printf(
                "Real dataset: %zu tokens, %zu batches at B=4 T=256\n",
                dataset.token_count(),
                dataset.batch_count(4, 256));
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Real-dataset test failed: %s\n", error.what());
            passed = false;
        }
    }

    std::filesystem::remove(path);
    std::printf("Dataset test: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

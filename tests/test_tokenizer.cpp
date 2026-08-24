// Verifies that the C++ tokenizer reads the Python binary format and applies
// ranked byte-level merges while preserving punctuation and special tokens.

#include "tokenizer.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void write_u32(std::ofstream& file, std::uint32_t value) {
    const std::array<char, 4> bytes = {
        static_cast<char>(value),
        static_cast<char>(value >> 8),
        static_cast<char>(value >> 16),
        static_cast<char>(value >> 24),
    };
    file.write(bytes.data(), bytes.size());
}

void write_test_tokenizer(const std::filesystem::path& path) {
    std::ofstream file(path, std::ios::binary);
    const std::array<char, 8> magic = {'D', 'S', 'B', 'P', 'E', '0', '1', '\0'};
    file.write(magic.data(), magic.size());
    write_u32(file, 1);
    write_u32(file, 262);
    write_u32(file, 4);
    write_u32(file, 256);
    write_u32(file, 257);
    write_u32(file, 0);

    const std::array<std::array<std::uint32_t, 3>, 4> merges = {{
        {{'l', 'l', 258}},
        {{'e', 258, 259}},
        {{'h', 259, 260}},
        {{260, 'o', 261}},
    }};
    for (const auto& merge : merges) {
        write_u32(file, merge[0]);
        write_u32(file, merge[1]);
        write_u32(file, merge[2]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const auto path = std::filesystem::temp_directory_path()
        / "dscuda_test_tokenizer.bin";
    write_test_tokenizer(path);

    bool passed = true;
    try {
        const dscuda::Tokenizer tokenizer(path.string());
        const std::vector<int> expected = {256, 261, ' ', 261, '!', 257};
        const std::vector<int> actual = tokenizer.encode("hello hello!", true, true);
        passed &= actual == expected;
        passed &= tokenizer.decode(actual) == "hello hello!";
        passed &= tokenizer.decode(actual, false)
            == "<bos>hello hello!<eos>";
        passed &= tokenizer.vocabulary_size() == 262;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Tokenizer test failed: %s\n", error.what());
        passed = false;
    }

    if (argc == 2) {
        try {
            const dscuda::Tokenizer tokenizer(argv[1]);
            const std::string sample =
                "Once upon a time, a tiny CUDA model learned to tell stories.";
            const std::vector<int> tokens = tokenizer.encode(sample, true, true);
            passed &= tokenizer.decode(tokens) == sample;
            std::printf(
                "Real tokenizer: %zu bytes -> %zu tokens, vocabulary %d\n",
                sample.size(),
                tokens.size(),
                tokenizer.vocabulary_size());
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Real-tokenizer test failed: %s\n", error.what());
            passed = false;
        }
    }

    std::filesystem::remove(path);
    std::printf("Tokenizer test: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

// Saves a versioned architecture header followed by contiguous parameters and optimizer state, mirroring llm.c's simple binary checkpoints.
// A DONE marker is published last so resume logic never selects a partially written checkpoint directory.

#include "checkpoint.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace dscuda {
namespace {

constexpr std::uint32_t kMagic0 = 0x4B435344U;  // DSCK
constexpr std::uint32_t kMagic1 = 0x31305450U;  // PT01
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kFp32 = 1;
constexpr std::uint32_t kAdamW = 1;
constexpr int kHeaderWords = 64;

void put_u64(
    std::array<std::uint32_t, kHeaderWords>& header,
    int index,
    std::uint64_t value) {
    header[index] = static_cast<std::uint32_t>(value);
    header[index + 1] = static_cast<std::uint32_t>(value >> 32);
}

std::uint64_t get_u64(
    const std::array<std::uint32_t, kHeaderWords>& header,
    int index) {
    return static_cast<std::uint64_t>(header[index])
        | (static_cast<std::uint64_t>(header[index + 1]) << 32);
}

void write_u32(std::ofstream& file, std::uint32_t value) {
    const std::array<char, 4> bytes = {
        static_cast<char>(value),
        static_cast<char>(value >> 8),
        static_cast<char>(value >> 16),
        static_cast<char>(value >> 24),
    };
    file.write(bytes.data(), bytes.size());
}

std::uint32_t read_u32(std::ifstream& file) {
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!file) {
        throw std::runtime_error("checkpoint header is truncated");
    }
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::filesystem::path model_path(const std::string& directory) {
    return std::filesystem::path(directory) / "model.bin";
}

CheckpointMetadata read_header(std::ifstream& file) {
    std::array<std::uint32_t, kHeaderWords> header{};
    for (std::uint32_t& value : header) {
        value = read_u32(file);
    }
    if (header[0] != kMagic0 || header[1] != kMagic1
        || header[2] != kVersion || header[4] != kFp32
        || header[5] != kAdamW) {
        throw std::runtime_error("unsupported checkpoint format");
    }
    if (header[3]
        != static_cast<std::uint32_t>(CheckpointArchitecture::dense_gpt)) {
        throw std::runtime_error("checkpoint is not a dense GPT model");
    }

    float epsilon = 0.0F;
    std::memcpy(&epsilon, &header[14], sizeof(float));
    CheckpointMetadata metadata{};
    metadata.format_version = header[2];
    metadata.architecture = static_cast<CheckpointArchitecture>(header[3]);
    metadata.config = {
        static_cast<int>(header[6]),
        static_cast<int>(header[7]),
        static_cast<int>(header[8]),
        static_cast<int>(header[9]),
        static_cast<int>(header[10]),
        static_cast<int>(header[11]),
        static_cast<int>(header[12]),
        static_cast<int>(header[13]),
        epsilon,
    };
    metadata.step = get_u64(header, 16);
    metadata.parameter_elements = get_u64(header, 18);
    return metadata;
}

void check_compatible(
    const ModelConfig& checkpoint,
    const ModelConfig& runtime) {
    if (checkpoint.vocabulary_size != runtime.vocabulary_size
        || checkpoint.layers != runtime.layers
        || checkpoint.hidden_size != runtime.hidden_size
        || checkpoint.heads != runtime.heads
        || checkpoint.ffn_size != runtime.ffn_size
        || checkpoint.rotary_size != runtime.rotary_size
        || checkpoint.rms_epsilon != runtime.rms_epsilon
        || runtime.sequence_length > checkpoint.sequence_length) {
        throw std::runtime_error(
            "checkpoint architecture does not match the runtime model");
    }
}

}  // namespace

std::string checkpoint_step_directory(
    const std::string& output_directory,
    std::uint64_t step) {
    std::ostringstream name;
    name << "step_" << std::setw(8) << std::setfill('0') << step;
    return (std::filesystem::path(output_directory) / name.str()).string();
}

void save_dense_gpt_checkpoint(
    const std::string& checkpoint_directory,
    const DenseGptModel& model,
    std::uint64_t step) {
    const std::filesystem::path directory(checkpoint_directory);
    std::filesystem::create_directories(directory);
    const std::filesystem::path destination = directory / "model.bin";
    const std::filesystem::path temporary = directory / "model.bin.part";
    const std::filesystem::path done = directory / "DONE";
    const std::filesystem::path done_temporary = directory / "DONE.part";

    const ModelTrainingState state = model.training_state_to_host();
    const ModelConfig& config = model.config();
    std::array<std::uint32_t, kHeaderWords> header{};
    header[0] = kMagic0;
    header[1] = kMagic1;
    header[2] = kVersion;
    header[3] = static_cast<std::uint32_t>(CheckpointArchitecture::dense_gpt);
    header[4] = kFp32;
    header[5] = kAdamW;
    header[6] = config.batch_size;
    header[7] = config.sequence_length;
    header[8] = config.vocabulary_size;
    header[9] = config.layers;
    header[10] = config.hidden_size;
    header[11] = config.heads;
    header[12] = config.ffn_size;
    header[13] = config.rotary_size;
    std::memcpy(&header[14], &config.rms_epsilon, sizeof(float));
    put_u64(header, 16, step);
    put_u64(header, 18, state.parameters.size());

    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("cannot create checkpoint file");
        }
        for (const std::uint32_t value : header) {
            write_u32(file, value);
        }
        auto write_values = [&](const std::vector<float>& values) {
            file.write(
                reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
        };
        write_values(state.parameters);
        write_values(state.first_moment);
        write_values(state.second_moment);
        if (!file) {
            throw std::runtime_error("failed while writing checkpoint state");
        }
    }

    std::filesystem::remove(destination);
    std::filesystem::rename(temporary, destination);
    {
        std::ofstream marker(done_temporary, std::ios::trunc);
        marker << step << '\n';
    }
    std::filesystem::remove(done);
    std::filesystem::rename(done_temporary, done);
}

CheckpointMetadata read_checkpoint_metadata(
    const std::string& checkpoint_directory) {
    const std::filesystem::path directory(checkpoint_directory);
    if (!std::filesystem::exists(directory / "DONE")) {
        throw std::runtime_error("checkpoint has no DONE marker");
    }
    std::ifstream file(model_path(checkpoint_directory), std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open checkpoint model.bin");
    }
    return read_header(file);
}

CheckpointMetadata load_dense_gpt_checkpoint(
    const std::string& checkpoint_directory,
    DenseGptModel& model) {
    if (!std::filesystem::exists(
            std::filesystem::path(checkpoint_directory) / "DONE")) {
        throw std::runtime_error("checkpoint has no DONE marker");
    }
    std::ifstream file(model_path(checkpoint_directory), std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open checkpoint model.bin");
    }
    const CheckpointMetadata metadata = read_header(file);
    check_compatible(metadata.config, model.config());
    if (metadata.parameter_elements != model.parameter_layout().elements) {
        throw std::runtime_error("checkpoint parameter count does not match model");
    }

    ModelTrainingState state;
    state.parameters.resize(metadata.parameter_elements);
    state.first_moment.resize(metadata.parameter_elements);
    state.second_moment.resize(metadata.parameter_elements);
    auto read_values = [&](std::vector<float>& values) {
        file.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!file) {
            throw std::runtime_error("checkpoint tensor data is truncated");
        }
    };
    read_values(state.parameters);
    read_values(state.first_moment);
    read_values(state.second_moment);
    if (file.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("checkpoint contains unexpected trailing data");
    }
    model.load_training_state(state);
    return metadata;
}

}  // namespace dscuda

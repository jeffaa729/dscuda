// Saves a versioned architecture header followed by contiguous parameters and optimizer state, mirroring llm.c's simple binary checkpoints.
// A DONE marker is published last so resume logic never selects a partially written checkpoint directory.

#include "checkpoint.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
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

void put_float(
    std::array<std::uint32_t, kHeaderWords>& header,
    int index,
    float value) {
    std::memcpy(&header[index], &value, sizeof(float));
}

float get_float(
    const std::array<std::uint32_t, kHeaderWords>& header,
    int index) {
    float value = 0.0F;
    std::memcpy(&value, &header[index], sizeof(float));
    return value;
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

    CheckpointMetadata metadata{};
    metadata.format_version = header[2];
    metadata.architecture = static_cast<CheckpointArchitecture>(header[3]);
    metadata.step = get_u64(header, 16);
    metadata.parameter_elements = get_u64(header, 18);
    if (metadata.architecture == CheckpointArchitecture::dense_gpt) {
        metadata.config = {
            static_cast<int>(header[6]),
            static_cast<int>(header[7]),
            static_cast<int>(header[8]),
            static_cast<int>(header[9]),
            static_cast<int>(header[10]),
            static_cast<int>(header[11]),
            static_cast<int>(header[12]),
            static_cast<int>(header[13]),
            get_float(header, 14),
        };
    } else if (metadata.architecture
               == CheckpointArchitecture::deepseek_v3) {
        metadata.deepseek_v3_config = {
            static_cast<int>(header[6]),
            static_cast<int>(header[7]),
            static_cast<int>(header[8]),
            static_cast<int>(header[9]),
            static_cast<int>(header[10]),
            static_cast<int>(header[11]),
            static_cast<int>(header[12]),
            static_cast<int>(header[13]),
            static_cast<int>(header[14]),
            static_cast<int>(header[15]),
            static_cast<int>(header[20]),
            static_cast<int>(header[21]),
            static_cast<int>(header[22]),
            static_cast<int>(header[23]),
            static_cast<int>(header[24]),
            get_float(header, 25),
            get_float(header, 26),
            get_float(header, 27),
            get_float(header, 30),
            static_cast<int>(header[31]),
            get_float(header, 32),
            static_cast<int>(header[33]),
            static_cast<int>(header[34]),
        };
        metadata.routing_bias_elements = get_u64(header, 28);
    } else {
        throw std::runtime_error("unsupported checkpoint architecture");
    }
    return metadata;
}

void check_dense_compatible(
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

void check_deepseek_v3_compatible(
    const DeepSeekV3Config& checkpoint,
    const DeepSeekV3Config& runtime) {
    if (checkpoint.vocabulary_size != runtime.vocabulary_size
        || checkpoint.layers != runtime.layers
        || checkpoint.hidden_size != runtime.hidden_size
        || checkpoint.heads != runtime.heads
        || checkpoint.query_rank != runtime.query_rank
        || checkpoint.kv_rank != runtime.kv_rank
        || checkpoint.nope_size != runtime.nope_size
        || checkpoint.rope_size != runtime.rope_size
        || checkpoint.value_size != runtime.value_size
        || checkpoint.expert_hidden_size != runtime.expert_hidden_size
        || checkpoint.routed_experts != runtime.routed_experts
        || checkpoint.shared_experts != runtime.shared_experts
        || checkpoint.top_k != runtime.top_k
        || checkpoint.rms_epsilon != runtime.rms_epsilon
        || checkpoint.route_scale != runtime.route_scale
        || checkpoint.routing_bias_update_speed
            != runtime.routing_bias_update_speed
        || checkpoint.balance_loss_weight != runtime.balance_loss_weight
        || checkpoint.mtp_depth != runtime.mtp_depth
        || checkpoint.mtp_loss_weight != runtime.mtp_loss_weight
        || checkpoint.dense_layers != runtime.dense_layers
        || checkpoint.dense_ffn_size != runtime.dense_ffn_size
        || runtime.sequence_length > checkpoint.sequence_length) {
        throw std::runtime_error(
            "checkpoint architecture does not match the runtime V3 model");
    }
}

void write_checkpoint(
    const std::string& checkpoint_directory,
    const std::array<std::uint32_t, kHeaderWords>& header,
    std::initializer_list<const std::vector<float>*> tensors,
    std::uint64_t step) {
    const std::filesystem::path directory(checkpoint_directory);
    std::filesystem::create_directories(directory);
    const std::filesystem::path destination = directory / "model.bin";
    const std::filesystem::path temporary = directory / "model.bin.part";
    const std::filesystem::path done = directory / "DONE";
    const std::filesystem::path done_temporary = directory / "DONE.part";

    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("cannot create checkpoint file");
        }
        for (const std::uint32_t value : header) {
            write_u32(file, value);
        }
        for (const std::vector<float>* tensor : tensors) {
            file.write(
                reinterpret_cast<const char*>(tensor->data()),
                static_cast<std::streamsize>(tensor->size() * sizeof(float)));
        }
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

void read_tensor(std::ifstream& file, std::vector<float>& tensor) {
    file.read(
        reinterpret_cast<char*>(tensor.data()),
        static_cast<std::streamsize>(tensor.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("checkpoint tensor data is truncated");
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
    put_float(header, 14, config.rms_epsilon);
    put_u64(header, 16, step);
    put_u64(header, 18, state.parameters.size());
    write_checkpoint(
        checkpoint_directory,
        header,
        {&state.parameters, &state.first_moment, &state.second_moment},
        step);
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
    if (metadata.architecture != CheckpointArchitecture::dense_gpt) {
        throw std::runtime_error("checkpoint is not a dense GPT model");
    }
    check_dense_compatible(metadata.config, model.config());
    if (metadata.parameter_elements != model.parameter_layout().elements) {
        throw std::runtime_error("checkpoint parameter count does not match model");
    }

    ModelTrainingState state;
    state.parameters.resize(metadata.parameter_elements);
    state.first_moment.resize(metadata.parameter_elements);
    state.second_moment.resize(metadata.parameter_elements);
    read_tensor(file, state.parameters);
    read_tensor(file, state.first_moment);
    read_tensor(file, state.second_moment);
    if (file.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("checkpoint contains unexpected trailing data");
    }
    model.load_training_state(state);
    return metadata;
}

void save_deepseek_v3_checkpoint(
    const std::string& checkpoint_directory,
    const DeepSeekV3Model& model,
    std::uint64_t step) {
    const DeepSeekV3TrainingState state = model.training_state_to_host();
    const DeepSeekV3Config& config = model.config();
    std::array<std::uint32_t, kHeaderWords> header{};
    header[0] = kMagic0;
    header[1] = kMagic1;
    header[2] = kVersion;
    header[3] = static_cast<std::uint32_t>(
        CheckpointArchitecture::deepseek_v3);
    header[4] = kFp32;
    header[5] = kAdamW;
    header[6] = config.batch_size;
    header[7] = config.sequence_length;
    header[8] = config.vocabulary_size;
    header[9] = config.layers;
    header[10] = config.hidden_size;
    header[11] = config.heads;
    header[12] = config.query_rank;
    header[13] = config.kv_rank;
    header[14] = config.nope_size;
    header[15] = config.rope_size;
    put_u64(header, 16, step);
    put_u64(header, 18, state.optimizer.parameters.size());
    header[20] = config.value_size;
    header[21] = config.expert_hidden_size;
    header[22] = config.routed_experts;
    header[23] = config.shared_experts;
    header[24] = config.top_k;
    put_float(header, 25, config.rms_epsilon);
    put_float(header, 26, config.route_scale);
    put_float(header, 27, config.routing_bias_update_speed);
    put_u64(header, 28, state.routing_bias.size());
    put_float(header, 30, config.balance_loss_weight);
    header[31] = config.mtp_depth;
    put_float(header, 32, config.mtp_loss_weight);
    header[33] = config.dense_layers;
    header[34] = config.dense_ffn_size;
    write_checkpoint(
        checkpoint_directory,
        header,
        {
            &state.optimizer.parameters,
            &state.optimizer.first_moment,
            &state.optimizer.second_moment,
            &state.routing_bias,
        },
        step);
}

CheckpointMetadata load_deepseek_v3_checkpoint(
    const std::string& checkpoint_directory,
    DeepSeekV3Model& model) {
    if (!std::filesystem::exists(
            std::filesystem::path(checkpoint_directory) / "DONE")) {
        throw std::runtime_error("checkpoint has no DONE marker");
    }
    std::ifstream file(model_path(checkpoint_directory), std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open checkpoint model.bin");
    }
    const CheckpointMetadata metadata = read_header(file);
    if (metadata.architecture != CheckpointArchitecture::deepseek_v3) {
        throw std::runtime_error("checkpoint is not a DeepSeek-V3 model");
    }
    check_deepseek_v3_compatible(
        metadata.deepseek_v3_config, model.config());
    if (metadata.parameter_elements != model.parameter_layout().elements) {
        throw std::runtime_error("checkpoint parameter count does not match model");
    }
    const std::size_t expected_biases =
        static_cast<std::size_t>(
            model.config().layers - model.config().dense_layers
            + model.config().mtp_depth)
        * model.config().routed_experts;
    if (metadata.routing_bias_elements != expected_biases) {
        throw std::runtime_error(
            "checkpoint routing-bias count does not match model");
    }

    DeepSeekV3TrainingState state;
    state.optimizer.parameters.resize(metadata.parameter_elements);
    state.optimizer.first_moment.resize(metadata.parameter_elements);
    state.optimizer.second_moment.resize(metadata.parameter_elements);
    state.routing_bias.resize(metadata.routing_bias_elements);
    read_tensor(file, state.optimizer.parameters);
    read_tensor(file, state.optimizer.first_moment);
    read_tensor(file, state.optimizer.second_moment);
    read_tensor(file, state.routing_bias);
    if (file.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("checkpoint contains unexpected trailing data");
    }
    model.load_training_state(state);
    return metadata;
}

}  // namespace dscuda

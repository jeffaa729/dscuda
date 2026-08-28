// Trains the first dense dscuda language model either on one fixed diagnostic batch or the prepared TinyStories token stream.
// The executable reports loss, gradient norm, throughput, parameter count, and allocated model memory while reusing the same model path for both modes.

#include "cuda_common.h"
#include "checkpoint.h"
#include "dataset.h"
#include "deepseek_v3_model.h"
#include "model.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string architecture;
    std::string mode = "overfit";
    dscuda::ModelPrecision precision = dscuda::ModelPrecision::fp32;
    dscuda::AttentionImplementation attention =
        dscuda::AttentionImplementation::composed;
    std::string data_directory = "data/tinystories";
    std::string output_directory = "checkpoints/tinystories";
    std::string resume_checkpoint;
    int steps = 0;
    int batch_size = 0;
    int sequence_length = 0;
    int layers = 0;
    int hidden_size = 0;
    int heads = 0;
    int ffn_size = 0;
    int rotary_size = 0;
    int query_rank = 0;
    int kv_rank = 0;
    int nope_size = 0;
    int rope_size = 0;
    int value_size = 0;
    int expert_hidden_size = 0;
    int routed_experts = 0;
    int shared_experts = 0;
    int top_k = 0;
    float route_scale = 0.0F;
    float routing_bias_update_speed = 0.0F;
    float balance_loss_weight = 0.0F;
    int mtp_depth = -1;
    float mtp_loss_weight = -1.0F;
    int dense_layers = -1;
    int dense_ffn_size = 0;
    float learning_rate = 0.0F;
    float maximum_gradient_norm = 1.0F;
    float peak_tflops = 14.56F;
    int log_interval = 10;
    int checkpoint_interval = 100;
    unsigned int seed = 1337;
};

int parse_int(const std::string& value) {
    return std::stoi(value);
}

float parse_float(const std::string& value) {
    return std::stof(value);
}

std::string trim(const std::string& text) {
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

void set_option(
    Options& options,
    const std::string& name,
    const std::string& value) {
    if (name == "mode") {
        options.mode = value;
    } else if (name == "architecture") {
        if (value != "dense" && value != "deepseek_v3") {
            throw std::runtime_error(
                "architecture must be dense or deepseek_v3");
        }
        options.architecture = value;
    } else if (name == "precision") {
        if (value == "fp32") {
            options.precision = dscuda::ModelPrecision::fp32;
        } else if (value == "bf16") {
            options.precision = dscuda::ModelPrecision::bf16;
        } else {
            throw std::runtime_error("precision must be fp32 or bf16");
        }
    } else if (name == "attention") {
        if (value == "composed") {
            options.attention = dscuda::AttentionImplementation::composed;
        } else if (value == "flash2") {
            options.attention = dscuda::AttentionImplementation::flash2;
        } else {
            throw std::runtime_error("attention must be composed or flash2");
        }
    } else if (name == "data-dir") {
        options.data_directory = value;
    } else if (name == "output-dir") {
        options.output_directory = value;
    } else if (name == "resume") {
        options.resume_checkpoint = value;
    } else if (name == "steps") {
        options.steps = parse_int(value);
    } else if (name == "batch") {
        options.batch_size = parse_int(value);
    } else if (name == "seq") {
        options.sequence_length = parse_int(value);
    } else if (name == "layers") {
        options.layers = parse_int(value);
    } else if (name == "hidden") {
        options.hidden_size = parse_int(value);
    } else if (name == "heads") {
        options.heads = parse_int(value);
    } else if (name == "ffn") {
        options.ffn_size = parse_int(value);
    } else if (name == "rotary") {
        options.rotary_size = parse_int(value);
    } else if (name == "query-rank") {
        options.query_rank = parse_int(value);
    } else if (name == "kv-rank") {
        options.kv_rank = parse_int(value);
    } else if (name == "nope-size") {
        options.nope_size = parse_int(value);
    } else if (name == "rope-size") {
        options.rope_size = parse_int(value);
    } else if (name == "value-size") {
        options.value_size = parse_int(value);
    } else if (name == "expert-hidden") {
        options.expert_hidden_size = parse_int(value);
    } else if (name == "routed-experts") {
        options.routed_experts = parse_int(value);
    } else if (name == "shared-experts") {
        options.shared_experts = parse_int(value);
    } else if (name == "top-k") {
        options.top_k = parse_int(value);
    } else if (name == "route-scale") {
        options.route_scale = parse_float(value);
    } else if (name == "routing-bias-speed") {
        options.routing_bias_update_speed = parse_float(value);
    } else if (name == "balance-loss-weight") {
        options.balance_loss_weight = parse_float(value);
    } else if (name == "mtp-depth") {
        options.mtp_depth = parse_int(value);
    } else if (name == "mtp-loss-weight") {
        options.mtp_loss_weight = parse_float(value);
    } else if (name == "dense-layers") {
        options.dense_layers = parse_int(value);
    } else if (name == "dense-ffn") {
        options.dense_ffn_size = parse_int(value);
    } else if (name == "lr") {
        options.learning_rate = parse_float(value);
    } else if (name == "max-grad-norm") {
        options.maximum_gradient_norm = parse_float(value);
    } else if (name == "peak-tflops") {
        options.peak_tflops = parse_float(value);
    } else if (name == "log-every") {
        options.log_interval = parse_int(value);
    } else if (name == "checkpoint-every") {
        options.checkpoint_interval = parse_int(value);
    } else if (name == "seed") {
        options.seed = static_cast<unsigned int>(parse_int(value));
    } else {
        throw std::runtime_error("unknown training option: " + name);
    }
}

void load_config(const std::string& path, Options& options) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open training config: " + path);
    }
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_number)
                + ": expected name = value");
        }
        set_option(
            options,
            trim(line.substr(0, equals)),
            trim(line.substr(equals + 1)));
    }
}

void print_help() {
    std::printf(
        "Usage: train_dscuda [options]\n"
        "  --config PATH                         load name = value settings\n"
        "  --architecture dense|deepseek_v3\n"
        "  --mode overfit|tinystories --precision fp32|bf16\n"
        "  --attention composed|flash2\n"
        "  --data-dir PATH --output-dir PATH --resume PATH\n"
        "  --steps N --batch N --seq N --layers N\n"
        "  --hidden N --heads N --ffn N --rotary N\n"
        "  V3: --query-rank N --kv-rank N --nope-size N --rope-size N\n"
        "      --value-size N --expert-hidden N --routed-experts N\n"
        "      --shared-experts N --top-k N --route-scale VALUE\n"
        "      --routing-bias-speed VALUE\n"
        "      --balance-loss-weight VALUE\n"
        "      --mtp-depth N --mtp-loss-weight VALUE\n"
        "      --dense-layers N --dense-ffn N\n"
        "  --lr VALUE --max-grad-norm VALUE --peak-tflops VALUE\n"
        "  --log-every N --checkpoint-every N --seed N\n"
        "Command-line settings override values loaded from --config.\n");
}

Options parse_options(int argc, char** argv) {
    std::string config_path;
    std::vector<std::pair<std::string, std::string>> overrides;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_help();
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value after " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--config") {
            config_path = value;
        } else if (argument.rfind("--", 0) == 0) {
            overrides.emplace_back(argument.substr(2), value);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }

    Options options;
    if (!config_path.empty()) {
        load_config(config_path, options);
        std::printf("Loaded training config: %s\n", config_path.c_str());
    }
    for (const auto& [name, value] : overrides) {
        set_option(options, name, value);
    }
    if (options.mode != "overfit" && options.mode != "tinystories") {
        throw std::runtime_error("--mode must be overfit or tinystories");
    }
    return options;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::size_t training_sample_start(
    std::uint64_t seed,
    std::uint64_t step,
    std::size_t possible_starts) {
    return static_cast<std::size_t>(
        splitmix64(seed + step) % possible_starts);
}

int use_default(int value, int fallback) {
    return value == 0 ? fallback : value;
}

float use_default(float value, float fallback) {
    return value == 0.0F ? fallback : value;
}

dscuda::ModelConfig make_config(
    const Options& options,
    int vocabulary_size) {
    const bool overfit = options.mode == "overfit";
    return {
        use_default(options.batch_size, overfit ? 2 : 4),
        use_default(options.sequence_length, overfit ? 32 : 256),
        vocabulary_size,
        use_default(options.layers, overfit ? 1 : 4),
        use_default(options.hidden_size, overfit ? 64 : 256),
        use_default(options.heads, 4),
        use_default(options.ffn_size, overfit ? 192 : 768),
        use_default(options.rotary_size, overfit ? 16 : 64),
        1.0e-5F,
        options.attention,
    };
}

dscuda::DeepSeekV3Config make_deepseek_v3_config(
    const Options& options,
    int vocabulary_size) {
    const bool overfit = options.mode == "overfit";
    const int layers = use_default(options.layers, overfit ? 1 : 4);
    const int dense_layers = options.dense_layers < 0
        ? (overfit ? 0 : std::min(3, layers))
        : options.dense_layers;
    return {
        use_default(options.batch_size, overfit ? 1 : 2),
        use_default(options.sequence_length, overfit ? 8 : 128),
        vocabulary_size,
        layers,
        use_default(options.hidden_size, overfit ? 32 : 256),
        use_default(options.heads, overfit ? 2 : 4),
        use_default(options.query_rank, overfit ? 16 : 128),
        use_default(options.kv_rank, overfit ? 16 : 64),
        use_default(options.nope_size, overfit ? 8 : 32),
        use_default(options.rope_size, overfit ? 8 : 32),
        use_default(options.value_size, overfit ? 8 : 32),
        use_default(options.expert_hidden_size, overfit ? 16 : 256),
        use_default(options.routed_experts, overfit ? 4 : 8),
        use_default(options.shared_experts, 1),
        use_default(options.top_k, 2),
        1.0e-5F,
        use_default(options.route_scale, 1.0F),
        use_default(options.routing_bias_update_speed, 1.0e-3F),
        options.balance_loss_weight,
        options.mtp_depth < 0 ? 1 : options.mtp_depth,
        options.mtp_loss_weight < 0.0F ? 0.1F : options.mtp_loss_weight,
        dense_layers,
        use_default(options.dense_ffn_size, overfit ? 64 : 768),
    };
}

double training_flops_per_token(
    const dscuda::ModelConfig& config,
    const dscuda::ModelMemoryReport& memory) {
    return 6.0 * static_cast<double>(memory.parameter_elements)
        + 12.0 * config.layers * config.heads
            * (config.hidden_size / config.heads)
            * config.sequence_length;
}

double training_flops_per_token(
    const dscuda::DeepSeekV3Config& config,
    const dscuda::ModelMemoryReport& memory) {
    (void)memory;
    const double hidden = config.hidden_size;
    const double mla_matrices =
        hidden * config.query_rank
        + static_cast<double>(config.query_rank) * config.heads
            * (config.nope_size + config.rope_size)
        + hidden * (config.kv_rank + config.rope_size)
        + static_cast<double>(config.heads) * config.nope_size
            * config.kv_rank
        + static_cast<double>(config.heads) * config.kv_rank
            * config.value_size
        + static_cast<double>(config.heads) * config.value_size * hidden;
    const double dense_ffn_matrices =
        3.0 * hidden * config.dense_ffn_size;
    const double active_moe_matrices =
        hidden * config.routed_experts
        + 3.0 * hidden * config.expert_hidden_size
            * (config.top_k + config.shared_experts);
    const double vocabulary_head = hidden * config.vocabulary_size;
    const int moe_layers = config.layers - config.dense_layers;
    const double main_active_matrices = vocabulary_head
        + config.layers * mla_matrices
        + config.dense_layers * dense_ffn_matrices
        + moe_layers * active_moe_matrices;
    double flops = 6.0 * main_active_matrices;

    const double attention_width =
        2.0 * config.kv_rank + config.rope_size;
    flops += 3.0 * config.layers * config.heads * attention_width
        * (config.sequence_length + 1);
    for (int depth = 1; depth <= config.mtp_depth; ++depth) {
        const int valid_length = config.sequence_length - depth;
        const double valid_fraction =
            static_cast<double>(valid_length) / config.sequence_length;
        const double mtp_active_matrices =
            2.0 * hidden * hidden + mla_matrices
            + active_moe_matrices + vocabulary_head;
        flops += 6.0 * mtp_active_matrices * valid_fraction;
        flops += 3.0 * config.heads * attention_width
            * valid_length * (valid_length + 1)
            / config.sequence_length;
    }
    return flops;
}

void print_model_summary(
    const dscuda::ModelConfig& config,
    const dscuda::ModelMemoryReport& memory,
    double peak_tflops,
    dscuda::ModelPrecision precision) {
    std::printf(
        "Model: L=%d H=%d heads=%d FFN=%d, B=%d T=%d V=%d, %s, %s attention\n",
        config.layers,
        config.hidden_size,
        config.heads,
        config.ffn_size,
        config.batch_size,
        config.sequence_length,
        config.vocabulary_size,
        precision == dscuda::ModelPrecision::bf16 ? "BF16 mixed" : "FP32",
        config.attention == dscuda::AttentionImplementation::flash2
            ? "Flash2"
            : "composed");
    std::printf(
        "Parameters: %.3f M, allocated model memory: %.1f MiB\n",
        static_cast<double>(memory.parameter_elements) / 1.0e6,
        static_cast<double>(memory.total_bytes) / (1024.0 * 1024.0));

    std::printf(
        "Model FLOPs/token: %.2f MFLOPs, peak used for MFU: %.2f TFLOP/s\n",
        training_flops_per_token(config, memory) / 1.0e6,
        peak_tflops);
}

void print_model_summary(
    const dscuda::DeepSeekV3Config& config,
    const dscuda::ModelMemoryReport& memory,
    double peak_tflops) {
    std::printf(
        "Model: DeepSeek-V3 L=%d H=%d heads=%d, q-rank=%d kv-rank=%d, "
        "experts=%d+%d top-k=%d, dense=%d, MTP=%d, B=%d T=%d V=%d\n",
        config.layers,
        config.hidden_size,
        config.heads,
        config.query_rank,
        config.kv_rank,
        config.routed_experts,
        config.shared_experts,
        config.top_k,
        config.dense_layers,
        config.mtp_depth,
        config.batch_size,
        config.sequence_length,
        config.vocabulary_size);
    std::printf(
        "Parameters: %.3f M, allocated model memory: %.1f MiB\n",
        static_cast<double>(memory.parameter_elements) / 1.0e6,
        static_cast<double>(memory.total_bytes) / (1024.0 * 1024.0));
    std::printf(
        "Model FLOPs/token: %.2f MFLOPs, peak used for MFU: %.2f TFLOP/s\n",
        training_flops_per_token(config, memory) / 1.0e6,
        peak_tflops);
}

struct TrainingPerformance {
    double tokens_per_second;
    double achieved_tflops;
    double mfu_percent;
};

TrainingPerformance training_performance(
    std::uint64_t tokens,
    double seconds,
    double flops_per_token,
    double peak_tflops) {
    const double tokens_per_second = tokens / seconds;
    const double achieved_tflops =
        flops_per_token * tokens_per_second / 1.0e12;
    return {
        tokens_per_second,
        achieved_tflops,
        100.0 * achieved_tflops / peak_tflops,
    };
}

int run_dense_overfit(const Options& options) {
    constexpr int kVocabularySize = 128;
    const dscuda::ModelConfig config = make_config(options, kVocabularySize);
    const int steps = use_default(options.steps, 200);
    const float learning_rate = use_default(options.learning_rate, 3.0e-3F);
    const int rows = config.batch_size * config.sequence_length;

    std::vector<int> stream(rows + 1);
    for (int index = 0; index <= rows; ++index) {
        stream[index] = (index * 37 + 11) % kVocabularySize;
    }
    std::vector<int> inputs(stream.begin(), stream.end() - 1);
    std::vector<int> targets(stream.begin() + 1, stream.end());

    dscuda::DenseGptModel model(config, options.precision);
    model.initialize(options.seed);
    const dscuda::ModelMemoryReport memory = model.memory_report();
    const double flops_per_token =
        training_flops_per_token(config, memory);
    print_model_summary(
        config, memory, options.peak_tflops, options.precision);
    const dscuda::AdamWConfig optimizer{
        learning_rate, 0.9F, 0.95F, 1.0e-8F, 0.0F};

    float initial_loss = 0.0F;
    double training_seconds = 0.0;
    std::uint64_t training_tokens = 0;
    for (int step = 1; step <= steps; ++step) {
        const auto step_start = std::chrono::steady_clock::now();
        const dscuda::TrainStepResult result = model.train_step(
            inputs,
            targets,
            step,
            optimizer,
            options.maximum_gradient_norm);
        training_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - step_start).count();
        training_tokens += rows;
        if (step == 1) {
            initial_loss = result.loss;
        }
        if (step == 1 || step % options.log_interval == 0 || step == steps) {
            const TrainingPerformance performance = training_performance(
                training_tokens,
                training_seconds,
                flops_per_token,
                options.peak_tflops);
            std::printf(
                "step %4d/%d  loss %.6f  grad %.4f  %.0f tokens/s  "
                "%.3f TFLOP/s  %.1f%% MFU\n",
                step,
                steps,
                result.loss,
                result.gradient_norm,
                performance.tokens_per_second,
                performance.achieved_tflops,
                performance.mfu_percent);
            training_seconds = 0.0;
            training_tokens = 0;
        }
    }
    const float final_loss = model.forward(inputs, targets);
    const bool passed = final_loss < 0.10F && final_loss < initial_loss * 0.05F;
    std::printf(
        "Fixed-batch overfit: %.6f -> %.6f  %s\n",
        initial_loss,
        final_loss,
        passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

int run_dense_tinystories(const Options& options) {
    const std::string tokenizer_path =
        options.data_directory + "/tokenizer.bin";
    const std::string train_path = options.data_directory + "/train.bin";
    const std::string validation_path = options.data_directory + "/val.bin";
    const dscuda::Tokenizer tokenizer(tokenizer_path);
    const dscuda::TokenDataset train(train_path);
    const dscuda::TokenDataset validation(validation_path);
    std::uint64_t completed_steps = 0;
    dscuda::ModelConfig config =
        make_config(options, tokenizer.vocabulary_size());
    if (!options.resume_checkpoint.empty()) {
        const dscuda::CheckpointMetadata metadata =
            dscuda::read_checkpoint_metadata(options.resume_checkpoint);
        config = metadata.config;
        config.attention = options.attention;
        completed_steps = metadata.step;
        if (config.vocabulary_size != tokenizer.vocabulary_size()) {
            throw std::runtime_error(
                "checkpoint vocabulary does not match tokenizer");
        }
    }
    const int steps = use_default(options.steps, 1000);
    const float learning_rate = use_default(options.learning_rate, 3.0e-4F);
    const std::size_t rows =
        static_cast<std::size_t>(config.batch_size) * config.sequence_length;
    if (train.token_count() <= rows || validation.token_count() <= rows) {
        throw std::runtime_error("token dataset is smaller than one model batch");
    }

    dscuda::DenseGptModel model(config, options.precision);
    if (options.resume_checkpoint.empty()) {
        model.initialize(options.seed);
    } else {
        dscuda::load_dense_gpt_checkpoint(options.resume_checkpoint, model);
        std::printf(
            "Resumed checkpoint %s at step %llu\n",
            options.resume_checkpoint.c_str(),
            static_cast<unsigned long long>(completed_steps));
    }
    const dscuda::ModelMemoryReport memory = model.memory_report();
    const double flops_per_token =
        training_flops_per_token(config, memory);
    print_model_summary(
        config, memory, options.peak_tflops, options.precision);
    std::printf(
        "Dataset: %zu training tokens, %zu validation tokens\n",
        train.token_count(),
        validation.token_count());
    const dscuda::AdamWConfig optimizer{
        learning_rate, 0.9F, 0.95F, 1.0e-8F, 0.1F};

    std::vector<int> inputs;
    std::vector<int> targets;
    std::vector<int> validation_inputs;
    std::vector<int> validation_targets;
    validation.get_batch(
        0,
        config.batch_size,
        config.sequence_length,
        validation_inputs,
        validation_targets);

    double training_seconds = 0.0;
    std::uint64_t training_tokens = 0;
    for (std::uint64_t step = completed_steps + 1;
         step <= static_cast<std::uint64_t>(steps);
         ++step) {
        train.get_batch(
            training_sample_start(
                options.seed,
                step,
                train.token_count() - rows),
            config.batch_size,
            config.sequence_length,
            inputs,
            targets);
        const auto step_start = std::chrono::steady_clock::now();
        const dscuda::TrainStepResult result = model.train_step(
            inputs,
            targets,
            static_cast<int>(step),
            optimizer,
            options.maximum_gradient_norm);
        training_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - step_start).count();
        training_tokens += rows;

        if (step == completed_steps + 1
            || step % static_cast<std::uint64_t>(options.log_interval) == 0
            || step == static_cast<std::uint64_t>(steps)) {
            const float validation_loss =
                model.forward(validation_inputs, validation_targets);
            const TrainingPerformance performance = training_performance(
                training_tokens,
                training_seconds,
                flops_per_token,
                options.peak_tflops);
            std::printf(
                "step %5d/%d  train %.5f  val %.5f  grad %.3f  "
                "%.0f tokens/s  %.3f TFLOP/s  %.1f%% MFU\n",
                static_cast<int>(step),
                steps,
                result.loss,
                validation_loss,
                result.gradient_norm,
                performance.tokens_per_second,
                performance.achieved_tflops,
                performance.mfu_percent);
            training_seconds = 0.0;
            training_tokens = 0;
        }

        if (options.checkpoint_interval > 0
            && (step % static_cast<std::uint64_t>(
                    options.checkpoint_interval) == 0
                || step == static_cast<std::uint64_t>(steps))) {
            const std::string checkpoint_directory =
                dscuda::checkpoint_step_directory(
                    options.output_directory, step);
            dscuda::save_dense_gpt_checkpoint(
                checkpoint_directory, model, step);
            std::printf(
                "Saved checkpoint: %s\n",
                checkpoint_directory.c_str());
        }
    }
    return 0;
}

int run_deepseek_v3_overfit(const Options& options) {
    constexpr int kVocabularySize = 64;
    const dscuda::DeepSeekV3Config config =
        make_deepseek_v3_config(options, kVocabularySize);
    const int steps = use_default(options.steps, 300);
    const float learning_rate = use_default(options.learning_rate, 3.0e-3F);
    const int rows = config.batch_size * config.sequence_length;

    std::vector<int> stream(rows + 1);
    for (int index = 0; index <= rows; ++index) {
        stream[index] = (index * 17 + 5) % kVocabularySize;
    }
    std::vector<int> inputs(stream.begin(), stream.end() - 1);
    std::vector<int> targets(stream.begin() + 1, stream.end());

    dscuda::DeepSeekV3Model model(config);
    model.initialize(options.seed);
    const dscuda::ModelMemoryReport memory = model.memory_report();
    const double flops_per_token = training_flops_per_token(config, memory);
    print_model_summary(config, memory, options.peak_tflops);
    std::printf("Precision: FP32 master state with BF16 compressed MLA operands\n");
    const dscuda::AdamWConfig optimizer{
        learning_rate, 0.9F, 0.95F, 1.0e-8F, 0.0F};

    float initial_loss = 0.0F;
    double training_seconds = 0.0;
    std::uint64_t training_tokens = 0;
    for (int step = 1; step <= steps; ++step) {
        const auto step_start = std::chrono::steady_clock::now();
        const dscuda::TrainStepResult result = model.train_step(
            inputs,
            targets,
            step,
            optimizer,
            options.maximum_gradient_norm);
        training_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - step_start).count();
        training_tokens += rows;
        if (step == 1) {
            initial_loss = result.loss;
        }
        if (step == 1 || step % options.log_interval == 0 || step == steps) {
            const TrainingPerformance performance = training_performance(
                training_tokens,
                training_seconds,
                flops_per_token,
                options.peak_tflops);
            std::printf(
                "step %4d/%d  loss %.6f  grad %.4f  %.0f tokens/s  "
                "%.3f TFLOP/s  %.1f%% MFU\n",
                step,
                steps,
                result.loss,
                result.gradient_norm,
                performance.tokens_per_second,
                performance.achieved_tflops,
                performance.mfu_percent);
            training_seconds = 0.0;
            training_tokens = 0;
        }
    }
    const float final_loss = model.forward(inputs, targets);
    const bool passed = final_loss < 0.10F && final_loss < initial_loss * 0.05F;
    std::printf(
        "DeepSeek-V3 fixed-batch overfit: %.6f -> %.6f  %s\n",
        initial_loss,
        final_loss,
        passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

int run_deepseek_v3_tinystories(const Options& options) {
    const std::string tokenizer_path =
        options.data_directory + "/tokenizer.bin";
    const std::string train_path = options.data_directory + "/train.bin";
    const std::string validation_path = options.data_directory + "/val.bin";
    const dscuda::Tokenizer tokenizer(tokenizer_path);
    const dscuda::TokenDataset train(train_path);
    const dscuda::TokenDataset validation(validation_path);
    std::uint64_t completed_steps = 0;
    dscuda::DeepSeekV3Config config =
        make_deepseek_v3_config(options, tokenizer.vocabulary_size());
    if (!options.resume_checkpoint.empty()) {
        const dscuda::CheckpointMetadata metadata =
            dscuda::read_checkpoint_metadata(options.resume_checkpoint);
        if (metadata.architecture
            != dscuda::CheckpointArchitecture::deepseek_v3) {
            throw std::runtime_error(
                "resume checkpoint is not a DeepSeek-V3 model");
        }
        config = metadata.deepseek_v3_config;
        completed_steps = metadata.step;
        if (config.vocabulary_size != tokenizer.vocabulary_size()) {
            throw std::runtime_error(
                "checkpoint vocabulary does not match tokenizer");
        }
    }
    const int steps = use_default(options.steps, 1000);
    const float learning_rate = use_default(options.learning_rate, 3.0e-4F);
    const std::size_t rows =
        static_cast<std::size_t>(config.batch_size) * config.sequence_length;
    if (train.token_count() <= rows || validation.token_count() <= rows) {
        throw std::runtime_error("token dataset is smaller than one model batch");
    }

    dscuda::DeepSeekV3Model model(config);
    if (options.resume_checkpoint.empty()) {
        model.initialize(options.seed);
    } else {
        dscuda::load_deepseek_v3_checkpoint(options.resume_checkpoint, model);
        std::printf(
            "Resumed V3 checkpoint %s at step %llu\n",
            options.resume_checkpoint.c_str(),
            static_cast<unsigned long long>(completed_steps));
    }
    const dscuda::ModelMemoryReport memory = model.memory_report();
    const double flops_per_token = training_flops_per_token(config, memory);
    print_model_summary(config, memory, options.peak_tflops);
    std::printf("Precision: FP32 master state with BF16 compressed MLA operands\n");
    std::printf(
        "Dataset: %zu training tokens, %zu validation tokens\n",
        train.token_count(),
        validation.token_count());
    const dscuda::AdamWConfig optimizer{
        learning_rate, 0.9F, 0.95F, 1.0e-8F, 0.1F};

    std::vector<int> inputs;
    std::vector<int> targets;
    std::vector<int> validation_inputs;
    std::vector<int> validation_targets;
    validation.get_batch(
        0,
        config.batch_size,
        config.sequence_length,
        validation_inputs,
        validation_targets);

    double training_seconds = 0.0;
    std::uint64_t training_tokens = 0;
    for (std::uint64_t step = completed_steps + 1;
         step <= static_cast<std::uint64_t>(steps);
         ++step) {
        train.get_batch(
            training_sample_start(
                options.seed,
                step,
                train.token_count() - rows),
            config.batch_size,
            config.sequence_length,
            inputs,
            targets);
        const auto step_start = std::chrono::steady_clock::now();
        const dscuda::TrainStepResult result = model.train_step(
            inputs,
            targets,
            static_cast<int>(step),
            optimizer,
            options.maximum_gradient_norm);
        training_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - step_start).count();
        training_tokens += rows;

        if (step == completed_steps + 1
            || step % static_cast<std::uint64_t>(options.log_interval) == 0
            || step == static_cast<std::uint64_t>(steps)) {
            const float validation_loss =
                model.forward(validation_inputs, validation_targets);
            const TrainingPerformance performance = training_performance(
                training_tokens,
                training_seconds,
                flops_per_token,
                options.peak_tflops);
            std::printf(
                "step %5d/%d  train %.5f  val %.5f  grad %.3f  "
                "%.0f tokens/s  %.3f TFLOP/s  %.1f%% MFU\n",
                static_cast<int>(step),
                steps,
                result.loss,
                validation_loss,
                result.gradient_norm,
                performance.tokens_per_second,
                performance.achieved_tflops,
                performance.mfu_percent);
            training_seconds = 0.0;
            training_tokens = 0;
        }

        if (options.checkpoint_interval > 0
            && (step % static_cast<std::uint64_t>(
                    options.checkpoint_interval) == 0
                || step == static_cast<std::uint64_t>(steps))) {
            const std::string checkpoint_directory =
                dscuda::checkpoint_step_directory(
                    options.output_directory, step);
            dscuda::save_deepseek_v3_checkpoint(
                checkpoint_directory, model, step);
            std::printf(
                "Saved V3 checkpoint: %s\n",
                checkpoint_directory.c_str());
        }
    }
    return 0;
}

bool use_deepseek_v3(const Options& options) {
    if (!options.resume_checkpoint.empty()) {
        const dscuda::CheckpointMetadata metadata =
            dscuda::read_checkpoint_metadata(options.resume_checkpoint);
        const bool checkpoint_is_v3 = metadata.architecture
            == dscuda::CheckpointArchitecture::deepseek_v3;
        if (!options.architecture.empty()
            && (options.architecture == "deepseek_v3") != checkpoint_is_v3) {
            throw std::runtime_error(
                "--architecture does not match the resume checkpoint");
        }
        return checkpoint_is_v3;
    }
    return options.architecture == "deepseek_v3";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        dscuda::print_device_summary();
        const bool v3 = use_deepseek_v3(options);
        if (options.mode == "overfit") {
            return v3
                ? run_deepseek_v3_overfit(options)
                : run_dense_overfit(options);
        }
        return v3
            ? run_deepseek_v3_tinystories(options)
            : run_dense_tinystories(options);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Training failed: %s\n", error.what());
        return 1;
    }
}

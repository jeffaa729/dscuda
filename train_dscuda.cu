// Trains the first dense dscuda language model either on one fixed diagnostic batch or the prepared TinyStories token stream.
// The executable reports loss, gradient norm, throughput, parameter count, and allocated model memory while reusing the same model path for both modes.

#include "cuda_common.h"
#include "dataset.h"
#include "model.h"
#include "tokenizer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string mode = "overfit";
    std::string data_directory = "data/tinystories";
    int steps = 0;
    int batch_size = 0;
    int sequence_length = 0;
    int layers = 0;
    int hidden_size = 0;
    int heads = 0;
    int ffn_size = 0;
    int rotary_size = 0;
    float learning_rate = 0.0F;
    float maximum_gradient_norm = 1.0F;
    int log_interval = 10;
    unsigned int seed = 1337;
};

int parse_int(const char* value) {
    return std::stoi(value);
}

float parse_float(const char* value) {
    return std::stof(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> const char* {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[++index];
        };

        if (argument == "--mode") {
            options.mode = value();
        } else if (argument == "--data-dir") {
            options.data_directory = value();
        } else if (argument == "--steps") {
            options.steps = parse_int(value());
        } else if (argument == "--batch") {
            options.batch_size = parse_int(value());
        } else if (argument == "--seq") {
            options.sequence_length = parse_int(value());
        } else if (argument == "--layers") {
            options.layers = parse_int(value());
        } else if (argument == "--hidden") {
            options.hidden_size = parse_int(value());
        } else if (argument == "--heads") {
            options.heads = parse_int(value());
        } else if (argument == "--ffn") {
            options.ffn_size = parse_int(value());
        } else if (argument == "--rotary") {
            options.rotary_size = parse_int(value());
        } else if (argument == "--lr") {
            options.learning_rate = parse_float(value());
        } else if (argument == "--max-grad-norm") {
            options.maximum_gradient_norm = parse_float(value());
        } else if (argument == "--log-every") {
            options.log_interval = parse_int(value());
        } else if (argument == "--seed") {
            options.seed = static_cast<unsigned int>(parse_int(value()));
        } else if (argument == "--help") {
            std::printf(
                "Usage: train_dscuda [options]\n"
                "  --mode overfit|tinystories\n"
                "  --data-dir PATH\n"
                "  --steps N --batch N --seq N --layers N\n"
                "  --hidden N --heads N --ffn N --rotary N\n"
                "  --lr VALUE --max-grad-norm VALUE --log-every N --seed N\n");
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.mode != "overfit" && options.mode != "tinystories") {
        throw std::runtime_error("--mode must be overfit or tinystories");
    }
    return options;
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
    };
}

void print_model_summary(
    const dscuda::ModelConfig& config,
    const dscuda::ModelMemoryReport& memory) {
    std::printf(
        "Model: L=%d H=%d heads=%d FFN=%d, B=%d T=%d V=%d\n",
        config.layers,
        config.hidden_size,
        config.heads,
        config.ffn_size,
        config.batch_size,
        config.sequence_length,
        config.vocabulary_size);
    std::printf(
        "Parameters: %.3f M, allocated model memory: %.1f MiB\n",
        static_cast<double>(memory.parameter_elements) / 1.0e6,
        static_cast<double>(memory.total_bytes) / (1024.0 * 1024.0));
}

int run_overfit(const Options& options) {
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

    dscuda::DenseGptModel model(config);
    model.initialize(options.seed);
    print_model_summary(config, model.memory_report());
    const dscuda::AdamWConfig optimizer{
        learning_rate, 0.9F, 0.95F, 1.0e-8F, 0.0F};

    float initial_loss = 0.0F;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 1; step <= steps; ++step) {
        const dscuda::TrainStepResult result = model.train_step(
            inputs,
            targets,
            step,
            optimizer,
            options.maximum_gradient_norm);
        if (step == 1) {
            initial_loss = result.loss;
        }
        if (step == 1 || step % options.log_interval == 0 || step == steps) {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            std::printf(
                "step %4d/%d  loss %.6f  grad %.4f  %.0f tokens/s\n",
                step,
                steps,
                result.loss,
                result.gradient_norm,
                static_cast<double>(step) * rows / seconds);
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

int run_tinystories(const Options& options) {
    const std::string tokenizer_path =
        options.data_directory + "/tokenizer.bin";
    const std::string train_path = options.data_directory + "/train.bin";
    const std::string validation_path = options.data_directory + "/val.bin";
    const dscuda::Tokenizer tokenizer(tokenizer_path);
    const dscuda::TokenDataset train(train_path);
    const dscuda::TokenDataset validation(validation_path);
    const dscuda::ModelConfig config =
        make_config(options, tokenizer.vocabulary_size());
    const int steps = use_default(options.steps, 1000);
    const float learning_rate = use_default(options.learning_rate, 3.0e-4F);
    const std::size_t rows =
        static_cast<std::size_t>(config.batch_size) * config.sequence_length;
    if (train.token_count() <= rows || validation.token_count() <= rows) {
        throw std::runtime_error("token dataset is smaller than one model batch");
    }

    dscuda::DenseGptModel model(config);
    model.initialize(options.seed);
    print_model_summary(config, model.memory_report());
    std::printf(
        "Dataset: %zu training tokens, %zu validation tokens\n",
        train.token_count(),
        validation.token_count());
    const dscuda::AdamWConfig optimizer{
        learning_rate, 0.9F, 0.95F, 1.0e-8F, 0.1F};

    std::mt19937_64 generator(options.seed);
    std::uniform_int_distribution<std::size_t> sample_start(
        0, train.token_count() - rows - 1);
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

    const auto start = std::chrono::steady_clock::now();
    for (int step = 1; step <= steps; ++step) {
        train.get_batch(
            sample_start(generator),
            config.batch_size,
            config.sequence_length,
            inputs,
            targets);
        const dscuda::TrainStepResult result = model.train_step(
            inputs,
            targets,
            step,
            optimizer,
            options.maximum_gradient_norm);

        if (step == 1 || step % options.log_interval == 0 || step == steps) {
            const float validation_loss =
                model.forward(validation_inputs, validation_targets);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            std::printf(
                "step %5d/%d  train %.5f  val %.5f  grad %.3f  "
                "%.0f tokens/s\n",
                step,
                steps,
                result.loss,
                validation_loss,
                result.gradient_norm,
                static_cast<double>(step) * rows / seconds);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        dscuda::print_device_summary();
        return options.mode == "overfit"
            ? run_overfit(options)
            : run_tinystories(options);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Training failed: %s\n", error.what());
        return 1;
    }
}

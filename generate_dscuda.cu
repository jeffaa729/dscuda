// Loads a trained dense GPT checkpoint, encodes a prompt, and performs no-cache autoregressive generation through the shared sampler.
// This correctness-first path recomputes the fixed context each token and can later be replaced by architecture-specific cache backends.

#include "checkpoint.h"
#include "cuda_common.h"
#include "generation.h"
#include "model.h"
#include "tokenizer.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string checkpoint;
    std::string tokenizer = "data/tinystories/tokenizer.bin";
    std::string prompt = "Once upon a time";
    int tokens = 80;
    float temperature = 0.8F;
    int top_k = 40;
    std::uint64_t seed = 1337;
};

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
        if (argument == "--checkpoint") {
            options.checkpoint = value();
        } else if (argument == "--tokenizer") {
            options.tokenizer = value();
        } else if (argument == "--prompt") {
            options.prompt = value();
        } else if (argument == "--tokens") {
            options.tokens = std::stoi(value());
        } else if (argument == "--temperature") {
            options.temperature = std::stof(value());
        } else if (argument == "--top-k") {
            options.top_k = std::stoi(value());
        } else if (argument == "--seed") {
            options.seed = std::stoull(value());
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.checkpoint.empty()) {
        throw std::runtime_error("--checkpoint is required");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        dscuda::print_device_summary();
        const dscuda::CheckpointMetadata metadata =
            dscuda::read_checkpoint_metadata(options.checkpoint);
        dscuda::ModelConfig generation_config = metadata.config;
        generation_config.batch_size = 1;
        dscuda::DenseGptModel model(generation_config);
        dscuda::load_dense_gpt_checkpoint(options.checkpoint, model);
        const dscuda::Tokenizer tokenizer(options.tokenizer);
        if (tokenizer.vocabulary_size() != model.vocabulary_size()) {
            throw std::runtime_error(
                "tokenizer vocabulary does not match checkpoint");
        }

        std::vector<int> prompt = tokenizer.encode(options.prompt);
        if (prompt.empty()) {
            prompt.push_back(tokenizer.eos_id());
        }
        const dscuda::GenerationConfig config{
            options.tokens,
            options.temperature,
            options.top_k,
            options.seed,
        };
        const std::vector<int> generated = dscuda::generate_tokens(
            model, prompt, tokenizer.eos_id(), config);

        std::printf(
            "Checkpoint step %llu, prompt tokens %zu, generated tokens %zu\n",
            static_cast<unsigned long long>(metadata.step),
            prompt.size(),
            generated.size() - prompt.size());
        std::printf("---\n%s\n---\n", tokenizer.decode(generated).c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Generation failed: %s\n", error.what());
        return 1;
    }
}

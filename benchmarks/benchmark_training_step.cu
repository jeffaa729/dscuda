// Profiles one complete BF16 dense-GPT training step after a warm-up step.
// The captured region includes forward, backward, gradient clipping, AdamW, and BF16 weight refresh for either composed or fused attention.

#include "cuda_common.h"
#include "model.h"
#include "optimizer.h"

#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        const char* attention_name = argc > 1 ? argv[1] : "flash2";
        dscuda::AttentionImplementation attention;
        if (std::strcmp(attention_name, "composed") == 0) {
            attention = dscuda::AttentionImplementation::composed;
        } else if (std::strcmp(attention_name, "flash2") == 0) {
            attention = dscuda::AttentionImplementation::flash2;
        } else {
            throw std::runtime_error("attention must be composed or flash2");
        }

        const int batch_size = argc > 2 ? std::atoi(argv[2]) : 4;
        const int sequence_length = argc > 3 ? std::atoi(argv[3]) : 256;
        const int layers = argc > 4 ? std::atoi(argv[4]) : 4;
        const int hidden_size = argc > 5 ? std::atoi(argv[5]) : 256;
        const int heads = argc > 6 ? std::atoi(argv[6]) : 4;
        const int ffn_size = argc > 7 ? std::atoi(argv[7]) : 768;
        const int vocabulary_size = argc > 8 ? std::atoi(argv[8]) : 4096;
        const int rotary_size = hidden_size / heads;
        const dscuda::ModelConfig config{
            batch_size,
            sequence_length,
            vocabulary_size,
            layers,
            hidden_size,
            heads,
            ffn_size,
            rotary_size,
            1.0e-5F,
            attention};
        const dscuda::AdamWConfig optimizer{
            3.0e-4F, 0.9F, 0.95F, 1.0e-8F, 0.1F};

        const int tokens = batch_size * sequence_length;
        std::vector<int> inputs(tokens);
        std::vector<int> targets(tokens);
        for (int index = 0; index < tokens; ++index) {
            inputs[index] = (index * 17 + 3) % vocabulary_size;
            targets[index] = (index * 29 + 5) % vocabulary_size;
        }

        dscuda::DenseGptModel model(config, dscuda::ModelPrecision::bf16);
        model.initialize(1337);
        model.train_step(inputs, targets, 1, optimizer, 1.0F);

        std::printf(
            "Training-step workload: attention=%s B=%d T=%d L=%d H=%d heads=%d FFN=%d V=%d\n",
            attention_name,
            batch_size,
            sequence_length,
            layers,
            hidden_size,
            heads,
            ffn_size,
            vocabulary_size);
        CUDA_CHECK(cudaProfilerStart());
        const dscuda::TrainStepResult result =
            model.train_step(inputs, targets, 2, optimizer, 1.0F);
        CUDA_CHECK(cudaProfilerStop());
        std::printf(
            "Profiled step: loss=%.6f gradient_norm=%.6f\n",
            result.loss,
            result.gradient_norm);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "Training-step benchmark failed: %s\n", error.what());
        return 1;
    }
}

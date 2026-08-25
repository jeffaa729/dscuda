// Assembles embeddings, repeated dense transformer blocks, final RMSNorm, a tied vocabulary head, loss, backward, clipping, and AdamW.
// Parameters and gradients share one flat layout so the optimizer and global-norm kernels can process the complete model in a single launch sequence.

#include "model.h"

#include "cross_entropy.h"
#include "cuda_common.h"
#include "embedding.h"
#include "global_norm.h"
#include "matmul.h"
#include "rmsnorm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace dscuda {
namespace {

std::size_t align_four(std::size_t value) {
    return (value + 3) & ~std::size_t{3};
}

void validate_config(const ModelConfig& config) {
    if (config.batch_size <= 0 || config.sequence_length <= 0
        || config.vocabulary_size <= 0 || config.layers <= 0
        || config.hidden_size <= 0 || config.heads <= 0
        || config.ffn_size <= 0 || config.hidden_size % config.heads != 0
        || config.hidden_size % 4 != 0 || config.ffn_size % 4 != 0
        || config.rotary_size <= 0
        || config.rotary_size > config.hidden_size / config.heads
        || config.rotary_size % 2 != 0) {
        throw std::runtime_error("invalid dense GPT model configuration");
    }
}

template <typename T>
T* allocate(std::size_t elements) {
    return static_cast<T*>(device_malloc(elements * sizeof(T)));
}

void fill_normal(
    std::vector<float>& values,
    std::size_t offset,
    std::size_t elements,
    float standard_deviation,
    std::mt19937_64& generator) {
    std::normal_distribution<float> distribution(0.0F, standard_deviation);
    for (std::size_t index = 0; index < elements; ++index) {
        values[offset + index] = distribution(generator);
    }
}

}  // namespace

ModelParameterLayout make_model_parameter_layout(const ModelConfig& config) {
    validate_config(config);
    ModelParameterLayout layout{};
    std::size_t offset = 0;
    auto reserve = [&](std::size_t elements) {
        offset = align_four(offset);
        const std::size_t result = offset;
        offset += elements;
        return result;
    };

    layout.token_embedding = reserve(
        static_cast<std::size_t>(config.vocabulary_size) * config.hidden_size);
    layout.blocks.resize(config.layers);
    for (ModelBlockOffsets& block : layout.blocks) {
        block.attention_norm = reserve(config.hidden_size);
        block.query = reserve(
            static_cast<std::size_t>(config.hidden_size) * config.hidden_size);
        block.key = reserve(
            static_cast<std::size_t>(config.hidden_size) * config.hidden_size);
        block.value = reserve(
            static_cast<std::size_t>(config.hidden_size) * config.hidden_size);
        block.output = reserve(
            static_cast<std::size_t>(config.hidden_size) * config.hidden_size);
        block.ffn_norm = reserve(config.hidden_size);
        block.gate = reserve(
            static_cast<std::size_t>(config.hidden_size) * config.ffn_size);
        block.up = reserve(
            static_cast<std::size_t>(config.hidden_size) * config.ffn_size);
        block.down = reserve(
            static_cast<std::size_t>(config.ffn_size) * config.hidden_size);
    }
    layout.final_norm = reserve(config.hidden_size);
    layout.elements = align_four(offset);
    return layout;
}

TransformerBlockParameters model_block_parameters(
    const float* parameters,
    const ModelParameterLayout& layout,
    int layer) {
    const ModelBlockOffsets& block = layout.blocks[layer];
    return {
        parameters + block.attention_norm,
        parameters + block.query,
        parameters + block.key,
        parameters + block.value,
        parameters + block.output,
        parameters + block.ffn_norm,
        parameters + block.gate,
        parameters + block.up,
        parameters + block.down,
    };
}

TransformerBlockGradients model_block_gradients(
    float* gradients,
    const ModelParameterLayout& layout,
    int layer) {
    const ModelBlockOffsets& block = layout.blocks[layer];
    return {
        gradients + block.attention_norm,
        gradients + block.query,
        gradients + block.key,
        gradients + block.value,
        gradients + block.output,
        gradients + block.ffn_norm,
        gradients + block.gate,
        gradients + block.up,
        gradients + block.down,
    };
}

struct DenseGptModel::Implementation {
    explicit Implementation(const ModelConfig& model_config)
        : config(model_config), layout(make_model_parameter_layout(config)) {
        rows = config.batch_size * config.sequence_length;
        activation_elements =
            static_cast<std::size_t>(rows) * config.hidden_size;
        logits_elements =
            static_cast<std::size_t>(rows) * config.vocabulary_size;
        block_config = {
            config.batch_size,
            config.sequence_length,
            config.hidden_size,
            config.heads,
            config.hidden_size / config.heads,
            config.ffn_size,
            config.rotary_size,
            config.rms_epsilon,
            1.0F / std::sqrt(
                static_cast<float>(config.hidden_size / config.heads)),
        };
        block_activation_elements =
            transformer_block_activation_elements(block_config);
        block_workspace_elements =
            transformer_block_backward_workspace_elements(block_config);
        norm_workspace_elements = global_norm_workspace_elements(
            static_cast<int>(layout.elements));

        parameters = allocate<float>(layout.elements);
        gradients = allocate<float>(layout.elements);
        first_moment = allocate<float>(layout.elements);
        second_moment = allocate<float>(layout.elements);
        input_tokens = allocate<int>(rows);
        target_tokens = allocate<int>(rows);

        const std::size_t frequencies =
            static_cast<std::size_t>(config.sequence_length)
            * config.rotary_size / 2;
        cosine = allocate<float>(frequencies);
        sine = allocate<float>(frequencies);
        hidden_states = allocate<float>(
            static_cast<std::size_t>(config.layers + 1) * activation_elements);
        block_activations = allocate<float>(
            static_cast<std::size_t>(config.layers) * block_activation_elements);
        final_norm = allocate<float>(activation_elements);
        final_inverse_rms = allocate<float>(rows);
        logits = allocate<float>(logits_elements);
        logsumexp = allocate<float>(rows);
        mean_loss = allocate<float>(1);

        logits_gradient = allocate<float>(logits_elements);
        activation_gradient_a = allocate<float>(activation_elements);
        activation_gradient_b = allocate<float>(activation_elements);
        block_workspace = allocate<float>(block_workspace_elements);
        gradient_norm = allocate<float>(1);
        norm_workspace = allocate<float>(norm_workspace_elements);

        std::vector<float> host_cosine(frequencies);
        std::vector<float> host_sine(frequencies);
        const int pairs = config.rotary_size / 2;
        for (int position = 0; position < config.sequence_length; ++position) {
            for (int pair = 0; pair < pairs; ++pair) {
                const float inverse_frequency = std::pow(
                    10000.0F,
                    -2.0F * static_cast<float>(pair) / config.rotary_size);
                const float angle = position * inverse_frequency;
                const int index = position * pairs + pair;
                host_cosine[index] = std::cos(angle);
                host_sine[index] = std::sin(angle);
            }
        }
        CUDA_CHECK(cudaMemcpy(
            cosine,
            host_cosine.data(),
            frequencies * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            sine,
            host_sine.data(),
            frequencies * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    ~Implementation() {
        device_free(norm_workspace);
        device_free(gradient_norm);
        device_free(block_workspace);
        device_free(activation_gradient_b);
        device_free(activation_gradient_a);
        device_free(logits_gradient);
        device_free(mean_loss);
        device_free(logsumexp);
        device_free(logits);
        device_free(final_inverse_rms);
        device_free(final_norm);
        device_free(block_activations);
        device_free(hidden_states);
        device_free(sine);
        device_free(cosine);
        device_free(target_tokens);
        device_free(input_tokens);
        device_free(second_moment);
        device_free(first_moment);
        device_free(gradients);
        device_free(parameters);
    }

    ModelConfig config;
    ModelParameterLayout layout;
    TransformerBlockConfig block_config{};
    int rows = 0;
    std::size_t activation_elements = 0;
    std::size_t logits_elements = 0;
    std::size_t block_activation_elements = 0;
    std::size_t block_workspace_elements = 0;
    std::size_t norm_workspace_elements = 0;

    float* parameters = nullptr;
    float* gradients = nullptr;
    float* first_moment = nullptr;
    float* second_moment = nullptr;
    int* input_tokens = nullptr;
    int* target_tokens = nullptr;
    float* cosine = nullptr;
    float* sine = nullptr;
    float* hidden_states = nullptr;
    float* block_activations = nullptr;
    float* final_norm = nullptr;
    float* final_inverse_rms = nullptr;
    float* logits = nullptr;
    float* logsumexp = nullptr;
    float* mean_loss = nullptr;
    float* logits_gradient = nullptr;
    float* activation_gradient_a = nullptr;
    float* activation_gradient_b = nullptr;
    float* block_workspace = nullptr;
    float* gradient_norm = nullptr;
    float* norm_workspace = nullptr;
    bool forward_ready = false;
};

DenseGptModel::DenseGptModel(const ModelConfig& config)
    : implementation_(std::make_unique<Implementation>(config)) {}

DenseGptModel::~DenseGptModel() = default;

void DenseGptModel::initialize(std::uint64_t seed) {
    Implementation& model = *implementation_;
    std::vector<float> host_parameters(model.layout.elements, 0.0F);
    std::mt19937_64 generator(seed);
    const int hidden = model.config.hidden_size;
    const int ffn = model.config.ffn_size;
    const float residual_scale =
        0.02F / std::sqrt(2.0F * model.config.layers);

    fill_normal(
        host_parameters,
        model.layout.token_embedding,
        static_cast<std::size_t>(model.config.vocabulary_size) * hidden,
        0.02F,
        generator);
    for (int layer = 0; layer < model.config.layers; ++layer) {
        const ModelBlockOffsets& block = model.layout.blocks[layer];
        std::fill_n(
            host_parameters.begin() + block.attention_norm,
            hidden,
            1.0F);
        std::fill_n(
            host_parameters.begin() + block.ffn_norm,
            hidden,
            1.0F);
        fill_normal(host_parameters, block.query, hidden * hidden, 0.02F, generator);
        fill_normal(host_parameters, block.key, hidden * hidden, 0.02F, generator);
        fill_normal(host_parameters, block.value, hidden * hidden, 0.02F, generator);
        fill_normal(
            host_parameters,
            block.output,
            hidden * hidden,
            residual_scale,
            generator);
        fill_normal(host_parameters, block.gate, hidden * ffn, 0.02F, generator);
        fill_normal(host_parameters, block.up, hidden * ffn, 0.02F, generator);
        fill_normal(
            host_parameters,
            block.down,
            ffn * hidden,
            residual_scale,
            generator);
    }
    std::fill_n(
        host_parameters.begin() + model.layout.final_norm,
        hidden,
        1.0F);
    load_parameters(host_parameters);
    CUDA_CHECK(cudaMemset(
        model.first_moment, 0, model.layout.elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(
        model.second_moment, 0, model.layout.elements * sizeof(float)));
    zero_gradients();
}

void DenseGptModel::load_parameters(const std::vector<float>& parameters) {
    Implementation& model = *implementation_;
    if (parameters.size() != model.layout.elements) {
        throw std::runtime_error("parameter vector has the wrong size");
    }
    CUDA_CHECK(cudaMemcpy(
        model.parameters,
        parameters.data(),
        parameters.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    model.forward_ready = false;
}

float DenseGptModel::forward(
    const std::vector<int>& input_tokens,
    const std::vector<int>& target_tokens) {
    Implementation& model = *implementation_;
    if (input_tokens.size() != static_cast<std::size_t>(model.rows)
        || target_tokens.size() != static_cast<std::size_t>(model.rows)) {
        throw std::runtime_error("training batch has the wrong number of tokens");
    }
    CUDA_CHECK(cudaMemcpy(
        model.input_tokens,
        input_tokens.data(),
        model.rows * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        model.target_tokens,
        target_tokens.data(),
        model.rows * sizeof(int),
        cudaMemcpyHostToDevice));

    embedding_forward_cuda(
        model.hidden_states,
        model.input_tokens,
        model.parameters + model.layout.token_embedding,
        model.rows,
        model.config.hidden_size);
    for (int layer = 0; layer < model.config.layers; ++layer) {
        const float* input =
            model.hidden_states + static_cast<std::size_t>(layer)
                * model.activation_elements;
        float* output =
            model.hidden_states + static_cast<std::size_t>(layer + 1)
                * model.activation_elements;
        float* activations =
            model.block_activations + static_cast<std::size_t>(layer)
                * model.block_activation_elements;
        transformer_block_forward_cuda(
            output,
            input,
            model_block_parameters(model.parameters, model.layout, layer),
            model.cosine,
            model.sine,
            activations,
            model.block_config);
    }

    const float* final_input =
        model.hidden_states + static_cast<std::size_t>(model.config.layers)
            * model.activation_elements;
    rmsnorm_forward_cuda(
        model.final_norm,
        model.final_inverse_rms,
        final_input,
        model.parameters + model.layout.final_norm,
        model.rows,
        model.config.hidden_size,
        model.config.rms_epsilon);
    matmul_fp32_strided_batched_cuda(
        model.logits,
        model.final_norm,
        model.parameters + model.layout.token_embedding,
        model.rows,
        model.config.vocabulary_size,
        model.config.hidden_size,
        1,
        0,
        0,
        0,
        false,
        true,
        false);
    cross_entropy_forward_cuda(
        model.mean_loss,
        model.logsumexp,
        model.logits,
        model.target_tokens,
        model.rows,
        model.config.vocabulary_size);

    float loss = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &loss, model.mean_loss, sizeof(float), cudaMemcpyDeviceToHost));
    model.forward_ready = true;
    return loss;
}

void DenseGptModel::zero_gradients() {
    Implementation& model = *implementation_;
    CUDA_CHECK(cudaMemset(
        model.gradients, 0, model.layout.elements * sizeof(float)));
}

void DenseGptModel::backward() {
    Implementation& model = *implementation_;
    if (!model.forward_ready) {
        throw std::runtime_error("model backward requires a matching forward pass");
    }

    cross_entropy_backward_cuda(
        model.logits_gradient,
        model.logits,
        model.logsumexp,
        model.target_tokens,
        model.rows,
        model.config.vocabulary_size);
    matmul_fp32_forward_cuda(
        model.activation_gradient_a,
        model.logits_gradient,
        model.parameters + model.layout.token_embedding,
        model.rows,
        model.config.hidden_size,
        model.config.vocabulary_size);
    matmul_fp32_strided_batched_cuda(
        model.gradients + model.layout.token_embedding,
        model.logits_gradient,
        model.final_norm,
        model.config.vocabulary_size,
        model.config.hidden_size,
        model.rows,
        1,
        0,
        0,
        0,
        true,
        false,
        true);

    CUDA_CHECK(cudaMemset(
        model.activation_gradient_b,
        0,
        model.activation_elements * sizeof(float)));
    const float* final_input =
        model.hidden_states + static_cast<std::size_t>(model.config.layers)
            * model.activation_elements;
    rmsnorm_backward_cuda(
        model.activation_gradient_b,
        model.gradients + model.layout.final_norm,
        model.activation_gradient_a,
        final_input,
        model.parameters + model.layout.final_norm,
        model.final_inverse_rms,
        model.rows,
        model.config.hidden_size);

    float* output_gradient = model.activation_gradient_b;
    float* input_gradient = model.activation_gradient_a;
    for (int layer = model.config.layers - 1; layer >= 0; --layer) {
        CUDA_CHECK(cudaMemset(
            input_gradient,
            0,
            model.activation_elements * sizeof(float)));
        const float* input =
            model.hidden_states + static_cast<std::size_t>(layer)
                * model.activation_elements;
        const float* activations =
            model.block_activations + static_cast<std::size_t>(layer)
                * model.block_activation_elements;
        transformer_block_backward_cuda(
            input_gradient,
            model_block_gradients(model.gradients, model.layout, layer),
            output_gradient,
            input,
            model_block_parameters(model.parameters, model.layout, layer),
            model.cosine,
            model.sine,
            activations,
            model.block_workspace,
            model.block_config);
        std::swap(output_gradient, input_gradient);
    }

    embedding_backward_cuda(
        model.gradients + model.layout.token_embedding,
        output_gradient,
        model.input_tokens,
        model.rows,
        model.config.hidden_size);
    model.forward_ready = false;
}

TrainStepResult DenseGptModel::train_step(
    const std::vector<int>& input_tokens,
    const std::vector<int>& target_tokens,
    int step,
    const AdamWConfig& optimizer,
    float maximum_gradient_norm) {
    Implementation& model = *implementation_;
    const float loss = forward(input_tokens, target_tokens);
    zero_gradients();
    backward();
    global_norm_cuda(
        model.gradient_norm,
        model.gradients,
        model.norm_workspace,
        static_cast<int>(model.layout.elements));
    float norm = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &norm, model.gradient_norm, sizeof(float), cudaMemcpyDeviceToHost));
    if (maximum_gradient_norm > 0.0F) {
        clip_gradients_cuda(
            model.gradients,
            model.gradient_norm,
            static_cast<int>(model.layout.elements),
            maximum_gradient_norm);
    }
    adamw_step_cuda(
        model.parameters,
        model.first_moment,
        model.second_moment,
        model.gradients,
        static_cast<int>(model.layout.elements),
        step,
        optimizer);
    synchronize();
    return {loss, norm};
}

std::vector<float> DenseGptModel::parameters_to_host() const {
    const Implementation& model = *implementation_;
    std::vector<float> result(model.layout.elements);
    CUDA_CHECK(cudaMemcpy(
        result.data(),
        model.parameters,
        result.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    return result;
}

std::vector<float> DenseGptModel::gradients_to_host() const {
    const Implementation& model = *implementation_;
    std::vector<float> result(model.layout.elements);
    CUDA_CHECK(cudaMemcpy(
        result.data(),
        model.gradients,
        result.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    return result;
}

std::vector<float> DenseGptModel::logits_to_host() const {
    const Implementation& model = *implementation_;
    std::vector<float> result(model.logits_elements);
    CUDA_CHECK(cudaMemcpy(
        result.data(),
        model.logits,
        result.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    return result;
}

const ModelConfig& DenseGptModel::config() const {
    return implementation_->config;
}

const ModelParameterLayout& DenseGptModel::parameter_layout() const {
    return implementation_->layout;
}

ModelMemoryReport DenseGptModel::memory_report() const {
    const Implementation& model = *implementation_;
    const std::size_t saved =
        static_cast<std::size_t>(model.config.layers + 1)
            * model.activation_elements
        + static_cast<std::size_t>(model.config.layers)
            * model.block_activation_elements
        + model.activation_elements + model.rows + model.logits_elements
        + model.rows + 1;
    const std::size_t workspace = model.logits_elements
        + 2 * model.activation_elements + model.block_workspace_elements
        + model.norm_workspace_elements + 1;
    const std::size_t float_bytes =
        (4 * model.layout.elements + saved + workspace
         + static_cast<std::size_t>(config().sequence_length)
             * config().rotary_size)
        * sizeof(float);
    const std::size_t token_bytes =
        static_cast<std::size_t>(2 * model.rows) * sizeof(int);
    return {model.layout.elements, saved, workspace, float_bytes + token_bytes};
}

}  // namespace dscuda

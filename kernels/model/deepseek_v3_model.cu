// Builds a trainable DeepSeek-V3-style language model from token embeddings, repeated MLA-plus-MoE blocks, a tied vocabulary head, loss, clipping, and AdamW.
// Trainable tensors use one flat FP32 layout while load-balancing biases remain separate non-optimizer state updated from each block's routed token counts.

#include "deepseek_v3_model.h"

#include "cross_entropy.h"
#include "cuda_common.h"
#include "embedding.h"
#include "global_norm.h"
#include "matmul.h"
#include "mtp.h"
#include "residual.h"
#include "rmsnorm.h"
#include "swiglu.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace dscuda {
namespace {

std::size_t align_four(std::size_t value) {
    return (value + 3) & ~std::size_t{3};
}

template <typename T>
T* allocate(std::size_t elements) {
    return static_cast<T*>(device_malloc(elements * sizeof(T)));
}

void validate_config(const DeepSeekV3Config& config) {
    if (config.batch_size <= 0 || config.sequence_length <= 0
        || config.vocabulary_size <= 0 || config.layers <= 0
        || config.hidden_size <= 0 || config.heads <= 0
        || config.query_rank <= 0 || config.kv_rank <= 0
        || config.nope_size <= 0 || config.rope_size <= 0
        || config.value_size <= 0 || config.expert_hidden_size <= 0
        || config.routed_experts <= 0 || config.shared_experts <= 0
        || config.top_k <= 0 || config.top_k > config.routed_experts
        || config.top_k > 8 || config.kv_rank > 512
        || config.rope_size > 256 || config.rope_size % 2 != 0
        || config.hidden_size % 4 != 0 || config.query_rank % 4 != 0
        || config.kv_rank % 4 != 0 || config.nope_size % 4 != 0
        || config.rope_size % 4 != 0 || config.value_size % 4 != 0
        || config.expert_hidden_size % 4 != 0) {
        throw std::runtime_error("invalid DeepSeek-V3 model configuration");
    }
    if (config.mtp_depth < 0
        || config.mtp_depth >= config.sequence_length
        || config.mtp_loss_weight < 0.0F) {
        throw std::runtime_error("invalid DeepSeek-V3 MTP configuration");
    }
    if (config.dense_layers < 0 || config.dense_layers > config.layers
        || (config.dense_layers > 0
            && (config.dense_ffn_size <= 0
                || config.dense_ffn_size % 4 != 0))) {
        throw std::runtime_error("invalid DeepSeek-V3 dense-layer configuration");
    }
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

DeepSeekV3BlockParameters block_parameters(
    const float* parameters,
    const float* routing_bias,
    const DeepSeekV3BlockOffsets& block) {
    return {
        parameters + block.attention_norm,
        {
            parameters + block.query_down,
            parameters + block.query_norm,
            parameters + block.query_up,
            parameters + block.kv_down,
            parameters + block.kv_norm,
            parameters + block.key_up,
            parameters + block.value_up,
            parameters + block.attention_output,
        },
        parameters + block.ffn_norm,
        {
            parameters + block.router,
            routing_bias,
            parameters + block.routed_gate,
            parameters + block.routed_up,
            parameters + block.routed_down,
            parameters + block.shared_gate,
            parameters + block.shared_up,
            parameters + block.shared_down,
        },
    };
}

MlaLayerParameters mla_parameters(
    const float* parameters,
    const DeepSeekV3BlockOffsets& block) {
    return {
        parameters + block.query_down,
        parameters + block.query_norm,
        parameters + block.query_up,
        parameters + block.kv_down,
        parameters + block.kv_norm,
        parameters + block.key_up,
        parameters + block.value_up,
        parameters + block.attention_output,
    };
}

DeepSeekV3BlockGradients block_gradients(
    float* gradients,
    const DeepSeekV3BlockOffsets& block) {
    return {
        gradients + block.attention_norm,
        {
            gradients + block.query_down,
            gradients + block.query_norm,
            gradients + block.query_up,
            gradients + block.kv_down,
            gradients + block.kv_norm,
            gradients + block.key_up,
            gradients + block.value_up,
            gradients + block.attention_output,
        },
        gradients + block.ffn_norm,
        {
            gradients + block.router,
            gradients + block.routed_gate,
            gradients + block.routed_up,
            gradients + block.routed_down,
            gradients + block.shared_gate,
            gradients + block.shared_up,
            gradients + block.shared_down,
        },
    };
}

MtpParameters mtp_parameters(
    const float* parameters,
    const float* routing_bias,
    const DeepSeekV3ParameterLayout::MtpOffsets& mtp) {
    return {
        parameters + mtp.hidden_norm,
        parameters + mtp.embedding_norm,
        parameters + mtp.projection,
        block_parameters(parameters, routing_bias, mtp.block),
    };
}

MtpGradients mtp_gradients(
    float* gradients,
    const DeepSeekV3ParameterLayout::MtpOffsets& mtp) {
    return {
        gradients + mtp.hidden_norm,
        gradients + mtp.embedding_norm,
        gradients + mtp.projection,
        block_gradients(gradients, mtp.block),
    };
}

DeepSeekV3DenseBlockParameters dense_block_parameters(
    const float* parameters,
    const DeepSeekV3BlockOffsets& block) {
    return {
        parameters + block.attention_norm,
        {
            parameters + block.query_down,
            parameters + block.query_norm,
            parameters + block.query_up,
            parameters + block.kv_down,
            parameters + block.kv_norm,
            parameters + block.key_up,
            parameters + block.value_up,
            parameters + block.attention_output,
        },
        parameters + block.ffn_norm,
        parameters + block.shared_gate,
        parameters + block.shared_up,
        parameters + block.shared_down,
    };
}

DeepSeekV3DenseBlockGradients dense_block_gradients(
    float* gradients,
    const DeepSeekV3BlockOffsets& block) {
    return {
        gradients + block.attention_norm,
        {
            gradients + block.query_down,
            gradients + block.query_norm,
            gradients + block.query_up,
            gradients + block.kv_down,
            gradients + block.kv_norm,
            gradients + block.key_up,
            gradients + block.value_up,
            gradients + block.attention_output,
        },
        gradients + block.ffn_norm,
        gradients + block.shared_gate,
        gradients + block.shared_up,
        gradients + block.shared_down,
    };
}

}  // namespace

DeepSeekV3ParameterLayout make_deepseek_v3_parameter_layout(
    const DeepSeekV3Config& config) {
    validate_config(config);
    DeepSeekV3ParameterLayout layout{};
    std::size_t offset = 0;
    auto reserve = [&](std::size_t elements) {
        offset = align_four(offset);
        const std::size_t result = offset;
        offset += elements;
        return result;
    };
    const std::size_t hidden = config.hidden_size;
    const std::size_t query_width = static_cast<std::size_t>(config.heads)
        * (config.nope_size + config.rope_size);
    const std::size_t shared_width =
        static_cast<std::size_t>(config.shared_experts)
        * config.expert_hidden_size;

    auto reserve_block = [&](DeepSeekV3BlockOffsets& block, bool dense) {
        block.attention_norm = reserve(hidden);
        block.query_down = reserve(hidden * config.query_rank);
        block.query_norm = reserve(config.query_rank);
        block.query_up = reserve(config.query_rank * query_width);
        block.kv_down = reserve(
            hidden * (config.kv_rank + config.rope_size));
        block.kv_norm = reserve(config.kv_rank);
        block.key_up = reserve(
            static_cast<std::size_t>(config.heads) * config.nope_size
            * config.kv_rank);
        block.value_up = reserve(
            static_cast<std::size_t>(config.heads) * config.kv_rank
            * config.value_size);
        block.attention_output = reserve(
            static_cast<std::size_t>(config.heads) * config.value_size
            * hidden);
        block.ffn_norm = reserve(hidden);
        if (dense) {
            block.shared_gate = reserve(hidden * config.dense_ffn_size);
            block.shared_up = reserve(hidden * config.dense_ffn_size);
            block.shared_down = reserve(
                static_cast<std::size_t>(config.dense_ffn_size) * hidden);
            return;
        }
        block.router = reserve(hidden * config.routed_experts);
        block.routed_gate = reserve(
            static_cast<std::size_t>(config.routed_experts) * hidden
            * config.expert_hidden_size);
        block.routed_up = reserve(
            static_cast<std::size_t>(config.routed_experts) * hidden
            * config.expert_hidden_size);
        block.routed_down = reserve(
            static_cast<std::size_t>(config.routed_experts)
            * config.expert_hidden_size * hidden);
        block.shared_gate = reserve(hidden * shared_width);
        block.shared_up = reserve(hidden * shared_width);
        block.shared_down = reserve(shared_width * hidden);
    };

    layout.token_embedding = reserve(
        static_cast<std::size_t>(config.vocabulary_size) * hidden);
    layout.blocks.resize(config.layers);
    for (int layer = 0; layer < config.layers; ++layer) {
        reserve_block(layout.blocks[layer], layer < config.dense_layers);
    }
    layout.final_norm = reserve(hidden);
    layout.mtp_modules.resize(config.mtp_depth);
    for (DeepSeekV3ParameterLayout::MtpOffsets& mtp : layout.mtp_modules) {
        mtp.hidden_norm = reserve(hidden);
        mtp.embedding_norm = reserve(hidden);
        mtp.projection = reserve(2 * hidden * hidden);
        reserve_block(mtp.block, false);
    }
    layout.elements = align_four(offset);
    return layout;
}

struct DeepSeekV3Model::Implementation {
    explicit Implementation(const DeepSeekV3Config& model_config)
        : config(model_config),
          layout(make_deepseek_v3_parameter_layout(config)) {
        rows = config.batch_size * config.sequence_length;
        routing_bias_elements = static_cast<std::size_t>(
            config.layers - config.dense_layers + config.mtp_depth)
            * config.routed_experts;
        activation_elements =
            static_cast<std::size_t>(rows) * config.hidden_size;
        logits_elements =
            static_cast<std::size_t>(rows) * config.vocabulary_size;
        block_config = {
            {
                config.batch_size,
                config.sequence_length,
                config.hidden_size,
                config.heads,
                config.query_rank,
                config.kv_rank,
                config.nope_size,
                config.rope_size,
                config.value_size,
                config.rms_epsilon,
                1.0F / std::sqrt(
                    static_cast<float>(config.nope_size + config.rope_size)),
            },
            {
                rows,
                config.hidden_size,
                config.expert_hidden_size,
                config.routed_experts,
                config.shared_experts,
                config.top_k,
                config.route_scale,
                config.batch_size,
                config.sequence_length,
                config.balance_loss_weight,
            },
            config.rms_epsilon,
        };
        dense_block_config = {
            block_config.mla,
            rows,
            config.hidden_size,
            config.dense_ffn_size,
            config.rms_epsilon,
        };
        block_activation_elements =
            deepseek_v3_block_activation_elements(block_config);
        block_integer_activation_elements =
            deepseek_v3_block_integer_activation_elements(block_config);
        block_workspace_elements =
            deepseek_v3_block_backward_workspace_elements(block_config);
        bf16_workspace_elements =
            deepseek_v3_block_bf16_workspace_elements(block_config);
        if (config.dense_layers > 0) {
            block_activation_elements = std::max(
                block_activation_elements,
                deepseek_v3_dense_block_activation_elements(
                    dense_block_config));
            block_workspace_elements = std::max(
                block_workspace_elements,
                deepseek_v3_dense_block_backward_workspace_elements(
                    dense_block_config));
            bf16_workspace_elements = std::max(
                bf16_workspace_elements,
                deepseek_v3_dense_block_bf16_workspace_elements(
                    dense_block_config));
        }
        std::size_t mtp_activation_total = 0;
        std::size_t mtp_integer_total = 0;
        std::size_t mtp_output_total = 0;
        std::size_t mtp_row_total = 0;
        std::size_t mtp_logits_total = 0;
        for (int depth = 1; depth <= config.mtp_depth; ++depth) {
            const int valid_length = config.sequence_length - depth;
            const int mtp_rows = config.batch_size * valid_length;
            DeepSeekV3BlockConfig mtp_block = block_config;
            mtp_block.mla.sequence_length = valid_length;
            mtp_block.moe.rows = mtp_rows;
            mtp_block.moe.sequence_length = valid_length;
            MtpConfig mtp_config{
                config.batch_size,
                config.sequence_length,
                config.hidden_size,
                depth,
                config.rms_epsilon,
                mtp_block,
            };
            mtp_configs.push_back(mtp_config);
            mtp_activation_offsets.push_back(mtp_activation_total);
            mtp_integer_offsets.push_back(mtp_integer_total);
            mtp_output_offsets.push_back(mtp_output_total);
            mtp_row_offsets.push_back(mtp_row_total);
            mtp_logits_offsets.push_back(mtp_logits_total);
            mtp_activation_total += mtp_activation_elements(mtp_config);
            mtp_integer_total += mtp_integer_activation_elements(mtp_config);
            mtp_output_total +=
                static_cast<std::size_t>(mtp_rows) * config.hidden_size;
            mtp_row_total += mtp_rows;
            mtp_logits_total +=
                static_cast<std::size_t>(mtp_rows) * config.vocabulary_size;
            mtp_workspace_elements = std::max(
                mtp_workspace_elements,
                mtp_backward_workspace_elements(mtp_config));
            bf16_workspace_elements = std::max(
                bf16_workspace_elements,
                mtp_bf16_workspace_elements(mtp_config));
        }
        mtp_activation_elements_total = mtp_activation_total;
        mtp_integer_elements_total = mtp_integer_total;
        mtp_output_elements_total = mtp_output_total;
        mtp_row_elements_total = mtp_row_total;
        mtp_logits_elements_total = mtp_logits_total;
        norm_workspace_elements = global_norm_workspace_elements(
            static_cast<int>(layout.elements));

        if (config.batch_size == 1) {
            decode_moe_config = block_config.moe;
            decode_moe_config.rows = 1;
            decode_moe_config.batch_size = 1;
            decode_moe_config.sequence_length = 1;
            decode_moe_config.balance_loss_weight = 0.0F;
            decode_mla_workspace_elements =
                mla_layer_decode_workspace_elements(
                    block_config.mla, decode_splits);
            decode_mla_bf16_elements =
                mla_layer_decode_bf16_workspace_elements(block_config.mla);
            decode_moe_activation_elements =
                deepseek_moe_activation_elements(decode_moe_config);
            decode_moe_integer_elements =
                deepseek_moe_integer_activation_elements(decode_moe_config);
            const std::size_t hidden = config.hidden_size;
            const std::size_t dense = std::max(1, config.dense_ffn_size);
            decode_hidden_a = allocate<float>(hidden);
            decode_hidden_b = allocate<float>(hidden);
            decode_attention_norm = allocate<float>(hidden);
            decode_attention_inverse_rms = allocate<float>(1);
            decode_attention_output = allocate<float>(hidden);
            decode_residual = allocate<float>(hidden);
            decode_ffn_norm = allocate<float>(hidden);
            decode_ffn_inverse_rms = allocate<float>(1);
            decode_dense_gate = allocate<float>(dense);
            decode_dense_up = allocate<float>(dense);
            decode_dense_hidden = allocate<float>(dense);
            decode_ffn_output = allocate<float>(hidden);
            decode_final_norm = allocate<float>(hidden);
            decode_final_inverse_rms = allocate<float>(1);
            decode_logits = allocate<float>(config.vocabulary_size);
            decode_kv_cache = allocate<__nv_bfloat16>(
                static_cast<std::size_t>(config.layers)
                * config.sequence_length * config.kv_rank);
            decode_key_rope_cache = allocate<__nv_bfloat16>(
                static_cast<std::size_t>(config.layers)
                * config.sequence_length * config.rope_size);
            decode_cache_lengths = allocate<int>(1);
            decode_mla_workspace = allocate<float>(
                decode_mla_workspace_elements);
            decode_mla_bf16 = allocate<__nv_bfloat16>(
                decode_mla_bf16_elements);
            decode_moe_activations = allocate<float>(
                decode_moe_activation_elements);
            decode_moe_integer_activations = allocate<int>(
                decode_moe_integer_elements);
        }

        parameters = allocate<float>(layout.elements);
        gradients = allocate<float>(layout.elements);
        first_moment = allocate<float>(layout.elements);
        second_moment = allocate<float>(layout.elements);
        if (routing_bias_elements > 0) {
            routing_bias = allocate<float>(routing_bias_elements);
        }
        input_tokens = allocate<int>(rows);
        target_tokens = allocate<int>(rows);
        const std::size_t frequencies =
            static_cast<std::size_t>(config.sequence_length)
            * config.rope_size / 2;
        cosine = allocate<float>(frequencies);
        sine = allocate<float>(frequencies);
        hidden_states = allocate<float>(
            static_cast<std::size_t>(config.layers + 1) * activation_elements);
        block_activations = allocate<float>(
            static_cast<std::size_t>(config.layers) * block_activation_elements);
        block_integer_activations = allocate<int>(
            static_cast<std::size_t>(config.layers)
            * block_integer_activation_elements);
        if (config.mtp_depth > 0) {
            mtp_outputs = allocate<float>(mtp_output_elements_total);
            mtp_activations = allocate<float>(mtp_activation_elements_total);
            mtp_integer_activations = allocate<int>(mtp_integer_elements_total);
            mtp_output_norm = allocate<float>(mtp_output_elements_total);
            mtp_output_inverse_rms = allocate<float>(mtp_row_elements_total);
            mtp_logits = allocate<float>(mtp_logits_elements_total);
            mtp_logsumexp = allocate<float>(mtp_row_elements_total);
            mtp_targets = allocate<int>(mtp_row_elements_total);
            mtp_losses = allocate<float>(config.mtp_depth);
            mtp_output_gradients = allocate<float>(mtp_output_elements_total);
            mtp_workspace = allocate<float>(mtp_workspace_elements);
        }
        final_norm = allocate<float>(activation_elements);
        final_inverse_rms = allocate<float>(rows);
        logits = allocate<float>(logits_elements);
        logsumexp = allocate<float>(rows);
        mean_loss = allocate<float>(1);
        logits_gradient = allocate<float>(logits_elements);
        activation_gradient_a = allocate<float>(activation_elements);
        activation_gradient_b = allocate<float>(activation_elements);
        block_workspace = allocate<float>(block_workspace_elements);
        bf16_workspace = allocate<__nv_bfloat16>(bf16_workspace_elements);
        gradient_norm = allocate<float>(1);
        norm_workspace = allocate<float>(norm_workspace_elements);

        std::vector<float> host_cosine(frequencies);
        std::vector<float> host_sine(frequencies);
        const int pairs = config.rope_size / 2;
        for (int position = 0; position < config.sequence_length; ++position) {
            for (int pair = 0; pair < pairs; ++pair) {
                const float inverse_frequency = std::pow(
                    10000.0F,
                    -2.0F * static_cast<float>(pair) / config.rope_size);
                const float angle = position * inverse_frequency;
                host_cosine[position * pairs + pair] = std::cos(angle);
                host_sine[position * pairs + pair] = std::sin(angle);
            }
        }
        CUDA_CHECK(cudaMemcpy(
            cosine, host_cosine.data(), frequencies * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            sine, host_sine.data(), frequencies * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    ~Implementation() {
        if (config.batch_size == 1) {
            device_free(decode_moe_integer_activations);
            device_free(decode_moe_activations);
            device_free(decode_mla_bf16);
            device_free(decode_mla_workspace);
            device_free(decode_cache_lengths);
            device_free(decode_key_rope_cache);
            device_free(decode_kv_cache);
            device_free(decode_logits);
            device_free(decode_final_inverse_rms);
            device_free(decode_final_norm);
            device_free(decode_ffn_output);
            device_free(decode_dense_hidden);
            device_free(decode_dense_up);
            device_free(decode_dense_gate);
            device_free(decode_ffn_inverse_rms);
            device_free(decode_ffn_norm);
            device_free(decode_residual);
            device_free(decode_attention_output);
            device_free(decode_attention_inverse_rms);
            device_free(decode_attention_norm);
            device_free(decode_hidden_b);
            device_free(decode_hidden_a);
        }
        if (config.mtp_depth > 0) {
            device_free(mtp_workspace);
            device_free(mtp_output_gradients);
            device_free(mtp_losses);
            device_free(mtp_targets);
            device_free(mtp_logsumexp);
            device_free(mtp_logits);
            device_free(mtp_output_inverse_rms);
            device_free(mtp_output_norm);
            device_free(mtp_integer_activations);
            device_free(mtp_activations);
            device_free(mtp_outputs);
        }
        device_free(norm_workspace);
        device_free(gradient_norm);
        device_free(bf16_workspace);
        device_free(block_workspace);
        device_free(activation_gradient_b);
        device_free(activation_gradient_a);
        device_free(logits_gradient);
        device_free(mean_loss);
        device_free(logsumexp);
        device_free(logits);
        device_free(final_inverse_rms);
        device_free(final_norm);
        device_free(block_integer_activations);
        device_free(block_activations);
        device_free(hidden_states);
        device_free(sine);
        device_free(cosine);
        device_free(target_tokens);
        device_free(input_tokens);
        if (routing_bias != nullptr) {
            device_free(routing_bias);
        }
        device_free(second_moment);
        device_free(first_moment);
        device_free(gradients);
        device_free(parameters);
    }

    DeepSeekV3Config config;
    DeepSeekV3ParameterLayout layout;
    DeepSeekV3BlockConfig block_config{};
    DeepSeekV3DenseBlockConfig dense_block_config{};
    int rows = 0;
    std::size_t activation_elements = 0;
    std::size_t logits_elements = 0;
    std::size_t block_activation_elements = 0;
    std::size_t block_integer_activation_elements = 0;
    std::size_t block_workspace_elements = 0;
    std::size_t bf16_workspace_elements = 0;
    std::size_t norm_workspace_elements = 0;
    std::size_t routing_bias_elements = 0;
    std::vector<MtpConfig> mtp_configs;
    std::vector<std::size_t> mtp_activation_offsets;
    std::vector<std::size_t> mtp_integer_offsets;
    std::vector<std::size_t> mtp_output_offsets;
    std::vector<std::size_t> mtp_row_offsets;
    std::vector<std::size_t> mtp_logits_offsets;
    std::size_t mtp_activation_elements_total = 0;
    std::size_t mtp_integer_elements_total = 0;
    std::size_t mtp_output_elements_total = 0;
    std::size_t mtp_row_elements_total = 0;
    std::size_t mtp_logits_elements_total = 0;
    std::size_t mtp_workspace_elements = 0;
    int decode_splits = 4;
    DeepSeekMoeConfig decode_moe_config{};
    std::size_t decode_mla_workspace_elements = 0;
    std::size_t decode_mla_bf16_elements = 0;
    std::size_t decode_moe_activation_elements = 0;
    std::size_t decode_moe_integer_elements = 0;
    float* parameters = nullptr;
    float* gradients = nullptr;
    float* first_moment = nullptr;
    float* second_moment = nullptr;
    float* routing_bias = nullptr;
    int* input_tokens = nullptr;
    int* target_tokens = nullptr;
    float* cosine = nullptr;
    float* sine = nullptr;
    float* hidden_states = nullptr;
    float* block_activations = nullptr;
    int* block_integer_activations = nullptr;
    float* mtp_outputs = nullptr;
    float* mtp_activations = nullptr;
    int* mtp_integer_activations = nullptr;
    float* mtp_output_norm = nullptr;
    float* mtp_output_inverse_rms = nullptr;
    float* mtp_logits = nullptr;
    float* mtp_logsumexp = nullptr;
    int* mtp_targets = nullptr;
    float* mtp_losses = nullptr;
    float* mtp_output_gradients = nullptr;
    float* mtp_workspace = nullptr;
    float* decode_hidden_a = nullptr;
    float* decode_hidden_b = nullptr;
    float* decode_attention_norm = nullptr;
    float* decode_attention_inverse_rms = nullptr;
    float* decode_attention_output = nullptr;
    float* decode_residual = nullptr;
    float* decode_ffn_norm = nullptr;
    float* decode_ffn_inverse_rms = nullptr;
    float* decode_dense_gate = nullptr;
    float* decode_dense_up = nullptr;
    float* decode_dense_hidden = nullptr;
    float* decode_ffn_output = nullptr;
    float* decode_final_norm = nullptr;
    float* decode_final_inverse_rms = nullptr;
    float* decode_logits = nullptr;
    __nv_bfloat16* decode_kv_cache = nullptr;
    __nv_bfloat16* decode_key_rope_cache = nullptr;
    int* decode_cache_lengths = nullptr;
    float* decode_mla_workspace = nullptr;
    __nv_bfloat16* decode_mla_bf16 = nullptr;
    float* decode_moe_activations = nullptr;
    int* decode_moe_integer_activations = nullptr;
    std::vector<int> cached_tokens;
    float* final_norm = nullptr;
    float* final_inverse_rms = nullptr;
    float* logits = nullptr;
    float* logsumexp = nullptr;
    float* mean_loss = nullptr;
    float* logits_gradient = nullptr;
    float* activation_gradient_a = nullptr;
    float* activation_gradient_b = nullptr;
    float* block_workspace = nullptr;
    __nv_bfloat16* bf16_workspace = nullptr;
    float* gradient_norm = nullptr;
    float* norm_workspace = nullptr;
    bool forward_ready = false;
};

DeepSeekV3Model::DeepSeekV3Model(const DeepSeekV3Config& config)
    : implementation_(std::make_unique<Implementation>(config)) {}

DeepSeekV3Model::~DeepSeekV3Model() = default;

void DeepSeekV3Model::initialize(std::uint64_t seed) {
    Implementation& model = *implementation_;
    std::vector<float> host(model.layout.elements, 0.0F);
    std::mt19937_64 generator(seed);
    const int hidden = model.config.hidden_size;
    const int query_width = model.config.heads
        * (model.config.nope_size + model.config.rope_size);
    const int shared_width =
        model.config.shared_experts * model.config.expert_hidden_size;
    const float residual_scale =
        0.02F / std::sqrt(2.0F * model.config.layers);
    fill_normal(
        host,
        model.layout.token_embedding,
        static_cast<std::size_t>(model.config.vocabulary_size) * hidden,
        0.02F,
        generator);
    auto initialize_block = [
        &](const DeepSeekV3BlockOffsets& block, bool dense) {
        std::fill_n(host.begin() + block.attention_norm, hidden, 1.0F);
        std::fill_n(
            host.begin() + block.query_norm, model.config.query_rank, 1.0F);
        std::fill_n(
            host.begin() + block.kv_norm, model.config.kv_rank, 1.0F);
        std::fill_n(host.begin() + block.ffn_norm, hidden, 1.0F);
        fill_normal(host, block.query_down,
            static_cast<std::size_t>(hidden) * model.config.query_rank,
            0.02F, generator);
        fill_normal(host, block.query_up,
            static_cast<std::size_t>(model.config.query_rank) * query_width,
            0.02F, generator);
        fill_normal(host, block.kv_down,
            static_cast<std::size_t>(hidden)
                * (model.config.kv_rank + model.config.rope_size),
            0.02F, generator);
        fill_normal(host, block.key_up,
            static_cast<std::size_t>(model.config.heads)
                * model.config.nope_size * model.config.kv_rank,
            0.02F, generator);
        fill_normal(host, block.value_up,
            static_cast<std::size_t>(model.config.heads)
                * model.config.kv_rank * model.config.value_size,
            0.02F, generator);
        fill_normal(host, block.attention_output,
            static_cast<std::size_t>(model.config.heads)
                * model.config.value_size * hidden,
            residual_scale, generator);
        if (dense) {
            fill_normal(
                host,
                block.shared_gate,
                static_cast<std::size_t>(hidden)
                    * model.config.dense_ffn_size,
                0.02F,
                generator);
            fill_normal(
                host,
                block.shared_up,
                static_cast<std::size_t>(hidden)
                    * model.config.dense_ffn_size,
                0.02F,
                generator);
            fill_normal(
                host,
                block.shared_down,
                static_cast<std::size_t>(model.config.dense_ffn_size)
                    * hidden,
                residual_scale,
                generator);
            return;
        }
        fill_normal(host, block.router,
            static_cast<std::size_t>(hidden) * model.config.routed_experts,
            0.02F, generator);
        const std::size_t routed_up =
            static_cast<std::size_t>(model.config.routed_experts) * hidden
            * model.config.expert_hidden_size;
        const std::size_t routed_down =
            static_cast<std::size_t>(model.config.routed_experts)
            * model.config.expert_hidden_size * hidden;
        fill_normal(host, block.routed_gate, routed_up, 0.02F, generator);
        fill_normal(host, block.routed_up, routed_up, 0.02F, generator);
        fill_normal(
            host, block.routed_down, routed_down, residual_scale, generator);
        fill_normal(host, block.shared_gate,
            static_cast<std::size_t>(hidden) * shared_width,
            0.02F, generator);
        fill_normal(host, block.shared_up,
            static_cast<std::size_t>(hidden) * shared_width,
            0.02F, generator);
        fill_normal(host, block.shared_down,
            static_cast<std::size_t>(shared_width) * hidden,
            residual_scale, generator);
    };
    for (int layer = 0; layer < model.config.layers; ++layer) {
        initialize_block(
            model.layout.blocks[layer], layer < model.config.dense_layers);
    }
    for (const DeepSeekV3ParameterLayout::MtpOffsets& mtp
         : model.layout.mtp_modules) {
        std::fill_n(host.begin() + mtp.hidden_norm, hidden, 1.0F);
        std::fill_n(host.begin() + mtp.embedding_norm, hidden, 1.0F);
        fill_normal(
            host,
            mtp.projection,
            static_cast<std::size_t>(2) * hidden * hidden,
            0.02F,
            generator);
        initialize_block(mtp.block, false);
    }
    std::fill_n(host.begin() + model.layout.final_norm, hidden, 1.0F);
    load_parameters(host);
    CUDA_CHECK(cudaMemset(
        model.first_moment, 0, model.layout.elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(
        model.second_moment, 0, model.layout.elements * sizeof(float)));
    if (model.routing_bias_elements > 0) {
        CUDA_CHECK(cudaMemset(
            model.routing_bias,
            0,
            model.routing_bias_elements * sizeof(float)));
    }
    zero_gradients();
}

void DeepSeekV3Model::load_parameters(const std::vector<float>& parameters) {
    Implementation& model = *implementation_;
    if (parameters.size() != model.layout.elements) {
        throw std::runtime_error("DeepSeek-V3 parameter vector has wrong size");
    }
    CUDA_CHECK(cudaMemcpy(
        model.parameters,
        parameters.data(),
        model.layout.elements * sizeof(float),
        cudaMemcpyHostToDevice));
    model.cached_tokens.clear();
    model.forward_ready = false;
}

DeepSeekV3TrainingState DeepSeekV3Model::training_state_to_host() const {
    const Implementation& model = *implementation_;
    DeepSeekV3TrainingState state;
    state.optimizer.parameters.resize(model.layout.elements);
    state.optimizer.first_moment.resize(model.layout.elements);
    state.optimizer.second_moment.resize(model.layout.elements);
    state.routing_bias.resize(model.routing_bias_elements);
    CUDA_CHECK(cudaMemcpy(
        state.optimizer.parameters.data(), model.parameters,
        model.layout.elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        state.optimizer.first_moment.data(), model.first_moment,
        model.layout.elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        state.optimizer.second_moment.data(), model.second_moment,
        model.layout.elements * sizeof(float), cudaMemcpyDeviceToHost));
    if (model.routing_bias_elements > 0) {
        CUDA_CHECK(cudaMemcpy(
            state.routing_bias.data(), model.routing_bias,
            state.routing_bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return state;
}

void DeepSeekV3Model::load_training_state(
    const DeepSeekV3TrainingState& state) {
    Implementation& model = *implementation_;
    const std::size_t bias_elements = model.routing_bias_elements;
    if (state.optimizer.parameters.size() != model.layout.elements
        || state.optimizer.first_moment.size() != model.layout.elements
        || state.optimizer.second_moment.size() != model.layout.elements
        || state.routing_bias.size() != bias_elements) {
        throw std::runtime_error("DeepSeek-V3 checkpoint state has wrong size");
    }
    CUDA_CHECK(cudaMemcpy(
        model.parameters, state.optimizer.parameters.data(),
        model.layout.elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        model.first_moment, state.optimizer.first_moment.data(),
        model.layout.elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        model.second_moment, state.optimizer.second_moment.data(),
        model.layout.elements * sizeof(float), cudaMemcpyHostToDevice));
    if (bias_elements > 0) {
        CUDA_CHECK(cudaMemcpy(
            model.routing_bias, state.routing_bias.data(),
            bias_elements * sizeof(float), cudaMemcpyHostToDevice));
    }
    model.cached_tokens.clear();
    model.forward_ready = false;
}

void DeepSeekV3Model::run_model_forward(
    const std::vector<int>& input_tokens) {
    Implementation& model = *implementation_;
    if (input_tokens.size() != static_cast<std::size_t>(model.rows)) {
        throw std::runtime_error("DeepSeek-V3 input has wrong token count");
    }
    CUDA_CHECK(cudaMemcpy(
        model.input_tokens, input_tokens.data(), model.rows * sizeof(int),
        cudaMemcpyHostToDevice));
    embedding_forward_cuda(
        model.hidden_states,
        model.input_tokens,
        model.parameters + model.layout.token_embedding,
        model.rows,
        model.config.hidden_size);
    for (int layer = 0; layer < model.config.layers; ++layer) {
        float* output = model.hidden_states
            + static_cast<std::size_t>(layer + 1)
                * model.activation_elements;
        const float* input = model.hidden_states
            + static_cast<std::size_t>(layer) * model.activation_elements;
        float* activations = model.block_activations
            + static_cast<std::size_t>(layer)
                * model.block_activation_elements;
        if (layer < model.config.dense_layers) {
            deepseek_v3_dense_block_forward_cuda(
                output,
                input,
                dense_block_parameters(
                    model.parameters, model.layout.blocks[layer]),
                model.cosine,
                model.sine,
                activations,
                model.bf16_workspace,
                model.dense_block_config);
        } else {
            deepseek_v3_block_forward_cuda(
                output,
                input,
                block_parameters(
                    model.parameters,
                    model.routing_bias
                        + static_cast<std::size_t>(
                            layer - model.config.dense_layers)
                            * model.config.routed_experts,
                    model.layout.blocks[layer]),
                model.cosine,
                model.sine,
                activations,
                model.block_integer_activations
                    + static_cast<std::size_t>(layer)
                        * model.block_integer_activation_elements,
                model.bf16_workspace,
                model.block_config);
        }
    }
    const float* final_input = model.hidden_states
        + static_cast<std::size_t>(model.config.layers)
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
    model.forward_ready = false;
}

float DeepSeekV3Model::forward(
    const std::vector<int>& input_tokens,
    const std::vector<int>& target_tokens) {
    Implementation& model = *implementation_;
    if (target_tokens.size() != static_cast<std::size_t>(model.rows)) {
        throw std::runtime_error("DeepSeek-V3 target has wrong token count");
    }
    run_model_forward(input_tokens);
    CUDA_CHECK(cudaMemcpy(
        model.target_tokens, target_tokens.data(), model.rows * sizeof(int),
        cudaMemcpyHostToDevice));
    cross_entropy_forward_cuda(
        model.mean_loss,
        model.logsumexp,
        model.logits,
        model.target_tokens,
        model.rows,
        model.config.vocabulary_size);
    for (int layer = model.config.dense_layers;
         layer < model.config.layers;
         ++layer) {
        deepseek_v3_block_add_balance_loss_cuda(
            model.mean_loss,
            model.block_activations
                + static_cast<std::size_t>(layer)
                    * model.block_activation_elements,
            model.block_config);
    }
    const float* main_hidden = model.hidden_states
        + static_cast<std::size_t>(model.config.layers)
            * model.activation_elements;
    for (int module = 0; module < model.config.mtp_depth; ++module) {
        const MtpConfig& mtp_config = model.mtp_configs[module];
        const int mtp_rows = mtp_config.block.moe.rows;
        const float* previous_hidden = module == 0
            ? main_hidden
            : model.mtp_outputs + model.mtp_output_offsets[module - 1];
        float* output =
            model.mtp_outputs + model.mtp_output_offsets[module];
        float* activations =
            model.mtp_activations + model.mtp_activation_offsets[module];
        mtp_forward_cuda(
            output,
            previous_hidden,
            model.input_tokens,
            model.parameters + model.layout.token_embedding,
            mtp_parameters(
                model.parameters,
                model.routing_bias
                    + static_cast<std::size_t>(
                        model.config.layers - model.config.dense_layers
                        + module)
                        * model.config.routed_experts,
                model.layout.mtp_modules[module]),
            model.cosine,
            model.sine,
            activations,
            model.mtp_integer_activations
                + model.mtp_integer_offsets[module],
            model.bf16_workspace,
            mtp_config);
        mtp_add_balance_loss_cuda(
            model.mean_loss, activations, mtp_config);
        rmsnorm_forward_cuda(
            model.mtp_output_norm + model.mtp_output_offsets[module],
            model.mtp_output_inverse_rms + model.mtp_row_offsets[module],
            output,
            model.parameters + model.layout.final_norm,
            mtp_rows,
            model.config.hidden_size,
            model.config.rms_epsilon);
        matmul_fp32_strided_batched_cuda(
            model.mtp_logits + model.mtp_logits_offsets[module],
            model.mtp_output_norm + model.mtp_output_offsets[module],
            model.parameters + model.layout.token_embedding,
            mtp_rows,
            model.config.vocabulary_size,
            model.config.hidden_size,
            1,
            0,
            0,
            0,
            false,
            true,
            false);
        mtp_gather_targets_cuda(
            model.mtp_targets + model.mtp_row_offsets[module],
            model.target_tokens,
            mtp_config);
        cross_entropy_forward_cuda(
            model.mtp_losses + module,
            model.mtp_logsumexp + model.mtp_row_offsets[module],
            model.mtp_logits + model.mtp_logits_offsets[module],
            model.mtp_targets + model.mtp_row_offsets[module],
            mtp_rows,
            model.config.vocabulary_size);
        const float loss_scale = model.config.mtp_loss_weight
            / model.config.mtp_depth
            * static_cast<float>(
                model.config.sequence_length - mtp_config.depth)
            / model.config.sequence_length;
        mtp_add_scaled_loss_cuda(
            model.mean_loss, model.mtp_losses + module, loss_scale);
    }
    float loss = 0.0F;
    CUDA_CHECK(cudaMemcpy(
        &loss, model.mean_loss, sizeof(float), cudaMemcpyDeviceToHost));
    model.forward_ready = true;
    return loss;
}

void DeepSeekV3Model::zero_gradients() {
    Implementation& model = *implementation_;
    CUDA_CHECK(cudaMemset(
        model.gradients, 0, model.layout.elements * sizeof(float)));
}

void DeepSeekV3Model::backward() {
    Implementation& model = *implementation_;
    if (!model.forward_ready) {
        throw std::runtime_error(
            "DeepSeek-V3 backward requires a matching forward pass");
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
    const float* final_input = model.hidden_states
        + static_cast<std::size_t>(model.config.layers)
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

    if (model.config.mtp_depth > 0) {
        CUDA_CHECK(cudaMemset(
            model.mtp_output_gradients,
            0,
            model.mtp_output_elements_total * sizeof(float)));
    }
    for (int module = model.config.mtp_depth - 1; module >= 0; --module) {
        const MtpConfig& mtp_config = model.mtp_configs[module];
        const int mtp_rows = mtp_config.block.moe.rows;
        const int mtp_logits_elements =
            mtp_rows * model.config.vocabulary_size;
        const float loss_scale = model.config.mtp_loss_weight
            / model.config.mtp_depth
            * static_cast<float>(
                model.config.sequence_length - mtp_config.depth)
            / model.config.sequence_length;
        cross_entropy_backward_cuda(
            model.logits_gradient,
            model.mtp_logits + model.mtp_logits_offsets[module],
            model.mtp_logsumexp + model.mtp_row_offsets[module],
            model.mtp_targets + model.mtp_row_offsets[module],
            mtp_rows,
            model.config.vocabulary_size);
        mtp_scale_gradients_cuda(
            model.logits_gradient, mtp_logits_elements, loss_scale);
        matmul_fp32_forward_cuda(
            model.activation_gradient_a,
            model.logits_gradient,
            model.parameters + model.layout.token_embedding,
            mtp_rows,
            model.config.hidden_size,
            model.config.vocabulary_size);
        matmul_fp32_strided_batched_cuda(
            model.gradients + model.layout.token_embedding,
            model.logits_gradient,
            model.mtp_output_norm + model.mtp_output_offsets[module],
            model.config.vocabulary_size,
            model.config.hidden_size,
            mtp_rows,
            1,
            0,
            0,
            0,
            true,
            false,
            true);
        float* module_output_gradient =
            model.mtp_output_gradients + model.mtp_output_offsets[module];
        rmsnorm_backward_cuda(
            module_output_gradient,
            model.gradients + model.layout.final_norm,
            model.activation_gradient_a,
            model.mtp_outputs + model.mtp_output_offsets[module],
            model.parameters + model.layout.final_norm,
            model.mtp_output_inverse_rms + model.mtp_row_offsets[module],
            mtp_rows,
            model.config.hidden_size);
        const float* previous_hidden = module == 0
            ? final_input
            : model.mtp_outputs + model.mtp_output_offsets[module - 1];
        float* previous_hidden_gradient = module == 0
            ? model.activation_gradient_b
            : model.mtp_output_gradients
                + model.mtp_output_offsets[module - 1];
        mtp_backward_cuda(
            previous_hidden_gradient,
            model.gradients + model.layout.token_embedding,
            mtp_gradients(
                model.gradients, model.layout.mtp_modules[module]),
            module_output_gradient,
            previous_hidden,
            model.input_tokens,
            model.parameters + model.layout.token_embedding,
            mtp_parameters(
                model.parameters,
                model.routing_bias
                    + static_cast<std::size_t>(
                        model.config.layers - model.config.dense_layers
                        + module)
                        * model.config.routed_experts,
                model.layout.mtp_modules[module]),
            model.cosine,
            model.sine,
            model.mtp_activations + model.mtp_activation_offsets[module],
            model.mtp_integer_activations
                + model.mtp_integer_offsets[module],
            model.mtp_workspace,
            model.bf16_workspace,
            mtp_config);
    }

    float* output_gradient = model.activation_gradient_b;
    float* input_gradient = model.activation_gradient_a;
    for (int layer = model.config.layers - 1; layer >= 0; --layer) {
        CUDA_CHECK(cudaMemset(
            input_gradient, 0, model.activation_elements * sizeof(float)));
        const float* input = model.hidden_states
            + static_cast<std::size_t>(layer) * model.activation_elements;
        const float* activations = model.block_activations
            + static_cast<std::size_t>(layer)
                * model.block_activation_elements;
        if (layer < model.config.dense_layers) {
            deepseek_v3_dense_block_backward_cuda(
                input_gradient,
                dense_block_gradients(
                    model.gradients, model.layout.blocks[layer]),
                output_gradient,
                input,
                dense_block_parameters(
                    model.parameters, model.layout.blocks[layer]),
                model.cosine,
                model.sine,
                activations,
                model.block_workspace,
                model.bf16_workspace,
                model.dense_block_config);
        } else {
            deepseek_v3_block_backward_cuda(
                input_gradient,
                block_gradients(
                    model.gradients, model.layout.blocks[layer]),
                output_gradient,
                input,
                block_parameters(
                    model.parameters,
                    model.routing_bias
                        + static_cast<std::size_t>(
                            layer - model.config.dense_layers)
                            * model.config.routed_experts,
                    model.layout.blocks[layer]),
                model.cosine,
                model.sine,
                activations,
                model.block_integer_activations
                    + static_cast<std::size_t>(layer)
                        * model.block_integer_activation_elements,
                model.block_workspace,
                model.bf16_workspace,
                model.block_config);
        }
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

TrainStepResult DeepSeekV3Model::train_step(
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
    for (int layer = model.config.dense_layers;
         layer < model.config.layers;
         ++layer) {
        deepseek_moe_update_bias_cuda(
            model.routing_bias
                + static_cast<std::size_t>(
                    layer - model.config.dense_layers)
                    * model.config.routed_experts,
            model.block_integer_activations
                + static_cast<std::size_t>(layer)
                    * model.block_integer_activation_elements,
            model.block_config.moe,
            model.config.routing_bias_update_speed);
    }
    for (int module = 0; module < model.config.mtp_depth; ++module) {
        deepseek_moe_update_bias_cuda(
                model.routing_bias
                    + static_cast<std::size_t>(
                        model.config.layers - model.config.dense_layers
                        + module)
                        * model.config.routed_experts,
            model.mtp_integer_activations
                + model.mtp_integer_offsets[module],
            model.mtp_configs[module].block.moe,
            model.config.routing_bias_update_speed);
    }
    model.cached_tokens.clear();
    synchronize();
    return {loss, norm};
}

std::vector<float> DeepSeekV3Model::parameters_to_host() const {
    const Implementation& model = *implementation_;
    std::vector<float> result(model.layout.elements);
    CUDA_CHECK(cudaMemcpy(
        result.data(), model.parameters, result.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    return result;
}

std::vector<float> DeepSeekV3Model::gradients_to_host() const {
    const Implementation& model = *implementation_;
    std::vector<float> result(model.layout.elements);
    CUDA_CHECK(cudaMemcpy(
        result.data(), model.gradients, result.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    return result;
}

std::vector<float> DeepSeekV3Model::logits_to_host() const {
    const Implementation& model = *implementation_;
    std::vector<float> result(model.logits_elements);
    CUDA_CHECK(cudaMemcpy(
        result.data(), model.logits, result.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    return result;
}

std::vector<float> DeepSeekV3Model::routing_bias_to_host() const {
    const Implementation& model = *implementation_;
    std::vector<float> result(model.routing_bias_elements);
    if (model.routing_bias_elements > 0) {
        CUDA_CHECK(cudaMemcpy(
            result.data(), model.routing_bias, result.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
    }
    return result;
}

int DeepSeekV3Model::vocabulary_size() const {
    return implementation_->config.vocabulary_size;
}

int DeepSeekV3Model::maximum_context_length() const {
    return implementation_->config.sequence_length;
}

std::size_t DeepSeekV3Model::kv_cache_bytes_per_token() const {
    const DeepSeekV3Config& config = implementation_->config;
    return static_cast<std::size_t>(config.layers)
        * (config.kv_rank + config.rope_size) * sizeof(__nv_bfloat16);
}

std::vector<float> DeepSeekV3Model::forward_last_logits(
    const std::vector<int>& tokens) {
    Implementation& model = *implementation_;
    if (model.config.batch_size != 1 || tokens.empty()
        || tokens.size() > static_cast<std::size_t>(model.config.sequence_length)) {
        throw std::runtime_error(
            "V3 generation requires B=1 and a non-empty context within T");
    }

    const bool extends_cache = model.cached_tokens.size() <= tokens.size()
        && std::equal(
            model.cached_tokens.begin(),
            model.cached_tokens.end(),
            tokens.begin());
    std::size_t first_position = model.cached_tokens.size();
    if (!extends_cache) {
        model.cached_tokens.clear();
        first_position = 0;
    }

    for (std::size_t position = first_position;
         position < tokens.size();
         ++position) {
        CUDA_CHECK(cudaMemcpy(
            model.input_tokens,
            tokens.data() + position,
            sizeof(int),
            cudaMemcpyHostToDevice));
        embedding_forward_cuda(
            model.decode_hidden_a,
            model.input_tokens,
            model.parameters + model.layout.token_embedding,
            1,
            model.config.hidden_size);

        float* hidden = model.decode_hidden_a;
        float* next_hidden = model.decode_hidden_b;
        for (int layer = 0; layer < model.config.layers; ++layer) {
            const DeepSeekV3BlockOffsets& offsets = model.layout.blocks[layer];
            rmsnorm_forward_cuda(
                model.decode_attention_norm,
                model.decode_attention_inverse_rms,
                hidden,
                model.parameters + offsets.attention_norm,
                1,
                model.config.hidden_size,
                model.config.rms_epsilon);
            mla_layer_decode_forward_cuda(
                model.decode_attention_output,
                model.decode_attention_norm,
                mla_parameters(model.parameters, offsets),
                model.cosine,
                model.sine,
                static_cast<int>(position),
                model.decode_kv_cache
                    + static_cast<std::size_t>(layer)
                        * model.config.sequence_length * model.config.kv_rank,
                model.decode_key_rope_cache
                    + static_cast<std::size_t>(layer)
                        * model.config.sequence_length * model.config.rope_size,
                model.decode_cache_lengths,
                model.decode_mla_workspace,
                model.decode_mla_bf16,
                model.block_config.mla,
                model.decode_splits);
            residual_forward_cuda(
                model.decode_residual,
                hidden,
                model.decode_attention_output,
                model.config.hidden_size);
            rmsnorm_forward_cuda(
                model.decode_ffn_norm,
                model.decode_ffn_inverse_rms,
                model.decode_residual,
                model.parameters + offsets.ffn_norm,
                1,
                model.config.hidden_size,
                model.config.rms_epsilon);

            if (layer < model.config.dense_layers) {
                matmul_fp32_forward_cuda(
                    model.decode_dense_gate,
                    model.decode_ffn_norm,
                    model.parameters + offsets.shared_gate,
                    1,
                    model.config.dense_ffn_size,
                    model.config.hidden_size);
                matmul_fp32_forward_cuda(
                    model.decode_dense_up,
                    model.decode_ffn_norm,
                    model.parameters + offsets.shared_up,
                    1,
                    model.config.dense_ffn_size,
                    model.config.hidden_size);
                swiglu_forward_cuda(
                    model.decode_dense_hidden,
                    model.decode_dense_gate,
                    model.decode_dense_up,
                    model.config.dense_ffn_size);
                matmul_fp32_forward_cuda(
                    model.decode_ffn_output,
                    model.decode_dense_hidden,
                    model.parameters + offsets.shared_down,
                    1,
                    model.config.hidden_size,
                    model.config.dense_ffn_size);
            } else {
                const DeepSeekV3BlockParameters parameters = block_parameters(
                    model.parameters,
                    model.routing_bias
                        + static_cast<std::size_t>(
                            layer - model.config.dense_layers)
                            * model.config.routed_experts,
                    offsets);
                deepseek_moe_forward_cuda(
                    model.decode_ffn_output,
                    model.decode_ffn_norm,
                    parameters.moe,
                    model.decode_moe_activations,
                    model.decode_moe_integer_activations,
                    model.decode_moe_config);
            }
            residual_forward_cuda(
                next_hidden,
                model.decode_residual,
                model.decode_ffn_output,
                model.config.hidden_size);
            std::swap(hidden, next_hidden);
        }

        rmsnorm_forward_cuda(
            model.decode_final_norm,
            model.decode_final_inverse_rms,
            hidden,
            model.parameters + model.layout.final_norm,
            1,
            model.config.hidden_size,
            model.config.rms_epsilon);
        matmul_fp32_strided_batched_cuda(
            model.decode_logits,
            model.decode_final_norm,
            model.parameters + model.layout.token_embedding,
            1,
            model.config.vocabulary_size,
            model.config.hidden_size,
            1,
            0,
            0,
            0,
            false,
            true,
            false);
    }

    model.cached_tokens = tokens;
    std::vector<float> result(model.config.vocabulary_size);
    CUDA_CHECK(cudaMemcpy(
        result.data(), model.decode_logits, result.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    return result;
}

const DeepSeekV3Config& DeepSeekV3Model::config() const {
    return implementation_->config;
}

const DeepSeekV3ParameterLayout& DeepSeekV3Model::parameter_layout() const {
    return implementation_->layout;
}

ModelMemoryReport DeepSeekV3Model::memory_report() const {
    const Implementation& model = *implementation_;
    const std::size_t saved =
        static_cast<std::size_t>(model.config.layers + 1)
            * model.activation_elements
        + static_cast<std::size_t>(model.config.layers)
            * model.block_activation_elements
        + model.activation_elements + model.rows + model.logits_elements
        + model.rows + 1
        + 2 * model.mtp_output_elements_total
        + model.mtp_activation_elements_total
        + 2 * model.mtp_row_elements_total
        + model.mtp_logits_elements_total
        + model.config.mtp_depth;
    const std::size_t workspace = model.logits_elements
        + 2 * model.activation_elements + model.block_workspace_elements
        + model.norm_workspace_elements + 1
        + model.mtp_output_elements_total + model.mtp_workspace_elements;
    const std::size_t float_bytes =
        (4 * model.layout.elements + saved + workspace
         + static_cast<std::size_t>(model.config.sequence_length)
             * model.config.rope_size
         + model.routing_bias_elements)
        * sizeof(float);
    const std::size_t int_bytes =
        (2 * static_cast<std::size_t>(model.rows)
         + static_cast<std::size_t>(model.config.layers)
             * model.block_integer_activation_elements
         + model.mtp_integer_elements_total + model.mtp_row_elements_total)
        * sizeof(int);
    const std::size_t bf16_bytes =
        model.bf16_workspace_elements * sizeof(__nv_bfloat16);
    std::size_t decode_bytes = 0;
    if (model.config.batch_size == 1) {
        const std::size_t decode_float_elements =
            static_cast<std::size_t>(8) * model.config.hidden_size
            + 3
            + static_cast<std::size_t>(3)
                * std::max(1, model.config.dense_ffn_size)
            + model.config.vocabulary_size
            + model.decode_mla_workspace_elements
            + model.decode_moe_activation_elements;
        const std::size_t decode_bf16_elements =
            static_cast<std::size_t>(model.config.layers)
                * model.config.sequence_length
                * (model.config.kv_rank + model.config.rope_size)
            + model.decode_mla_bf16_elements;
        const std::size_t decode_int_elements =
            1 + model.decode_moe_integer_elements;
        decode_bytes = decode_float_elements * sizeof(float)
            + decode_bf16_elements * sizeof(__nv_bfloat16)
            + decode_int_elements * sizeof(int);
    }
    return {
        model.layout.elements,
        saved,
        workspace,
        float_bytes + int_bytes + bf16_bytes + decode_bytes,
    };
}

}  // namespace dscuda

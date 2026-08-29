// Compares DeepSeek sigmoid routing, parallel no-drop dispatch, grouped linear algebra, combination, and backward maps with CPU references.
// Uneven expert loads and non-tile-aligned tensor sizes exercise the irregular paths that distinguish MoE from dense feed-forward layers.

#include "cuda_common.h"
#include "expert_dispatch.h"
#include "expert_dispatch_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int ROWS = 7;
constexpr int D = 11;
constexpr int E = 4;
constexpr int K = 2;
constexpr int O = 13;
constexpr int ROUTES = ROWS * K;
constexpr float ROUTE_SCALE = 1.25F;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements)
        : elements_(elements),
          data_(static_cast<T*>(dscuda::device_malloc(elements * sizeof(T)))) {}
    ~DeviceBuffer() { dscuda::device_free(data_); }
    T* data() { return data_; }
    void upload(const std::vector<T>& values) {
        CUDA_CHECK(cudaMemcpy(
            data_, values.data(), elements_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(elements_);
        CUDA_CHECK(cudaMemcpy(
            values.data(), data_, elements_ * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
    void zero() { CUDA_CHECK(cudaMemset(data_, 0, elements_ * sizeof(T))); }
private:
    std::size_t elements_;
    T* data_;
};

std::vector<float> make_values(int elements, float scale, float phase) {
    std::vector<float> values(elements);
    for (int index = 0; index < elements; ++index) {
        values[index] = scale *
            (std::sin(0.19F * index + phase) +
             0.3F * std::cos(0.07F * index - phase));
    }
    return values;
}

template <typename T>
bool check_exact(
    const char* name,
    const std::vector<T>& expected,
    const std::vector<T>& actual) {
    const bool passed = expected == actual;
    std::printf("  %-25s %s\n", name, passed ? "PASS" : "FAIL");
    return passed;
}

bool check_float(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance = 2.0e-5F) {
    float error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    const bool passed = error < tolerance;
    std::printf(
        "  %-25s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool check_dispatched_by_route(
    const char* name,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const std::vector<int>& expected_route_to_slot,
    const std::vector<int>& actual_route_to_slot,
    int width,
    float tolerance = 2.0e-5F) {
    float error = 0.0F;
    for (int route = 0; route < ROUTES; ++route) {
        const int expected_slot = expected_route_to_slot[route];
        const int actual_slot = actual_route_to_slot[route];
        for (int column = 0; column < width; ++column) {
            error = std::max(
                error,
                std::abs(
                    expected[expected_slot * width + column]
                    - actual[actual_slot * width + column]));
        }
    }
    const bool passed = error < tolerance;
    std::printf(
        "  %-25s max error = %.3e  %s\n",
        name,
        error,
        passed ? "PASS" : "FAIL");
    return passed;
}

bool check_dispatch_map(
    const std::vector<int>& indices,
    const std::vector<int>& route_to_slot,
    const std::vector<int>& slot_to_route,
    const std::vector<int>& slot_expert) {
    std::vector<int> seen(ROUTES);
    bool passed = true;
    for (int route = 0; route < ROUTES; ++route) {
        const int slot = route_to_slot[route];
        passed &= slot >= 0 && slot < ROUTES;
        if (slot >= 0 && slot < ROUTES) {
            passed &= seen[slot]++ == 0;
            passed &= slot_to_route[slot] == route;
            passed &= slot_expert[slot] == indices[route];
        }
    }
    std::printf("  %-25s %s\n", "parallel dispatch map", passed ? "PASS" : "FAIL");
    return passed;
}

}  // namespace

int main() {
    const auto input = make_values(ROWS * D, 0.17F, 0.1F);
    const auto logits = make_values(ROWS * E, 0.9F, 0.3F);
    const std::vector<float> bias = {0.08F, -0.03F, 0.02F, -0.01F};
    const auto weight = make_values(E * D * O, 0.12F, 0.5F);
    const auto shared = make_values(ROWS * O, 0.08F, 0.7F);
    const auto output_gradient = make_values(ROWS * O, 0.11F, 0.9F);

    std::vector<float> scores(ROWS * E);
    std::vector<int> indices(ROUTES);
    std::vector<float> route_weights(ROUTES);
    std::vector<int> counts(E);
    dscuda::expert_route_forward_cpu(
        scores.data(), indices.data(), route_weights.data(), counts.data(),
        logits.data(), bias.data(), ROWS, E, K, ROUTE_SCALE);
    std::vector<float> dispatched(ROUTES * D);
    std::vector<int> offsets(E + 1);
    std::vector<int> route_to_slot(ROUTES);
    std::vector<int> slot_to_route(ROUTES);
    std::vector<int> slot_expert(ROUTES);
    dscuda::expert_dispatch_forward_cpu(
        dispatched.data(), offsets.data(), route_to_slot.data(),
        slot_to_route.data(), slot_expert.data(), input.data(), indices.data(),
        counts.data(), ROWS, D, E, K);
    std::vector<float> grouped_output(ROUTES * O);
    dscuda::grouped_linear_forward_cpu(
        grouped_output.data(), dispatched.data(), weight.data(),
        slot_expert.data(), ROUTES, O, D);
    std::vector<float> combined(ROWS * O);
    dscuda::expert_combine_forward_cpu(
        combined.data(), shared.data(), grouped_output.data(),
        route_weights.data(), route_to_slot.data(), ROWS, O, K);

    std::vector<float> grouped_output_gradient(ROUTES * O);
    std::vector<float> route_weight_gradient(ROUTES);
    std::vector<float> shared_gradient(ROWS * O);
    dscuda::expert_combine_backward_cpu(
        grouped_output_gradient.data(), route_weight_gradient.data(),
        shared_gradient.data(), output_gradient.data(), grouped_output.data(),
        route_weights.data(), route_to_slot.data(), ROWS, O, K);
    std::vector<float> dispatched_gradient(ROUTES * D, 0.0F);
    std::vector<float> weight_gradient(E * D * O, 0.0F);
    dscuda::grouped_linear_backward_cpu(
        dispatched_gradient.data(), weight_gradient.data(),
        grouped_output_gradient.data(), dispatched.data(), weight.data(),
        offsets.data(), slot_expert.data(), ROUTES, E, O, D, false);
    std::vector<float> input_gradient(ROWS * D, 0.0F);
    dscuda::expert_unroute_backward_cpu(
        input_gradient.data(), dispatched_gradient.data(), route_to_slot.data(),
        ROWS, D, K);
    std::vector<float> logit_gradient(ROWS * E);
    dscuda::expert_route_backward_cpu(
        logit_gradient.data(), route_weight_gradient.data(), scores.data(),
        indices.data(), ROWS, E, K, ROUTE_SCALE);
    std::vector<float> updated_bias = bias;
    dscuda::update_routing_bias_cpu(
        updated_bias.data(), counts.data(), E, ROUTES, 0.01F);

    DeviceBuffer<float> g_input(input.size());
    DeviceBuffer<float> g_logits(logits.size());
    DeviceBuffer<float> g_bias(bias.size());
    DeviceBuffer<float> g_weight(weight.size());
    DeviceBuffer<float> g_shared(shared.size());
    DeviceBuffer<float> g_output_gradient(output_gradient.size());
    DeviceBuffer<float> g_scores(scores.size());
    DeviceBuffer<int> g_indices(indices.size());
    DeviceBuffer<float> g_route_weights(route_weights.size());
    DeviceBuffer<int> g_counts(counts.size());
    DeviceBuffer<float> g_dispatched(dispatched.size());
    DeviceBuffer<int> g_offsets(offsets.size());
    DeviceBuffer<int> g_route_to_slot(route_to_slot.size());
    DeviceBuffer<int> g_slot_to_route(slot_to_route.size());
    DeviceBuffer<int> g_slot_expert(slot_expert.size());
    DeviceBuffer<float> g_grouped_output(grouped_output.size());
    DeviceBuffer<float> g_combined(combined.size());
    DeviceBuffer<float> g_grouped_output_gradient(grouped_output_gradient.size());
    DeviceBuffer<float> g_route_weight_gradient(route_weight_gradient.size());
    DeviceBuffer<float> g_shared_gradient(shared_gradient.size());
    DeviceBuffer<float> g_dispatched_gradient(dispatched_gradient.size());
    DeviceBuffer<float> g_weight_gradient(weight_gradient.size());
    DeviceBuffer<float> g_input_gradient(input_gradient.size());
    DeviceBuffer<float> g_logit_gradient(logit_gradient.size());
    g_input.upload(input);
    g_logits.upload(logits);
    g_bias.upload(bias);
    g_weight.upload(weight);
    g_shared.upload(shared);
    g_output_gradient.upload(output_gradient);
    g_weight_gradient.zero();
    g_input_gradient.zero();

    dscuda::expert_route_forward_cuda(
        g_scores.data(), g_indices.data(), g_route_weights.data(),
        g_counts.data(), g_logits.data(), g_bias.data(), ROWS, E, K,
        ROUTE_SCALE);
    dscuda::expert_dispatch_forward_cuda(
        g_dispatched.data(), g_offsets.data(), g_route_to_slot.data(),
        g_slot_to_route.data(), g_slot_expert.data(), g_input.data(),
        g_indices.data(), g_counts.data(), ROWS, D, E, K);
    dscuda::grouped_linear_forward_cuda(
        g_grouped_output.data(), g_dispatched.data(), g_weight.data(),
        g_slot_expert.data(), ROUTES, O, D);
    dscuda::expert_combine_forward_cuda(
        g_combined.data(), g_shared.data(), g_grouped_output.data(),
        g_route_weights.data(), g_route_to_slot.data(), ROWS, O, K);
    dscuda::expert_combine_backward_cuda(
        g_grouped_output_gradient.data(), g_route_weight_gradient.data(),
        g_shared_gradient.data(), g_output_gradient.data(),
        g_grouped_output.data(), g_route_weights.data(),
        g_route_to_slot.data(), ROWS, O, K);
    dscuda::grouped_linear_backward_cuda(
        g_dispatched_gradient.data(), g_weight_gradient.data(),
        g_grouped_output_gradient.data(), g_dispatched.data(), g_weight.data(),
        g_offsets.data(), g_slot_expert.data(), ROUTES, E, O, D, false);
    dscuda::expert_unroute_backward_cuda(
        g_input_gradient.data(), g_dispatched_gradient.data(),
        g_route_to_slot.data(), ROWS, D, K);
    dscuda::expert_route_backward_cuda(
        g_logit_gradient.data(), g_route_weight_gradient.data(),
        g_scores.data(), g_indices.data(), ROWS, E, K, ROUTE_SCALE);
    dscuda::update_routing_bias_cuda(
        g_bias.data(), g_counts.data(), E, ROUTES, 0.01F);
    dscuda::synchronize();

    const auto gpu_route_to_slot = g_route_to_slot.download();
    const auto gpu_slot_to_route = g_slot_to_route.download();
    const auto gpu_slot_expert = g_slot_expert.download();
    const auto gpu_dispatched = g_dispatched.download();
    const auto gpu_grouped_output = g_grouped_output.download();
    const auto gpu_grouped_output_gradient =
        g_grouped_output_gradient.download();
    const auto gpu_dispatched_gradient = g_dispatched_gradient.download();

    std::printf("DeepSeek expert dispatch test\n");
    bool passed = true;
    passed &= check_float("scores", scores, g_scores.download());
    passed &= check_exact("indices", indices, g_indices.download());
    passed &= check_float("route weights", route_weights, g_route_weights.download());
    passed &= check_exact("expert counts", counts, g_counts.download());
    passed &= check_exact("expert offsets", offsets, g_offsets.download());
    passed &= check_dispatch_map(
        indices, gpu_route_to_slot, gpu_slot_to_route, gpu_slot_expert);
    passed &= check_dispatched_by_route(
        "dispatched input", dispatched, gpu_dispatched,
        route_to_slot, gpu_route_to_slot, D);
    passed &= check_dispatched_by_route(
        "grouped output", grouped_output, gpu_grouped_output,
        route_to_slot, gpu_route_to_slot, O);
    passed &= check_float("combined output", combined, g_combined.download());
    passed &= check_float(
        "route weight gradient", route_weight_gradient,
        g_route_weight_gradient.download());
    passed &= check_float("shared gradient", shared_gradient, g_shared_gradient.download());
    passed &= check_dispatched_by_route(
        "grouped output gradient", grouped_output_gradient,
        gpu_grouped_output_gradient, route_to_slot, gpu_route_to_slot, O);
    passed &= check_dispatched_by_route(
        "dispatched gradient", dispatched_gradient,
        gpu_dispatched_gradient, route_to_slot, gpu_route_to_slot, D);
    passed &= check_float("weight gradient", weight_gradient, g_weight_gradient.download());
    passed &= check_float("unrouted gradient", input_gradient, g_input_gradient.download());
    passed &= check_float("router logit gradient", logit_gradient, g_logit_gradient.download());
    passed &= check_float("updated routing bias", updated_bias, g_bias.download());
    return passed ? 0 : 1;
}

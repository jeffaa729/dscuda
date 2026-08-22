// Implements interleaved FP32 Rotary Positional Embeddings and their inverse-rotation gradient on the CPU.
// The reference rotates adjacent scalar pairs and copies the non-rotary suffix without optimization.

#include "rope_cpu.h"

namespace dscuda {

void rope_forward_cpu(
    float* output,
    const float* input,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size) {
    const int rotary_pairs = rotary_size / 2;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int position = 0; position < sequence_length; ++position) {
            for (int head = 0; head < heads; ++head) {
                const int head_offset =
                    ((batch * sequence_length + position) * heads + head) * head_size;
                const int frequency_offset = position * rotary_pairs;

                for (int pair = 0; pair < rotary_pairs; ++pair) {
                    const int column = pair * 2;
                    const float first = input[head_offset + column];
                    const float second = input[head_offset + column + 1];
                    const float cosine_value = cosine[frequency_offset + pair];
                    const float sine_value = sine[frequency_offset + pair];
                    output[head_offset + column] =
                        first * cosine_value - second * sine_value;
                    output[head_offset + column + 1] =
                        first * sine_value + second * cosine_value;
                }

                for (int column = rotary_size; column < head_size; ++column) {
                    output[head_offset + column] = input[head_offset + column];
                }
            }
        }
    }
}

void rope_backward_cpu(
    float* input_gradient,
    const float* output_gradient,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size) {
    const int rotary_pairs = rotary_size / 2;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int position = 0; position < sequence_length; ++position) {
            for (int head = 0; head < heads; ++head) {
                const int head_offset =
                    ((batch * sequence_length + position) * heads + head) * head_size;
                const int frequency_offset = position * rotary_pairs;

                for (int pair = 0; pair < rotary_pairs; ++pair) {
                    const int column = pair * 2;
                    const float first_gradient = output_gradient[head_offset + column];
                    const float second_gradient = output_gradient[head_offset + column + 1];
                    const float cosine_value = cosine[frequency_offset + pair];
                    const float sine_value = sine[frequency_offset + pair];
                    input_gradient[head_offset + column] +=
                        first_gradient * cosine_value + second_gradient * sine_value;
                    input_gradient[head_offset + column + 1] +=
                        -first_gradient * sine_value + second_gradient * cosine_value;
                }

                for (int column = rotary_size; column < head_size; ++column) {
                    input_gradient[head_offset + column] += output_gradient[head_offset + column];
                }
            }
        }
    }
}

}  // namespace dscuda

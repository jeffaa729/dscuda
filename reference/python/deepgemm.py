"""Thin adapter for the official DeepGEMM grouped BF16 NN interface."""

import deep_gemm


def grouped_alignment(expected_rows):
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(
        expected_rows)
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    return alignment


def grouped_gemm_nn(inputs, weights, output, grouped_layout):
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
        inputs, weights, output, grouped_layout)
    return output

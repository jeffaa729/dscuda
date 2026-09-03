"""Thin adapters for the official DeepGEMM BF16 NN interfaces."""

import deep_gemm


def gemm_nn(left, right, output):
    deep_gemm.bf16_gemm_nn(left, right, output)
    return output


def grouped_alignment(expected_rows):
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(
        expected_rows)
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    return alignment


def grouped_gemm_nn(inputs, weights, output, grouped_layout):
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
        inputs, weights, output, grouped_layout)
    return output

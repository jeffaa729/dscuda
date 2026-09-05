"""Packed BF16 expert NN GEMMs with cuBLAS-loop and DeepGEMM references."""

import ctypes
import importlib

from common import I, P, Operation, bind, checked, library, pointers, stream, torch
from reference.python.moe import grouped_gemm


def expert_counts(rows, experts, distribution, alignment=1):
    if rows % alignment:
        raise ValueError("rows must be divisible by the expert alignment")
    units = rows // alignment
    if distribution == "uniform":
        counts = [units // experts] * experts
    elif distribution == "hot":
        counts = [units * 4 // 5] + [
            units // (5 * (experts - 1))] * (experts - 1)
    else:
        counts = [0, units // 3, 0] + [
            (units - units // 3) // (experts - 3)] * (experts - 3)
    counts[-1] += units - sum(counts)
    return [count * alignment for count in counts]


def cases(args, family):
    reference = args.reference or ("both" if args.suite == "h100" else "cublas")
    if reference not in ("cublas", "deepgemm", "both"):
        raise ValueError("Grouped GEMM references: cublas, deepgemm, or both")
    use_cublas = reference in ("cublas", "both")
    use_deepgemm = reference in ("deepgemm", "both")
    deepgemm = importlib.import_module("reference.python.deepgemm") if use_deepgemm else None

    lib = library("operator")
    if use_cublas:
        checked(lib, "operator", bind(lib, "dscuda_cublas_init", [])())
    gemm = bind(lib, "dscuda_grouped_gemm", [P] * 5 + [I] * 5 + [P])
    test_shapes = (
        ((512, 4, 128, 128), (1024, 8, 256, 256)) if use_deepgemm else
        ((64, 5, 32, 48), (129, 8, 37, 49)))
    shapes = test_shapes if args.test else (
        ((512, 8, 256, 512),) if args.suite == "quick" else
        ((4096, 8, 512, 1536), (8192, 16, 512, 1536)))

    try:
        for m, e, k, n in shapes:
            alignment = deepgemm.grouped_alignment(m // e) if use_deepgemm else 1
            for distribution in ("uniform", "hot", "empty"):
                counts = expert_counts(m, e, distribution, alignment)
                offsets = [0]
                for count in counts:
                    offsets.append(offsets[-1] + count)
                host_offsets = (I * len(offsets))(*offsets)
                device_offsets = torch.tensor(
                    offsets, device="cuda", dtype=torch.int32)
                inputs = torch.randn(
                    m, k, device="cuda", dtype=torch.bfloat16) * .1
                weights = torch.randn(
                    e, k, n, device="cuda", dtype=torch.bfloat16) * .1
                expected = grouped_gemm(
                    inputs, weights, tuple(offsets)).bfloat16()
                custom_output = torch.empty(
                    m, n, device="cuda", dtype=torch.bfloat16)

                def native(output, use_reference):
                    checked(lib, "operator", gemm(
                        *pointers((output, inputs, weights, device_offsets)),
                        ctypes.cast(host_offsets, P),
                        m, e, n, k, use_reference, stream()))
                    return output

                functions = {
                    "custom": lambda output=custom_output: native(output, 0)
                }
                if use_cublas:
                    cublas_output = torch.empty_like(custom_output)
                    functions["cuBLAS loop"] = (
                        lambda output=cublas_output: native(output, 1))
                if use_deepgemm:
                    grouped_layout = torch.repeat_interleave(
                        torch.arange(e, device="cuda", dtype=torch.int32),
                        torch.tensor(counts, device="cuda"),
                        output_size=m)
                    deepgemm_output = torch.empty_like(custom_output)
                    functions["DeepGEMM"] = (
                        lambda output=deepgemm_output:
                            deepgemm.grouped_gemm_nn(
                                inputs, weights, output, grouped_layout))

                size = f"M={m},N={n},K={k},E={e},{distribution}"
                yield Operation(
                    size, "bf16", "NN", functions, (expected,),
                    1e-2, 1e-2)
    finally:
        if use_cublas:
            bind(lib, "dscuda_cublas_destroy", [], None)()

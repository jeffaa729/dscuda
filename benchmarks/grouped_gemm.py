"""Packed BF16 expert NN GEMMs: PyTorch correctness and cuBLAS-loop timing."""

import ctypes

from common import I, P, Operation, bind, checked, library, pointers, stream, torch
from reference.python.moe import grouped_gemm


def expert_counts(rows, experts, distribution):
    if distribution == "uniform":
        counts = [rows // experts] * experts
    elif distribution == "hot":
        counts = [rows * 4 // 5] + [rows // (5 * (experts - 1))] * (experts - 1)
    else:
        counts = [0, rows // 3, 0] + [(rows - rows // 3) // (experts - 3)] * (experts - 3)
    counts[-1] += rows - sum(counts)
    return counts


def cases(args, family):
    if args.reference not in (None, "cublas"):
        raise ValueError("Grouped GEMM uses cuBLAS as its only benchmark reference")
    lib = library("operator")
    checked(lib, "operator", bind(lib, "dscuda_cublas_init", [])())
    gemm = bind(lib, "dscuda_grouped_gemm", [P] * 5 + [I] * 5 + [P])
    shapes = ((64, 5, 32, 48), (129, 8, 37, 49)) if args.test else (
        ((512, 8, 256, 512),) if args.suite == "quick" else
        ((4096, 8, 512, 1536), (8192, 16, 512, 1536)))
    try:
        for m, e, k, n in shapes:
            for distribution in ("uniform", "hot", "empty"):
                counts = expert_counts(m, e, distribution)
                offsets = [0]
                for count in counts:
                    offsets.append(offsets[-1] + count)
                host_offsets = (I * len(offsets))(*offsets)
                device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * .1
                w = torch.randn(e, k, n, device="cuda", dtype=torch.bfloat16) * .1
                outputs = [torch.empty(m, n, device="cuda") for _ in range(2)]

                def pytorch():
                    return grouped_gemm(x, w, tuple(offsets))

                expected = pytorch()

                def call(ref=0):
                    checked(lib, "operator", gemm(
                        *pointers((outputs[ref], x, w, device_offsets)),
                        ctypes.cast(host_offsets, P), m, e, n, k, ref, stream()))
                    return outputs[ref]

                size = f"M={m},N={n},K={k},E={e},{distribution}"
                yield Operation(
                    size, "bf16", "NN",
                    {"custom": call, "cuBLAS loop": lambda: call(1)},
                    (expected.detach(),))
    finally:
        bind(lib, "dscuda_cublas_destroy", [], None)()

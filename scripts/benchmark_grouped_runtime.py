#!/usr/bin/env python3
"""Packed BF16 grouped GEMM versus a cuBLAS loop, both with FP32 output."""

import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_operators_runtime import checked, load_library, measure

from reference.python.moe import grouped_gemm


class GroupedWorkload:
    def __init__(self, torch, library, counts, k, n, distribution):
        self.torch, self.lib = torch, library
        p, i = ctypes.c_void_p, ctypes.c_int
        library.dscuda_grouped_gemm.argtypes = [p] * 6 + [i] * 6 + [p]
        library.dscuda_grouped_gemm.restype = i
        self.rows, self.experts, self.k, self.n = sum(counts), len(counts), k, n
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        self.host_offsets = (ctypes.c_int * len(offsets))(*offsets)
        self.offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        self.slot_expert = torch.repeat_interleave(
            torch.arange(len(counts), device="cuda", dtype=torch.int32),
            torch.tensor(counts, device="cuda"),
        )
        self.x = torch.randn(self.rows, k, device="cuda", dtype=torch.bfloat16) * 0.1
        self.w = (
            torch.randn(self.experts, k, n, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        self.initial = torch.zeros(self.rows, n, device="cuda")
        self.buffers = {name: self.initial.clone() for name in ("custom", "reference")}
        self.expected = grouped_gemm(self.x, self.w, tuple(offsets))
        self.stream = torch.cuda.current_stream().cuda_stream
        self.dtype, self.operation, self.reference = "bf16", "NN", "cuBLAS loop"
        self.size = f"M={self.rows},N={n},K={k},E={self.experts},{distribution}"

    def reset(self, name):
        self.buffers[name].zero_()

    def run(self, name):
        checked(
            self.lib,
            self.lib.dscuda_grouped_gemm(
                self.buffers[name].data_ptr(),
                self.x.data_ptr(),
                self.w.data_ptr(),
                self.offsets.data_ptr(),
                self.slot_expert.data_ptr(),
                ctypes.cast(self.host_offsets, ctypes.c_void_p),
                self.rows,
                self.experts,
                self.n,
                self.k,
                1,
                int(name == "reference"),
                self.stream,
            ),
        )
        return (self.buffers[name],)

    def check(self, graphs=None):
        errors = []
        for name in self.buffers:
            # Poison every output to reject unwritten expert rows on eager AND graph paths.
            self.buffers[name].fill_(float("nan"))
            if graphs is None:
                self.run(name)
            else:
                graphs[name].graph.replay()
            self.torch.testing.assert_close(
                self.buffers[name], self.expected, atol=2e-4, rtol=2e-3
            )
            first = self.buffers[name].clone()
            self.run(name)
            self.torch.testing.assert_close(self.buffers[name], first, atol=0, rtol=0)
            errors.append((self.buffers[name] - self.expected).abs().max().item())
        return errors


def counts_for(rows, experts, distribution):
    if distribution == "uniform":
        counts = [rows // experts] * experts
    elif distribution == "hot":
        counts = [rows * 4 // 5] + [rows // (5 * (experts - 1))] * (experts - 1)
    elif distribution == "empty":
        counts = [0, rows // 3, 0] + [(rows - rows // 3) // (experts - 3)] * (
            experts - 3
        )
    else:
        raise ValueError(distribution)
    counts[-1] += rows - sum(counts)
    return counts


def run(torch, args):
    lib = load_library(args.library)
    checked(lib, lib.dscuda_cublas_init())
    results, checks = [], []
    try:
        shapes = (
            [(512, 8, 256, 512)]
            if args.suite == "quick"
            else [(4096, 8, 512, 1536), (8192, 16, 512, 1536)]
        )
        for m, e, k, n in shapes:
            for distribution in ("uniform", "hot", "empty"):
                workload = GroupedWorkload(
                    torch, lib, counts_for(m, e, distribution), k, n, distribution
                )
                if args.check_only:
                    errors = workload.check()
                else:
                    rows, errors = measure(workload, args)
                    results.extend(rows)
                checks.append(
                    dict(
                        size=workload.size,
                        errors=errors,
                        counts=counts_for(m, e, distribution),
                    )
                )
    finally:
        lib.dscuda_cublas_destroy()
    return results, checks

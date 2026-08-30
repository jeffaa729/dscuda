#!/usr/bin/env python3
"""Same-process CUDA Graph comparisons: GEMM/cuBLAS and AdamW/PyTorch equations."""

import argparse
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from benchmark_flash_attention_runtime import (
    Captured, add_reference_percentages, format_percentage, positive, summarize, table)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))
from adamw import adamw_step

BACKENDS = ("custom", "reference")


def cases(family, suite):
    if family == "matmul":
        sizes = (2048,) if suite == "quick" else (2048, 4096, 8192)
        return [(size, dtype, operation) for size in sizes for dtype in ("fp32", "bf16")
                for operation in ("forward", "left_backward", "right_backward")]
    return [(size, "fp32", "update") for size in
            ((1 << 22,) if suite == "quick" else (1 << 18, 1 << 22, 1 << 24))]


def load_library(path):
    lib = ctypes.CDLL(str(path.resolve()))
    pointer, integer, scalar = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    lib.dscuda_operator_last_error.restype = ctypes.c_char_p
    lib.dscuda_cublas_init.argtypes = []
    lib.dscuda_cublas_init.restype = integer
    lib.dscuda_cublas_destroy.argtypes = []
    lib.dscuda_cublas_destroy.restype = None
    lib.dscuda_cublas_version.argtypes = []
    lib.dscuda_cublas_version.restype = integer
    lib.dscuda_gemm.argtypes = [pointer] * 3 + [integer] * 8 + [pointer]
    lib.dscuda_gemm.restype = integer
    lib.dscuda_adamw.argtypes = [pointer] * 4 + [integer] * 2 + [scalar] * 5 + [pointer]
    lib.dscuda_adamw.restype = integer
    return lib


def checked(lib, status):
    if status:
        raise RuntimeError(lib.dscuda_operator_last_error().decode())


class Workload:
    def __init__(self, torch, lib, family, size, dtype, operation):
        self.torch, self.lib = torch, lib
        self.family, self.size, self.dtype, self.operation = family, size, dtype, operation
        self.stream = torch.cuda.current_stream().cuda_stream
        self.reference = "cuBLAS" if family == "matmul" else "pytorch_unfused"
        torch.manual_seed(2026)
        if family == "matmul":
            precision = torch.bfloat16 if dtype == "bf16" else torch.float32
            self.left = torch.randn((size, size), device="cuda", dtype=precision) * 0.1
            self.right = torch.randn_like(self.left) * 0.1
            # Backward is a GEMM with the indicated transpose and += contract.
            self.transpose_left = operation == "right_backward"
            self.transpose_right = operation == "left_backward"
            self.initial = (torch.zeros((size, size), device="cuda", dtype=torch.float32),)
        else:
            self.gradient = torch.randn(size, device="cuda") * 0.01
            self.initial = (torch.randn_like(self.gradient),
                            torch.randn_like(self.gradient) * 0.001,
                            torch.rand_like(self.gradient) * 0.01 + 0.001)
            # Quantize scalar constants to FP32 on both sides, including beta.
            self.config = tuple(ctypes.c_float(v).value for v in (3e-4, .9, .95, 1e-8, .1))
            self.step = 100
        self.buffers = {name: tuple(t.clone() for t in self.initial) for name in BACKENDS}

    def reset(self, name):
        for output, initial in zip(self.buffers[name], self.initial):
            output.copy_(initial)

    def run(self, name):
        outputs = self.buffers[name]
        if self.family == "matmul":
            checked(self.lib, self.lib.dscuda_gemm(
                outputs[0].data_ptr(), self.left.data_ptr(), self.right.data_ptr(),
                self.size, self.size, self.size, int(self.dtype == "bf16"),
                int(self.transpose_left), int(self.transpose_right), int(self.operation != "forward"),
                int(name == "reference"), self.stream))
        elif name == "custom":
            checked(self.lib, self.lib.dscuda_adamw(
                *(t.data_ptr() for t in outputs), self.gradient.data_ptr(),
                self.size, self.step, *self.config, self.stream))
        else:
            adamw_step(*outputs, self.gradient, self.step, *self.config)
        return outputs

    def check(self, graphs=None):
        # Repeated execution checks accumulation/state updates, not only one call.
        for name in BACKENDS:
            self.reset(name)
            if graphs is None:
                self.run(name)
                self.run(name)
            else:
                graphs[name].graph.replay()
        checks = []
        for actual, expected in zip(self.buffers["custom"], self.buffers["reference"]):
            if not bool(self.torch.isfinite(actual).all() & self.torch.isfinite(expected).all()):
                raise AssertionError("non-finite comparison output")
            self.torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
            checks.append((actual - expected).abs().max().item())
        return checks


def measure(workload, args):
    torch = workload.torch
    precheck = workload.check()
    graphs = {name: Captured(torch, lambda name=name: workload.run(name), args.graph_operations)
              for name in BACKENDS}
    graph_check = workload.check(graphs)
    deadline = time.perf_counter() + args.warmup_ms / 1000
    while time.perf_counter() < deadline:
        for name in BACKENDS:
            workload.reset(name)
            graphs[name].graph.replay()
        torch.cuda.current_stream().synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {name: [] for name in BACKENDS}
    for trial in range(args.trials):
        for name in (BACKENDS if trial % 2 == 0 else BACKENDS[::-1]):
            workload.reset(name)
            # Reset is ordered before the start event and excluded from timing.
            start.record()
            for _ in range(args.graph_replays):
                graphs[name].graph.replay()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) /
                                 (args.graph_replays * args.graph_operations))
    postcheck = workload.check(graphs)
    rows = []
    for name in BACKENDS:
        summary = summarize(samples[name])
        rows.append(dict(size=workload.size, dtype=workload.dtype, operation=workload.operation,
                         backend=name, reference=workload.reference, **summary))
    return rows, dict(before=precheck, graph=graph_check, after=postcheck)


def result_table(rows):
    add_reference_percentages(rows, "reference", ("size", "dtype", "operation", "reference"))
    headers = ("size", "dtype", "operation", "backend", "median us", "reference %")
    values = [(r["size"], r["dtype"], r["operation"],
               r["reference"] if r["backend"] == "reference" else "custom",
               f'{1000*r["median_ms"]:.2f}', format_percentage(r["reference_pct"]))
              for r in rows]
    return table(headers, values, {0, 4, 5})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("matmul", "adamw"))
    parser.add_argument("--suite", choices=("quick", "full", "h100"), default="quick")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiles" / "runtime")
    parser.add_argument("--graph-operations", type=positive, default=10)
    parser.add_argument("--graph-replays", type=positive, default=3)
    parser.add_argument("--warmup-ms", type=positive, default=1000)
    parser.add_argument("--trials", type=positive, default=9)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    lib = load_library(args.library)
    checked(lib, lib.dscuda_cublas_init())
    cublas_version = lib.dscuda_cublas_version()
    rows, checks = [], []
    # Keep capture off the default stream, as in the FlashAttention comparison.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.inference_mode(), torch.cuda.stream(stream):
            for size, dtype, operation in cases(args.family, args.suite):
                workload = Workload(torch, lib, args.family, size, dtype, operation)
                if args.check_only:
                    errors = workload.check()
                else:
                    results, errors = measure(workload, args)
                    rows.extend(results)
                checks.append(dict(size=size, dtype=dtype, operation=operation, errors=errors))
                del workload
            stream.synchronize()
    finally:
        lib.dscuda_cublas_destroy()
    if args.check_only:
        return
    report = result_table(rows)
    print(report, end="", flush=True)
    note = ("FP32 disables TF32; BF16 inputs accumulate and output FP32. "
            "Backward GEMMs accumulate FP32 gradients." if args.family == "matmul" else
            "FP32 fixed-step AdamW update (step=100), versus unfused PyTorch equations, "
            "not torch.optim.AdamW(fused=True). Both replay the same fixed bias correction; "
            "state is reset outside timing. This is not an optimizer training trajectory.")
    note += ("\nWarm-cache CUDA Graph measurements; allocations, state resets, and graph construction "
             "are excluded.\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ("size", "dtype", "operation", "backend", "reference", "median_ms", "reference_pct")
    with (args.output_dir / f"{args.family}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / f"{args.family}.md").write_text(report)
    metadata = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    cublas=cublas_version,
                    cuda=torch.version.cuda, python=sys.version, contract=note,
                    git_commit=subprocess.check_output(
                        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                    library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                    reference_sha256=hashlib.sha256(
                        (ROOT / "reference/python/adamw.py").read_bytes()).hexdigest()
                        if args.family == "adamw" else None,
                    controls={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output_dir / f"{args.family}_samples.json").write_text(
        json.dumps(dict(environment=metadata, checks=checks, results=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()

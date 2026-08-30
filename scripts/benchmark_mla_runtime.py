#!/usr/bin/env python3
"""Compressed MLA: custom CUDA versus unfused PyTorch, with matched GPU Graph timing."""

import argparse
from dataclasses import asdict, dataclass
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from benchmark_flash_attention_runtime import (
    Captured, add_reference_percentages, comparison_table, positive, summarize)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/python"))
from mla import mla_forward, mla_backward

BACKENDS = ("custom", "pytorch")
FLASHMLA_SOURCE = "https://github.com/deepseek-ai/FlashMLA#requirements"
FLASHMLA_STATUS = "not_implemented"
OUTPUT_NAMES = ("output", "LSE")
GRADIENT_NAMES = ("dQ_latent", "dQ_rope", "dKV_latent", "dK_rope")


@dataclass(frozen=True)
class Case:
    mode: str
    batch: int
    sequence: int
    heads: int
    rank: int = 512
    rope: int = 64
    splits: int = 8
    lengths: tuple = ()

    def __post_init__(self):
        if self.mode not in ("prefill", "decode"):
            raise ValueError("mode must be prefill or decode")
        if min(self.batch, self.sequence, self.heads, self.rank, self.rope, self.splits) < 1:
            raise ValueError("shape dimensions and splits must be positive")
        if (self.rank, self.rope) != (512, 64):
            raise ValueError("MLA requires C=512 and RoPE=64")
        if self.lengths and (len(self.lengths) != self.batch
                             or any(n < 1 or n > self.sequence for n in self.lengths)):
            raise ValueError("one cache length in [1, sequence] is required per batch")

    @property
    def cache_lengths(self):
        return self.lengths or tuple(max(1, self.sequence - b * self.sequence // (2 * self.batch))
                                     for b in range(self.batch))

    @property
    def operations(self):
        return ("forward", "backward") if self.mode == "prefill" else ("decode",)


def benchmark_cases(suite):
    quick = (Case("prefill", 1, 128, 8), Case("decode", 2, 1024, 16))
    if suite == "quick":
        return quick
    return (quick[0], Case("prefill", 2, 257, 4), Case("prefill", 1, 512, 8),
            Case("decode", 1, 256, 16), quick[1], Case("decode", 4, 4096, 32))


def correctness_cases():
    return (Case("prefill", 1, 1, 1), Case("prefill", 2, 17, 3),
            Case("prefill", 1, 65, 4),
            Case("decode", 2, 23, 3, lengths=(1, 23)),
            Case("decode", 3, 129, 5, lengths=(129, 7, 64)))


def load_library(path):
    lib = ctypes.CDLL(str(path.resolve()))
    pointer, integer, scalar = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    lib.dscuda_mla_last_error.restype = ctypes.c_char_p
    lib.dscuda_mla_forward.argtypes = [pointer] * 6 + [integer] * 5 + [scalar, pointer]
    lib.dscuda_mla_backward.argtypes = [pointer] * 11 + [integer] * 5 + [scalar, integer, pointer]
    lib.dscuda_mla_decode.argtypes = [pointer] * 8 + [integer] * 6 + [scalar, pointer]
    for name in ("forward", "backward", "decode"):
        getattr(lib, "dscuda_mla_" + name).restype = integer
    lib.dscuda_mla_workspace_elements.argtypes = [integer] * 4
    lib.dscuda_mla_workspace_elements.restype = ctypes.c_size_t
    return lib


class Workload:
    def __init__(self, torch, lib, case):
        self.torch, self.lib, self.case = torch, lib, case
        self.stream = torch.cuda.current_stream().cuda_stream
        b, t, h, c, r = case.batch, case.sequence, case.heads, case.rank, case.rope
        q = t if case.mode == "prefill" else 1
        torch.manual_seed(2026)
        def operand(shape):
            return (torch.randn(shape, device="cuda") * .25).to(torch.bfloat16)
        self.inputs = (operand((b, q, h, c)), operand((b, q, h, r)),
                       operand((b, t, c)), operand((b, t, r)))
        self.scale = ctypes.c_float((c + r)**-.5).value
        self.shape = (b, t, h, c, r)
        self.output = torch.empty((b, q, h, c), device="cuda")
        self.lse = torch.empty((b, h, q), device="cuda")
        if case.mode == "prefill":
            self.mask = torch.ones((t, t), dtype=torch.bool, device="cuda").tril()
            self.dout = torch.randn_like(self.output) * .2
            self.gradients = tuple(torch.empty_like(x, dtype=torch.float32) for x in self.inputs)
        else:
            self.lengths = torch.tensor(case.cache_lengths, dtype=torch.int32, device="cuda")
            self.mask = torch.arange(t, device="cuda")[None, :] < self.lengths[:, None]
            self.mask = self.mask[:, None, None, :]
            count = lib.dscuda_mla_workspace_elements(b, h, case.splits, c)
            self.workspace = torch.empty(count, device="cuda", dtype=torch.float32)
            # Finite poison in padding detects accidental use of invalid cache positions.
            for batch, length in enumerate(case.cache_lengths):
                self.inputs[2][batch, length:].fill_(100)
                self.inputs[3][batch, length:].fill_(-100)
        # Both backward implementations consume exactly the same saved output/LSE.
        self.saved = self.reference_forward()
        self.expected = {"forward" if case.mode == "prefill" else "decode": self.saved}
        if case.mode == "prefill":
            self.expected["backward"] = self.reference_backward()

    def checked(self, status):
        if status:
            raise RuntimeError(self.lib.dscuda_mla_last_error().decode())

    def reference_forward(self):
        return mla_forward(*self.inputs, self.scale, self.mask)

    def reference_backward(self):
        return mla_backward(self.dout, *self.saved, *self.inputs, self.scale, self.mask)

    def custom_forward(self):
        pointers = tuple(x.data_ptr() for x in (self.output, self.lse, *self.inputs))
        if self.case.mode == "prefill":
            self.checked(self.lib.dscuda_mla_forward(
                *pointers, *self.shape, self.scale, self.stream))
        else:
            self.checked(self.lib.dscuda_mla_decode(
                *pointers, self.lengths.data_ptr(), self.workspace.data_ptr(),
                *self.shape, self.case.splits, self.scale, self.stream))
        return self.output, self.lse

    def custom_backward(self, accumulate=False):
        self.checked(self.lib.dscuda_mla_backward(
            *(x.data_ptr() for x in (*self.gradients, self.dout, *self.saved, *self.inputs)),
            *self.shape, self.scale, int(accumulate), self.stream))
        return self.gradients

    def function(self, backend, operation):
        if operation == "backward":
            return self.custom_backward if backend == "custom" else self.reference_backward
        return self.custom_forward if backend == "custom" else self.reference_forward

    def compare(self, actual, operation):
        names = GRADIENT_NAMES if operation == "backward" else OUTPUT_NAMES
        checks = []
        for name, output, expected in zip(names, actual, self.expected[operation]):
            if output.dtype != self.torch.float32 or output.shape != expected.shape:
                raise AssertionError(f"{name}: output dtype/shape mismatch")
            if not bool(self.torch.isfinite(output).all() & self.torch.isfinite(expected).all()):
                raise AssertionError(f"{name}: non-finite output")
            atol, rtol = (2e-5, 1e-5) if name == "LSE" else (2e-4, 2e-3)
            self.torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
            difference = output - expected
            checks.append(dict(tensor=name, max_abs=difference.abs().max().item(),
                               rms=difference.square().mean().sqrt().item(),
                               atol=atol, rtol=rtol))
        return checks

    def check(self, operation):
        destinations = self.gradients if operation == "backward" else (self.output, self.lse)
        for tensor in destinations:
            tensor.fill_(float("nan"))
        first = self.function("custom", operation)()
        checks = self.compare(first, operation)
        previous = tuple(x.clone() for x in first)
        second = self.function("custom", operation)()
        for a, b in zip(second, previous):
            self.torch.testing.assert_close(a, b, atol=0, rtol=0)
        if operation == "backward":
            common_saved = self.saved
            self.saved = self.custom_forward()
            self.compare(self.custom_backward(), operation)
            self.saved = common_saved
            for tensor in self.gradients:
                tensor.fill_(1)
            self.custom_backward(accumulate=True)
            for actual, expected in zip(self.gradients, self.expected["backward"]):
                self.torch.testing.assert_close(actual, expected + 1, atol=2e-4, rtol=2e-3)
            self.custom_backward()
        return checks


def verify_graph(workload, graph, operation):
    for tensor in graph.result:
        tensor.fill_(float("nan"))
    graph.graph.replay()
    return workload.compare(graph.result, operation)


def check_extreme_inputs(torch, lib):
    # Large logits exercise online max rescaling; zero logits give uniform P.
    # These checks are separate from, and never part of, the timed workload.
    for factor in (16, 0):
        workload = Workload(torch, lib, Case("prefill", 2, 33, 3))
        for tensor in workload.inputs:
            tensor.mul_(factor)
        workload.saved = workload.reference_forward()
        workload.expected = {"forward": workload.saved, "backward": workload.reference_backward()}
        for operation in workload.case.operations:
            workload.check(operation)
        original = workload.custom_forward()[0].clone()
        workload.inputs[2][:, 17:].fill_(100)
        workload.inputs[3][:, 17:].fill_(-100)
        changed = workload.custom_forward()[0]
        torch.testing.assert_close(changed[:, :17], original[:, :17], atol=0, rtol=0)


def measure(workload, operation, args):
    torch = workload.torch
    checks = {"before": workload.check(operation)}
    graphs = {name: Captured(torch, workload.function(name, operation), args.graph_operations)
              for name in BACKENDS}
    for name, graph in graphs.items():
        checks[name + "_graph"] = verify_graph(workload, graph, operation)
    deadline = time.perf_counter() + args.warmup_ms / 1000
    while time.perf_counter() < deadline:
        for graph in graphs.values():
            graph.graph.replay()
        torch.cuda.current_stream().synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {name: [] for name in BACKENDS}
    replays = {}
    for name in BACKENDS:
        start.record()
        for _ in range(args.graph_replays):
            graphs[name].graph.replay()
        stop.record()
        stop.synchronize()
        graph_ms = start.elapsed_time(stop) / args.graph_replays
        replays[name] = max(args.graph_replays, math.ceil(args.min_sample_ms / graph_ms))
    for trial in range(args.trials):
        for name in (BACKENDS if trial % 2 == 0 else BACKENDS[::-1]):
            start.record()
            for _ in range(replays[name]):
                graphs[name].graph.replay()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) /
                                 (args.graph_operations * replays[name]))
    for name, graph in graphs.items():
        checks[name + "_after"] = verify_graph(workload, graph, operation)
    rows = []
    for name in BACKENDS:
        summary = summarize(samples[name])
        dimensions = asdict(workload.case)
        dimensions["lengths"] = workload.case.cache_lengths if workload.case.mode == "decode" else ()
        rows.append(dict(**dimensions, operation=operation, backend=name,
                         graph_operations=args.graph_operations, graph_replays=replays[name],
                         **summary))
    return rows, checks


def result_table(rows):
    add_reference_percentages(rows, "pytorch",
                              ("mode", "batch", "sequence", "heads", "rank", "rope", "lengths", "splits", "operation"))
    def size(row):
        query = row["sequence"] if row["mode"] == "prefill" else 1
        shape = (f'B={row["batch"]},Q={query},KV={row["sequence"]},H={row["heads"]},'
                 f'C={row["rank"]},RoPE={row["rope"]}')
        if row["mode"] == "decode":
            shape += f',lengths=({",".join(map(str, row["lengths"]))}),splits={row["splits"]}'
        return shape

    return comparison_table(
        rows, ("mode", "batch", "sequence", "heads", "rank", "rope", "lengths", "splits", "operation"),
        size, lambda r: "bf16", "PyTorch unfused", reference_backend="pytorch")


def write_results(args, torch, rows, checks):
    report = result_table(rows)
    contract = (
        "Absorbed-query MQA: C=512, RoPE=64, one shared KV latent per token; QK width=576, V width=512. "
        "BF16 input storage; FP32 output, natural-log LSE, output gradient and input gradients. "
        "Causal square prefill; single-query decode attends positive per-batch cache lengths. "
        "Decode uses contiguous split latent/RoPE caches, not FlashMLA's paged layout. "
        "PyTorch is an unfused materialized FP32 reference with BF16 upcasts INSIDE each measured call; TF32 is disabled. "
        "C512/R64 forward uses BF16 Tensor Cores with FP32 accumulation and two-part BF16 softmax weights. "
        "Backward and decode use SIMT FP32 arithmetic. There is no alternate CUDA forward path. "
        "Backward overwrites gradients and recomputes probabilities; common saved output/LSE and masks are prepared outside timing. "
        "Graph construction and warmup are excluded; these are not API latency measurements. "
        "Replay counts are calibrated per backend to a minimum sample duration, then normalized per operation.")
    files = ("kernels/attention/mla.cu", "benchmarks/bindings/mla_bridge.cu",
             "reference/python/mla.py", "reference/python/flashmla.py",
             "scripts/benchmark_mla_runtime.py")
    metadata = dict(gpu=torch.cuda.get_device_name(),
                    compute_capability=torch.cuda.get_device_capability(),
                    torch=torch.__version__, cuda=torch.version.cuda, python=sys.version,
                    flashmla=dict(status=FLASHMLA_STATUS, source=FLASHMLA_SOURCE,
                                  reason="No SM80 support; dense decoding is SM90-only"),
                    contract=contract, seed=2026,
                    library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                    sources={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
                    git_commit=subprocess.check_output(
                        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                    controls={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "mla.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("mode", "batch", "sequence", "heads", "rank",
                                                "rope", "splits", "lengths", "operation", "backend", "median_ms", "reference_pct"),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "mla.md").write_text(report)
    (args.output_dir / "mla_samples.json").write_text(
        json.dumps(dict(environment=metadata, checks=checks, results=rows), indent=2) + "\n")
    print(report, end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--suite", choices=("quick", "full", "h100"), default="quick")
    parser.add_argument("--mode", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--reference", choices=("pytorch", "flashmla"), default="pytorch")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiles/runtime")
    parser.add_argument("--graph-operations", type=positive, default=5)
    parser.add_argument("--graph-replays", type=positive, default=3)
    parser.add_argument("--min-sample-ms", type=positive, default=50)
    parser.add_argument("--warmup-ms", type=positive, default=1000)
    parser.add_argument("--trials", type=positive, default=9)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.reference == "flashmla":
        parser.error("FlashMLA adapter is intentionally unimplemented; SM80 is unsupported. Use --reference pytorch.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    lib = load_library(args.library)
    rows, checks = [], []
    cases = benchmark_cases(args.suite)
    if args.check_only:
        cases = correctness_cases() + cases
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        for case in cases:
            if args.mode != "all" and case.mode != args.mode:
                continue
            workload = Workload(torch, lib, case)
            for operation in case.operations:
                if args.check_only:
                    check = workload.check(operation)
                else:
                    result, check = measure(workload, operation, args)
                    rows.extend(result)
                checks.append(dict(case=asdict(case), operation=operation, checks=check))
            del workload
        if args.check_only:
            check_extreme_inputs(torch, lib)
        stream.synchronize()
    if not args.check_only:
        write_results(args, torch, rows, checks)


if __name__ == "__main__":
    main()

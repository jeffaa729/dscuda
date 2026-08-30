#!/usr/bin/env python3

"""Same-process BF16 FlashAttention comparison: GPU graphs first, API wall time second."""

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


QUICK_CASES = ((1, 512, 8, 128),)
BACKENDS = ("custom", "official")


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--suite", choices=("quick", "full", "h100"), default="quick")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--graph-operations", type=positive, default=100)
    parser.add_argument("--graph-replays", type=positive, default=10)
    parser.add_argument("--warmup-ms", type=positive, default=1000)
    parser.add_argument("--iterations", type=positive, default=50)
    parser.add_argument("--trials", type=positive, default=9)
    return parser.parse_args()


def benchmark_cases(suite):
    if suite == "quick":
        return QUICK_CASES
    return tuple(
        (batch, sequence, heads, 128)
        for batch in (1, 4)
        for sequence in (128, 256, 512, 1024, 2048)
        for heads in (4, 8)
    )


def load_library(path):
    library = ctypes.CDLL(str(path.resolve()))
    suffix = [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
    library.dscuda_flash_forward.argtypes = [ctypes.c_void_p] * 5 + suffix
    library.dscuda_flash_backward.argtypes = [ctypes.c_void_p] * 9 + suffix
    library.dscuda_flash_forward.restype = ctypes.c_int
    library.dscuda_flash_backward.restype = ctypes.c_int
    library.dscuda_flash_last_error.restype = ctypes.c_char_p
    return library


class Workload:
    def __init__(self, torch, library, shape):
        from flash_attn import flash_attn_func
        self.torch, self.library, self.flash = torch, library, flash_attn_func
        self.shape = shape
        batch, sequence, heads, dimension = shape
        self.scale = 1.0 / math.sqrt(dimension)
        generator = torch.Generator(device="cuda").manual_seed(2026)

        def tensor():
            return (torch.randn(shape, generator=generator, device="cuda") * 0.5).to(torch.bfloat16)

        self.inputs = tuple(tensor().requires_grad_(True) for _ in range(3))
        self.dout = tensor()
        self.output = torch.empty_like(self.inputs[0])
        self.lse = torch.empty((batch, heads, sequence), device="cuda", dtype=torch.float32)
        self.gradients = tuple(torch.empty_like(x) for x in self.inputs)
        dimensions = (*shape, self.scale)
        self.forward_args = (self.output.data_ptr(), self.lse.data_ptr(),
                             *(x.data_ptr() for x in self.inputs), *dimensions)
        self.backward_args = (*(x.data_ptr() for x in self.gradients),
                              self.dout.data_ptr(), self.output.data_ptr(), self.lse.data_ptr(),
                              *(x.data_ptr() for x in self.inputs), *dimensions)
        # This forward runs on the same non-default stream later used for capture,
        # so autograd backward never has to cross into the legacy default stream.
        self.official_output, self.official_lse, _ = self.flash(
            *self.inputs, dropout_p=0.0, softmax_scale=self.scale,
            causal=True, return_attn_probs=True)
        self.custom_forward()
        self.reference_gradients = self.official_backward()
        self.reference = {
            "forward": (self.official_output.detach(), self.official_lse.detach()),
            "backward": tuple(x.detach() for x in self.reference_gradients),
        }

    def call(self, function, args):
        status = function(*args, self.torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(self.library.dscuda_flash_last_error().decode())

    def custom_forward(self):
        self.call(self.library.dscuda_flash_forward, self.forward_args)
        return self.output, self.lse

    def custom_backward(self):
        self.call(self.library.dscuda_flash_backward, self.backward_args)
        return self.gradients

    def official_forward(self):
        with self.torch.no_grad():
            output, lse, _ = self.flash(
                *self.inputs, dropout_p=0.0, softmax_scale=self.scale,
                causal=True, return_attn_probs=True)
        return output, lse

    def official_backward(self):
        return self.torch.autograd.grad(
            self.official_output, self.inputs, self.dout, retain_graph=True)

    def operation(self, backend, operation):
        return getattr(self, f"{backend}_{operation}")

    def compare(self, actual, expected, name):
        torch = self.torch
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"{name}: dtype/shape mismatch")
        if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
            raise AssertionError(f"{name}: non-finite values")
        atol, rtol = (1.0e-4, 1.0e-5) if name == "LSE" else (1.0e-2, 1.0e-2)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        error = actual.float() - expected.float()
        return {"tensor": name, "max_abs": error.abs().max().item(),
                "rms": error.square().mean().sqrt().item(), "atol": atol, "rtol": rtol}

    def check(self):
        # Poison gradients to reject a partial write or accidental += semantics.
        for tensor in self.gradients:
            tensor.fill_(float("nan"))
        results = (*self.custom_forward(), *self.custom_backward())
        expected = (*self.reference["forward"], *self.reference["backward"])
        names = ("output", "LSE", "dQ", "dK", "dV")
        checks = [self.compare(x, y, name) for x, y, name in zip(results, expected, names)]
        first = tuple(x.clone() for x in self.gradients)
        self.custom_backward()
        for actual, previous in zip(self.gradients, first):
            self.torch.testing.assert_close(actual, previous, atol=0, rtol=0)
        print(table(("tensor", "max abs", "RMS", "result"),
                    [(c["tensor"], f'{c["max_abs"]:.3e}', f'{c["rms"]:.3e}', "PASS")
                     for c in checks], {1, 2}), flush=True)
        return checks


class Captured:
    def __init__(self, torch, function, operations):
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.current_stream()
        for _ in range(10):
            function()
        stream.synchronize()
        with torch.cuda.graph(self.graph, stream=stream):
            for _ in range(operations):
                result = function()
        # Keep the captured output allocations alive throughout every replay.
        self.result = result
        self.graph.replay()
        stream.synchronize()

    def verify(self, workload, operation):
        with workload.torch.no_grad():
            for tensor in self.result:
                tensor.fill_(float("nan"))
        self.graph.replay()
        names = ("output", "LSE") if operation == "forward" else ("dQ", "dK", "dV")
        for actual, expected, name in zip(self.result, workload.reference[operation], names):
            workload.compare(actual, expected, name)


def percentile(samples, fraction):
    ordered = sorted(samples)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    median = statistics.median(samples)
    q25, q75 = percentile(samples, 0.25), percentile(samples, 0.75)
    return {"median_ms": median, "minimum_ms": min(samples), "maximum_ms": max(samples),
            "iqr_pct": 100 * (q75 - q25) / median, "samples_ms": list(samples)}


def measure_operation(workload, operation, args):
    torch = workload.torch
    functions = {name: workload.operation(name, operation) for name in BACKENDS}
    graphs = {name: Captured(torch, function, args.graph_operations)
              for name, function in functions.items()}
    for graph in graphs.values():
        graph.verify(workload, operation)

    # Warm both implementations together for time, not just a handful of calls.
    deadline = time.perf_counter() + args.warmup_ms / 1000
    while time.perf_counter() < deadline:
        for graph in graphs.values():
            graph.graph.replay()
        torch.cuda.current_stream().synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = {mode: {name: [] for name in BACKENDS} for mode in ("graph", "api")}

    def graph_sample(name):
        start.record()
        for _ in range(args.graph_replays):
            graphs[name].graph.replay()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) / (args.graph_replays * args.graph_operations)

    # Alternating AB/BA trials avoid measuring one backend only on a cold GPU.
    for trial in range(args.trials):
        order = BACKENDS if trial % 2 == 0 else BACKENDS[::-1]
        for name in order:
            samples["graph"][name].append(graph_sample(name))
    for trial in range(args.trials):
        order = BACKENDS if trial % 2 == 0 else BACKENDS[::-1]
        for name in order:
            torch.cuda.current_stream().synchronize()
            begin = time.perf_counter()
            for _ in range(args.iterations):
                functions[name]()
            torch.cuda.current_stream().synchronize()
            samples["api"][name].append((time.perf_counter() - begin) * 1000 / args.iterations)

    # Verify after long replay as well, especially the backward overwrite contract.
    for graph in graphs.values():
        graph.verify(workload, operation)
    rows = []
    for mode in ("graph", "api"):
        for name in BACKENDS:
            row = dict(zip(("batch", "sequence", "heads", "head_size"), workload.shape))
            row.update(mode=mode, backend=name, operation=operation, **summarize(samples[mode][name]))
            rows.append(row)
    return rows


def attention_flops(row):
    pairs = row["batch"] * row["heads"] * row["sequence"] * (row["sequence"] + 1) // 2
    return (4 if row["operation"] == "forward" else 8) * pairs * row["head_size"]


def api_bytes(row):
    elements = row["batch"] * row["sequence"] * row["heads"] * row["head_size"]
    lse_bytes = 4 * row["batch"] * row["heads"] * row["sequence"]
    return (8 if row["operation"] == "forward" else 16) * elements + lse_bytes


def row_key(row):
    return tuple(row[key] for key in ("batch", "sequence", "heads", "head_size", "operation", "mode"))


def add_derived_metrics(rows):
    official = {row_key(row): row["median_ms"] for row in rows if row["backend"] == "official"}
    for row in rows:
        row["tflops"] = attention_flops(row) / (row["median_ms"] * 1.0e9)
        row["io_gb_s"] = api_bytes(row) / (row["median_ms"] * 1.0e6)
        row["relative_pct"] = official[row_key(row)] / row["median_ms"] * 100


def table(headers, rows, numeric):
    widths = [max(3, len(h), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
    def line(values):
        return "|" + "|".join(
            f" {str(value):{('>' if i in numeric else '<')}{width}} "
            for i, (value, width) in enumerate(zip(values, widths))) + "|"
    separator = "|" + "|".join(
        "-" * (width + 1) + ":" if i in numeric else ":" + "-" * (width + 1)
        for i, width in enumerate(widths)) + "|"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)]) + "\n"


def result_table(rows, mode):
    headers = ("B", "T", "H", "D", "backend", "operation", "median us", "min us",
               "max us", "IQR %", "TFLOP/s", "min IO GB/s", "official %")
    values = [(r["batch"], r["sequence"], r["heads"], r["head_size"], r["backend"],
               r["operation"], f'{1000*r["median_ms"]:.2f}', f'{1000*r["minimum_ms"]:.2f}',
               f'{1000*r["maximum_ms"]:.2f}', f'{r["iqr_pct"]:.1f}',
               f'{r["tflops"]:.2f}', f'{r["io_gb_s"]:.2f}', f'{r["relative_pct"]:.1f}%')
              for r in rows if r["mode"] == mode]
    return table(headers, values, set(range(4)) | set(range(6, len(headers))))


def write_results(directory, rows, checks, args):
    directory.mkdir(parents=True, exist_ok=True)
    fields = [key for key in rows[0] if key != "samples_ms"]
    with (directory / "flash_attention.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (directory / "flash_attention_samples.json").write_text(
        json.dumps({"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "correctness": checks, "measurements": rows}, indent=2) + "\n")
    text = """# FlashAttention matched BF16 comparison

Both implementations run in one process on the same GPU stream, reading the same
contiguous BF16 Q/K/V and upstream gradient. Output and all three gradients are
BF16; log-sum-exp and accumulation are FP32. Both overwrite gradient outputs.
Attention is causal MHA with D=128, scale=1/sqrt(D), dropout=0, no ALiBi/window/softcap.
Output, LSE, gradients, overwrite behavior and replay outputs are checked before
accepting timings. Backward-only measurements exclude forward setup.

## Primary: CUDA Graph GPU operator time

Graphs capture repeated full forward or backward operations, including all kernels
and GPU scratch-buffer work required by each implementation. Replay reuses captured
allocations; capture, per-operation Python dispatch and host allocations are outside
timing. The remaining graph-replay launch overhead is amortized across captured operations.
No cast or gradient-zeroing kernels are silently excluded. These are warm, repeated-input
operator measurements, not cold-cache inference or end-to-end training throughput.

"""
    text += result_table(rows, "graph")
    text += """
## Secondary: Python API wall time

This is synchronized wall time per call averaged over a loop: custom ctypes bridge
with caller-owned buffers versus official flash_attn_func / torch.autograd.grad.
It includes Python, allocation bookkeeping and dispatch; it is NOT a CUDA kernel
speedup. Forward disables autograd recording but still computes O and LSE. Backward
uses a previously built autograd graph. Buffer-ownership differences are intentional
in this API-level table and absent as repeated host costs during graph replay.

"""
    text += result_table(rows, "api")
    text += """
Each value is the median of alternating custom/official trials. IQR is the
interquartile range divided by the median; graph rows above 10% are flagged as
unstable and should not be used for performance claims. Raw samples and correctness
errors are saved in flash_attention_samples.json.

"official %" means official latency / backend latency * 100 within the SAME timing
mode and shape. TFLOP/s counts useful causal matmul work (4 per pair in forward,
8 in backward); recomputation and softmax are excluded from that FLOP count.
"min IO GB/s" uses the same minimum tensor-byte count for both backends. It is NOT
measured DRAM bandwidth, peak-memory usage or hardware utilization.
"""
    unstable = [r for r in rows if r["mode"] == "graph" and r["iqr_pct"] > 10]
    if unstable:
        warning = "WARNING: Unstable graph rows (IQR > 10%; do not use for performance claims): " + ", ".join(
            f'{r["backend"]}/{r["operation"]}/B{r["batch"]}T{r["sequence"]}H{r["heads"]}'
            for r in unstable)
        text += "\n" + warning + "\n"
        print(warning, flush=True)
    (directory / "flash_attention.md").write_text(text)
    print("\nCUDA Graph GPU operator time\n" + result_table(rows, "graph"), flush=True)
    print("Python API wall time\n" + result_table(rows, "api"), flush=True)


def write_environment(directory, args, torch):
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()
    def digest(path):
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(source.read()).hexdigest()
    try:
        gpu = subprocess.check_output(
            ("nvidia-smi", "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.draw",
             "--format=csv,noheader"), text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = "nvidia-smi telemetry unavailable"
    content = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "telemetry_after_run": gpu, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "flash_attn": importlib.metadata.version("flash-attn"), "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "library_sha256": digest(args.library),
        "runner_sha256": digest(Path(__file__)),
        "kernel_sha256": digest(root / "kernels/attention/flash_attention.cu"),
        "settings": vars(args),
    }
    cache = args.library.parent / "CMakeCache.txt"
    if cache.exists():
        content["custom_build"] = [line for line in cache.read_text().splitlines()
                                   if line.startswith(("CMAKE_CUDA_COMPILER:", "CMAKE_CUDA_ARCHITECTURES:",
                                                       "CMAKE_BUILD_TYPE:", "CMAKE_CUDA_FLAGS:"))]
    (directory / "flash_attention_environment.md").write_text(
        "# FlashAttention benchmark environment\n\n" +
        "\n".join(f"- {key}: {value}" for key, value in content.items()) + "\n")


def main():
    args = arguments()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    library = load_library(args.library)
    print(f"GPU: {torch.cuda.get_device_name()}; matched BF16 IO, D128, causal", flush=True)
    rows, checks = [], []
    stream = torch.cuda.Stream()
    for shape in benchmark_cases(args.suite):
        print(f"\nChecking B={shape[0]}, T={shape[1]}, H={shape[2]}, D={shape[3]}", flush=True)
        with torch.cuda.stream(stream):
            workload = Workload(torch, library, shape)
            checks.append({"shape": shape, "errors": workload.check()})
            for operation in ("forward", "backward"):
                print(f"Capturing and measuring {operation}", flush=True)
                rows.extend(measure_operation(workload, operation, args))
        stream.synchronize()
        del workload
        gc.collect()
    add_derived_metrics(rows)
    write_results(args.output_dir, rows, checks, args)
    write_environment(args.output_dir, args, torch)
    print(f"\nReport: {args.output_dir / 'flash_attention.md'}", flush=True)


if __name__ == "__main__":
    main()

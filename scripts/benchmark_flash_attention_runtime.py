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


def add_reference_percentages(rows, reference, keys, time_field="median_ms"):
    """Reference time / backend time * 100; match shape, operation and timing mode."""
    def key(row):
        return tuple(tuple(row[k]) if isinstance(row[k], (list, tuple)) else row[k] for k in keys)

    times = {(row["backend"], key(row)): float(row[time_field]) for row in rows}
    for row in rows:
        backend = reference.get(row["backend"]) if isinstance(reference, dict) else reference
        reference_time = times.get((backend, key(row)))
        elapsed = float(row[time_field])
        row["reference_pct"] = (
            100 * reference_time / elapsed
            if reference_time is not None and math.isfinite(reference_time) and reference_time > 0
            and math.isfinite(elapsed) and elapsed > 0 else None)


def format_percentage(value):
    return "-" if value is None else f"{value:.1f}"


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


def comparison_table(rows, keys, size, dtype, reference, *,
                     custom_backend="custom", reference_backend="reference",
                     time_field="median_ms", time_scale=1000.0):
    """Pair identical workloads without changing the raw per-backend records."""
    groups = {}
    for row in rows:
        backend = row["backend"]
        if backend not in (custom_backend, reference_backend):
            raise ValueError(f"unexpected comparison backend: {backend}")
        key = tuple(tuple(row[k]) if isinstance(row[k], (list, tuple)) else row[k] for k in keys)
        group = groups.setdefault(key, {})
        if backend in group:
            raise ValueError(f"duplicate measurement for {backend}: {key}")
        group[backend] = row

    def elapsed(row):
        value = float(row[time_field]) * time_scale if row is not None else math.nan
        return value if math.isfinite(value) and value > 0 else None

    def number(value):
        return "-" if value is None else f"{value:.2f}"

    values = []
    for group in groups.values():
        row = next(iter(group.values()))
        custom = elapsed(group.get(custom_backend))
        baseline = elapsed(group.get(reference_backend))
        percent = 100 * baseline / custom if custom is not None and baseline is not None else None
        values.append((size(row), dtype(row), row["operation"],
                       number(custom), number(baseline), format_percentage(percent)))
    return table(("size", "dtype", "operation", "custom us", f"reference ({reference}) us",
                  "reference %"), values, {0, 3, 4, 5})


def result_table(rows, mode):
    add_reference_percentages(rows, "official",
                              ("batch", "sequence", "heads", "head_size", "operation", "mode"))
    return comparison_table(
        [r for r in rows if r["mode"] == mode],
        ("batch", "sequence", "heads", "head_size", "operation", "mode"),
        lambda r: f'B={r["batch"]},T={r["sequence"]},H={r["heads"]},D={r["head_size"]}',
        lambda r: "bf16", "FA-2", reference_backend="official")


def write_results(directory, rows, checks, args):
    directory.mkdir(parents=True, exist_ok=True)
    report = ("CUDA Graph GPU time\n\n" + result_table(rows, "graph")
              + "\nPython API wall time\n\n" + result_table(rows, "api"))
    fields = ("batch", "sequence", "heads", "head_size", "mode", "backend", "operation", "median_ms", "reference_pct")
    with (directory / "flash_attention.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (directory / "flash_attention_samples.json").write_text(
        json.dumps({"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "contract": {"input_dtype": "bf16", "output_and_gradient_dtype": "bf16",
                                 "lse_dtype": "fp32", "layout": "BTHD", "causal": True,
                                 "dropout": 0, "softmax_scale": "1/sqrt(D)",
                                 "backward": "overwrite, forward setup excluded",
                                 "graph": "CUDA events, warm repeated-input graph replay",
                                 "api": "synchronized wall time, includes Python dispatch and allocations"},
                    "correctness": checks, "measurements": rows}, indent=2) + "\n")
    (directory / "flash_attention.md").write_text(report)
    print(report, end="", flush=True)


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
    rows, checks = [], []
    stream = torch.cuda.Stream()
    for shape in benchmark_cases(args.suite):
        with torch.cuda.stream(stream):
            workload = Workload(torch, library, shape)
            checks.append({"shape": shape, "errors": workload.check()})
            for operation in ("forward", "backward"):
                rows.extend(measure_operation(workload, operation, args))
        stream.synchronize()
        del workload
        gc.collect()
    write_results(args.output_dir, rows, checks, args)
    write_environment(args.output_dir, args, torch)


if __name__ == "__main__":
    main()

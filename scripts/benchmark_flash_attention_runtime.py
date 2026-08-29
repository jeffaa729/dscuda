#!/usr/bin/env python3

"""Compares custom and official FlashAttention with portable CUDA-event timing."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile


TIMING_PREFIX = "DSCUDA_TIMING "
QUICK_CASES = ((1, 512, 8, 64), (1, 512, 8, 128))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", required=True, type=Path)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--compare", required=True, type=Path)
    parser.add_argument("--suite", choices=("quick", "full", "h100"), default="quick")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    return parser.parse_args()


def benchmark_cases(suite):
    if suite == "quick":
        return QUICK_CASES
    return tuple(
        (batch, sequence, heads, head_size)
        for batch in (1, 4)
        for sequence in (128, 256, 512, 1024, 2048)
        for heads in (4, 8)
        for head_size in (64, 128)
    )


def run(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout


def timing_record(output):
    for line in output.splitlines():
        if line.startswith(TIMING_PREFIX):
            return json.loads(line[len(TIMING_PREFIX):])
    raise RuntimeError("timing result was not emitted")


def attention_flops(row):
    pairs = (
        row["batch"] * row["heads"] * row["sequence"]
        * (row["sequence"] + 1) // 2
    )
    matmul_factor = 4 if row["operation"] == "forward" else 8
    return matmul_factor * pairs * row["head_size"]


def api_bytes(row):
    activations = (
        row["batch"] * row["sequence"] * row["heads"] * row["head_size"]
    )
    rows = row["batch"] * row["sequence"] * row["heads"]
    if row["backend"] == "custom":
        return (
            10 * activations + 4 * rows
            if row["operation"] == "forward"
            else 26 * activations + 4 * rows
        )
    return 8 * activations if row["operation"] == "forward" else 16 * activations


def add_derived_metrics(rows):
    official_times = {
        (
            row["batch"],
            row["sequence"],
            row["heads"],
            row["head_size"],
            row["operation"],
        ): row["median_ms"]
        for row in rows
        if row["backend"] == "official"
    }
    for row in rows:
        key = (
            row["batch"],
            row["sequence"],
            row["heads"],
            row["head_size"],
            row["operation"],
        )
        row["tflops"] = attention_flops(row) / (row["median_ms"] * 1.0e9)
        row["api_gb_s"] = api_bytes(row) / (row["median_ms"] * 1.0e6)
        row["relative_pct"] = official_times[key] / row["median_ms"] * 100.0


def write_csv(path, rows):
    fields = (
        "batch",
        "sequence",
        "heads",
        "head_size",
        "backend",
        "operation",
        "median_ms",
        "minimum_ms",
        "maximum_ms",
        "tflops",
        "api_gb_s",
        "relative_pct",
        "warmup",
        "iterations",
        "trials",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    with path.open("w", encoding="utf-8") as destination:
        destination.write("# FlashAttention CUDA-event runtime\n\n")
        destination.write(
            "All listed shapes passed the custom-versus-official output and gradient "
            "comparison before timing. Each value is the median trial average.\n\n"
        )
        destination.write(
            "| B | T | H | D | backend | operation | median ms | min ms | max ms | "
            "TFLOP/s | API GB/s | official % |\n"
        )
        destination.write(
            "|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            destination.write(
                f"| {row['batch']} | {row['sequence']} | {row['heads']} | "
                f"{row['head_size']} | {row['backend']} | {row['operation']} | "
                f"{row['median_ms']:.4f} | {row['minimum_ms']:.4f} | "
                f"{row['maximum_ms']:.4f} | {row['tflops']:.2f} | "
                f"{row['api_gb_s']:.2f} | {row['relative_pct']:.1f}% |\n"
            )
        destination.write(
            "\n`official %` is official latency divided by backend latency; official is "
            "therefore 100%. TFLOP/s counts causal QK/PV matmuls in forward and their "
            "four matmul gradients in backward; softmax arithmetic is omitted. `API GB/s` "
            "uses minimum visible tensor bytes for each backend and is not measured DRAM "
            "traffic.\n"
        )


def write_environment(path, args, torch):
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    capability = torch.cuda.get_device_capability()
    with path.open("w", encoding="utf-8") as destination:
        destination.write("# FlashAttention runtime environment\n\n")
        destination.write(f"- GPU: {torch.cuda.get_device_name()}\n")
        destination.write(f"- CUDA capability: {capability[0]}.{capability[1]}\n")
        destination.write(f"- PyTorch: {torch.__version__}\n")
        destination.write(f"- PyTorch CUDA: {torch.version.cuda}\n")
        destination.write(
            f"- flash-attn: {importlib.metadata.version('flash-attn')}\n"
        )
        destination.write(f"- dscuda commit: {commit}\n")
        destination.write(f"- Shape suite: {args.suite}\n")
        destination.write(f"- Warm-up operations: {args.warmup}\n")
        destination.write(f"- Operations per trial: {args.iterations}\n")
        destination.write(f"- Trials: {args.trials}\n")
        destination.write("- Timer: CUDA events\n")


def main():
    args = arguments()
    import torch
    from flash_attn import flash_attn_func  # noqa: F401

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="dscuda_attention_") as temporary:
        temporary = Path(temporary)
        for batch, sequence, heads, head_size in benchmark_cases(args.suite):
            shape = (str(batch), str(sequence), str(heads), str(head_size))
            print(
                f"\nChecking B={batch},T={sequence},H={heads},D={head_size}",
                flush=True,
            )
            custom_dump = temporary / "custom.bin"
            official_dump = temporary / "official.bin"
            run((str(args.custom), *shape, "all", str(custom_dump)))
            run((
                sys.executable,
                str(args.official),
                *shape,
                "all",
                str(official_dump),
            ))
            run((
                sys.executable,
                str(args.compare),
                str(custom_dump),
                str(official_dump),
                *shape,
            ))

            for operation in ("forward", "backward"):
                timing_flags = (
                    "--timing",
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--trials",
                    str(args.trials),
                )
                print(f"Timing custom {operation}", flush=True)
                custom_output = run((
                    str(args.custom),
                    *shape,
                    operation,
                    *timing_flags,
                ))
                rows.append(timing_record(custom_output))

                print(f"Timing official {operation}", flush=True)
                official_output = run((
                    sys.executable,
                    str(args.official),
                    *shape,
                    operation,
                    *timing_flags,
                ))
                rows.append(timing_record(official_output))

    add_derived_metrics(rows)
    write_csv(args.output_dir / "flash_attention.csv", rows)
    write_markdown(args.output_dir / "flash_attention.md", rows)
    write_environment(args.output_dir / "flash_attention_environment.md", args, torch)
    print(f"\nRuntime table: {args.output_dir / 'flash_attention.md'}", flush=True)


if __name__ == "__main__":
    main()

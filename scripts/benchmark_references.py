#!/usr/bin/env python3
"""Reference-first CUDA Graph timings; no custom performance claim without a kernel."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_flash_attention_runtime import (
    Captured,
    comparison_table,
    positive,
    summarize,
)


def attention_stages(torch, family, tokens):
    from reference.python.attention import sparse_attention
    from reference.python.compression import compress
    from reference.python.csa import csa_forward
    from reference.python.dsa import dsa_forward, index_scores, select_indices
    from reference.python.hca import hca_forward

    b, h, d, hi, di = 1, 4, 512, 4, 128
    rand = lambda *s: (
        torch.randn(*s, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    qi, weights = rand(b, tokens, hi, di), rand(b, tokens, hi)
    q = rand(b, tokens, h, 576 if family == "dsa" else d)
    local = rand(b, tokens, q.shape[-1])
    positions = torch.arange(tokens, device="cuda")
    if family == "dsa":
        ki = rand(b, tokens, di)
        visible = positions[None] <= positions[:, None]
        indexer = lambda: (select_indices(index_scores(qi, ki, weights, visible), 64),)
        ids = indexer()[0]
        attention = lambda: sparse_attention(q, local, local[..., :512], ids)
        stages = dict(
            indexer=indexer,
            attention=attention,
            pipeline=lambda: dsa_forward(q, local, qi, ki, weights, 64)[:2],
        )
        differentiable = (q, local)
        size = f"B={b},Q={tokens},KV={tokens},H={h},C=512,R=64,Hi={hi},Di={di},topk=64"
    else:
        ratio = 128 if family == "hca" else 4
        overlap = family == "csa"
        channels = d * (2 if overlap else 1)
        # Compressor projections/gates accumulate in FP32; final compressed cache stores BF16.
        values, gates = (
            rand(b, tokens, channels).float(),
            rand(b, tokens, channels).float(),
        )
        bias = torch.zeros(ratio, channels, device="cuda")
        norm = torch.ones(d, device="cuda")
        compress_main = lambda: compress(
            values,
            gates,
            bias,
            ratio,
            norm,
            overlap=overlap,
            storage_dtype=torch.bfloat16,
        )
        compressed = compress_main().detach().requires_grad_()
        sink = torch.zeros(h, device="cuda", requires_grad=True)
        if family == "hca":
            attention = lambda: hca_forward(q, local, compressed, sink=sink)
            stages = dict(
                compression=lambda: (compress_main(),),
                attention=attention,
                pipeline=lambda: hca_forward(q, local, compress_main(), sink=sink),
            )
        else:
            iv, ig = rand(b, tokens, 2 * di).float(), rand(b, tokens, 2 * di).float()
            ib, iw = (
                torch.zeros(ratio, 2 * di, device="cuda"),
                torch.ones(di, device="cuda"),
            )
            compress_index = lambda: compress(
                iv, ig, ib, ratio, iw, overlap=True, storage_dtype=torch.bfloat16
            )
            ik = compress_index().detach()
            visible = (
                torch.arange(tokens // ratio, device="cuda")[None] + 1
            ) * ratio <= positions[:, None] + 1
            indexer = lambda: (
                select_indices(index_scores(qi, ik, weights, visible), 32),
            )
            selected = indexer()[0]
            from reference.python.compression import local_indices

            local_ids = local_indices(positions, 128)[None]
            ids = torch.cat(
                (local_ids, torch.where(selected >= 0, selected + tokens, -1)), -1
            )

            def attention():
                cache = torch.cat((local, compressed), 1)
                return sparse_attention(q, cache, cache, ids, sink=sink)

            stages = dict(
                compression=lambda: (compress_main(), compress_index()),
                indexer=indexer,
                attention=attention,
                pipeline=lambda: csa_forward(
                    q,
                    local,
                    compress_main(),
                    qi,
                    compress_index(),
                    weights,
                    32,
                    sink=sink,
                )[:2],
            )
        differentiable = (q, local, compressed, sink)
        size = f"B={b},Q={tokens},KV={tokens},H={h},D={d},ratio={ratio},window=128"
        if family == "csa":
            size += f",Hi={hi},Di={di},topk=32"
    # Fixed saved forward and fixed discrete selection; no indexer gradient is implied.
    saved = attention()[0]
    dout = torch.randn_like(saved)
    stages["backward"] = lambda: torch.autograd.grad(
        saved, differentiable, dout, retain_graph=True
    )
    return size, stages


def kda_stages(torch, tokens, backend):
    from reference.python.kda import fla_forward, kda_forward

    b, h, k, v = 1, 4, 128, 128
    normalize = lambda: (
        torch.nn.functional.normalize(
            torch.randn(b, tokens, h, k, device="cuda"), dim=-1
        )
        .bfloat16()
        .requires_grad_()
    )
    q, key = normalize(), normalize()
    value = (
        torch.randn(b, tokens, h, v, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    g = (-torch.rand(b, tokens, h, k, device="cuda") * 0.1).requires_grad_()
    beta = torch.rand(b, tokens, h, device="cuda", requires_grad=True)
    initial = torch.zeros(b, h, k, v, device="cuda", requires_grad=True)
    inputs = (q, key, value, g, beta, initial)
    function = fla_forward if backend == "fla" else kda_forward
    forward = lambda: function(*inputs)
    saved = forward()
    # Validate the official path at the ACTUAL benchmark shape, including final-state loss.
    expected = kda_forward(*inputs)
    for a, e in zip(saved, expected):
        torch.testing.assert_close(a.float(), e.float(), atol=1e-3, rtol=2e-2)
    dout = (torch.randn_like(saved[0]), torch.randn_like(saved[1]) * 0.1)
    gradients = torch.autograd.grad(saved, inputs, dout, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, inputs, dout, retain_graph=True)
    for a, e in zip(gradients, expected_gradients):
        torch.testing.assert_close(a.float(), e.float(), atol=3e-3, rtol=4e-2)
    stages = dict(
        forward=forward,
        backward=lambda: torch.autograd.grad(saved, inputs, dout, retain_graph=True),
    )
    return f"B={b},T={tokens},H={h},K={k},V={v}", stages


def measure_reference(torch, function, args):
    expected = tuple(x.detach().clone() for x in function())
    captured = Captured(torch, function, args.graph_operations)

    def check():
        captured.graph.replay()
        for actual, reference in zip(captured.result, expected):
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)

    check()
    deadline = time.perf_counter() + args.warmup_ms / 1000
    while time.perf_counter() < deadline:
        captured.graph.replay()
        torch.cuda.current_stream().synchronize()
    start, stop = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    def elapsed(replays):
        start.record()
        for _ in range(replays):
            captured.graph.replay()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop)

    replays = args.graph_replays
    while replays < 4096 and elapsed(replays) < 5.0:
        replays *= 2
    samples = [
        elapsed(replays) / (replays * args.graph_operations) for _ in range(args.trials)
    ]
    check()
    return dict(**summarize(samples), graph_replays=replays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("grouped_gemm", "hca", "dsa", "csa", "kda"))
    parser.add_argument("--suite", choices=("quick", "full", "h100"), default="quick")
    parser.add_argument("--backend", choices=("pytorch", "fla", "cublas"))
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/libdscuda_operator_bench.so"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiles/runtime")
    parser.add_argument("--graph-operations", type=positive, default=1)
    parser.add_argument("--graph-replays", type=positive, default=3)
    parser.add_argument("--warmup-ms", type=positive, default=1000)
    parser.add_argument("--trials", type=positive, default=9)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    backend = args.backend or (
        "cublas"
        if args.family == "grouped_gemm"
        else "fla"
        if args.family == "kda"
        else "pytorch"
    )
    if (backend == "fla" and args.family != "kda") or (backend == "cublas") != (
        args.family == "grouped_gemm"
    ):
        parser.error(
            "cuBLAS is for grouped_gemm; FLA is for KDA; other families use PyTorch"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    rows, checks = [], []
    label = {"pytorch": "PyTorch", "fla": "FLA", "cublas": "cuBLAS loop"}[backend]
    with torch.cuda.stream(stream):
        if args.family == "grouped_gemm":
            from benchmark_grouped_runtime import run

            rows, checks = run(torch, args)
        else:
            for tokens in (128,) if args.suite == "quick" else (256, 512, 1024):
                size, stages = (
                    kda_stages(torch, tokens, backend)
                    if args.family == "kda"
                    else attention_stages(torch, args.family, tokens)
                )
                for operation, function in stages.items():
                    if args.check_only:
                        result = function()
                        if any(torch.isnan(x).any().item() for x in result):
                            raise AssertionError(f"{operation}: NaN output")
                    else:
                        timing = measure_reference(torch, function, args)
                        rows.append(
                            dict(
                                size=size,
                                dtype="bf16",
                                operation=operation,
                                backend="reference",
                                reference=label,
                                **timing,
                            )
                        )
                    checks.append(
                        dict(
                            size=size,
                            operation=operation,
                            graph_checked=not args.check_only,
                        )
                    )
                del stages
        stream.synchronize()
    if args.check_only:
        return
    report = comparison_table(
        rows,
        ("size", "dtype", "operation"),
        lambda r: r["size"],
        lambda r: r["dtype"],
        label,
    )
    print(report, end="")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.family if args.family == "grouped_gemm" else args.family + "_reference"
    # Emit the actual backend used and raw samples; missing custom time stays missing.
    for row in rows:
        match = next(
            (
                r
                for r in rows
                if r["size"] == row["size"]
                and r["operation"] == row["operation"]
                and r["backend"] == "reference"
            ),
            None,
        )
        row["reference_pct"] = (
            100 * match["median_ms"] / row["median_ms"]
            if row["backend"] == "custom" and match
            else None
        )
    with (args.output_dir / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "size",
                "dtype",
                "operation",
                "backend",
                "reference",
                "median_ms",
                "reference_pct",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / f"{stem}.md").write_text(report)
    metadata = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        backend=backend,
        fla=importlib.metadata.version("fla-core"),
        git_commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        sources={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "reference/python").glob("*.py")
        },
        contract="Warm-cache CUDA Graph GPU time. Prepared projection outputs; BF16 attention storage, FP32 math. No quantization. Pipeline includes compression/selection, not model projections or RoPE. Attention backward uses saved forward/fixed indices. KDA has prepared gates and includes final-state gradients. PyTorch upcasts are inside calls. No custom timing or ratio when no custom kernel exists.",
        controls={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )
    (args.output_dir / f"{stem}_samples.json").write_text(
        json.dumps(dict(environment=metadata, checks=checks, results=rows), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()

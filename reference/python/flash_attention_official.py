#!/usr/bin/env python3

"""Profiles or times official FlashAttention on deterministic tensors shared with the CUDA benchmark.
The optional raw dump contains output, dQ, dK, and dV as consecutive FP32 arrays."""

import argparse
import json
import math
from pathlib import Path
import statistics


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=int)
    parser.add_argument("sequence", type=int)
    parser.add_argument("heads", type=int)
    parser.add_argument("head_size", type=int)
    parser.add_argument(
        "operation", choices=("forward", "backward", "all"))
    parser.add_argument("dump", nargs="?", type=Path)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    return parser.parse_args()


def checked_profiler_call(call, name):
    result = call()
    status = result[0] if isinstance(result, tuple) else result
    if status != 0:
        raise RuntimeError(f"{name} failed with CUDA status {status}")


def main():
    args = arguments()
    try:
        import torch
        from flash_attn import flash_attn_func
    except ImportError as error:
        raise SystemExit(
            "official FlashAttention reference requires torch and flash-attn"
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit("official FlashAttention reference requires CUDA")

    elements = args.batch * args.sequence * args.heads * args.head_size
    index = torch.arange(elements, dtype=torch.int64, device="cuda")

    def bf16_values(multiplier, modulus, offset, divisor):
        return (((index * multiplier) % modulus).float() - offset).div_(
            divisor
        ).to(torch.bfloat16).reshape(
            args.batch, args.sequence, args.heads, args.head_size
        )

    query = bf16_values(17, 101, 50, 64).requires_grad_(True)
    key = bf16_values(23, 97, 48, 61).requires_grad_(True)
    value = bf16_values(31, 89, 44, 59).requires_grad_(True)
    output_gradient = bf16_values(37, 83, 41, 67)
    scale = 1.0 / math.sqrt(args.head_size)

    def forward():
        return flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True,
        )

    def backward(output, retain_graph=False):
        return torch.autograd.grad(
            output,
            (query, key, value),
            output_gradient,
            retain_graph=retain_graph,
        )

    def measure(operation):
        for _ in range(args.warmup):
            operation()
        torch.cuda.synchronize()

        measurements = []
        for _ in range(args.trials):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                operation()
            stop.record()
            stop.synchronize()
            measurements.append(
                start.elapsed_time(stop) / args.iterations)
        return measurements

    # Warm the same operation that will be captured. Backward setup remains
    # outside capture so only the reference backward kernels are measured.
    if args.operation == "forward":
        with torch.no_grad():
            forward()
    elif args.operation == "backward":
        backward(forward())
    else:
        backward(forward())
    torch.cuda.synchronize()

    timing_result = None
    if args.timing:
        if args.operation == "all":
            raise SystemExit("timing requires forward or backward operation")
        if args.operation == "forward":
            def measured_operation():
                with torch.no_grad():
                    forward()
        else:
            backward_output = forward()
            torch.cuda.synchronize()

            def measured_operation():
                backward(backward_output, retain_graph=True)

        measurements = measure(measured_operation)
        timing_result = {
            "backend": "official",
            "operation": args.operation,
            "batch": args.batch,
            "sequence": args.sequence,
            "heads": args.heads,
            "head_size": args.head_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "median_ms": statistics.median(measurements),
            "minimum_ms": min(measurements),
            "maximum_ms": max(measurements),
        }
    else:
        backward_output = None
        if args.operation == "backward":
            backward_output = forward()
            torch.cuda.synchronize()

        cudart = torch.cuda.cudart()
        checked_profiler_call(cudart.cudaProfilerStart, "cudaProfilerStart")
        if args.operation == "forward":
            with torch.no_grad():
                forward()
        elif args.operation == "backward":
            backward(backward_output)
        else:
            backward(forward())
        torch.cuda.synchronize()
        checked_profiler_call(cudart.cudaProfilerStop, "cudaProfilerStop")

    if args.dump is not None:
        output = forward()
        gradients = backward(output)
        torch.cuda.synchronize()
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("wb") as destination:
            for tensor in (output, *gradients):
                destination.write(
                    tensor.detach().float().cpu().contiguous().numpy().tobytes()
                )

    print(
        "Official FlashAttention workload: "
        f"B={args.batch} T={args.sequence} H={args.heads} "
        f"D={args.head_size} operation={args.operation}"
    )
    if timing_result is not None:
        print("DSCUDA_TIMING " + json.dumps(timing_result, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import csv
import io
import re
import subprocess
import sys


def read_report(ncu, report):
    result = subprocess.run(
        [ncu, "--import", report, "--csv", "--page", "details"],
        check=True,
        capture_output=True,
        text=True,
    )
    kernels = {}
    order = []
    for row in csv.DictReader(io.StringIO(result.stdout)):
        kernel_id = row["ID"]
        if kernel_id not in kernels:
            kernels[kernel_id] = {"name": kernel_name(row["Kernel Name"]), "metrics": {}}
            order.append(kernel_id)

        section = row["Section Name"]
        metric = row["Metric Name"]
        value = row["Metric Value"]
        if metric and value:
            kernels[kernel_id]["metrics"][(section, metric)] = (
                value,
                row["Metric Unit"],
            )
    return [kernels[kernel_id] for kernel_id in order]


def kernel_name(full_name):
    matmul = re.search(
        r"matmul(?:_tensor_core(?:_mma|_edge)?)?_kernel<([^>]*)>",
        full_name,
    )
    if matmul:
        arguments = (
            matmul.group(1).replace("(bool)", "").replace(" ", "").split(",")
        )
        flags = ",".join(arguments[-3:])
        names = {
            "0,0,0": "forward",
            "false,false,false": "forward",
            "0,1,1": "left_backward",
            "false,true,true": "left_backward",
            "1,0,1": "right_backward",
            "true,false,true": "right_backward",
        }
        return names.get(flags, "matmul")

    match = re.search(r"([a-z0-9_]+)_(forward|backward)_kernel", full_name)
    return match.group(2) if match else full_name.split("(", 1)[0]


def value(metrics, section, metric):
    return metrics.get((section, metric), ("-", ""))[0]


def value_with_unit(metrics, section, metric):
    metric_value, unit = metrics.get((section, metric), ("-", ""))
    return f"{metric_value} {unit}".strip()


def duration_us(metrics):
    metric_value, unit = metrics.get(
        ("GPU Speed Of Light Throughput", "Duration"), ("0", "us")
    )
    factors = {"ns": 0.001, "us": 1.0, "ms": 1000.0, "s": 1_000_000.0}
    return float(metric_value.replace(",", "")) * factors[unit]


def print_performance(rows):
    print("\nPerformance")
    print(
        f"{'workload':<20} {'kernel':<14} {'time':>12} {'memory':>14} "
        f"{'DRAM %':>8} {'SM %':>8} {'occupancy %':>12}"
    )
    for workload, kernel in rows:
        metrics = kernel["metrics"]
        print(
            f"{workload:<20} {kernel['name']:<14} "
            f"{value_with_unit(metrics, 'GPU Speed Of Light Throughput', 'Duration'):>12} "
            f"{value_with_unit(metrics, 'Memory Workload Analysis', 'Memory Throughput'):>14} "
            f"{value(metrics, 'GPU Speed Of Light Throughput', 'DRAM Throughput'):>8} "
            f"{value(metrics, 'GPU Speed Of Light Throughput', 'Compute (SM) Throughput'):>8} "
            f"{value(metrics, 'Occupancy', 'Achieved Occupancy'):>12}"
        )


def print_memory(rows):
    print("\nMemory behavior")
    print(
        f"{'workload':<20} {'kernel':<14} {'DRAM read':>13} {'DRAM write':>13} "
        f"{'L1 hit %':>9} {'L2 hit %':>9} {'regs':>6} {'static/block':>14} "
        f"{'dynamic/block':>14} {'spills':>8}"
    )
    for workload, kernel in rows:
        metrics = kernel["metrics"]
        print(
            f"{workload:<20} {kernel['name']:<14} "
            f"{value_with_unit(metrics, 'Command line profiler metrics', 'dram__bytes_read.sum'):>13} "
            f"{value_with_unit(metrics, 'Command line profiler metrics', 'dram__bytes_write.sum'):>13} "
            f"{value(metrics, 'Memory Workload Analysis', 'L1/TEX Hit Rate'):>9} "
            f"{value(metrics, 'Memory Workload Analysis', 'L2 Hit Rate'):>9} "
            f"{value(metrics, 'Launch Statistics', 'Registers Per Thread'):>6} "
            f"{value_with_unit(metrics, 'Launch Statistics', 'Static Shared Memory Per Block'):>14} "
            f"{value_with_unit(metrics, 'Launch Statistics', 'Dynamic Shared Memory Per Block'):>14} "
            f"{value(metrics, 'Memory Workload Analysis', 'Local Memory Spilling Requests'):>8}"
        )


def print_comparison(rows):
    measurements = {}
    for workload, kernel in rows:
        if "/" not in workload:
            continue
        shape, backend = workload.rsplit("/", 1)
        measurements[(shape, kernel["name"], backend)] = duration_us(kernel["metrics"])

    comparisons = []
    for shape, operation, backend in measurements:
        if backend != "fp32":
            continue
        comparisons.append(
            (
                shape,
                operation,
                measurements.get((shape, operation, "fp32")),
                measurements.get((shape, operation, "cublas_fp32")),
                measurements.get((shape, operation, "bf16")),
                measurements.get((shape, operation, "cublas_bf16")),
            )
        )

    if not comparisons:
        return

    print("\nMatmul backend comparison")
    print(
        f"{'workload':<20} {'kernel':<14} {'CUDA FP32 us':>14} "
        f"{'cuBLAS FP32 us':>15} {'FP32 %':>8} {'BF16 Tensor us':>15} "
        f"{'cuBLAS BF16 us':>15} {'BF16 %':>8}"
    )
    for shape, operation, fp32, cublas_fp32, bf16, cublas_bf16 in comparisons:
        fp32_percent = 100.0 * cublas_fp32 / fp32
        bf16_percent = 100.0 * cublas_bf16 / bf16
        print(
            f"{shape:<20} {operation:<14} {fp32:>14.2f} "
            f"{cublas_fp32:>15.2f} {fp32_percent:>8.1f} {bf16:>15.2f} "
            f"{cublas_bf16:>15.2f} {bf16_percent:>8.1f}"
        )


def main():
    ncu = sys.argv[1]
    rows = []
    for index in range(2, len(sys.argv), 3):
        workload = sys.argv[index]
        speed_kernels = read_report(ncu, sys.argv[index + 1])
        metric_kernels = read_report(ncu, sys.argv[index + 2])
        if len(speed_kernels) != len(metric_kernels):
            raise RuntimeError(f"report kernel count mismatch for {workload}")
        kernels = []
        for speed_kernel, metric_kernel in zip(speed_kernels, metric_kernels):
            merged = metric_kernel["metrics"].copy()
            merged.update(speed_kernel["metrics"])
            kernels.append({"name": speed_kernel["name"], "metrics": merged})
        operation = workload.rsplit("/", 1)[-1]
        if operation in {"forward", "left_backward", "right_backward"}:
            workload = workload.rsplit("/", 1)[0]
            for kernel in kernels:
                kernel["name"] = operation
        elif "/cublas" in workload:
            operations = ["forward", "left_backward", "right_backward"]
            for operation, kernel in zip(operations, kernels):
                kernel["name"] = operation
        rows.extend((workload, kernel) for kernel in kernels)

    print_performance(rows)
    print_memory(rows)
    print_comparison(rows)


if __name__ == "__main__":
    main()

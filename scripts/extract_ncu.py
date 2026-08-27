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
    flash_attention = re.search(
        r"flash_attention_(forward|backward_query|backward_key_value)(?:_tensor_core)?_kernel",
        full_name,
    )
    if flash_attention:
        names = {
            "forward": "flash_fwd",
            "backward_query": "flash_bwd_dq",
            "backward_key_value": "flash_bwd_dkv",
        }
        return names[flash_attention.group(1)]

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
            "0,0,0": "matmul_fwd",
            "false,false,false": "matmul_fwd",
            "0,1,1": "matmul_dinput",
            "false,true,true": "matmul_dinput",
            "1,0,1": "matmul_dweight",
            "true,false,true": "matmul_dweight",
        }
        return names.get(flags, "matmul")

    match = re.search(r"([a-z0-9_]+)_(forward|backward)_kernel", full_name)
    if match:
        direction = "fwd" if match.group(2) == "forward" else "bwd"
        return f"{match.group(1)}_{direction}"
    match = re.search(r"attention_(pack_qkv|unpack_output|pack_backward|unpack_gradients)_kernel", full_name)
    if match:
        return f"attention_{match.group(1)}"
    match = re.search(r"([a-z0-9_]+)_kernel", full_name)
    return match.group(1) if match else full_name.split("(", 1)[0]


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


def bytes_value(metrics, metric):
    metric_value, unit = metrics.get(
        ("Command line profiler metrics", metric), ("0", "byte")
    )
    factors = {
        "byte": 1.0,
        "Kbyte": 1_000.0,
        "Mbyte": 1_000_000.0,
        "Gbyte": 1_000_000_000.0,
    }
    return float(metric_value.replace(",", "")) * factors.get(unit, 1.0)


def print_aggregate(rows):
    workloads = {}
    for workload, kernel in rows:
        workloads.setdefault(workload, []).append(kernel)

    groups = [(workload, kernels) for workload, kernels in workloads.items() if len(kernels) > 1]
    if not groups:
        return

    print("\nEnd-to-end kernel totals")
    print(
        f"{'workload':<20} {'launches':>9} {'time ms':>12} "
        f"{'DRAM read MB':>14} {'DRAM write MB':>15} {'effective GB/s':>16}"
    )
    for workload, kernels in groups:
        total_us = sum(duration_us(kernel["metrics"]) for kernel in kernels)
        read_bytes = sum(
            bytes_value(kernel["metrics"], "dram__bytes_read.sum")
            for kernel in kernels
        )
        write_bytes = sum(
            bytes_value(kernel["metrics"], "dram__bytes_write.sum")
            for kernel in kernels
        )
        bandwidth = (read_bytes + write_bytes) / (total_us * 1_000.0)
        print(
            f"{workload:<20} {len(kernels):>9} {total_us / 1000.0:>12.3f} "
            f"{read_bytes / 1_000_000.0:>14.2f} "
            f"{write_bytes / 1_000_000.0:>15.2f} {bandwidth:>16.2f}"
        )


def numeric_value(metrics, section, metric):
    metric_value = value(metrics, section, metric)
    if metric_value == "-":
        return 0.0
    return float(metric_value.replace(",", ""))


def stage_name(kernel):
    name = kernel["name"]
    if name.startswith("matmul"):
        return "gemm"
    if name.startswith("convert_fp32_to_bf16"):
        return "bf16_conversion"
    if name.startswith("rmsnorm"):
        return "rmsnorm"
    if name.startswith("rope"):
        return "rope"
    if name.startswith("causal_softmax"):
        return "softmax"
    if name.startswith("attention"):
        return "attention_layout"
    if name.startswith("flash_"):
        return "flash_attention"
    if name.startswith("swiglu"):
        return "swiglu"
    if name.startswith("residual"):
        return "residual"
    if name.startswith("cross_entropy"):
        return "loss"
    if name.startswith("embedding"):
        return "embedding"
    if name.startswith("global_norm") or name.startswith("clip_gradients"):
        return "gradient_norm"
    if name.startswith("adamw"):
        return "optimizer"
    return "other"


def print_stage_totals(rows):
    grouped = {}
    workload_totals = {}
    for workload, kernel in rows:
        grouped.setdefault((workload, stage_name(kernel)), []).append(kernel)
        workload_totals[workload] = workload_totals.get(workload, 0.0) + duration_us(
            kernel["metrics"]
        )

    print("\nStage totals")
    print(
        f"{'workload':<20} {'stage':<18} {'launches':>9} {'time us':>11} "
        f"{'time %':>8} {'DRAM read MB':>13} {'SM %':>8} "
        f"{'occupancy %':>12} {'L2 hit %':>9}"
    )
    for (workload, stage), kernels in grouped.items():
        durations = [duration_us(kernel["metrics"]) for kernel in kernels]
        total_us = sum(durations)
        read_bytes = sum(
            bytes_value(kernel["metrics"], "dram__bytes_read.sum")
            for kernel in kernels
        )

        def weighted(section, metric):
            return sum(
                duration * numeric_value(kernel["metrics"], section, metric)
                for duration, kernel in zip(durations, kernels)
            ) / total_us

        sm = weighted(
            "GPU Speed Of Light Throughput", "Compute (SM) Throughput"
        )
        occupancy = weighted("Occupancy", "Achieved Occupancy")
        l2_hit = weighted("Memory Workload Analysis", "L2 Hit Rate")
        print(
            f"{workload:<20} {stage:<18} {len(kernels):>9} {total_us:>11.2f} "
            f"{100.0 * total_us / workload_totals[workload]:>8.1f} "
            f"{read_bytes / 1_000_000.0:>13.2f} {sm:>8.2f} "
            f"{occupancy:>12.2f} {l2_hit:>9.2f}"
        )


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

    print_aggregate(rows)
    workload_counts = {}
    for workload, _ in rows:
        workload_counts[workload] = workload_counts.get(workload, 0) + 1
    if max(workload_counts.values()) > 12:
        print_stage_totals(rows)
    else:
        print_performance(rows)
        print_memory(rows)
    print_comparison(rows)


if __name__ == "__main__":
    main()

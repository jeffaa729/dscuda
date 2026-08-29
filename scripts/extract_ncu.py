#!/usr/bin/env python3

"""Extracts comparable timing and hardware metrics from paired Nsight Compute reports.
It prints concise tables and can overwrite one CSV and Markdown summary for a kernel family."""

import argparse
import csv
import io
import math
import re
import subprocess
from pathlib import Path


SPEED_SECTION = "GPU Speed Of Light Throughput"
MEMORY_SECTION = "Memory Workload Analysis"
LAUNCH_SECTION = "Launch Statistics"
OCCUPANCY_SECTION = "Occupancy"
COMMAND_SECTION = "Command line profiler metrics"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("ncu")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    if len(args.reports) % 3 != 0:
        parser.error("reports must be LABEL SPEED_REPORT METRICS_REPORT triples")
    return args


def short_kernel_name(full_name):
    patterns = (
        (r"flash_attention_forward(?:_tensor_core)?_kernel", "flash_fwd"),
        (r"flash_attention_backward_query(?:_tensor_core)?_kernel", "flash_bwd_dq"),
        (r"flash_attention_backward_key_value(?:_tensor_core)?_kernel", "flash_bwd_dkv"),
        (r"flash_fwd", "official_flash_fwd"),
        (r"flash_bwd", "official_flash_bwd"),
        (r"grouped_linear_bf16_tensor_core_kernel", "grouped_gemm_bf16"),
        (r"build_dispatch_map_kernel", "dispatch_map"),
        (r"dispatch_copy_kernel", "dispatch_copy"),
        (r"route_forward_kernel", "router_topk"),
        (r"combine_forward_kernel", "expert_combine"),
    )
    for pattern, name in patterns:
        if re.search(pattern, full_name, re.IGNORECASE):
            return name

    matmul = re.search(
        r"matmul(?:_tensor_core(?:_mma|_edge)?)?_kernel<([^>]*)>",
        full_name,
    )
    if matmul:
        template_arguments = (
            matmul.group(1).replace("(bool)", "").replace(" ", "").split(",")
        )
        flags = ",".join(template_arguments[-3:])
        return {
            "0,0,0": "matmul_fwd",
            "false,false,false": "matmul_fwd",
            "0,1,1": "matmul_dinput",
            "false,true,true": "matmul_dinput",
            "1,0,1": "matmul_dweight",
            "true,false,true": "matmul_dweight",
        }.get(flags, "matmul")

    match = re.search(r"([a-z0-9_]+)_(forward|backward)_kernel", full_name)
    if match:
        direction = "fwd" if match.group(2) == "forward" else "bwd"
        return f"{match.group(1)}_{direction}"
    match = re.search(r"([a-z0-9_]+)_kernel", full_name)
    if match:
        return match.group(1)
    return full_name.split("(", 1)[0].strip()[-60:]


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
            kernels[kernel_id] = {
                "name": short_kernel_name(row["Kernel Name"]),
                "metrics": {},
            }
            order.append(kernel_id)
        metric = row["Metric Name"]
        value = row["Metric Value"]
        if metric and value:
            kernels[kernel_id]["metrics"][(row["Section Name"], metric)] = (
                value,
                row["Metric Unit"],
            )
    return [kernels[kernel_id] for kernel_id in order]


def numeric(metrics, section, metric):
    raw, unit = metrics.get((section, metric), ("0", ""))
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return 0.0
    return value, unit


def duration_us(metrics):
    value, unit = numeric(metrics, SPEED_SECTION, "Duration")
    return value * {"ns": 0.001, "us": 1.0, "ms": 1000.0, "s": 1e6}.get(
        unit, 1.0
    )


def byte_count(metrics, metric):
    value, unit = numeric(metrics, COMMAND_SECTION, metric)
    return value * {
        "byte": 1.0,
        "Kbyte": 1e3,
        "Mbyte": 1e6,
        "Gbyte": 1e9,
    }.get(unit, 1.0)


def size_bytes(metrics, section, metric):
    value, unit = numeric(metrics, section, metric)
    return value * {
        "byte": 1.0,
        "Kbyte": 1e3,
        "Mbyte": 1e6,
        "Gbyte": 1e9,
    }.get(unit, 1.0)


def metric_value(metrics, section, metric):
    return numeric(metrics, section, metric)[0]


def parse_label(label):
    parts = label.split("/")
    if len(parts) >= 3:
        return "/".join(parts[:-2]), parts[-2], parts[-1]
    return label, "custom", "all"


def merge_reports(ncu, label, speed_path, metrics_path):
    speed = read_report(ncu, speed_path)
    hardware = read_report(ncu, metrics_path)
    if len(speed) != len(hardware):
        raise RuntimeError(
            f"report kernel count mismatch for {label}: "
            f"speed={len(speed)} metrics={len(hardware)}"
        )
    workload, backend, operation = parse_label(label)
    rows = []
    for speed_kernel, hardware_kernel in zip(speed, hardware):
        metrics = hardware_kernel["metrics"].copy()
        metrics.update(speed_kernel["metrics"])
        elapsed = duration_us(metrics)
        read_bytes = byte_count(metrics, "dram__bytes_read.sum")
        write_bytes = byte_count(metrics, "dram__bytes_write.sum")
        rows.append(
            {
                "workload": workload,
                "backend": backend,
                "operation": operation,
                "kernel": speed_kernel["name"],
                "time_us": elapsed,
                "dram_read_mb": read_bytes / 1e6,
                "dram_write_mb": write_bytes / 1e6,
                "effective_gbps": (
                    (read_bytes + write_bytes) / (elapsed * 1e3)
                    if elapsed
                    else 0.0
                ),
                "dram_pct": metric_value(
                    metrics, SPEED_SECTION, "DRAM Throughput"
                ),
                "sm_pct": metric_value(
                    metrics, SPEED_SECTION, "Compute (SM) Throughput"
                ),
                "occupancy_pct": metric_value(
                    metrics, OCCUPANCY_SECTION, "Achieved Occupancy"
                ),
                "l1_hit_pct": metric_value(
                    metrics, MEMORY_SECTION, "L1/TEX Hit Rate"
                ),
                "l2_hit_pct": metric_value(
                    metrics, MEMORY_SECTION, "L2 Hit Rate"
                ),
                "registers": metric_value(
                    metrics, LAUNCH_SECTION, "Registers Per Thread"
                ),
                "static_smem_bytes": size_bytes(
                    metrics, LAUNCH_SECTION, "Static Shared Memory Per Block"
                ),
                "dynamic_smem_bytes": size_bytes(
                    metrics, LAUNCH_SECTION, "Dynamic Shared Memory Per Block"
                ),
                "spills": metric_value(
                    metrics, MEMORY_SECTION, "Local Memory Spilling Requests"
                ),
            }
        )
    return rows


def weighted(rows, field):
    elapsed = sum(row["time_us"] for row in rows)
    if elapsed == 0:
        return 0.0
    return sum(row[field] * row["time_us"] for row in rows) / elapsed


def totals(rows):
    grouped = {}
    for row in rows:
        key = (row["workload"], row["backend"], row["operation"])
        grouped.setdefault(key, []).append(row)

    result = []
    for (workload, backend, operation), kernels in grouped.items():
        elapsed = sum(row["time_us"] for row in kernels)
        read_mb = sum(row["dram_read_mb"] for row in kernels)
        write_mb = sum(row["dram_write_mb"] for row in kernels)
        result.append(
            {
                "workload": workload,
                "backend": backend,
                "operation": operation,
                "launches": len(kernels),
                "time_us": elapsed,
                "dram_read_mb": read_mb,
                "dram_write_mb": write_mb,
                "effective_gbps": (
                    (read_mb + write_mb) * 1000.0 / elapsed if elapsed else 0.0
                ),
                "dram_pct": weighted(kernels, "dram_pct"),
                "sm_pct": weighted(kernels, "sm_pct"),
                "occupancy_pct": weighted(kernels, "occupancy_pct"),
                "l1_hit_pct": weighted(kernels, "l1_hit_pct"),
                "l2_hit_pct": weighted(kernels, "l2_hit_pct"),
                "max_registers": max(row["registers"] for row in kernels),
                "max_smem_bytes": max(
                    row["static_smem_bytes"] + row["dynamic_smem_bytes"]
                    for row in kernels
                ),
                "spills": sum(row["spills"] for row in kernels),
                "tflops": operation_tflops(workload, elapsed),
            }
        )
    return result


def operation_tflops(workload, elapsed_us):
    dimensions = {
        name: int(value)
        for name, value in re.findall(r"(?:^|,)(M|N|K)=([0-9]+)", workload)
    }
    if not all(name in dimensions for name in ("M", "N", "K")):
        return math.nan
    flops = 2.0 * dimensions["M"] * dimensions["N"] * dimensions["K"]
    return flops / (elapsed_us * 1.0e6) if elapsed_us else math.nan


def reference_backend(backend):
    return {
        "fp32": "cublas_fp32",
        "bf16": "cublas_bf16",
        "custom": "official",
        "custom_bf16": "cublas_bf16",
    }.get(backend)


def add_relative_performance(summary):
    lookup = {
        (row["workload"], row["backend"], row["operation"]): row
        for row in summary
    }
    for row in summary:
        reference = reference_backend(row["backend"])
        reference_row = lookup.get(
            (row["workload"], reference, row["operation"])
        )
        row["relative_pct"] = (
            100.0 * reference_row["time_us"] / row["time_us"]
            if reference_row and row["time_us"]
            else math.nan
        )
        row["bottleneck"] = bottleneck(row)


def bottleneck(row):
    if row["spills"] > 0:
        return "local-memory spills"
    if row["max_registers"] >= 120 and row["occupancy_pct"] < 40:
        return "register pressure"
    if row["dram_pct"] >= 70:
        return "DRAM bandwidth"
    if row["sm_pct"] >= 70:
        return "compute throughput"
    if row["sm_pct"] < 35 and row["dram_pct"] < 35:
        return "low parallelism/latency"
    return "mixed compute/memory"


def print_totals(summary):
    print("Operation totals")
    print(
        f"{'workload':<30} {'backend':<14} {'operation':<10} {'launches':>8} "
        f"{'time us':>11} {'TFLOP/s':>9} {'DRAM MB':>10} {'GB/s':>10} {'SM %':>8} "
        f"{'occ %':>8} {'regs':>7} {'spills':>8} {'ref %':>8} bottleneck"
    )
    for row in summary:
        relative = (
            "-"
            if math.isnan(row["relative_pct"])
            else f"{row['relative_pct']:.1f}"
        )
        tflops = "-" if math.isnan(row["tflops"]) else f"{row['tflops']:.2f}"
        print(
            f"{row['workload']:<30} {row['backend']:<14} "
            f"{row['operation']:<10} {row['launches']:>8} "
            f"{row['time_us']:>11.2f} {tflops:>9} "
            f"{row['dram_read_mb'] + row['dram_write_mb']:>10.2f} "
            f"{row['effective_gbps']:>10.2f} {row['sm_pct']:>8.2f} "
            f"{row['occupancy_pct']:>8.2f} {row['max_registers']:>7.0f} "
            f"{row['spills']:>8.0f} {relative:>8} {row['bottleneck']}"
        )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        output.write(
            "| workload | backend | operation | launches | time us | TFLOP/s | DRAM MB | "
            "effective GB/s | DRAM % | SM % | occupancy % | L2 hit % | "
            "registers | shared KiB | spills | ref % | bottleneck |\n"
        )
        output.write(
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
        )
        for row in summary:
            relative = (
                "-"
                if math.isnan(row["relative_pct"])
                else f"{row['relative_pct']:.1f}"
            )
            tflops = (
                "-" if math.isnan(row["tflops"])
                else f"{row['tflops']:.2f}"
            )
            output.write(
                f"| {row['workload']} | {row['backend']} | "
                f"{row['operation']} | {row['launches']} | "
                f"{row['time_us']:.2f} | {tflops} | "
                f"{row['dram_read_mb'] + row['dram_write_mb']:.2f} | "
                f"{row['effective_gbps']:.2f} | {row['dram_pct']:.2f} | "
                f"{row['sm_pct']:.2f} | {row['occupancy_pct']:.2f} | "
                f"{row['l2_hit_pct']:.2f} | {row['max_registers']:.0f} | "
                f"{row['max_smem_bytes'] / 1024.0:.2f} | "
                f"{row['spills']:.0f} | {relative} | "
                f"{row['bottleneck']} |\n"
            )


def main():
    args = arguments()
    rows = []
    for index in range(0, len(args.reports), 3):
        rows.extend(
            merge_reports(
                args.ncu,
                args.reports[index],
                args.reports[index + 1],
                args.reports[index + 2],
            )
        )
    summary = totals(rows)
    add_relative_performance(summary)
    print_totals(summary)
    if args.csv_out:
        write_csv(args.csv_out, summary)
    if args.markdown_out:
        write_markdown(args.markdown_out, summary)


if __name__ == "__main__":
    main()

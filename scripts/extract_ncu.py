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
    match = re.search(r"([a-z0-9_]+)_(forward|backward)_kernel", full_name)
    return match.group(2) if match else full_name.split("(", 1)[0]


def value(metrics, section, metric):
    return metrics.get((section, metric), ("-", ""))[0]


def value_with_unit(metrics, section, metric):
    metric_value, unit = metrics.get((section, metric), ("-", ""))
    return f"{metric_value} {unit}".strip()


def print_performance(rows):
    print("\nPerformance")
    print(
        f"{'workload':<20} {'kernel':<9} {'time us':>10} {'mem GB/s':>10} "
        f"{'DRAM %':>8} {'SM %':>8} {'occupancy %':>12}"
    )
    for workload, kernel in rows:
        metrics = kernel["metrics"]
        print(
            f"{workload:<20} {kernel['name']:<9} "
            f"{value(metrics, 'GPU Speed Of Light Throughput', 'Duration'):>10} "
            f"{value(metrics, 'Memory Workload Analysis', 'Memory Throughput'):>10} "
            f"{value(metrics, 'GPU Speed Of Light Throughput', 'DRAM Throughput'):>8} "
            f"{value(metrics, 'GPU Speed Of Light Throughput', 'Compute (SM) Throughput'):>8} "
            f"{value(metrics, 'Occupancy', 'Achieved Occupancy'):>12}"
        )


def print_memory(rows):
    print("\nMemory behavior")
    print(
        f"{'workload':<20} {'kernel':<9} {'DRAM read':>13} {'DRAM write':>13} "
        f"{'L1 hit %':>9} {'L2 hit %':>9} {'regs':>6} {'shared/block':>14} {'spills':>8}"
    )
    for workload, kernel in rows:
        metrics = kernel["metrics"]
        print(
            f"{workload:<20} {kernel['name']:<9} "
            f"{value_with_unit(metrics, 'Command line profiler metrics', 'dram__bytes_read.sum'):>13} "
            f"{value_with_unit(metrics, 'Command line profiler metrics', 'dram__bytes_write.sum'):>13} "
            f"{value(metrics, 'Memory Workload Analysis', 'L1/TEX Hit Rate'):>9} "
            f"{value(metrics, 'Memory Workload Analysis', 'L2 Hit Rate'):>9} "
            f"{value(metrics, 'Launch Statistics', 'Registers Per Thread'):>6} "
            f"{value_with_unit(metrics, 'Launch Statistics', 'Dynamic Shared Memory Per Block'):>14} "
            f"{value(metrics, 'Memory Workload Analysis', 'Local Memory Spilling Requests'):>8}"
        )


def main():
    ncu = sys.argv[1]
    rows = []
    for index in range(2, len(sys.argv), 2):
        workload = sys.argv[index]
        report = sys.argv[index + 1]
        rows.extend((workload, kernel) for kernel in read_report(ncu, report))

    print_performance(rows)
    print_memory(rows)


if __name__ == "__main__":
    main()

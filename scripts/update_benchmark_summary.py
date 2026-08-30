#!/usr/bin/env python3

"""Rebuild the timing index from saved Nsight family CSV files."""

import csv
import sys
from pathlib import Path


from benchmark_flash_attention_runtime import add_reference_percentages, format_percentage, table
from extract_ncu import REFERENCE_BACKENDS


def main():
    result_dir = Path(sys.argv[1])
    rows = []
    for path in sorted(result_dir.glob("*.csv")):
        if path.stem not in {"matmul", "grouped_gemm", "flash_attention", "mla"}:
            continue
        with path.open(newline="", encoding="utf-8") as source:
            measurements = list(csv.DictReader(source))
        add_reference_percentages(measurements, REFERENCE_BACKENDS, ("workload", "operation"), "time_us")
        for row in measurements:
            rows.append((path.stem, row["workload"], row["backend"], row["operation"],
                         f'{float(row["time_us"]):.2f}', format_percentage(row["reference_pct"])))
    (result_dir / "summary.md").write_text(
        table(("family", "workload", "backend", "operation", "time us", "reference %"), rows, {4, 5}),
        encoding="utf-8")


if __name__ == "__main__":
    main()

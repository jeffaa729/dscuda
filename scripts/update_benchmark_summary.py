#!/usr/bin/env python3

"""Rebuild the timing index from saved Nsight family CSV files."""

import csv
import sys
from pathlib import Path


from benchmark_flash_attention_runtime import table


def main():
    result_dir = Path(sys.argv[1])
    rows = []
    for path in sorted(result_dir.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                rows.append((path.stem, row["workload"], row["backend"], row["operation"],
                             f'{float(row["time_us"]):.2f}'))
    (result_dir / "summary.md").write_text(
        table(("family", "workload", "backend", "operation", "time us"), rows, {4}),
        encoding="utf-8")


if __name__ == "__main__":
    main()

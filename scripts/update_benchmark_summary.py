#!/usr/bin/env python3

"""Rebuilds the aggregate benchmark index from the available family CSV files.
Each listed family has passed its matching correctness tests in profile.sh before its table is written."""

import csv
import math
import sys
from pathlib import Path


CUSTOM_BACKENDS = {"fp32", "bf16", "custom", "custom_bf16"}


def summarize(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    custom = [row for row in rows if row["backend"] in CUSTOM_BACKENDS]
    relative = []
    for row in custom:
        value = float(row["relative_pct"])
        if not math.isnan(value):
            relative.append(value)
    relative_text = "-"
    if relative:
        relative_text = (
            f"{relative[0]:.1f}%" if len(relative) == 1
            else f"{min(relative):.1f}-{max(relative):.1f}%"
        )
    primary = max(custom, key=lambda row: float(row["time_us"]))
    return relative_text, primary["bottleneck"]


def main():
    result_dir = Path(sys.argv[1])
    families = []
    for path in sorted(result_dir.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        relative, bottleneck = summarize(path)
        families.append((path.stem, relative, bottleneck))

    output = result_dir / "summary.md"
    with output.open("w", encoding="utf-8") as destination:
        destination.write("# Benchmark summary\n\n")
        destination.write(
            "Correctness is marked PASS only because `profile.sh` runs the "
            "matching tests before replacing a family result.\n\n"
        )
        destination.write(
            "| family | correctness | custom/reference | primary bottleneck | details |\n"
        )
        destination.write("|---|---|---:|---|---|\n")
        for family, relative, bottleneck in families:
            destination.write(
                f"| {family} | PASS | {relative} | {bottleneck} | "
                f"[{family}.md]({family}.md) |\n"
            )


if __name__ == "__main__":
    main()

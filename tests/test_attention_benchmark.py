"""CPU-only regression tests for the comparison's shape, metric and report contract."""

import argparse
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_flash_attention_runtime as benchmark


class BenchmarkTests(unittest.TestCase):
    def test_fixed_dimension_suites(self):
        self.assertEqual(benchmark.benchmark_cases("quick"), ((1, 512, 8, 128),))
        full = benchmark.benchmark_cases("full")
        self.assertEqual(len(full), 20)
        self.assertTrue(all(shape[3] == 128 for shape in full))
        self.assertEqual(full, benchmark.benchmark_cases("h100"))

    def test_positive_controls(self):
        self.assertEqual(benchmark.positive("9"), 9)
        for value in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                benchmark.positive(value)

    def test_sample_summary(self):
        result = benchmark.summarize([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(result["median_ms"], 2.5)
        self.assertEqual(result["minimum_ms"], 1.0)
        self.assertEqual(result["maximum_ms"], 4.0)
        self.assertEqual(result["iqr_pct"], 60.0)

    def test_rejects_invalid_measurements(self):
        for samples in ([], [0.0], [-1.0], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                benchmark.summarize(samples)

    def test_modes_never_share_a_reference_time(self):
        rows = []
        for mode, custom, official in (("graph", 2.0, 1.0), ("api", 3.0, 6.0)):
            for name, elapsed in (("custom", custom), ("official", official)):
                rows.append(dict(batch=1, sequence=512, heads=8, head_size=128,
                                 mode=mode, backend=name, operation="forward", median_ms=elapsed))
        benchmark.add_derived_metrics(rows)
        self.assertEqual([r["relative_pct"] for r in rows], [50.0, 100.0, 200.0, 100.0])
        self.assertEqual(benchmark.api_bytes(rows[0]), benchmark.api_bytes(rows[1]))

    def test_bf16_io_byte_counts_include_lse(self):
        row = dict(batch=1, sequence=512, heads=8, head_size=128, operation="forward")
        self.assertEqual(benchmark.api_bytes(row), 8 * 524288 + 4 * 4096)
        row["operation"] = "backward"
        self.assertEqual(benchmark.api_bytes(row), 16 * 524288 + 4 * 4096)

    def test_table_columns_remain_aligned(self):
        table = benchmark.table(("backend", "median us"),
                                [("custom", "8.0"), ("official", "128.000")], {1})
        widths = [[len(cell) for cell in line.split("|")] for line in table.splitlines()]
        self.assertTrue(all(row == widths[0] for row in widths))


if __name__ == "__main__":
    unittest.main()

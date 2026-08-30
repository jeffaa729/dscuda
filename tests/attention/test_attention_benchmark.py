"""CPU-only regression tests for comparison shapes, timing and concise reports."""

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
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

    def test_timing_modes_remain_separate(self):
        rows = []
        for mode, custom, official in (("graph", 2.0, 1.0), ("api", 3.0, 6.0)):
            for name, elapsed in (("custom", custom), ("official", official)):
                rows.append(dict(batch=1, sequence=512, heads=8, head_size=128,
                                 mode=mode, backend=name, operation="forward", median_ms=elapsed))
        graph = benchmark.result_table(rows, "graph")
        api = benchmark.result_table(rows, "api")
        self.assertIn("2000.00", graph)
        self.assertNotIn("6000.00", graph)
        self.assertIn("6000.00", api)
        self.assertNotIn("2000.00", api)
        self.assertEqual([r["reference_pct"] for r in rows], [50, 100, 200, 100])

    def test_percentages_require_matching_shape_and_operation(self):
        rows = [dict(batch=1, sequence=128, heads=8, head_size=128, mode="graph",
                     operation=operation, backend=backend, median_ms=elapsed)
                for operation, backend, elapsed in (("forward", "custom", 2),
                                                     ("forward", "official", 1),
                                                     ("backward", "custom", 3))]
        rows.append(dict(rows[0], sequence=256))
        benchmark.result_table(rows, "graph")
        self.assertEqual([r["reference_pct"] for r in rows], [50, 100, None, None])
        self.assertEqual(benchmark.format_percentage(None), "-")

    def test_invalid_durations_have_no_percentage(self):
        for invalid in (0, float("nan"), float("inf")):
            for backend in ("custom", "official"):
                rows = [dict(backend=name, median_ms=invalid if name == backend else 1)
                        for name in ("custom", "official")]
                benchmark.add_reference_percentages(rows, "official", ())
                self.assertIsNone(rows[0]["reference_pct"])

    def test_reports_show_parameters_time_and_percentage(self):
        rows = [dict(batch=1, sequence=512, heads=8, head_size=128, mode=mode,
                     backend="custom", operation="forward", **benchmark.summarize([1, 2, 9]))
                for mode in ("graph", "api")]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            path = Path(directory)
            benchmark.write_results(path, rows, [], SimpleNamespace())
            self.assertEqual(output.getvalue(), (path / "flash_attention.md").read_text())
            with (path / "flash_attention.csv").open() as file:
                fields = csv.DictReader(file).fieldnames
            self.assertEqual(set(fields), {"batch", "sequence", "heads", "head_size", "mode",
                                          "backend", "operation", "median_ms", "reference_pct"})
            saved = json.loads((path / "flash_attention_samples.json").read_text())
            self.assertEqual(saved["measurements"][0]["samples_ms"], [1, 2, 9])
            self.assertIsNone(saved["measurements"][0]["reference_pct"])
        for unwanted in ("IQR", "GB/s", "TFLOP", "official %", "WARNING", "Report:", "PASS"):
            self.assertNotIn(unwanted, output.getvalue())

    def test_table_columns_remain_aligned(self):
        table = benchmark.table(("backend", "median us"),
                                [("custom", "8.0"), ("official", "128.000")], {1})
        widths = [[len(cell) for cell in line.split("|")] for line in table.splitlines()]
        self.assertTrue(all(row == widths[0] for row in widths))


if __name__ == "__main__":
    unittest.main()

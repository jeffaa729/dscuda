"""CPU-only checks for MLA comparison shapes, concise reports, and backend status."""
from contextlib import redirect_stdout
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_mla_runtime as benchmark


class MlaBenchmarkTests(unittest.TestCase):
    def test_primary_dimensions(self):
        for suite in ("quick", "full", "h100"):
            cases = benchmark.benchmark_cases(suite)
            self.assertEqual({c.mode for c in cases}, {"prefill", "decode"})
            self.assertTrue(all((c.rank, c.rope) == (512, 64) for c in cases))

    def test_positive_cache_contract(self):
        with self.assertRaises(ValueError):
            benchmark.Case("decode", 1, 8, 2, lengths=(0,))
        with self.assertRaises(ValueError):
            benchmark.Case("decode", 2, 8, 2, lengths=(4,))
        with self.assertRaises(ValueError):
            benchmark.Case("prefill", 1, 8, 2, rank=513)
        with self.assertRaises(ValueError):
            benchmark.Case("prefill", 1, 8, 2, rope=257)
        self.assertTrue(any(c.mode == "decode" and min(c.cache_lengths) < c.splits
                            for c in benchmark.correctness_cases()))

    def test_flashmla_is_really_blank(self):
        self.assertEqual(benchmark.FLASHMLA_STATUS, "not_implemented")
        self.assertEqual((ROOT / "reference/python/flashmla.py").stat().st_size, 0)

    def test_table_alignment(self):
        row = dict(**asdict(benchmark.Case("prefill", 1, 128, 8)), operation="backward",
                   backend="pytorch", median_ms=.5)
        report = benchmark.result_table([row])
        self.assertEqual(len(set(map(len, report.strip().splitlines()))), 1)
        self.assertIn("512", report)
        self.assertIn("64", report)
        self.assertIn("500.00", report)

    def test_saved_and_printed_report_is_only_the_table(self):
        row = dict(**asdict(benchmark.Case("decode", 2, 1024, 16, lengths=(1024, 768))),
                   operation="decode", backend="pytorch", **benchmark.summarize([1, 2, 9]))
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda: "test GPU",
                                                    get_device_capability=lambda: (8, 9)),
                                __version__="test", version=SimpleNamespace(cuda="test"))
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output), \
                patch.object(benchmark.subprocess, "check_output", return_value="test-commit"):
            path = Path(directory)
            args = SimpleNamespace(output_dir=path, library=Path(__file__))
            benchmark.write_results(args, torch, [row], [])
            self.assertEqual(output.getvalue(), (path / "mla.md").read_text())
            with (path / "mla.csv").open() as file:
                fields = csv.DictReader(file).fieldnames
            self.assertEqual(set(fields), {"mode", "batch", "sequence", "heads", "rank", "rope",
                                          "splits", "lengths", "operation", "backend", "median_ms"})
            saved = json.loads((path / "mla_samples.json").read_text())
            self.assertEqual(saved["results"][0]["samples_ms"], [1, 2, 9])
            self.assertEqual(saved["environment"]["flashmla"]["status"], "not_implemented")
        self.assertIn("1024,768", output.getvalue())
        self.assertIn("pytorch_unfused", output.getvalue())
        self.assertTrue(all(line.startswith("|") for line in output.getvalue().splitlines()))
        for unwanted in ("IQR", "GB/s", "TFLOP", "PyTorch %", "FlashMLA:", "Results:", "PASS"):
            self.assertNotIn(unwanted, output.getvalue())


if __name__ == "__main__":
    unittest.main()

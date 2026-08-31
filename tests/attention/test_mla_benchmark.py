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
        self.assertTrue(all((c.rank, c.rope) == (512, 64) for c in benchmark.correctness_cases()))

    def test_no_legacy_shape(self):
        for mode in ("prefill", "decode"):
            for rank, rope in ((64, 32), (64, 16), (512, 32)):
                with self.assertRaisesRegex(ValueError, "C=512 and RoPE=64"):
                    benchmark.Case(mode, 1, 64, 4, rank=rank, rope=rope)

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

    def test_flashmla_has_a_separate_precision_contract(self):
        self.assertEqual(benchmark.FLASHMLA_STATUS, "separate_bf16_adapter")
        source = (ROOT / "reference/python/flashmla.py").read_text()
        self.assertIn("class FlashMLADecode", source)
        self.assertIn("class FlashMLASparse", source)

    def test_table_alignment(self):
        row = dict(**asdict(benchmark.Case("prefill", 1, 128, 8)), operation="backward",
                   backend="pytorch", median_ms=.5)
        report = benchmark.result_table([row])
        self.assertEqual(len(set(map(len, report.strip().splitlines()))), 1)
        self.assertIn("512", report)
        self.assertIn("64", report)
        self.assertIn("500.00", report)
        self.assertEqual(row["reference_pct"], 100)

    def test_paired_decode_row_keeps_input_dimensions(self):
        case = asdict(benchmark.Case("decode", 2, 1024, 16, lengths=(1024, 768)))
        rows = [dict(**case, operation="decode", backend=backend, median_ms=time)
                for backend, time in (("custom", 2), ("pytorch", 1))]
        lines = benchmark.result_table(rows).splitlines()
        self.assertEqual(len(lines), 3)
        cells = [cell.strip() for cell in lines[2].split("|")[1:-1]]
        self.assertEqual(cells[1:], ["bf16", "decode", "2000.00", "1000.00", "50.0"])
        self.assertIn("B=2,Q=1,KV=1024,H=16,C=512,RoPE=64", cells[0])
        self.assertIn("lengths=(1024,768),splits=8", cells[0])

    def test_percentage_matches_decode_lengths(self):
        case = asdict(benchmark.Case("decode", 2, 1024, 16, lengths=(1024, 768)))
        rows = [dict(**case, operation="decode", backend="custom", median_ms=2),
                dict(**case, operation="decode", backend="pytorch", median_ms=1)]
        rows[1]["lengths"] = [1024, 768]  # JSON uses lists rather than tuples.
        rows.append(dict(rows[0], lengths=(1024, 512)))
        rows.append(dict(rows[0], splits=4))
        benchmark.result_table(rows)
        self.assertEqual([r["reference_pct"] for r in rows], [50, 100, None, None])

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
                                          "splits", "lengths", "operation", "backend", "median_ms", "reference_pct"})
            saved = json.loads((path / "mla_samples.json").read_text())
            self.assertEqual(saved["results"][0]["samples_ms"], [1, 2, 9])
            self.assertEqual(saved["results"][0]["reference_pct"], 100)
            self.assertEqual(saved["environment"]["flashmla"]["status"], "separate_bf16_adapter")
        self.assertIn("1024,768", output.getvalue())
        self.assertIn("reference (PyTorch unfused) us", output.getvalue())
        self.assertTrue(all(line.startswith("|") for line in output.getvalue().splitlines()))
        for unwanted in ("IQR", "GB/s", "TFLOP", "PyTorch %", "FlashMLA:", "Results:", "PASS"):
            self.assertNotIn(unwanted, output.getvalue())


if __name__ == "__main__":
    unittest.main()

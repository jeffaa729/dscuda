"""CPU-only checks for the standalone operator comparison/report contract."""
from contextlib import redirect_stdout
import io
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_operators_runtime as benchmark
import extract_ncu
import update_benchmark_summary


class OperatorBenchmarkTests(unittest.TestCase):
    def test_matmul_suite(self):
        cases = benchmark.cases("matmul", "full")
        self.assertEqual({c[0] for c in cases}, {2048, 4096, 8192})
        self.assertEqual({c[1] for c in cases}, {"fp32", "bf16"})
        self.assertEqual({c[2] for c in cases}, {"forward", "left_backward", "right_backward"})
        self.assertEqual(len(cases), 18)
        self.assertEqual(len(benchmark.cases("matmul", "quick")), 6)

    def test_adamw_suite(self):
        self.assertEqual(benchmark.cases("adamw", "quick"), [(1 << 22, "fp32", "update")])
        self.assertTrue(all(c[0] % 4 == 0 for c in benchmark.cases("adamw", "full")))

    def test_table(self):
        row = dict(size=2048, dtype="fp32", operation="forward", backend="reference",
                   reference="cuBLAS", median_ms=0.5)
        report = benchmark.result_table([row])
        self.assertIn("cuBLAS", report)
        self.assertIn("500.00", report)
        self.assertEqual(len(set(map(len, report.strip().splitlines()))), 1)
        self.assertEqual([cell.strip() for cell in report.splitlines()[0].split("|")[1:-1]],
                         ["size", "dtype", "operation", "backend", "median us"])

    def test_ncu_only_reports_collected_metrics(self):
        kernel = dict(workload="M=2048,N=2048,K=2048", backend="fp32", operation="forward",
                      time_us=100, dram_read_mb=2, dram_write_mb=1, dram_pct=20, sm_pct=80,
                      occupancy_pct=50, l1_hit_pct=30, l2_hit_pct=90, registers=64,
                      static_smem_bytes=1024, dynamic_smem_bytes=0, spills=0)
        rows = extract_ncu.totals([kernel])
        report = extract_ncu.result_table(rows)
        self.assertIn("100.00", report)
        self.assertIn("DRAM MB", report)
        self.assertIn("SM %", report)
        for unwanted in ("TFLOP", "GB/s", "ref %", "bottleneck", "IQR"):
            self.assertNotIn(unwanted, report)
        self.assertEqual(len(set(map(len, report.strip().splitlines()))), 1)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
            path = Path(directory) / "ncu.md"
            extract_ncu.write_markdown(path, rows)
            extract_ncu.print_totals(rows)
            self.assertEqual(output.getvalue(), path.read_text())
        self.assertTrue(math.isnan(extract_ncu.metric_value({}, "missing", "metric")))
        rows[0]["sm_pct"] = math.nan
        self.assertNotIn("nan", extract_ncu.result_table(rows))

    def test_profile_summary_accepts_runtime_only_csv(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            extract_ncu.write_csv(path / "matmul.csv", [dict(workload="M=2048,N=2048,K=2048",
                                  backend="custom", operation="forward", time_us=100)])
            with patch.object(sys, "argv", ["update_benchmark_summary.py", directory]):
                update_benchmark_summary.main()
            report = (path / "summary.md").read_text()
            self.assertIn("100.00", report)
            self.assertNotIn("bottleneck", report)
            self.assertNotIn("reference", report)

    def test_runner_hides_success_but_preserves_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "bin").mkdir()
            shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/benchmark.sh",
                            root / "scripts/benchmark.sh")
            fixtures = {
                "scripts/build.sh": 'echo build-output\nexit "${TEST_BUILD_STATUS:-0}"\n',
                "bin/ctest": 'echo test-output\nexit "${TEST_TEST_STATUS:-0}"\n',
                "bin/python": "echo '| median us |'\n",
            }
            for name, body in fixtures.items():
                path = root / name
                path.write_text("#!/usr/bin/env bash\n" + body)
                path.chmod(0o755)
            for build_status, test_status, error in ((0, 0, ""), (3, 0, "build-output\n"),
                                                      (0, 4, "test-output\n")):
                with self.subTest(build=build_status, test=test_status):
                    result = subprocess.run(
                        ["bash", str(root / "scripts/benchmark.sh"), "mla", "quick"],
                        capture_output=True, text=True,
                        env={**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                             "DSCUDA_PYTHON": str(root / "bin/python"),
                             "TEST_BUILD_STATUS": str(build_status), "TEST_TEST_STATUS": str(test_status)})
                    self.assertEqual(result.stdout, "" if error else "| median us |\n")
                    self.assertEqual(result.stderr, error)
                    self.assertEqual(result.returncode, 1 if error else 0)
                    self.assertEqual((root / "build/mla_build.log").read_text(), "build-output\n")


if __name__ == "__main__":
    unittest.main()

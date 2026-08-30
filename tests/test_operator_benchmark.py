"""CPU-only checks for the standalone operator comparison/report contract."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_operators_runtime as benchmark


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
                   reference="cuBLAS", median_ms=0.5, iqr_pct=3, tflops=30,
                   min_io_gb_s=100, reference_pct=100)
        report = benchmark.result_table([row])
        self.assertIn("cuBLAS", report)
        self.assertIn("500.00", report)
        self.assertEqual(len(set(map(len, report.strip().splitlines()))), 1)


if __name__ == "__main__":
    unittest.main()

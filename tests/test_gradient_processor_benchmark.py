import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark" / "gradient_processor_benchmark.py"


def extract_json(stdout):
    start = stdout.find("{")
    if start < 0:
        raise AssertionError(f"benchmark did not print JSON: {stdout}")
    return json.loads(stdout[start:])


class GradientProcessorBenchmarkCliTests(unittest.TestCase):
    def test_gradient_processor_benchmark_cpu_json_smoke(self):
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--nx",
            "8",
            "--nz",
            "6",
            "--cases",
            "norm,marine_smooth,land_smooth,illumination",
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = extract_json(proc.stdout)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["backend"]["name"], "cpu")
        self.assertEqual(report["seed"], 20240523)
        self.assertEqual(report["config"]["repeat"], 1)
        self.assertEqual(report["config"]["cases"], ["norm", "marine_smooth", "land_smooth", "illumination"])
        self.assertEqual(report["config"]["tolerance_profile"], "strict")
        self.assertEqual(report["config"]["compare_rtol"], 1e-5)
        self.assertEqual(report["config"]["compare_atol"], 2e-3)
        self.assertEqual(len(report["cases"]), 4)
        for case in report["cases"]:
            self.assertEqual(case["status"], "ok")
            self.assertLessEqual(case["max_abs_diff"], case["atol"])
            self.assertGreaterEqual(case["legacy_seconds"]["mean"], 0.0)
            self.assertGreaterEqual(case["torch_seconds"]["mean"], 0.0)
            self.assertIn("speedup", case)

    def test_gradient_processor_benchmark_tolerance_profile(self):
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--nx",
            "8",
            "--nz",
            "6",
            "--cases",
            "marine_smooth,illumination",
            "--tolerance-profile",
            "npu-float32",
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = extract_json(proc.stdout)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["config"]["tolerance_profile"], "npu-float32")
        self.assertEqual(report["config"]["compare_rtol"], 2e-4)
        self.assertEqual(report["config"]["compare_atol"], 5e-1)
        for case in report["cases"]:
            self.assertEqual(case["rtol"], 2e-4)
            self.assertEqual(case["atol"], 5e-1)

    def test_gradient_processor_benchmark_explicit_tolerance_override(self):
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--nx",
            "8",
            "--nz",
            "6",
            "--cases",
            "norm",
            "--tolerance-profile",
            "npu-float32",
            "--compare-rtol",
            "1e-6",
            "--compare-atol",
            "1e-4",
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = extract_json(proc.stdout)

        self.assertEqual(report["config"]["tolerance_profile"], "npu-float32")
        self.assertEqual(report["config"]["compare_rtol"], 1e-6)
        self.assertEqual(report["config"]["compare_atol"], 1e-4)
        self.assertEqual(report["cases"][0]["rtol"], 1e-6)
        self.assertEqual(report["cases"][0]["atol"], 1e-4)


if __name__ == "__main__":
    unittest.main()

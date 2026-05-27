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
        self.assertEqual(len(report["cases"]), 4)
        for case in report["cases"]:
            self.assertEqual(case["status"], "ok")
            self.assertLessEqual(case["max_abs_diff"], case["atol"])
            self.assertGreaterEqual(case["legacy_seconds"]["mean"], 0.0)
            self.assertGreaterEqual(case["torch_seconds"]["mean"], 0.0)
            self.assertIn("speedup", case)


if __name__ == "__main__":
    unittest.main()

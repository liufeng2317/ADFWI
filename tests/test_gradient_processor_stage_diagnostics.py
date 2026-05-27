import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark" / "gradient_processor_stage_diagnostics.py"


def extract_json(stdout):
    start = stdout.find("{")
    if start < 0:
        raise AssertionError(f"diagnostics did not print JSON: {stdout}")
    return json.loads(stdout[start:])


class GradientProcessorStageDiagnosticsCliTests(unittest.TestCase):
    def test_gradient_processor_stage_diagnostics_cpu_json_smoke(self):
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--device",
            "cpu",
            "--nx",
            "8",
            "--nz",
            "6",
            "--smooth-span",
            "2",
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = extract_json(proc.stdout)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["backend"]["name"], "cpu")
        self.assertEqual(report["seed"], 20240523)
        self.assertEqual(report["config"]["smooth_span"], 2)
        self.assertIn("smooth2d_raw", report["cases"])
        self.assertIn("marine_smooth_without_norm", report["cases"])
        self.assertIn("illumination_preconditioner", report["cases"])

        raw = report["cases"]["smooth2d_raw"]
        self.assertLessEqual(raw["legacy_vs_torch_cpu"]["max_abs_diff"], 1e-3)
        self.assertLessEqual(raw["legacy_vs_torch_device"]["max_abs_diff"], 1e-3)


if __name__ == "__main__":
    unittest.main()

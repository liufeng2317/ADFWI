import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark" / "acoustic_backend_benchmark.py"


def extract_json(stdout):
    start = stdout.find("{")
    if start < 0:
        raise AssertionError(f"benchmark did not print JSON: {stdout}")
    return json.loads(stdout[start:])


class AcousticBenchmarkCliTests(unittest.TestCase):
    def test_acoustic_benchmark_cpu_json_smoke(self):
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
            "--nabc",
            "2",
            "--nt",
            "8",
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = extract_json(proc.stdout)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["backend"]["name"], "cpu")
        self.assertEqual(report["seed"], 20240523)
        self.assertEqual(report["config"]["repeat"], 1)
        self.assertEqual(len(report["runs"]), 1)
        self.assertIn("environment", report)
        self.assertIn("git", report)
        self.assertIn("forward_seconds", report["metrics"])
        self.assertIn("backward_seconds", report["metrics"])
        self.assertGreaterEqual(report["runs"][0]["forward_seconds"], 0.0)
        self.assertGreaterEqual(report["runs"][0]["backward_seconds"], 0.0)
        self.assertGreater(report["runs"][0]["vp_grad_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()

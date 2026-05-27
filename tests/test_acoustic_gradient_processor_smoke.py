import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "smoke" / "acoustic_mini_inversion_smoke.py"
METRICS = ("loss", "vp_grad_norm", "vp_update_norm")


def extract_json(stdout):
    start = stdout.find("{")
    if start < 0:
        raise AssertionError(f"child process did not print JSON: {stdout}")
    return json.loads(stdout[start:])


def run_smoke(processor):
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--device",
        "cpu",
        "--gradient-processor",
        processor,
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
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    return extract_json(proc.stdout)


class AcousticGradientProcessorSmokeTests(unittest.TestCase):
    def test_torch_gradient_processor_matches_legacy_mini_inversion_metrics_on_cpu(self):
        legacy = run_smoke("legacy")
        torch_native = run_smoke("torch")

        self.assertEqual(legacy["status"], "ok")
        self.assertEqual(torch_native["status"], "ok")
        self.assertEqual(legacy["inversion"]["gradient_processor"], "legacy")
        self.assertEqual(torch_native["inversion"]["gradient_processor"], "torch")

        for metric in METRICS:
            reference = float(legacy["inversion"][metric])
            candidate = float(torch_native["inversion"][metric])
            abs_diff = abs(candidate - reference)
            rel_diff = abs_diff / max(abs(reference), abs(candidate), 1e-30)
            self.assertLessEqual(abs_diff, 1e-12, metric)
            self.assertLessEqual(rel_diff, 1e-6, metric)


if __name__ == "__main__":
    unittest.main()

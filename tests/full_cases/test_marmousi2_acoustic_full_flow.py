import json
import math
import os
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "examples" / "marmousi2_acoustic_reduced_inversion.py"


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def extract_json(stdout: str):
    start = stdout.find("{")
    if start < 0:
        raise AssertionError(f"full-case command did not print JSON: {stdout}")
    return json.loads(stdout[start:])


def full_flow_command():
    device = os.environ.get("ADFWI_FULL_CASE_DEVICE", "npu:0")
    shot_count = env_int("ADFWI_FULL_CASE_SHOT_COUNT", 1)
    nt_samples = env_int("ADFWI_FULL_CASE_NT_SAMPLES", 3000)
    iterations = env_int("ADFWI_FULL_CASE_ITERATIONS", 2)
    lr = env_float("ADFWI_FULL_CASE_LR", 10.0)
    checkpoint_segments = env_int("ADFWI_FULL_CASE_CHECKPOINT_SEGMENTS", 10)
    gradient_processor = os.environ.get("ADFWI_FULL_CASE_GRADIENT_PROCESSOR", "legacy")
    output_dir = os.environ.get("ADFWI_FULL_CASE_OUTPUT_DIR")

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--device",
        device,
        "--shot-count",
        str(shot_count),
        "--nt-samples",
        str(nt_samples),
        "--observed-source",
        "synthetic-true",
        "--iterations",
        str(iterations),
        "--optimizer",
        "adam",
        "--lr",
        str(lr),
        "--scheduler-step-size",
        "200",
        "--scheduler-gamma",
        "0.75",
        "--misfit",
        "legacy-l2",
        "--waveform-normalize",
        "--auto-update-rho",
        "--checkpoint-segments",
        str(checkpoint_segments),
        "--gradient-processor",
        gradient_processor,
    ]
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    return cmd


class Marmousi2AcousticFullFlowTests(unittest.TestCase):
    def test_full_flow_command_uses_synthetic_true_observations(self):
        cmd = full_flow_command()

        self.assertIn("--observed-source", cmd)
        self.assertEqual(cmd[cmd.index("--observed-source") + 1], "synthetic-true")
        self.assertIn("--optimizer", cmd)
        self.assertEqual(cmd[cmd.index("--optimizer") + 1], "adam")
        self.assertIn("--misfit", cmd)
        self.assertEqual(cmd[cmd.index("--misfit") + 1], "legacy-l2")
        self.assertIn("--gradient-processor", cmd)
        self.assertEqual(cmd[cmd.index("--gradient-processor") + 1], "legacy")

    def test_full_flow_command_accepts_torch_gradient_processor(self):
        with mock.patch.dict(os.environ, {"ADFWI_FULL_CASE_GRADIENT_PROCESSOR": "torch"}):
            cmd = full_flow_command()

        self.assertIn("--gradient-processor", cmd)
        self.assertEqual(cmd[cmd.index("--gradient-processor") + 1], "torch")

    @unittest.skipUnless(env_flag("ADFWI_RUN_FULL_CASES"), "set ADFWI_RUN_FULL_CASES=1 to run full Marmousi2 case")
    def test_marmousi2_synthetic_true_forward_and_inversion(self):
        proc = subprocess.run(
            full_flow_command(),
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = extract_json(proc.stdout)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["observed"]["source"], "synthetic-true")
        self.assertGreater(report["observed"]["synthetic_true_forward_seconds"], 0.0)

        inversion = report["inversion"]
        self.assertEqual(inversion["optimizer"], "Adam")
        self.assertEqual(inversion["misfit"], "legacy-l2")
        self.assertEqual(inversion["gradient_processor"], os.environ.get("ADFWI_FULL_CASE_GRADIENT_PROCESSOR", "legacy"))
        self.assertTrue(inversion["waveform_normalize"])
        self.assertTrue(inversion["auto_update_rho"])
        self.assertEqual(len(inversion["loss_history"]), inversion["iterations"])
        self.assertTrue(all(math.isfinite(value) for value in inversion["loss_history"]))
        self.assertLess(inversion["loss_delta"], 0.0)
        self.assertLess(inversion["loss_relative_delta"], 0.0)
        self.assertGreater(inversion["vp_grad_norm"], 0.0)
        self.assertGreater(inversion["vp_update_norm"], 0.0)

        output_dir = os.environ.get("ADFWI_FULL_CASE_OUTPUT_DIR")
        if output_dir:
            outputs = report["outputs"]
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), path)


if __name__ == "__main__":
    unittest.main()

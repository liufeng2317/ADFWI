import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark" / "run_marmousi2_full_case.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_marmousi2_full_case", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module()


class Marmousi2FullCasePresetTests(unittest.TestCase):
    def test_shot3_preset_matches_current_baseline(self):
        preset = runner.PRESETS["shot3"]

        self.assertEqual(preset.shot_count, 3)
        self.assertEqual(preset.iterations, 10)
        self.assertEqual(preset.checkpoint_segments, 10)
        self.assertEqual(preset.nt_samples, 3000)

    def test_build_command_uses_notebook_like_full_case_options(self):
        preset = runner.PRESETS["shot5"]
        cmd = runner.build_command(
            preset,
            device="npu:0",
            output_dir=Path("tests/full_cases/outputs/custom"),
            python="python",
        )

        self.assertIn("--observed-source", cmd)
        self.assertEqual(cmd[cmd.index("--observed-source") + 1], "synthetic-true")
        self.assertIn("--optimizer", cmd)
        self.assertEqual(cmd[cmd.index("--optimizer") + 1], "adam")
        self.assertIn("--misfit", cmd)
        self.assertEqual(cmd[cmd.index("--misfit") + 1], "legacy-l2")
        self.assertIn("--waveform-normalize", cmd)
        self.assertIn("--auto-update-rho", cmd)
        self.assertEqual(cmd[cmd.index("--checkpoint-segments") + 1], "10")
        self.assertEqual(cmd[cmd.index("--shot-count") + 1], "5")
        self.assertIn("--gradient-processor", cmd)
        self.assertEqual(cmd[cmd.index("--gradient-processor") + 1], "legacy")

    def test_build_command_accepts_torch_gradient_processor(self):
        preset = runner.PRESETS["shot3"]
        cmd = runner.build_command(
            preset,
            device="npu:0",
            output_dir=Path("tests/full_cases/outputs/custom"),
            python="python",
            gradient_processor="torch",
        )

        self.assertIn("--gradient-processor", cmd)
        self.assertEqual(cmd[cmd.index("--gradient-processor") + 1], "torch")

    def test_cli_dry_run_outputs_plan_without_running(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "shot3",
                "--dry-run",
                "--output-dir",
                "tests/full_cases/outputs/dry_run",
                "--gradient-processor",
                "torch",
            ],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["preset"]["name"], "shot3")
        self.assertEqual(report["output_dir"], "tests/full_cases/outputs/dry_run")
        self.assertEqual(report["gradient_processor"], "torch")
        self.assertIn("--output-dir", report["command"])
        self.assertEqual(report["command"][report["command"].index("--gradient-processor") + 1], "torch")
        self.assertIsNone(report["compare_command"])

    def test_cli_dry_run_includes_compare_command(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "shot3",
                "--dry-run",
                "--output-dir",
                "tests/full_cases/outputs/candidate",
                "--compare-to",
                "tests/full_cases/outputs/baseline",
                "--compare-labels",
                "baseline,candidate",
                "--fail-on-loss-drift",
            ],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = json.loads(proc.stdout)
        compare_command = report["compare_command"]
        self.assertIsNotNone(compare_command)
        self.assertIn(str(runner.COMPARE_SCRIPT), compare_command)
        self.assertIn("tests/full_cases/outputs/baseline", compare_command)
        self.assertIn("tests/full_cases/outputs/candidate", compare_command)
        self.assertIn("--fail-on-loss-drift", compare_command)

    def test_cli_dry_run_includes_profile_command(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "shot3",
                "--dry-run",
                "--profile",
                "--output-dir",
                "tests/full_cases/outputs/profile_candidate",
            ],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = json.loads(proc.stdout)
        self.assertTrue(report["profile"])
        self.assertEqual(report["profile_output"], "tests/full_cases/outputs/profile_candidate/python_profile.prof")
        self.assertEqual(report["command"][1:4], ["-m", "cProfile", "-o"])
        self.assertIn(str(runner.INVERSION_SCRIPT), report["command"])

    def test_cli_lists_presets(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "shot3", "--list-presets"],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertIn("shot3", report["presets"])
        self.assertIn("shot5", report["presets"])


if __name__ == "__main__":
    unittest.main()

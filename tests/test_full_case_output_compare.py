import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark" / "compare_full_case_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_full_case_outputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load_module()


def write_summary(path: Path, seconds: float, loss_history):
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "case": "marmousi2-acoustic",
        "seed": 20240524,
        "backend": {"name": "npu", "device": "npu:0", "dtype": "float32", "memory_allocated": 1234},
        "subset": {"shot_count": 3, "nt_samples": 3000, "checkpoint_segments": 10},
        "observed": {"synthetic_true_forward_seconds": 1.5},
        "inversion": {
            "iterations": len(loss_history),
            "seconds": seconds,
            "initial_loss": loss_history[0],
            "loss": loss_history[-1],
            "loss_delta": loss_history[-1] - loss_history[0],
            "loss_relative_delta": (loss_history[-1] - loss_history[0]) / loss_history[0],
            "loss_min": min(loss_history),
            "loss_max": max(loss_history),
            "loss_history": loss_history,
            "vp_grad_norm": 0.25,
            "vp_update_norm": 100.0,
        },
    }
    (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


class FullCaseOutputCompareTests(unittest.TestCase):
    def test_build_report_summarizes_and_compares_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root / "reference", 20.0, [10.0, 8.0, 7.0])
            write_summary(root / "candidate", 18.0, [10.0, 8.0001, 7.0])

            report = compare.build_report(
                [root / "reference", root / "candidate"],
                ["ref", "candidate"],
                reference_index=0,
            )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["reference_label"], "ref")
        self.assertEqual(report["runs"][0]["seconds_per_iteration"], 20.0 / 3.0)
        self.assertTrue(report["runs"][1]["loss_monotonic_nonincreasing"])
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["label"], "candidate")
        self.assertAlmostEqual(comparison["seconds_per_iteration"]["abs_diff"], 2.0 / 3.0)
        self.assertAlmostEqual(comparison["loss_history"]["max_abs_diff"], 0.0001)

    def test_cli_outputs_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root / "reference", 20.0, [10.0, 8.0, 7.0])
            write_summary(root / "candidate", 18.0, [10.0, 8.0, 7.0])
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(root / "reference"),
                    str(root / "candidate"),
                    "--labels",
                    "ref,candidate",
                ],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["comparisons"][0]["final_loss"]["abs_diff"], 0.0)

    def test_fail_on_loss_drift_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root / "reference", 20.0, [10.0, 8.0, 7.0])
            write_summary(root / "candidate", 18.0, [10.0, 8.0, 7.5])
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(root / "reference"),
                    str(root / "candidate"),
                    "--fail-on-loss-drift",
                    "--loss-abs-tol",
                    "0.01",
                ],
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()

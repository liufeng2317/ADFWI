import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "validation" / "marmousi2_acoustic_bv12" / "scripts" / "run_validation.py"
NOTEBOOK_DIR = REPO_ROOT / "examples" / "validation" / "marmousi2_acoustic_bv12" / "notebooks"


class Marmousi2ValidationExampleTests(unittest.TestCase):
    def run_dry_run(self, *args):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), *args, "--dry-run"],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        return json.loads(proc.stdout)

    def test_all_stage_plans_check_forward_and_short_inversion(self):
        payload = self.run_dry_run("all", "--device", "cpu")

        self.assertEqual(payload["status"], "ok")
        self.assertEqual([stage["stage"] for stage in payload["stages"]], ["check", "forward", "inversion10"])
        self.assertIn("--run-forward", payload["stages"][1]["command"])
        inversion = payload["stages"][2]["command"]
        self.assertIn("--observed-source", inversion)
        self.assertEqual(inversion[inversion.index("--observed-source") + 1], "synthetic-true")
        self.assertEqual(inversion[inversion.index("--iterations") + 1], "10")
        self.assertIn("--output-dir", inversion)

    def test_inversion100_stage_uses_longer_default_iterations(self):
        payload = self.run_dry_run("inversion100", "--device", "cpu")

        command = payload["stages"][0]["command"]
        self.assertEqual(command[command.index("--iterations") + 1], "100")

    def test_iterations_override_changes_inversion_stage(self):
        payload = self.run_dry_run("inversion10", "--iterations", "4", "--device", "cpu")

        command = payload["stages"][0]["command"]
        self.assertEqual(command[command.index("--iterations") + 1], "4")

    def test_forward_and_inversion_notebooks_are_separate(self):
        self.assertTrue((NOTEBOOK_DIR / "01_forward_modeling.ipynb").exists())
        self.assertTrue((NOTEBOOK_DIR / "02_inversion.ipynb").exists())
        self.assertFalse((NOTEBOOK_DIR / "marmousi2_acoustic_bv12_validation.ipynb").exists())


if __name__ == "__main__":
    unittest.main()

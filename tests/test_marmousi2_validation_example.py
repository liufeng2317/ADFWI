import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = REPO_ROOT / "examples" / "validation" / "marmousi2_acoustic_bv12"
SCRIPT_DIR = VALIDATION_DIR / "scripts"
SCRIPT = SCRIPT_DIR / "run_validation.py"
NOTEBOOK_DIR = VALIDATION_DIR / "notebooks"


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
        self.assertIn("forward_modeling.py", payload["stages"][1]["description"])
        self.assertIn("inversion.py", payload["stages"][2]["description"])
        self.assertEqual(payload["stages"][0]["parameters"]["device"], "cpu")
        self.assertEqual(payload["stages"][0]["parameters"]["f0"], 5.0)
        self.assertEqual(payload["stages"][1]["parameters"]["checkpoint_segments"], 1)
        inversion = payload["stages"][2]["parameters"]
        self.assertEqual(inversion["iterations"], 10)
        self.assertEqual(inversion["shots"], 3)
        self.assertEqual(inversion["lr"], 10.0)

    def test_inversion100_stage_uses_longer_default_iterations(self):
        payload = self.run_dry_run("inversion100", "--device", "cpu")

        self.assertEqual(payload["stages"][0]["parameters"]["iterations"], 100)

    def test_iterations_override_changes_inversion_stage(self):
        payload = self.run_dry_run("inversion10", "--iterations", "4", "--device", "cpu")

        self.assertEqual(payload["stages"][0]["parameters"]["iterations"], 4)

    def test_forward_and_inversion_notebooks_are_separate(self):
        self.assertTrue((NOTEBOOK_DIR / "01_forward_modeling.ipynb").exists())
        self.assertTrue((NOTEBOOK_DIR / "02_inversion.ipynb").exists())
        self.assertFalse((NOTEBOOK_DIR / "marmousi2_acoustic_bv12_validation.ipynb").exists())

    def test_forward_and_inversion_scripts_are_separate(self):
        self.assertTrue((SCRIPT_DIR / "forward_modeling.py").exists())
        self.assertTrue((SCRIPT_DIR / "inversion.py").exists())
        text = SCRIPT.read_text()
        self.assertIn("from forward_modeling import", text)
        self.assertIn("from inversion import", text)
        self.assertNotIn("scripts/examples", text)

    def test_notebooks_define_case_setup_inline(self):
        for notebook in ("01_forward_modeling.ipynb", "02_inversion.ipynb"):
            text = (NOTEBOOK_DIR / notebook).read_text()
            self.assertIn("ADFWI.set_backend", text)
            self.assertIn("AcousticPropagator", text)
            self.assertIn("load_marmousi_model", text)
            self.assertIn("validation_shots", text)
            self.assertNotIn("subprocess", text)
            self.assertNotIn("run_validation.py", text)
            self.assertNotIn("from case_definition import", text)
            self.assertNotIn("marmousi2_acoustic_backend_check", text)

    def test_notebooks_follow_original_example_order(self):
        forward_text = (NOTEBOOK_DIR / "01_forward_modeling.ipynb").read_text()
        inversion_text = (NOTEBOOK_DIR / "02_inversion.ipynb").read_text()

        for marker in (
            "## Basic Parameter",
            "## Define the observed System",
            "## Define the propagator",
        ):
            self.assertIn(marker, forward_text)

        for marker in (
            "## Define the basic model parameter",
            "## Load observed datasets",
            "## Inversion",
            "## Visualize the inverted results",
        ):
            self.assertIn(marker, inversion_text)


if __name__ == "__main__":
    unittest.main()

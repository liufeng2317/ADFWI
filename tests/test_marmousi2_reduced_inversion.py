import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "examples"
SCRIPT = SCRIPT_DIR / "marmousi2_acoustic_reduced_inversion.py"


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("marmousi2_acoustic_reduced_inversion", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


case_script = load_module()


class Marmousi2ReducedInversionTests(unittest.TestCase):
    def test_summarize_losses_reports_delta_and_relative_delta(self):
        summary = case_script.summarize_losses([10.0, 9.5, 9.0])

        self.assertEqual(summary["initial_loss"], 10.0)
        self.assertEqual(summary["loss"], 9.0)
        self.assertEqual(summary["loss_min"], 9.0)
        self.assertEqual(summary["loss_max"], 10.0)
        self.assertEqual(summary["loss_delta"], -1.0)
        self.assertEqual(summary["loss_relative_delta"], -0.1)


if __name__ == "__main__":
    unittest.main()

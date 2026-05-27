import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


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

    def test_finite_value_accepts_negative_legacy_loss(self):
        case_script.finite_value(-1228.7, "legacy_l2_loss")

    def test_build_optimizer_supports_notebook_adam_setting(self):
        model = Mock()
        param = case_script.torch.nn.Parameter(case_script.torch.ones(()))
        model.parameters.return_value = [param]
        args = SimpleNamespace(optimizer="adam", lr=10.0)

        optimizer, name = case_script.build_optimizer(model, args)

        self.assertEqual(name, "Adam")
        self.assertIsInstance(optimizer, case_script.torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["lr"], 10.0)

    def test_build_case_model_can_disable_training_for_synthetic_observations(self):
        model_npz = {
            "vp": case_script.np.ones((2, 3), dtype=case_script.np.float32) * 1500.0,
            "rho": case_script.np.ones((2, 3), dtype=case_script.np.float32) * 1900.0,
            "vp_bound": case_script.np.array([1400.0, 2200.0], dtype=case_script.np.float32),
            "rho_bound": case_script.np.array([1500.0, 2500.0], dtype=case_script.np.float32),
            "ox": 0.0,
            "oz": 0.0,
            "nx": 3,
            "nz": 2,
            "dx": 10.0,
            "dz": 10.0,
            "free_surface": False,
            "nabc": 1,
        }
        args = SimpleNamespace(abc_type="PML", abc_jerjan_alpha=0.007)

        model = case_script.build_case_model(model_npz, args, vp_grad=False, auto_update_rho=False)

        self.assertFalse(model.get_requires_grad("vp"))


if __name__ == "__main__":
    unittest.main()

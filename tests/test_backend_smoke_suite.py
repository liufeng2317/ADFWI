import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "smoke" / "run_backend_smoke_suite.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_backend_smoke_suite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


suite = load_module()


class BackendSmokeSuiteTests(unittest.TestCase):
    def test_example_command_passes_gradient_processor_option(self):
        args = SimpleNamespace(
            prefer="npu,cpu",
            dtype="float32",
            checkpoint_segments=1,
            seed=20240523,
            fallback_cpu=False,
            example_gradient_processor="torch",
        )

        cmd = suite.command_for_example("acoustic", "cpu", args)

        self.assertIn("--gradient-processor", cmd)
        index = cmd.index("--gradient-processor")
        self.assertEqual(cmd[index + 1], "torch")

    def test_example_command_accepts_explicit_gradient_processor(self):
        args = SimpleNamespace(
            prefer="npu,cpu",
            dtype="float32",
            checkpoint_segments=1,
            seed=20240523,
            fallback_cpu=False,
            example_gradient_processor="legacy",
        )

        cmd = suite.command_for_example("acoustic", "cpu", args, "torch")

        index = cmd.index("--gradient-processor")
        self.assertEqual(cmd[index + 1], "torch")

    def test_example_gradient_processor_comparison_reports_metric_drift(self):
        args = SimpleNamespace(
            devices=["cpu"],
            example_problems=["acoustic"],
            example_gradient_processors=["legacy", "torch"],
            example_gradient_rtol=1e-6,
            example_gradient_atol=1e-12,
        )
        runs = [
            {
                "status": "ok",
                "problem": "acoustic",
                "device_request": "cpu",
                "gradient_processor": "legacy",
                "result": {"inversion": {"loss": 1.0, "vp_grad_norm": 2.0, "vp_update_norm": 3.0}},
            },
            {
                "status": "ok",
                "problem": "acoustic",
                "device_request": "cpu",
                "gradient_processor": "torch",
                "result": {"inversion": {"loss": 1.0, "vp_grad_norm": 2.0, "vp_update_norm": 3.0}},
            },
        ]

        comparisons = suite.compare_example_gradient_processors(runs, args)

        self.assertEqual(len(comparisons), 1)
        comparison = comparisons[0]
        self.assertEqual(comparison["comparison_type"], "gradient_processor")
        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["reference_gradient_processor"], "legacy")
        self.assertEqual(comparison["gradient_processor"], "torch")
        self.assertTrue(all(metric["status"] == "ok" for metric in comparison["metrics"]))

    def test_case_inversion_command_passes_iteration_count(self):
        args = SimpleNamespace(
            prefer="npu,cpu",
            dtype="float32",
            seed=20240523,
            case_model_file="init_model.npz",
            checkpoint_segments=1,
            case_inversion_shot_count=1,
            case_inversion_nt_samples=300,
            case_inversion_iterations=10,
            case_inversion_misfit="safe-squared-l2",
            case_inversion_lr=1e12,
            case_inversion_dt_for_loss=1.0,
            case_inversion_grad_mute_top=12,
            fallback_cpu=False,
            case_inversion_norm_grad=False,
            case_inversion_forw_illumination=False,
            case_inversion_auto_update_rho=False,
            case_inversion_waveform_normalize=False,
        )

        cmd = suite.command_for_case_inversion("marmousi2-acoustic-reduced", "cpu", args)

        self.assertIn("--iterations", cmd)
        index = cmd.index("--iterations")
        self.assertEqual(cmd[index + 1], "10")


if __name__ == "__main__":
    unittest.main()

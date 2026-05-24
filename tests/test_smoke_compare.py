import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke" / "compare_backend_smoke.py"
    spec = importlib.util.spec_from_file_location("compare_backend_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load_module()


class SmokeCompareTests(unittest.TestCase):
    def test_extract_json_ignores_warning_prefix(self):
        payload = compare.extract_json("warning before json\n{\"status\": \"ok\"}")
        self.assertEqual(payload["status"], "ok")

    def test_compare_value_accepts_relative_tolerance(self):
        result = compare.compare_value(100.0, 100.0005, rel_tol=1e-5, abs_tol=1e-10)
        self.assertTrue(result["passed"])
        self.assertGreater(result["abs_diff"], result["abs_tol"])

    def test_compare_value_rejects_large_drift(self):
        result = compare.compare_value(1.0, 1.1, rel_tol=1e-5, abs_tol=1e-10)
        self.assertFalse(result["passed"])

    def test_compare_runs_uses_first_successful_run_as_reference(self):
        runs = [
            {
                "status": "ok",
                "device_request": "cpu",
                "result": {"inversion": {"loss": 1.0, "vp_grad_norm": 2.0, "vp_update_norm": 3.0}},
            },
            {
                "status": "ok",
                "device_request": "npu:0",
                "result": {"inversion": {"loss": 1.0, "vp_grad_norm": 2.000001, "vp_update_norm": 3.0}},
            },
        ]
        result = compare.compare_runs(runs, compare.METRICS, rel_tol=1e-5, abs_tol=1e-10)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reference_device"], "cpu")
        self.assertIn("npu:0", result["comparisons"])

    def test_case_args_for_mute_late_sets_long_record(self):
        self.assertEqual(
            compare.case_args("acoustic", "mute-late"),
            ["--nt", "160", "--mute-late-window", "0.01"],
        )

    def test_case_args_for_trace_missing_selects_receivers(self):
        self.assertEqual(compare.case_args("elastic", "trace-missing"), ["--receiver-mask-mode", "select"])

    def test_parse_cases_rejects_unknown_case(self):
        with self.assertRaises(Exception):
            compare.parse_cases("baseline,unknown")

    def test_selected_matrix_uses_explicit_lists(self):
        class Args:
            problem = "acoustic"
            case = "baseline"
            problems = ["acoustic", "elastic"]
            cases = ["baseline", "mute-offset"]

        self.assertEqual(compare.selected_matrix(Args()), (["acoustic", "elastic"], ["baseline", "mute-offset"]))

    def test_selected_matrix_falls_back_to_single_shortcuts(self):
        class Args:
            problem = "elastic"
            case = "mute-late"
            problems = None
            cases = None

        self.assertEqual(compare.selected_matrix(Args()), (["elastic"], ["mute-late"]))


if __name__ == "__main__":
    unittest.main()

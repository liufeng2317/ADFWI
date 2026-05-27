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


if __name__ == "__main__":
    unittest.main()

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".md", ".rst", ".ipynb"}

FORBIDDEN_IMPORT_SURFACES = (
    "ADFWI.fwi.multiScaleProcessing",
    "ADFWI.fwi.normalization",
    "ADFWI.fwi.transforms.waveform",
    "from ADFWI.fwi import iteration",
    "from ADFWI.fwi import runtime",
    "from ADFWI.fwi.iteration import",
    "from ADFWI.fwi.runtime import",
    "import ADFWI.fwi.iteration",
    "import ADFWI.fwi.runtime",
)

DATA_FACADE_MARKERS = (
    "from ADFWI.fwi.data import",
    "ADFWI.fwi.data import",
    "import ADFWI.fwi.data",
)

HISTORICAL_DOC_PREFIXES = (
    "docs/version-plans/",
)

POLICY_TEST_PATH = "tests/test_import_surface_policy.py"

DATA_FACADE_ALLOWED_PATHS = {
    "docs/backend-usage.md",
    "tests/test_fwi_data_public_api.py",
}


def tracked_text_files():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    for relpath in result.stdout.splitlines():
        path = Path(relpath)
        if path.suffix in TEXT_SUFFIXES:
            yield relpath, REPO_ROOT / relpath


def is_historical_doc(relpath):
    return relpath.startswith(HISTORICAL_DOC_PREFIXES)


def is_policy_definition(relpath):
    return relpath == POLICY_TEST_PATH


class ImportSurfacePolicyTests(unittest.TestCase):
    def test_removed_or_namespace_only_import_surfaces_do_not_reappear(self):
        violations = []
        for relpath, path in tracked_text_files():
            if is_historical_doc(relpath) or is_policy_definition(relpath):
                continue
            text = path.read_text(errors="ignore")
            for marker in FORBIDDEN_IMPORT_SURFACES:
                if marker in text:
                    violations.append(f"{relpath}: {marker}")

        self.assertEqual(violations, [])

    def test_data_facade_is_only_used_by_public_docs_and_api_tests(self):
        violations = []
        for relpath, path in tracked_text_files():
            if is_historical_doc(relpath) or is_policy_definition(relpath) or relpath in DATA_FACADE_ALLOWED_PATHS:
                continue
            text = path.read_text(errors="ignore")
            for marker in DATA_FACADE_MARKERS:
                if marker in text:
                    violations.append(f"{relpath}: {marker}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()

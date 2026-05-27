import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".md", ".rst", ".ipynb"}

FORBIDDEN_IMPORT_SURFACES = (
    "ADFWI.fwi.data",
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

HISTORICAL_DOC_PREFIXES = (
    "docs/version-plans/",
)

POLICY_TEST_PATH = "tests/test_import_surface_policy.py"


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
        fullpath = REPO_ROOT / relpath
        if path.suffix in TEXT_SUFFIXES and fullpath.exists():
            yield relpath, fullpath


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

if __name__ == "__main__":
    unittest.main()

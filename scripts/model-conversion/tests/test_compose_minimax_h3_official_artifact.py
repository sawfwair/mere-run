import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).parents[1]
SCRIPT = SCRIPT_DIRECTORY / "compose_minimax_h3_official_artifact.py"
SPEC = importlib.util.spec_from_file_location(
    "compose_minimax_h3_official_artifact",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(SCRIPT_DIRECTORY))
COMPOSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPOSER)


class MiniMaxH3ArtifactComposerTests(unittest.TestCase):
    def test_atomic_text_does_not_mutate_hardlinked_base_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            output = root / "output.json"
            base.write_text("base\n", encoding="utf-8")
            os.link(base, output)
            self.assertEqual(base.stat().st_ino, output.stat().st_ino)

            COMPOSER.atomic_text(output, "release\n")

            self.assertEqual(base.read_text(encoding="utf-8"), "base\n")
            self.assertEqual(output.read_text(encoding="utf-8"), "release\n")
            self.assertNotEqual(base.stat().st_ino, output.stat().st_ino)


if __name__ == "__main__":
    unittest.main()

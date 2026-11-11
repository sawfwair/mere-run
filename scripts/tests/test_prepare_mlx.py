import importlib.util
from pathlib import Path
import sys
import unittest

MODULE = Path(__file__).resolve().parents[1] / "prepare_mlx.py"
spec = importlib.util.spec_from_file_location("prepare_mlx", MODULE)
prepare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prepare
spec.loader.exec_module(prepare)


class MLXHostPatchPolicyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = prepare.PatchManifest("swift", "core", "host.cpp", "base", "patched", "patch")

    def verify(self, outer=None, inner=None, digest="patched", swift="swift", core="core"):
        return prepare.validate_state(
            self.manifest, swift, core,
            {prepare.CORE_PATH} if outer is None else outer,
            {"host.cpp"} if inner is None else inner, digest)

    def test_pristine_pin(self):
        self.assertEqual(self.verify(set(), set(), "base"), "original")

    def test_exact_host_patch(self):
        self.assertEqual(self.verify(), "patched")

    def test_extra_shader_edit_is_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(inner={"host.cpp", "kernels/quantized_nax.h"})

    def test_extra_swift_edit_is_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(outer={prepare.CORE_PATH, "Source/MLX/Ops.swift"})

    def test_unknown_host_bytes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(digest="unreviewed")

    def test_wrong_pin_is_rejected(self):
        for field in ["swift", "core"]:
            with self.assertRaises(ValueError):
                self.verify(**{field: "different"})

    def test_pristine_source_does_not_hide_other_edits(self):
        with self.assertRaises(ValueError):
            self.verify(digest="base", inner={"kernels/quantized_nax.h"})


if __name__ == "__main__":
    unittest.main()

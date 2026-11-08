"""Contract checks for the audited Q4/Q8 converter; no checkpoint required."""
import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import convert_muse_glimmer_mlx as converter


class ConversionTests(unittest.TestCase):
    def test_precision_is_consistent_across_config_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({"model_type": "muse_glimmer"}))
            for bits in [4, 8]:
                config = converter.converted_config(root, bits)
                converter.write_manifest(root, bits)
                manifest = json.loads((root / "mererun_model.json").read_text())
                self.assertEqual(config["quantization"], dict(bits=bits, group_size=64, mode="affine"))
                self.assertEqual(manifest["precision"], f"int{bits}")
                self.assertEqual(manifest["quantization"]["bits"], bits)
            self.assertEqual(converter.converted_config(root)["quantization"]["bits"], 4)

    def test_selective_scope_preserves_embedding_and_vision(self):
        matrix = mx.zeros((64, 64), dtype=mx.bfloat16)
        for key in ["model.language_model.embed_tokens.weight", "model.vision_tower.layers.0.mlp.fc1.weight"]:
            self.assertFalse(converter.should_quantize(key, matrix, "selective"))
            self.assertTrue(converter.should_quantize(key, matrix, "compact"))
        for key in ["lm_head.weight", "model.language_model.layers.0.self_attn.q_proj.weight"]:
            self.assertTrue(converter.should_quantize(key, matrix, "selective"))

    def test_q8_reduces_error_and_keeps_bf16_quantization_metadata(self):
        source = mx.sin(mx.arange(4096, dtype=mx.float32) * 0.037).reshape(64, 64).astype(mx.bfloat16)
        errors = []
        for bits in [4, 8]:
            packed, scales, biases = converter.quantize_affine(source, bits)
            self.assertEqual(packed.shape, (64, 64 * bits // 32))
            self.assertEqual(scales.dtype, mx.bfloat16)
            self.assertEqual(biases.dtype, mx.bfloat16)
            restored = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
            errors.append(float(mx.mean(mx.square(restored.astype(mx.float32) - source.astype(mx.float32))).item()))
        self.assertLess(errors[1], errors[0] / 10)

    def test_symmetric_codes_preserve_zero_and_both_signs(self):
        self.assertEqual(converter.resolved_affine_method(4, "auto"), "standard")
        self.assertEqual(converter.resolved_affine_method(8, "auto"), "symmetric")
        self.assertEqual(converter.resolved_affine_method(8, "standard"), "standard")
        source = mx.tile(mx.array([0, -1, 1, 0], dtype=mx.bfloat16), (2, 16))
        for bits in [4, 8]:
            packed, scales, biases = converter.quantize_affine(source, bits, "symmetric")
            self.assertEqual(packed.shape, (2, 64 * bits // 32))
            self.assertEqual(scales.dtype, mx.bfloat16)
            restored = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
            self.assertTrue(bool(mx.all(mx.isfinite(restored)).item()))
            self.assertEqual(restored[:, ::4].tolist(), [[0] * 16] * 2)
            self.assertLess(float(mx.max(mx.abs(restored - source)).item()), 0.02)
        zero = mx.zeros((2, 64), dtype=mx.bfloat16)
        packed, scales, biases = converter.quantize_affine(zero, 8, "symmetric")
        self.assertEqual(mx.dequantize(packed, scales, biases, group_size=64, bits=8).tolist(), zero.tolist())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).parents[1] / "convert_qwen38_flash_next_mlx.py"
SPEC = importlib.util.spec_from_file_location("convert_qwen38_flash_next_mlx", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


class Qwen38FlashNextConverterTests(unittest.TestCase):
    def test_sanitizes_public_checkpoint_paths(self) -> None:
        self.assertEqual(
            CONVERTER.sanitize_key("model.language_model.layers.7.linear_attn.out_proj.weight"),
            "language_model.model.layers.7.linear_attn.out_proj.weight",
        )
        self.assertEqual(
            CONVERTER.sanitize_key("model.visual.patch_embed.proj.weight"),
            "vision_tower.patch_embed.proj.weight",
        )
        self.assertEqual(
            CONVERTER.sanitize_key("lm_head.weight"),
            "language_model.lm_head.weight",
        )

    def test_splits_fused_experts_into_switch_mlp_contract(self) -> None:
        plans = CONVERTER.expected_output_plans(
            "mixed",
            "model.language_model.layers.3.mlp.experts.gate_up_proj",
            (512, 1280, 2560),
            floating=True,
        )
        self.assertEqual(
            [plan.key for plan in plans],
            [
                "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight",
                "language_model.model.layers.3.mlp.switch_mlp.up_proj.weight",
            ],
        )
        self.assertTrue(all(plan.quantization == CONVERTER.EXPERT_Q2 for plan in plans))

    def test_mixed_profile_keeps_sensitive_surfaces_dense(self) -> None:
        dense = (
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight",
            "vision_tower.blocks.0.attn.qkv.weight",
            "mtp.fc_hidden.weight",
        )
        for key in dense:
            with self.subTest(key=key):
                self.assertIsNone(
                    CONVERTER.quantization_for("mixed", key, (2560, 2560))
                )

    def test_ngram_width_uses_group32_in_both_profiles(self) -> None:
        key = (
            "language_model.model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_127.weight"
        )
        for profile in ("q4", "mixed"):
            with self.subTest(profile=profile):
                self.assertEqual(
                    CONVERTER.quantization_for(profile, key, (2_500_012, 160)),
                    CONVERTER.NGRAM_Q4,
                )

    def test_mtp_experts_remain_q4_in_mixed_profile(self) -> None:
        key = "mtp.layers.0.mlp.switch_mlp.down_proj.weight"
        self.assertEqual(
            CONVERTER.quantization_for("mixed", key, (512, 2560, 640)),
            CONVERTER.Q4,
        )

    def test_only_zero_centered_text_norms_are_shifted(self) -> None:
        self.assertTrue(
            CONVERTER.is_shifted_text_norm(
                "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"
            )
        )
        self.assertTrue(
            CONVERTER.is_shifted_text_norm(
                "language_model.model.layers.3.self_attn.indexer.q_layernorm.weight"
            )
        )
        self.assertFalse(
            CONVERTER.is_shifted_text_norm(
                "language_model.model.layers.0.linear_attn.norm.weight"
            )
        )
        self.assertFalse(CONVERTER.is_shifted_text_norm("vision_tower.blocks.0.norm1.weight"))

    def test_pinned_full_artifact_preflight_contract(self) -> None:
        self.assertEqual(
            CONVERTER.EXPECTED_OUTPUT_LOGICAL_BYTES,
            {"q4": 104_741_817_208, "mixed": 73_094_818_808},
        )
        self.assertEqual(
            CONVERTER.EXPECTED_OUTPUT_TENSORS,
            {"q4": 3_817, "mixed": 3_615},
        )
        self.assertEqual(
            CONVERTER.EXPECTED_QUANTIZED_MODULES,
            {"q4": 1_055, "mixed": 954},
        )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "merge_minimax_h3_fasth3_q8.py"
SPEC = importlib.util.spec_from_file_location("merge_minimax_h3_fasth3_q8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


class MiniMaxH3FastH3Q8MergeTests(unittest.TestCase):
    def test_adapter_targets_match_runtime_global_qkv_layout(self) -> None:
        self.assertEqual(
            CONVERTER.adapter_target("transformer_blocks.17.attn.to_q"),
            CONVERTER.AdapterTarget("blocks.17.attn.qkv_proj", "query"),
        )
        self.assertEqual(
            CONVERTER.adapter_target("transformer_blocks.17.attn.to_out.0"),
            CONVERTER.AdapterTarget("blocks.17.attn.out_proj", None),
        )
        self.assertEqual(
            CONVERTER.adapter_target(
                "token_refiner.refiner_blocks.1.ff.net.0.proj"
            ),
            CONVERTER.AdapterTarget("token_refiner.blocks.1.mlp.fc1", None),
        )

    def test_difference_targets_match_compact_runtime_parameters(self) -> None:
        self.assertEqual(
            CONVERTER.difference_target("proj_in.diff"),
            "video_patch_proj.weight",
        )
        self.assertEqual(
            CONVERTER.difference_target("transformer_blocks.9.norm2.diff"),
            "blocks.9.norm2.weight",
        )
        self.assertTrue(
            CONVERTER.is_cache_covered(
                CONVERTER.difference_target("time_embedder.linear_1.diff")
            )
        )

    def test_inventory_requires_exact_fast_h3_tensor_closure(self) -> None:
        header = {}
        modules = []
        for prefix, count in (("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2)):
            for index in range(count):
                block = f"{prefix}.{index}"
                modules.extend(
                    [
                        f"{block}.attn.to_q",
                        f"{block}.attn.to_k",
                        f"{block}.attn.to_v",
                        f"{block}.attn.to_out.0",
                        f"{block}.ff.net.0.proj",
                        f"{block}.ff.net.2",
                    ]
                )
                if prefix == "transformer_blocks":
                    modules.append(f"{block}.adaln_proj.linear")
                    header[f"{block}.attn.to_gate_compress.set_weight"] = {}
        for module in modules:
            header[f"{module}.lora_A.weight"] = {}
            header[f"{module}.lora_B.weight"] = {}
        for index in range(CONVERTER.EXPECTED_DIFFS):
            header[f"transformer_blocks.{index % 50}.synthetic_{index}.diff"] = {}
        inventory = CONVERTER.adapter_inventory(
            header,
            {
                "format": CONVERTER.SOURCE_ADAPTER_FORMAT,
                "finetuned_model": CONVERTER.FAST_H3_MODEL,
            },
        )
        self.assertEqual(inventory["lora_pair_count"], 362)
        self.assertEqual(inventory["core_lora_pair_count"], 312)
        self.assertEqual(inventory["core_target_count"], 208)
        self.assertEqual(inventory["compression_gate_count"], 50)


if __name__ == "__main__":
    unittest.main()

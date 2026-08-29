import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "refresh_minimax_h3_adaln_cache_pack.py"
SPEC = importlib.util.spec_from_file_location("refresh_minimax_h3_adaln_cache_pack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFRESH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFRESH)


class MiniMaxH3AdaLNCacheRefreshTests(unittest.TestCase):
    def test_source_closure_has_exact_schedule_only_inventory(self) -> None:
        keys = REFRESH.required_source_keys()

        self.assertEqual(len(keys), REFRESH.SOURCE_TENSOR_COUNT)
        self.assertEqual(len(set(keys)), REFRESH.SOURCE_TENSOR_COUNT)
        self.assertIn("time_embedder.proj_in.weight", keys)
        self.assertIn("blocks.49.adaln_proj.linear.weight", keys)
        self.assertIn("final_layer.adaln_proj.linear.bias", keys)

    def test_content_range_requires_exact_requested_span(self) -> None:
        self.assertEqual(REFRESH.validate_content_range("bytes 8-15/64", 8, 15), 64)
        with self.assertRaisesRegex(ValueError, "does not match"):
            REFRESH.validate_content_range("bytes 8-14/64", 8, 15)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            REFRESH.validate_content_range(None, 8, 15)

    def test_subset_header_rewrites_offsets_without_changing_tensor_contract(self) -> None:
        source = {
            "second": {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]},
            "first": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        }

        encoded, selected = REFRESH.subset_header(source, ("second", "first"), {"format": "pt"})
        header_length = int.from_bytes(encoded[:8], "little")
        header = __import__("json").loads(encoded[8:8 + header_length])

        self.assertEqual([key for key, _ in selected], ["first", "second"])
        self.assertEqual(header["first"]["data_offsets"], [0, 8])
        self.assertEqual(header["second"]["data_offsets"], [8, 12])
        self.assertEqual(header["__metadata__"], {"format": "pt"})

    def test_refresh_includes_exact_lightx2v_eight_step_schedule(self) -> None:
        self.assertIn((9, 6.0, 3.0), REFRESH.CONVERTER.CACHE_SCHEDULES)
        self.assertEqual(len(REFRESH.CONVERTER.CACHE_SCHEDULES), 8)

    def test_tensor_closure_ignores_safetensors_header_padding(self) -> None:
        tensor = bytes(range(8))
        header = {
            "__metadata__": {"source_identity": "source"},
            "value": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"cache-{padding}.safetensors" for padding in (0, 8)]
            for path, extra_padding in zip(paths, (0, 8)):
                encoded = json.dumps(header, separators=(",", ":")).encode()
                encoded += b" " * ((-len(encoded) % 8) + extra_padding)
                path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + tensor)

            self.assertEqual(
                REFRESH.safetensors_tensor_closure(paths[0]),
                REFRESH.safetensors_tensor_closure(paths[1]),
            )


if __name__ == "__main__":
    unittest.main()

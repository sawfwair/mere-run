import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "convert_minimax_h3_official_mlx.py"
SPEC = importlib.util.spec_from_file_location("convert_minimax_h3_official_mlx", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class MiniMaxH3OfficialConverterTests(unittest.TestCase):
    def test_metal_cache_receipt_requires_exact_real_generation_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "adaln_cache.index.json"
            index.write_text("{}\n", encoding="utf-8")
            digest = CONVERTER.sha256_file(index)
            source_identity = "official-source"
            receipt = {
                "schema_version": CONVERTER.CACHE_RECEIPT_SCHEMA_VERSION,
                "format": CONVERTER.CACHE_RECEIPT_FORMAT,
                "source_identity": source_identity,
                "evaluation_backend": "mlx-metal",
                "generator": "mere.run model optimize",
                "cache_pack_index_sha256": digest,
                "source_tensor_closure_sha256": "a" * 64,
                "hardware": {"chip": "Apple M4 Max"},
                "real_generation_parity": [
                    {
                        "point_count": 9,
                        "full_mp4_sha256": "b" * 64,
                        "compact_mp4_sha256": "b" * 64,
                    },
                    {
                        "point_count": 21,
                        "full_mp4_sha256": "c" * 64,
                        "compact_mp4_sha256": "c" * 64,
                    },
                ],
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(
                CONVERTER.load_metal_cache_receipt(
                    receipt_path,
                    source_identity,
                    index,
                ),
                receipt,
            )

            receipt["real_generation_parity"][1]["compact_mp4_sha256"] = "d" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-identical"):
                CONVERTER.load_metal_cache_receipt(
                    receipt_path,
                    source_identity,
                    index,
                )


if __name__ == "__main__":
    unittest.main()

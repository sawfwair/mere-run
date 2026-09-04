import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "convert_parakeet_coreml.py"
SPEC = importlib.util.spec_from_file_location("convert_parakeet_coreml", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


class ParakeetCoreMLConverterTests(unittest.TestCase):
    def test_plan_pins_the_nvidia_source_and_static_encoder_shape(self) -> None:
        value = CONVERTER.plan()
        self.assertEqual(value["sourceRepository"], "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(
            value["sourceRevision"],
            "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        )
        self.assertEqual(value["sourceLicense"], "CC-BY-4.0")
        self.assertEqual(value["encoder"]["inputShape"], [1, 1501, 128])
        self.assertEqual(value["encoder"]["outputFeatures"], 1024)
        self.assertEqual(value["decoder"]["format"], "coreml-hybrid-v1")
        self.assertEqual(value["decoder"]["tensorCount"], 13)
        self.assertEqual(value["decoder"]["coreMLModelDirectory"], "decoder.mlmodelc")
        self.assertEqual(value["decoder"]["embeddingFile"], "embedding.f16")
        self.assertEqual(value["decoder"]["lanes"], 16)
        self.assertEqual(value["decoder"]["windowFrames"], 8)

    def test_source_verification_rejects_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                CONVERTER.verify_file(
                    path,
                    CONVERTER.FilePin(byte_count=7, sha256="0" * 64),
                )

    def test_materializes_verified_hub_symlinks_as_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repository = workspace / "hub-cache" / "models--test--model"
            snapshot = repository / "snapshots" / "revision"
            blobs = repository / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            payload = b"pinned"
            blob = blobs / "blob"
            blob.write_bytes(payload)
            (snapshot / "config.json").symlink_to(blob)
            pins = {
                "config.json": CONVERTER.FilePin(
                    byte_count=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            }

            source = CONVERTER.materialize_source(snapshot, workspace, pins)

            materialized = source / "config.json"
            self.assertTrue(materialized.is_file())
            self.assertFalse(materialized.is_symlink())
            self.assertEqual(materialized.read_bytes(), payload)

    def test_artifact_closure_is_sorted_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / CONVERTER.COMPILED_MODEL_DIRECTORY
            model.mkdir()
            (model / "z.bin").write_bytes(b"z")
            (model / "a.bin").write_bytes(b"a")
            pins = CONVERTER.artifact_pins(root, CONVERTER.COMPILED_MODEL_DIRECTORY)
            self.assertEqual(
                [pin["filename"] for pin in pins],
                ["encoder.mlmodelc/a.bin", "encoder.mlmodelc/z.bin"],
            )
            self.assertEqual(pins[0]["byteCount"], 1)
            self.assertEqual(
                pins[0]["sha256"],
                "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
            )

    def test_plan_is_json_serializable_without_conversion_dependencies(self) -> None:
        encoded = json.dumps(CONVERTER.plan(), sort_keys=True)
        self.assertIn(CONVERTER.SOURCE_REVISION, encoded)

    def test_builds_standalone_runtime_config_from_pinned_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["ParakeetForTDT"],
                        "blank_token_id": 2,
                        "decoder_hidden_size": 640,
                        "durations": [0, 1, 2, 3, 4],
                        "encoder_config": {
                            "attention_bias": False,
                            "conv_kernel_size": 9,
                            "hidden_size": 1024,
                            "intermediate_size": 4096,
                            "max_position_embeddings": 5000,
                            "num_attention_heads": 8,
                            "num_hidden_layers": 24,
                            "scale_input": False,
                            "subsampling_conv_channels": 256,
                            "subsampling_factor": 8,
                        },
                        "hidden_act": "relu",
                        "max_symbols_per_step": 10,
                        "num_decoder_layers": 2,
                        "vocab_size": 3,
                    }
                ),
                encoding="utf-8",
            )
            (source / "processor_config.json").write_text(
                json.dumps(
                    {
                        "feature_extractor": {
                            "feature_size": 128,
                            "hop_length": 160,
                            "n_fft": 512,
                            "padding_value": 0.0,
                            "preemphasis": 0.97,
                            "sampling_rate": 16000,
                            "win_length": 400,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (source / "tokenizer.json").write_text(
                json.dumps({"model": {"vocab": {"hello": 0, "world": 1}}}),
                encoding="utf-8",
            )

            config, vocabulary = CONVERTER.build_runtime_config(source)

            self.assertEqual(vocabulary, ["hello", "world"])
            self.assertEqual(config["mere"]["format"], "coreml-hybrid-v1")
            self.assertEqual(config["decoder"]["vocab_size"], 2)
            self.assertEqual(config["joint"]["num_extra_outputs"], 5)

    def test_decoder_mapping_has_exact_runtime_tensor_inventory(self) -> None:
        mapped = set(CONVERTER.DECODER_KEY_MAP.values())
        mapped.update(
            f"decoder.prediction.dec_rnn.lstm.{layer}.bias"
            for layer in (0, 1)
        )
        self.assertEqual(len(mapped), CONVERTER.DECODER_TENSOR_COUNT)


if __name__ == "__main__":
    unittest.main()

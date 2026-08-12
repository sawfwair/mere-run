#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.14.4"
# dependencies = [
#   "mlx==0.32.0",
# ]
# ///

"""Convert the exact Muse Glimmer 30B BF16 release to native MLX Q4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import mlx.core as mx


SOURCE_REPOSITORY = "meta-models/Muse-Glimmer-30B"
SOURCE_REVISION = "f84ecc3a0ea984a4c04542a84269e3d065350a6e"
MODEL_ID = "vision-chat-muse-glimmer-30b"
GROUP_SIZE = 64
BITS = 4
MODE = "affine"
EXPECTED_SOURCE_TENSORS = 1_436
EXPECTED_QUANTIZED_TENSORS = {
    "selective": 420,
    "compact": 721,
}
DEFAULT_SHARD_BYTES = 4 * 1024 * 1024 * 1024

PINNED_FILES: dict[str, dict[str, int | str]] = {
    "model-00001-of-00002.safetensors": {
        "byte_count": 49_950_112_952,
        "sha256": "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694",
    },
    "model-00002-of-00002.safetensors": {
        "byte_count": 9_603_322_320,
        "sha256": "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381",
    },
    "model.safetensors.index.json": {
        "byte_count": 132_674,
        "sha256": "7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba",
    },
    "config.json": {
        "byte_count": 5_109,
        "sha256": "5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14",
    },
    "generation_config.json": {
        "byte_count": 148,
        "sha256": "b0e427c998641420eb4091cc85d4e15643fb57f89222834a14ec76430625b6fb",
    },
    "processor_config.json": {
        "byte_count": 1_084,
        "sha256": "97e2a486dd9866b81f40cf4b8bc0c9ced9a7cd8a5bc65aa4cc2f4de0712dae77",
    },
    "tokenizer.json": {
        "byte_count": 28_129_897,
        "sha256": "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d",
    },
    "tokenizer_config.json": {
        "byte_count": 79_936,
        "sha256": "781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5",
    },
    "chat_template.jinja": {
        "byte_count": 7_167,
        "sha256": "114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e",
    },
    "README.md": {
        "byte_count": 16_878,
        "sha256": "1647ac8916f9c3c7c8ad508f6509d79f0c61d434b03b73f57ec6ad629b6adc13",
    },
    "LICENSE": {
        "byte_count": 11_358,
        "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    },
    "USAGE_POLICY.md": {
        "byte_count": 5_230,
        "sha256": "98a14dab9fd97de1666dc8589d399efab1e7fc3ba4e6230d037d6637ba9481d3",
    },
}

METADATA_FILES = [
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
    "LICENSE",
    "USAGE_POLICY.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, pin: dict[str, int | str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned source file is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != pin["byte_count"]:
        raise ValueError(
            f"{path.name} has {actual_bytes} bytes; expected {pin['byte_count']}"
        )
    actual_hash = sha256(path)
    if actual_hash != pin["sha256"]:
        raise ValueError(
            f"{path.name} has SHA-256 {actual_hash}; expected {pin['sha256']}"
        )


def verify_environment() -> None:
    actual = importlib.metadata.version("mlx")
    if actual != "0.32.0":
        raise RuntimeError(f"mlx {actual} is installed; expected 0.32.0")


def load_source_index(source: Path) -> dict[str, Any]:
    value = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Pinned model index has no weight_map object")
    if len(weight_map) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(
            f"Pinned model index has {len(weight_map)} tensors; "
            f"expected {EXPECTED_SOURCE_TENSORS}"
        )
    expected_shards = {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    if set(weight_map.values()) != expected_shards:
        raise ValueError("Pinned model index references an unexpected shard set")
    return value


def should_quantize(key: str, value: mx.array, scope: str) -> bool:
    if not key.endswith(".weight") or value.ndim != 2:
        return False
    if key.endswith("position_embedding_table.weight"):
        return False
    if value.shape[-1] % GROUP_SIZE != 0:
        return False
    if scope == "compact":
        return True
    if key == "lm_head.weight":
        return True
    if key in {
        "model.vision_adapter.fc1.weight",
        "model.vision_adapter.fc2.weight",
        "model.vision_projection.weight",
    }:
        return True
    if not key.startswith("model.language_model.layers."):
        return False
    return any(
        fragment in key
        for fragment in (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".self_attn.gate_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        )
    )


def array_bytes(value: mx.array) -> int:
    return int(value.nbytes)


class ShardWriter:
    def __init__(self, root: Path, target_bytes: int) -> None:
        self.root = root
        self.target_bytes = target_bytes
        self.arrays: dict[str, mx.array] = {}
        self.bytes = 0
        self.parts: list[tuple[Path, list[str]]] = []

    def append(self, key: str, value: mx.array) -> None:
        value_bytes = array_bytes(value)
        if self.arrays and self.bytes + value_bytes > self.target_bytes:
            self.flush()
        self.arrays[key] = value
        self.bytes += value_bytes

    def flush(self) -> None:
        if not self.arrays:
            return
        mx.eval(*self.arrays.values())
        path = self.root / f"part-{len(self.parts) + 1:05d}.safetensors"
        mx.save_safetensors(str(path), self.arrays, metadata={"format": "mlx"})
        self.parts.append((path, sorted(self.arrays)))
        self.arrays = {}
        self.bytes = 0

    def finalize(self) -> tuple[dict[str, str], list[Path]]:
        self.flush()
        total = len(self.parts)
        weight_map: dict[str, str] = {}
        outputs: list[Path] = []
        for index, (temporary, keys) in enumerate(self.parts, start=1):
            filename = f"model-{index:05d}-of-{total:05d}.safetensors"
            final = temporary.with_name(filename)
            temporary.rename(final)
            outputs.append(final)
            for key in keys:
                weight_map[key] = filename
        return weight_map, outputs


def converted_config(source: Path) -> dict[str, Any]:
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "muse_glimmer":
        raise ValueError("Pinned config is not a muse_glimmer model")
    config["quantization"] = {
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "mode": MODE,
    }
    return config


def write_manifest(output: Path) -> None:
    manifest = {
        "schemaVersion": 3,
        "id": MODEL_ID,
        "engine": "muse-glimmer",
        "family": "muse",
        "tier": "latest",
        "variant": "standard",
        "precision": "int4",
        "quantization": {
            "bits": BITS,
            "groupSize": GROUP_SIZE,
            "scheme": "mlx-affine",
        },
        "supports": ["chat", "code_generation", "vision_chat"],
        "components": {
            "tokenizer": {"type": "local", "path": "."},
            "text_encoder": {"type": "local", "path": "."},
        },
        "upstreamRepoId": f"{SOURCE_REPOSITORY}@{SOURCE_REVISION}",
        "sources": [
            {
                "role": "base-model",
                "repository": SOURCE_REPOSITORY,
                "revision": SOURCE_REVISION,
                "destination_path": ".",
            }
        ],
    }
    (output / "mererun_model.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def convert(source: Path, output: Path, shard_bytes: int, quantization_scope: str) -> None:
    for filename, pin in PINNED_FILES.items():
        verify_file(source / filename, pin)
    source_index = load_source_index(source)
    writer = ShardWriter(output, shard_bytes)
    quantized_count = 0
    source_count = 0

    for shard_name in sorted(set(source_index["weight_map"].values())):
        arrays = mx.load(str(source / shard_name))
        expected_keys = sorted(
            key for key, value in source_index["weight_map"].items()
            if value == shard_name
        )
        if sorted(arrays) != expected_keys:
            raise ValueError(f"{shard_name} tensor keys do not match the pinned index")
        for key in expected_keys:
            value = arrays[key]
            source_count += 1
            if should_quantize(key, value, quantization_scope):
                packed, scales, biases = mx.quantize(
                    value,
                    group_size=GROUP_SIZE,
                    bits=BITS,
                    mode=MODE,
                )
                writer.append(key, packed)
                writer.append(key.removesuffix(".weight") + ".scales", scales)
                writer.append(key.removesuffix(".weight") + ".biases", biases)
                quantized_count += 1
            else:
                dense = value.astype(mx.bfloat16) if mx.issubdtype(value.dtype, mx.floating) else value
                writer.append(key, dense)
        del arrays

    if source_count != EXPECTED_SOURCE_TENSORS:
        raise ValueError(f"Converted {source_count} source tensors; expected {EXPECTED_SOURCE_TENSORS}")
    expected_quantized = EXPECTED_QUANTIZED_TENSORS[quantization_scope]
    if quantized_count != expected_quantized:
        raise ValueError(
            f"Quantized {quantized_count} tensors; expected {expected_quantized} "
            f"for {quantization_scope} scope"
        )

    weight_map, weights = writer.finalize()
    expected_output_arrays = EXPECTED_SOURCE_TENSORS + (2 * expected_quantized)
    if len(weight_map) != expected_output_arrays:
        raise ValueError(
            f"Converted index has {len(weight_map)} arrays; expected {expected_output_arrays}"
        )
    total_size = sum(path.stat().st_size for path in weights)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    (output / "config.json").write_text(
        json.dumps(converted_config(source), indent=2, sort_keys=True) + "\n"
    )
    for filename in METADATA_FILES:
        shutil.copyfile(source / filename, output / filename)
        verify_file(output / filename, PINNED_FILES[filename])
    write_manifest(output)

    artifacts = [
        {
            "filename": path.name,
            "byte_count": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in [*weights, output / "model.safetensors.index.json", output / "config.json"]
    ]
    receipt = {
        "converter": "scripts/model-conversion/convert_muse_glimmer_mlx.py",
        "converter_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "files": PINNED_FILES,
            "tensor_count": source_count,
        },
        "quantization": {
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "mode": MODE,
            "scope": quantization_scope,
            "quantized_tensor_count": quantized_count,
            "dense_dtype": "bfloat16",
        },
        "tools": {"python": "3.14.4", "mlx": "0.32.0"},
        "artifacts": artifacts,
    }
    (output / "MERERUN_CONVERSION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and convert Meta's pinned Muse Glimmer 30B BF16 release "
            "to native MLX affine Q4/group-64 with selective or compact scope."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument(
        "--quantization-scope",
        choices=sorted(EXPECTED_QUANTIZED_TENSORS),
        default="selective",
        help=(
            "selective keeps the vision tower and token embedding BF16; compact "
            "quantizes every eligible matrix (default: selective)"
        ),
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    if args.shard_bytes <= 0:
        raise ValueError("--shard-bytes must be greater than zero")

    verify_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.converting-", dir=output.parent)
    )
    try:
        convert(source, temporary, args.shard_bytes, args.quantization_scope)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(output)


if __name__ == "__main__":
    main()

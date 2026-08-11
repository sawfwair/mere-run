#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.14.4"
# dependencies = [
#   "mlx==0.32.0",
# ]
# ///

"""Convert Meta's exact Muse Glimmer DFlash assistant to native MLX Q4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import mlx.core as mx


SOURCE_REPOSITORY = "meta-models/Muse-Glimmer-30B-assistant"
SOURCE_REVISION = "2c86316d689027b91123638739743fef1d425233"
MODEL_ID = "vision-chat-muse-glimmer-30b-assistant"
GROUP_SIZE = 64
BITS = 4
MODE = "affine"
EXPECTED_SOURCE_TENSORS = 58
EXPECTED_QUANTIZED_TENSORS = 36
EXPECTED_OUTPUT_ARRAYS = 130

PINNED_FILES: dict[str, dict[str, int | str]] = {
    "model.safetensors": {
        "byte_count": 5_111_976_608,
        "sha256": "fd88d337eb84f8d0e6ba33a7684d7efa6722d4460ba4d6badca9699418392a84",
    },
    "config.json": {
        "byte_count": 883,
        "sha256": "38915167b64b1e6405492aacae5b1b4511b6431163d2960b9bd25821df6fa30a",
    },
    "README.md": {
        "byte_count": 17_339,
        "sha256": "4a2bcc36dbb8088cd887def9233f3c5524d4ed1e27622f3c4a521d3676952c46",
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


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(file_path: Path, pin: dict[str, int | str]) -> None:
    if not file_path.is_file():
        raise FileNotFoundError(f"Pinned source file is missing: {file_path}")
    if file_path.stat().st_size != pin["byte_count"]:
        raise ValueError(
            f"{file_path.name} has {file_path.stat().st_size} bytes; "
            f"expected {pin['byte_count']}"
        )
    actual_hash = sha256(file_path)
    if actual_hash != pin["sha256"]:
        raise ValueError(
            f"{file_path.name} has SHA-256 {actual_hash}; expected {pin['sha256']}"
        )


def verify_environment() -> None:
    actual = importlib.metadata.version("mlx")
    if actual != "0.32.0":
        raise RuntimeError(f"mlx {actual} is installed; expected 0.32.0")


def should_quantize(key: str, value: mx.array) -> bool:
    return (
        key.endswith(".weight")
        and value.ndim == 2
        and value.shape[-1] % GROUP_SIZE == 0
    )


def converted_config(source: Path) -> dict[str, Any]:
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "muse_glimmer_assistant":
        raise ValueError("Pinned config is not a muse_glimmer_assistant model")
    if config.get("target_layer_ids") != [1, 13, 25, 37, 49]:
        raise ValueError("Pinned assistant target_layer_ids changed")
    if config.get("block_size") != 16 or config.get("mask_token_id") != 201818:
        raise ValueError("Pinned assistant DFlash block contract changed")
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
        "supports": ["chat"],
        "components": {"text_encoder": {"type": "local", "path": "."}},
        "upstreamRepoId": f"{SOURCE_REPOSITORY}@{SOURCE_REVISION}",
        "sources": [
            {
                "role": "draft-model",
                "repository": SOURCE_REPOSITORY,
                "revision": SOURCE_REVISION,
                "destination_path": ".",
            }
        ],
    }
    (output / "mererun_model.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def convert(source: Path, output: Path) -> None:
    for filename, pin in PINNED_FILES.items():
        verify_file(source / filename, pin)
    arrays = mx.load(str(source / "model.safetensors"))
    if len(arrays) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(
            f"Pinned assistant has {len(arrays)} tensors; "
            f"expected {EXPECTED_SOURCE_TENSORS}"
        )

    converted: dict[str, mx.array] = {}
    quantized_count = 0
    for key in sorted(arrays):
        value = arrays[key]
        if should_quantize(key, value):
            packed, scales, biases = mx.quantize(
                value,
                group_size=GROUP_SIZE,
                bits=BITS,
                mode=MODE,
            )
            prefix = key.removesuffix(".weight")
            converted[key] = packed
            converted[prefix + ".scales"] = scales
            converted[prefix + ".biases"] = biases
            quantized_count += 1
        else:
            converted[key] = (
                value.astype(mx.bfloat16)
                if mx.issubdtype(value.dtype, mx.floating)
                else value
            )
    if quantized_count != EXPECTED_QUANTIZED_TENSORS:
        raise ValueError(
            f"Quantized {quantized_count} tensors; expected {EXPECTED_QUANTIZED_TENSORS}"
        )
    if len(converted) != EXPECTED_OUTPUT_ARRAYS:
        raise ValueError(
            f"Converted {len(converted)} arrays; expected {EXPECTED_OUTPUT_ARRAYS}"
        )

    mx.eval(*converted.values())
    weights = output / "model.safetensors"
    mx.save_safetensors(str(weights), converted, metadata={"format": "mlx"})
    (output / "config.json").write_text(
        json.dumps(converted_config(source), indent=2, sort_keys=True) + "\n"
    )
    for filename in ["README.md", "LICENSE", "USAGE_POLICY.md"]:
        shutil.copyfile(source / filename, output / filename)
        verify_file(output / filename, PINNED_FILES[filename])
    write_manifest(output)

    receipt = {
        "converter": "scripts/model-conversion/convert_muse_glimmer_assistant_mlx.py",
        "converter_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "files": PINNED_FILES,
            "tensor_count": EXPECTED_SOURCE_TENSORS,
        },
        "quantization": {
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "mode": MODE,
            "quantized_tensor_count": quantized_count,
            "dense_dtype": "bfloat16",
        },
        "tools": {"python": "3.14.4", "mlx": "0.32.0"},
        "artifacts": [
            {
                "filename": weights.name,
                "byte_count": weights.stat().st_size,
                "sha256": sha256(weights),
            },
            {
                "filename": "config.json",
                "byte_count": (output / "config.json").stat().st_size,
                "sha256": sha256(output / "config.json"),
            },
        ],
    }
    (output / "MERERUN_CONVERSION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Meta's pinned Muse Glimmer DFlash assistant to MLX Q4."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    verify_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        convert(source, temporary)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

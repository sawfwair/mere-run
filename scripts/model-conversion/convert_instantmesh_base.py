#!/usr/bin/env python3
"""Convert the exact Apache-2.0 InstantMesh Base reconstruction CKPT.

This audited release tool is never an inference dependency. It accepts only
the pinned TencentARC checkpoint, uses PyTorch's weights-only unpickler, keeps
only `lrm_generator` reconstruction tensors, verifies the committed inventory,
and emits a deterministic non-executable safetensors package.

The Zero123++/custom diffusion weights are intentionally never accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import safetensors
import torch
from safetensors.torch import save_file


SOURCE = {
    "repository": "TencentARC/InstantMesh",
    "revision": "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
    "filename": "instant_mesh_base.ckpt",
    "byteCount": 1_253_574_354,
    "sha256": "22701cd25201d624ebb1568b93cf91b43a2c32006835c08fe73e1f3c9f6c44b5",
    "modelID": "image-3d-instantmesh-base",
    "sourceCodeRepository": "TencentARC/InstantMesh",
    "sourceCodeRevision": "08822c52fdc399b93ea00e4fa9e596344ed52ccc",
}
EXPECTED_TENSOR_COUNT = 455
EXPECTED_SCALAR_COUNT = 313_352_516
PREFIX = "lrm_generator."
CONVERTER_VERSION = 1
CONVERSION_ENVIRONMENT = {
    "python": "3.11.15",
    "torch": "2.13.0",
    "safetensors": "0.8.0",
}
EXPECTED_OUTPUTS = {
    "model.safetensors": {
        "byteCount": 1_253_463_832,
        "sha256": "2380601d17f6a817de0bf5328188ccea397af9d75c07b4b3cc476322dcca76af",
    },
    "config.json": {
        "byteCount": 486,
        "sha256": "33f89581172ab2d46759a1632b6e57ca9f9f1c6c23567468157cb4b48a3bc781",
    },
    "SOURCE.json": {
        "byteCount": 1_074,
        "sha256": "9fbda0d3875744353a4ca6ee9ee836182cb46f72aa0d241c30ee62b746d60061",
    },
    "LICENSE": {
        "byteCount": 11_357,
        "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_environment() -> None:
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__.split("+")[0],
        "safetensors": safetensors.__version__,
    }
    if actual != CONVERSION_ENVIRONMENT:
        raise ValueError(
            "conversion environment mismatch: "
            f"expected {CONVERSION_ENVIRONMENT}, found {actual}; "
            "install scripts/model-conversion/requirements-vfx.txt with Python 3.11.15"
        )


def validate_output(path: Path) -> None:
    expected = EXPECTED_OUTPUTS[path.name]
    actual_byte_count = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_byte_count != expected["byteCount"] or actual_sha256 != expected["sha256"]:
        raise ValueError(
            f"{path.name} reproducibility mismatch: expected "
            f"{expected['byteCount']} bytes/{expected['sha256']}, found "
            f"{actual_byte_count} bytes/{actual_sha256}"
        )


def validate_source(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source checkpoint must be a regular non-symlink file: {path}")
    if path.stat().st_size != SOURCE["byteCount"]:
        raise ValueError(
            f"source byte count mismatch: expected {SOURCE['byteCount']}, found {path.stat().st_size}"
        )
    actual_hash = sha256(path)
    if actual_hash != SOURCE["sha256"]:
        raise ValueError(
            f"source SHA-256 mismatch: expected {SOURCE['sha256']}, found {actual_hash}"
        )


def load_state(path: Path) -> dict[str, torch.Tensor]:
    root = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(root, dict) or set(root) != {"state_dict"}:
        raise ValueError("checkpoint root must contain only the Lightning state_dict")
    source_state = root["state_dict"]
    if not isinstance(source_state, dict) or not all(isinstance(key, str) for key in source_state):
        raise ValueError("checkpoint state_dict must be string-keyed")
    unexpected = sorted(
        key for key in source_state if not key.startswith(PREFIX) or "source_camera" in key
    )
    if unexpected:
        raise ValueError(f"checkpoint contains excluded/non-reconstruction tensors: {unexpected[:8]}")
    state: dict[str, torch.Tensor] = {}
    for source_key, tensor in sorted(source_state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint value is not a tensor: {source_key}")
        key = source_key.removeprefix(PREFIX)
        if key in state:
            raise ValueError(f"duplicate stripped tensor key: {key}")
        value = tensor.detach().cpu()
        state[key] = value if value.is_contiguous() else value.contiguous()
    scalar_count = sum(tensor.numel() for tensor in state.values())
    if len(state) != EXPECTED_TENSOR_COUNT or scalar_count != EXPECTED_SCALAR_COUNT:
        raise ValueError(
            "checkpoint inventory totals mismatch: "
            f"expected {EXPECTED_TENSOR_COUNT} tensors/{EXPECTED_SCALAR_COUNT} scalars, "
            f"found {len(state)} tensors/{scalar_count} scalars"
        )
    return state


def tensor_inventory(state: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {"dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape)}
        for key, value in sorted(state.items())
    }


def validate_inventory(
    actual: dict[str, dict[str, Any]],
    expected_path: Path,
    write_inventory_only: bool,
) -> None:
    if write_inventory_only:
        write_json(expected_path, actual)
        return
    if not expected_path.is_file():
        raise ValueError(f"frozen tensor inventory does not exist: {expected_path}")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if actual == expected:
        return
    actual_keys = set(actual)
    expected_keys = set(expected)
    mismatched = sorted(
        key for key in actual_keys & expected_keys if actual[key] != expected[key]
    )
    raise ValueError(
        "checkpoint tensor inventory mismatch: "
        f"missing={sorted(expected_keys - actual_keys)[:8]}, "
        f"extra={sorted(actual_keys - expected_keys)[:8]}, "
        f"shapeOrDtype={mismatched[:8]}"
    )


def convert(args: argparse.Namespace) -> None:
    validate_environment()
    source = args.source.resolve()
    output = args.output.resolve()
    inventory_path = args.inventory.resolve()
    validate_source(source)
    state = load_state(source)
    inventory = tensor_inventory(state)
    validate_inventory(inventory, inventory_path, args.write_inventory_only)
    if args.write_inventory_only:
        print(inventory_path)
        return
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        weights_path = staging / "model.safetensors"
        save_file(state, str(weights_path))
        weights_hash = sha256(weights_path)
        write_json(
            staging / "config.json",
            {
                "architecture": "instantmesh-base-reconstruction-only",
                "cameraDimension": 16,
                "conditioningImageSize": 320,
                "decoderHiddenLayers": 3,
                "decoderHiddenSize": 64,
                "gridResolution": 128,
                "gridScale": 2.1,
                "imageEncoder": "facebook/dino-vitb16-with-camera-adaln",
                "inputViewCounts": [4, 6],
                "planeChannels": 40,
                "planeResolution": 64,
                "transformerAttentionHeads": 16,
                "transformerLayers": 12,
                "transformerDimension": 1024,
                "viewGenerator": None,
            },
        )
        write_json(
            staging / "SOURCE.json",
            {
                "conversion": {
                    "converter": Path(__file__).name,
                    "converterVersion": CONVERTER_VERSION,
                    "environment": CONVERSION_ENVIRONMENT,
                    "outputByteCount": weights_path.stat().st_size,
                    "outputFile": weights_path.name,
                    "outputSHA256": weights_hash,
                    "scalarCount": EXPECTED_SCALAR_COUNT,
                    "tensorCount": EXPECTED_TENSOR_COUNT,
                },
                "exclusions": {
                    "diffusion_pytorch_model.bin": "Zero123++ view generation is excluded",
                    "runtimePython": False,
                    "viewGeneration": False,
                },
                "license": "Apache-2.0 reconstruction checkpoint",
                "modelID": SOURCE["modelID"],
                "source": {key: value for key, value in SOURCE.items() if key != "modelID"},
            },
        )
        license_file = args.license_file.resolve()
        if not license_file.is_file() or license_file.is_symlink():
            raise ValueError(f"license must be a regular non-symlink file: {license_file}")
        shutil.copyfile(license_file, staging / "LICENSE")
        for name in EXPECTED_OUTPUTS:
            validate_output(staging / name)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument(
        "--inventory",
        type=Path,
        default=Path(__file__).with_name("instantmesh-base-tensor-inventory.json"),
    )
    result.add_argument("--license-file", required=True, type=Path)
    result.add_argument("--write-inventory-only", action="store_true", help=argparse.SUPPRESS)
    return result


def main() -> int:
    try:
        convert(parser().parse_args())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

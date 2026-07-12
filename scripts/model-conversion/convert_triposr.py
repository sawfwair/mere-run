#!/usr/bin/env python3
"""Convert the exact pinned MIT TripoSR checkpoint to safetensors.

This is audited release tooling, never a mere.run inference dependency. The
checkpoint bytes are verified before PyTorch's weights-only loader is invoked,
and every tensor is compared with the committed key/dtype/shape inventory.
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
    "repository": "stabilityai/TripoSR",
    "revision": "5b521936b01fbe1890f6f9baed0254ab6351c04a",
    "filename": "model.ckpt",
    "byteCount": 1_677_246_742,
    "sha256": "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee",
    "modelID": "image-3d-triposr",
    "sourceCodeRepository": "VAST-AI-Research/TripoSR",
    "sourceCodeRevision": "107cefdc244c39106fa830359024f6a2f1c78871",
}
EXPECTED_TENSOR_COUNT = 549
EXPECTED_SCALAR_COUNT = 419_275_628
CONVERTER_VERSION = 1
CONVERSION_ENVIRONMENT = {
    "python": "3.11.15",
    "torch": "2.13.0",
    "safetensors": "0.8.0",
}
EXPECTED_OUTPUTS = {
    "model.safetensors": {
        "byteCount": 1_677_170_936,
        "sha256": "f72bb520b8b1a5639600ac818496f22d6ccb3b42d3942412bd1e2375ef780a2b",
    },
    "config.json": {
        "byteCount": 378,
        "sha256": "89bd2abd8024fba7474ca584b962aa1f50c67db2c6317cb86d04a3bfddd8f22c",
    },
    "SOURCE.json": {
        "byteCount": 855,
        "sha256": "5c12adbc30f80524007d946f78df11da077a0df6ba25b3409e566cda6afb902c",
    },
    "LICENSE": {
        "byteCount": 1_080,
        "sha256": "ade0a66629bdd7e01e46b3296b3851cff0fd27989bca53da470ad6e96ed620fb",
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
    # mmap keeps peak host memory bounded for this 1.67 GB state dict. The
    # weights-only unpickler does not permit arbitrary globals or reducers.
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("checkpoint root must be a string-keyed state dictionary")
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise ValueError("checkpoint contains a non-tensor state-dict value")
    state = {
        key: tensor.detach().cpu() if tensor.is_contiguous() else tensor.detach().cpu().contiguous()
        for key, tensor in sorted(value.items())
    }
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
        # A sorted tensor table without metadata is byte-stable across runs;
        # provenance lives in canonical sorted JSON beside it.
        save_file(state, str(weights_path))
        weights_hash = sha256(weights_path)
        write_json(
            staging / "config.json",
            {
                "architecture": "triposr",
                "conditioningImageSize": 512,
                "decoderHiddenLayers": 9,
                "decoderHiddenSize": 64,
                "densityBias": -1.0,
                "densityThreshold": 25.0,
                "imageEncoder": "facebook/dino-vitb16",
                "planeChannels": 40,
                "planeSize": 64,
                "rendererRadius": 0.87,
                "transformerAttentionHeads": 16,
                "transformerLayers": 16,
                "transformerTokenChannels": 1024,
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
                "license": "MIT",
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
        default=Path(__file__).with_name("triposr-tensor-inventory.json"),
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

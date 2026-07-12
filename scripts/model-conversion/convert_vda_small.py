#!/usr/bin/env python3
"""Convert a pinned Video Depth Anything Small checkpoint to safetensors.

This is release tooling, not a mere.run inference dependency. It accepts only
the two audited Apache-2.0 Small checkpoints, verifies their bytes before
deserialization, checks every tensor against a frozen inventory, and emits a
deterministic native package with provenance and checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch
from safetensors.torch import save_file


CHECKPOINTS: dict[str, dict[str, Any]] = {
    "relative": {
        "repository": "depth-anything/Video-Depth-Anything-Small",
        "revision": "256875362cff76724b920335dfb4b29dd611f66e",
        "filename": "video_depth_anything_vits.pth",
        "byteCount": 116_440_756,
        "sha256": "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
        "modelID": "vision-depth-vda-small",
        "depthSemantics": "affine-relative",
    },
    "metric": {
        "repository": "depth-anything/Metric-Video-Depth-Anything-Small",
        "revision": "273d090f2ce17df50c2872d82c8322c45da5b4dd",
        "filename": "metric_video_depth_anything_vits.pth",
        "byteCount": 116_444_063,
        "sha256": "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33",
        "modelID": "vision-depth-vda-small-metric",
        "depthSemantics": "metric-meters",
    },
}

EXPECTED_TENSOR_COUNT = 351
EXPECTED_SCALAR_COUNT = 29_080_193


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_inventory(state: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {"dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape)}
        for key, value in sorted(state.items())
    }


def validate_source(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise ValueError(f"source checkpoint does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected["byteCount"]:
        raise ValueError(
            f"source byte count mismatch: expected {expected['byteCount']}, found {actual_size}"
        )
    actual_hash = sha256(path)
    if actual_hash != expected["sha256"]:
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected['sha256']}, found {actual_hash}"
        )


def load_state(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("checkpoint root must be a string-keyed state dictionary")
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise ValueError("checkpoint contains a non-tensor state-dict value")
    state = {key: tensor.detach().cpu().contiguous() for key, tensor in sorted(value.items())}
    scalar_count = sum(tensor.numel() for tensor in state.values())
    if len(state) != EXPECTED_TENSOR_COUNT or scalar_count != EXPECTED_SCALAR_COUNT:
        raise ValueError(
            "checkpoint inventory totals mismatch: "
            f"expected {EXPECTED_TENSOR_COUNT} tensors/{EXPECTED_SCALAR_COUNT} scalars, "
            f"found {len(state)} tensors/{scalar_count} scalars"
        )
    return state


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
    if actual != expected:
        actual_keys = set(actual)
        expected_keys = set(expected)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        mismatched = sorted(
            key for key in actual_keys & expected_keys if actual[key] != expected[key]
        )
        raise ValueError(
            "checkpoint tensor inventory mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}, shapeOrDtype={mismatched[:8]}"
        )


def convert(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    inventory_path = args.inventory.resolve()
    expected = CHECKPOINTS[args.variant]
    validate_source(source, expected)
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
        # Keep provenance in sorted SOURCE.json. safetensors' metadata map is
        # serialized through an unordered native map and is not byte-stable
        # across processes, while the sorted tensor table is deterministic.
        save_file(state, str(weights_path))
        weights_hash = sha256(weights_path)
        config = {
            "architecture": "video-depth-anything-small",
            "backbone": "dinov2-vits14",
            "depthSemantics": expected["depthSemantics"],
            "featureChannels": 64,
            "intermediateLayers": [2, 5, 8, 11],
            "projectedChannels": [48, 96, 192, 384],
            "temporalAttentionBlocks": 2,
            "temporalAttentionHeads": 8,
            "temporalFrameCount": 32,
            "temporalOverlap": 10,
            "temporalTransformerBlocks": 1,
        }
        write_json(staging / "config.json", config)
        source_record = {
            "conversion": {
                "converter": Path(__file__).name,
                "converterSHA256": sha256(Path(__file__).resolve()),
                "outputByteCount": weights_path.stat().st_size,
                "outputFile": weights_path.name,
                "outputSHA256": weights_hash,
                "tensorCount": len(state),
                "scalarCount": sum(tensor.numel() for tensor in state.values()),
            },
            "license": "Apache-2.0",
            "modelID": expected["modelID"],
            "source": {
                "byteCount": expected["byteCount"],
                "filename": expected["filename"],
                "repository": expected["repository"],
                "revision": expected["revision"],
                "sha256": expected["sha256"],
            },
        }
        write_json(staging / "SOURCE.json", source_record)
        if args.license_file:
            shutil.copyfile(args.license_file.resolve(), staging / "LICENSE")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--variant", required=True, choices=sorted(CHECKPOINTS))
    result.add_argument("--source", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument(
        "--inventory",
        type=Path,
        default=Path(__file__).with_name("vda-small-tensor-inventory.json"),
    )
    result.add_argument("--license-file", type=Path)
    result.add_argument(
        "--write-inventory-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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

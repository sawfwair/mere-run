#!/usr/bin/env python3
"""Deterministically convert the pinned TerraMind Flood Lightning checkpoint to MLX safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

MODEL_ID = "vision-flood-terramind-base"
SOURCE_REPOSITORY = "ibm-esa-geospatial/TerraMind-base-Flood"
SOURCE_REVISION = "1e4b2429d17234922f8d92beb0d725af4db85c08"
SOURCE_CHECKPOINT = "TerraMind_v1_base_ImpactMesh_flood.pt"
SOURCE_CONFIGURATION = "terramind_v1_base_impactmesh_flood.yaml"
SOURCE_CHECKPOINT_SHA256 = "22627584c2db618c2f6ddb64b411a95762a893becb25104e3f66bfebecaa71e9"
FORMAT = "mere.run/terramind-flood-mlx-v1"
CONVERTER = "scripts/convert-terramind-flood-mlx.py@v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, help="Pinned Lightning .pt checkpoint")
    parser.add_argument("--configuration", type=pathlib.Path, help="Pinned TerraTorch YAML configuration")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Converted model directory")
    parser.add_argument(
        "--dtype",
        choices=["float32"],
        default="float32",
        help="Conversion precision. Flood boundary parity requires float32.",
    )
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sources(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    if args.checkpoint and args.configuration:
        return args.checkpoint.resolve(), args.configuration.resolve()
    if bool(args.checkpoint) != bool(args.configuration):
        raise SystemExit("--checkpoint and --configuration must be supplied together")
    from huggingface_hub import hf_hub_download

    checkpoint = pathlib.Path(
        hf_hub_download(
            repo_id=SOURCE_REPOSITORY,
            filename=SOURCE_CHECKPOINT,
            revision=SOURCE_REVISION,
        )
    )
    configuration = pathlib.Path(
        hf_hub_download(
            repo_id=SOURCE_REPOSITORY,
            filename=SOURCE_CONFIGURATION,
            revision=SOURCE_REVISION,
        )
    )
    return checkpoint.resolve(), configuration.resolve()


def interpolate_position_embedding(value: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape) != (1, 196, 768):
        raise ValueError(f"unexpected positional embedding shape: {tuple(value.shape)}")
    resized = F.interpolate(
        value.reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
        size=(16, 16),
        mode="bicubic",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).reshape(1, 256, 768).contiguous()


def convert_tensor(name: str, value: torch.Tensor, dtype: torch.dtype) -> tuple[str, torch.Tensor] | None:
    if name.endswith(".num_batches_tracked"):
        return None
    if not name.startswith("model."):
        raise ValueError(f"unexpected checkpoint key: {name}")
    target = name.removeprefix("model.")
    tensor = value.detach().cpu()
    if target.endswith(".pos_emb"):
        tensor = interpolate_position_embedding(tensor)
    if (
        target.startswith("decoder.decoder.blocks.") and target.endswith(".0.weight")
    ) or target == "head.head.2.weight":
        # PyTorch Conv2d [out, in, height, width] -> MLX [out, height, width, in].
        tensor = tensor.permute(0, 2, 3, 1)
    if target in {
        "neck.2.fpn1.0.weight",
        "neck.2.fpn1.3.weight",
        "neck.2.fpn2.0.weight",
    }:
        # PyTorch ConvTranspose2d [in, out, height, width] -> MLX [out, height, width, in].
        tensor = tensor.permute(1, 2, 3, 0)
    return target, tensor.to(dtype=dtype).contiguous()


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    checkpoint, configuration = resolve_sources(args)
    actual_checkpoint_hash = sha256(checkpoint)
    if actual_checkpoint_hash != SOURCE_CHECKPOINT_SHA256:
        raise SystemExit(
            f"checkpoint SHA-256 mismatch: expected {SOURCE_CHECKPOINT_SHA256}, found {actual_checkpoint_hash}"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("state_dict"), dict):
        raise SystemExit("checkpoint does not contain a weights-only state_dict")

    dtype = torch.float32
    converted: dict[str, torch.Tensor] = {}
    for source_name, source_tensor in state["state_dict"].items():
        item = convert_tensor(source_name, source_tensor, dtype)
        if item is not None:
            converted[item[0]] = item[1]
    converted = dict(sorted(converted.items()))
    scalar_count = sum(value.numel() for value in converted.values())

    args.output.mkdir(parents=True, exist_ok=True)
    weights_path = args.output / "model.safetensors"
    save_file(
        converted,
        weights_path,
        metadata={
            "format": FORMAT,
            "model_id": MODEL_ID,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "dtype": args.dtype,
        },
    )
    config = {
        "format": FORMAT,
        "model_id": MODEL_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_configuration_sha256": sha256(configuration),
        "converter": CONVERTER,
        "dtype": args.dtype,
        "tensor_count": len(converted),
        "scalar_count": scalar_count,
        "tile_size": 256,
        "timestamps": 4,
    }
    write_json(args.output / "config.json", config)
    result = {
        **config,
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256(weights_path),
        "configuration_bytes": (args.output / "config.json").stat().st_size,
        "configuration_sha256": sha256(args.output / "config.json"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

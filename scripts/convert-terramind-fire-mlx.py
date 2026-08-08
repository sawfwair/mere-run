#!/usr/bin/env python3
"""Deterministically convert the pinned TerraMind Fire Lightning checkpoint to MLX safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
from typing import Any

import torch
import torch.nn.functional as F

MODEL_ID = "vision-fire-terramind-base"
SOURCE_REPOSITORY = "ibm-esa-geospatial/TerraMind-base-Fire"
SOURCE_REVISION = "6eb5178aac4f8a4191796258ae26e796195cc00d"
SOURCE_CHECKPOINT = "TerraMind_v1_base_ImpactMesh_fire.pt"
SOURCE_CONFIGURATION = "terramind_v1_base_impactmesh_fire.yaml"
SOURCE_CHECKPOINT_SHA256 = "c16c070d95e9944c4b1a14c56cdd16f7821b133c8f2521985b579d08c2d8c72e"
SOURCE_CONFIGURATION_SHA256 = "0ccbd6b9464f5a95198204b31dda77a1e7637611bb69bff7d569315f8ccc863f"
FORMAT = "mere.run/terramind-fire-mlx-v1"
CONVERTER = "scripts/convert-terramind-fire-mlx.py@v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--configuration", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
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

    checkpoint = pathlib.Path(hf_hub_download(
        repo_id=SOURCE_REPOSITORY,
        filename=SOURCE_CHECKPOINT,
        revision=SOURCE_REVISION,
    ))
    configuration = pathlib.Path(hf_hub_download(
        repo_id=SOURCE_REPOSITORY,
        filename=SOURCE_CONFIGURATION,
        revision=SOURCE_REVISION,
    ))
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


def convert_tensor(name: str, value: torch.Tensor) -> tuple[str, torch.Tensor] | None:
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
        tensor = tensor.permute(0, 2, 3, 1)
    if target in {
        "neck.2.fpn1.0.weight",
        "neck.2.fpn1.3.weight",
        "neck.2.fpn2.0.weight",
    }:
        tensor = tensor.permute(1, 2, 3, 0)
    return target, tensor.to(dtype=torch.float32).contiguous()


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_deterministic_safetensors(
    path: pathlib.Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> None:
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError(f"{name} is not a contiguous CPU float32 tensor")
        byte_count = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + byte_count],
        }
        offset += byte_count

    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    partial = path.with_suffix(f"{path.suffix}.partial")
    with partial.open("wb") as target:
        target.write(struct.pack("<Q", len(encoded)))
        target.write(encoded)
        for tensor in tensors.values():
            target.write(tensor.numpy().tobytes(order="C"))
    partial.replace(path)


def main() -> int:
    args = parse_args()
    checkpoint, configuration = resolve_sources(args)
    actual_checkpoint_hash = sha256(checkpoint)
    actual_configuration_hash = sha256(configuration)
    if actual_checkpoint_hash != SOURCE_CHECKPOINT_SHA256:
        raise SystemExit(
            f"checkpoint SHA-256 mismatch: expected {SOURCE_CHECKPOINT_SHA256}, found {actual_checkpoint_hash}"
        )
    if actual_configuration_hash != SOURCE_CONFIGURATION_SHA256:
        raise SystemExit(
            f"configuration SHA-256 mismatch: expected {SOURCE_CONFIGURATION_SHA256}, found {actual_configuration_hash}"
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("state_dict"), dict):
        raise ValueError("checkpoint does not contain a weights-only state_dict")
    converted: dict[str, torch.Tensor] = {}
    for source_name, source_tensor in state["state_dict"].items():
        item = convert_tensor(source_name, source_tensor)
        if item is not None:
            converted[item[0]] = item[1]
    converted = dict(sorted(converted.items()))
    scalar_count = sum(value.numel() for value in converted.values())
    if len(converted) != 171 or scalar_count != 168_416_386:
        raise ValueError(
            f"tensor inventory mismatch: expected 171/168416386, found {len(converted)}/{scalar_count}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    weights_path = args.output / "model.safetensors"
    write_deterministic_safetensors(
        weights_path,
        converted,
        metadata={
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "dtype": args.dtype,
            "format": FORMAT,
            "model_id": MODEL_ID,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
    )
    config = {
        "format": FORMAT,
        "model_id": MODEL_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_configuration_sha256": SOURCE_CONFIGURATION_SHA256,
        "converter": CONVERTER,
        "dtype": args.dtype,
        "tensor_count": len(converted),
        "scalar_count": scalar_count,
        "tile_size": 256,
        "timestamps": 4,
    }
    config_path = args.output / "config.json"
    write_json(config_path, config)
    print(json.dumps({
        **config,
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256(weights_path),
        "configuration_bytes": config_path.stat().st_size,
        "configuration_sha256": sha256(config_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

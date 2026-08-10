#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.2.6",
#   "packaging==26.3",
#   "safetensors==0.6.2",
#   "torch==2.7.1",
# ]
# ///
"""Convert a pinned Comfy-Kitchen ConvRot INT8 H3 transformer to MLX affine INT8.

This is release tooling only. The native Swift runtime never invokes Python.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_REPOSITORY = "Comfy-Org/MiniMax-H3"
SOURCE_REVISION = "fd70b39279d1ae6eb214c903f53e1bec3af19a77"
SOURCE_FILES = {
    "fl2va": {
        "name": "minimax_h3_fl2va_int8_convrot.safetensors",
        "byte_count": 34_038_892_334,
        "sha256": "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5",
    },
    "ref2va": {
        "name": "minimax_h3_ref2va_int8_convrot.safetensors",
        "byte_count": 34_038_894_550,
        "sha256": "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9",
    },
}
MLX_GROUP_SIZE = 64
BITS = 8


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path, partition: str) -> None:
    pin = SOURCE_FILES[partition]
    if path.name != pin["name"]:
        raise ValueError(f"Expected {pin['name']}, got {path.name}")
    if path.stat().st_size != pin["byte_count"]:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; expected {pin['byte_count']}"
        )
    digest = file_sha256(path)
    if digest != pin["sha256"]:
        raise ValueError(f"{path} has SHA-256 {digest}; expected {pin['sha256']}")


def regular_hadamard(size: int, device: torch.device) -> torch.Tensor:
    """Return Comfy-Kitchen's normalized regular-Hadamard ConvRot basis."""
    if size < 4 or size & (size - 1) or not math.log(size, 4).is_integer():
        raise ValueError(f"ConvRot group size must be a power of four, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
        device=device,
    )
    matrix = h4
    current = 4
    while current < size:
        matrix = torch.kron(matrix, h4)
        current *= 4
    return matrix / math.sqrt(size)


def decode_convrot_group_size(config: torch.Tensor, key: str) -> int:
    """Decode and validate one Comfy-Kitchen tensorwise ConvRot record."""
    try:
        payload = json.loads(bytes(config.tolist()).decode("utf-8").rstrip("\x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{key} is not valid UTF-8 JSON") from error
    if payload.get("format") != "int8_tensorwise" or payload.get("convrot") is not True:
        raise ValueError(f"{key} is not an INT8 tensorwise ConvRot record")
    group_size = payload.get("convrot_groupsize")
    if not isinstance(group_size, int):
        raise ValueError(f"{key} has no integer convrot_groupsize")
    if group_size < 4 or group_size & (group_size - 1) or not math.log(group_size, 4).is_integer():
        raise ValueError(f"{key} has invalid ConvRot group size {group_size}")
    return group_size


def undo_convrot(
    codes: torch.Tensor,
    row_scales: torch.Tensor,
    hadamard: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize per-row symmetric INT8 and restore the original weight basis."""
    rows, columns = codes.shape
    if hadamard.shape != (group_size, group_size):
        raise ValueError(
            f"ConvRot basis has shape {tuple(hadamard.shape)}; expected {(group_size, group_size)}"
        )
    if columns % group_size:
        raise ValueError(f"ConvRot input width {columns} is not divisible by {group_size}")
    rotated = codes.to(torch.float32) * row_scales.to(torch.float32)
    return torch.matmul(
        rotated.reshape(rows, -1, group_size),
        hadamard.T,
    ).reshape(rows, columns)


def mlx_affine_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce MLX affine 8-bit/group-64 quantization and uint32 packing."""
    rows, columns = weight.shape
    if columns % MLX_GROUP_SIZE:
        raise ValueError(
            f"MLX affine input width {columns} is not divisible by {MLX_GROUP_SIZE}"
        )
    grouped = weight.reshape(rows, -1, MLX_GROUP_SIZE)
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    scale = torch.maximum((maximum - minimum) / 255.0, torch.tensor(1e-7, device=weight.device))
    use_minimum = minimum.abs() > maximum.abs()
    scale = torch.where(use_minimum, scale, -scale)
    edge = torch.where(use_minimum, minimum, maximum)
    q0 = torch.round(edge / scale)
    at_zero = q0 == 0
    scale = torch.where(at_zero, scale, edge / q0)
    bias = torch.where(at_zero, torch.zeros_like(edge), edge)
    codes = torch.round((grouped - bias[..., None]) / scale[..., None]).clamp(0, 255).to(torch.uint8)
    packed_bytes = codes.reshape(rows, columns).contiguous()
    packed = packed_bytes.view(torch.uint32).reshape(rows, columns // 4)
    return packed.cpu(), scale.to(torch.bfloat16).cpu(), bias.to(torch.bfloat16).cpu()


def convert(source: Path, output: Path, device: torch.device) -> dict[int, int]:
    if output.exists():
        raise ValueError(f"Output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    converted: dict[str, torch.Tensor] = {}
    with safe_open(source, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        hadamards: dict[int, torch.Tensor] = {}
        convrot_group_counts: Counter[int] = Counter()
        used_config_keys: set[str] = set()
        quantized_count = 0
        dense_count = 0
        for index, key in enumerate(sorted(keys), start=1):
            if key.endswith(".comfy_quant") or key.endswith(".weight_scale"):
                continue
            value = archive.get_tensor(key)
            scale_key = f"{key}_scale"
            if key.endswith(".weight") and value.dtype == torch.int8 and scale_key in keys:
                base = key.removesuffix(".weight")
                config_key = f"{base}.comfy_quant"
                if config_key not in keys:
                    raise ValueError(f"{key} has {scale_key} but no {config_key}")
                group_size = decode_convrot_group_size(
                    archive.get_tensor(config_key),
                    config_key,
                )
                hadamard = hadamards.get(group_size)
                if hadamard is None:
                    hadamard = regular_hadamard(group_size, device)
                    hadamards[group_size] = hadamard
                row_scales = archive.get_tensor(scale_key)
                dense = undo_convrot(
                    value.to(device),
                    row_scales.to(device),
                    hadamard,
                    group_size,
                )
                packed, scales, biases = mlx_affine_quantize(dense)
                converted[key] = packed
                converted[f"{base}.scales"] = scales
                converted[f"{base}.biases"] = biases
                quantized_count += 1
                convrot_group_counts[group_size] += 1
                used_config_keys.add(config_key)
                del dense, packed, scales, biases
            else:
                converted[key] = value.contiguous()
                dense_count += 1
            print(f"[{index}/{len(keys)}] {key}", flush=True)

        config_keys = {key for key in keys if key.endswith(".comfy_quant")}
        if used_config_keys != config_keys:
            unused = ", ".join(sorted(config_keys - used_config_keys))
            raise ValueError(f"Unused ConvRot records: {unused}")

    save_file(
        converted,
        output,
        metadata={
            "quantization": "affine 8-bit g64",
            "quantized_tensors": str(quantized_count),
            "dense_tensors": str(dense_count),
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
    )
    return dict(convrot_group_counts)


def write_receipt(
    source: Path,
    output: Path,
    partition: str,
    convrot_group_counts: dict[int, int],
    device: torch.device,
) -> None:
    receipt = {
        "converter": "scripts/model-conversion/convert_minimax_h3_convrot.py",
        "converter_version": 2,
        "partition": partition,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "filename": source.name,
            "byte_count": source.stat().st_size,
            "sha256": file_sha256(source),
        },
        "output": {
            "filename": output.name,
            "byte_count": output.stat().st_size,
            "sha256": file_sha256(output),
        },
        "source_convrot_groups": {
            str(group_size): count
            for group_size, count in sorted(convrot_group_counts.items())
        },
        "toolchain": {
            "device": str(device),
            "python": platform.python_version(),
            "safetensors": importlib_metadata.version("safetensors"),
            "torch": torch.__version__,
        },
        "quantization": {
            "bits": BITS,
            "group_size": MLX_GROUP_SIZE,
            "mode": "affine",
        },
    }
    receipt_path = output.with_suffix(".conversion.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pinned MiniMax-H3 ConvRot INT8 weights to MLX affine INT8."
    )
    parser.add_argument("--partition", choices=sorted(SOURCE_FILES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = torch.device(args.device)
    verify_source(source, args.partition)
    convrot_group_counts = convert(source, output, device)
    write_receipt(source, output, args.partition, convrot_group_counts, device)
    print(output)


if __name__ == "__main__":
    main()

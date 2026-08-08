#!/usr/bin/env python3
"""Convert an immutable OlmoEarth v1.2 encoder tier to native MLX safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
from typing import Any

import torch

FORMAT = "mere.run/olmoearth-v1.2-mlx-v1"
CONVERTER = "scripts/convert-olmoearth-v12-mlx.py@v1"
SUPPORTED_MODALITIES = ("sentinel2_l2a", "sentinel1", "landsat")

VARIANTS: dict[str, dict[str, Any]] = {
    "nano": {
        "model_id": "vision-embed-olmoearth-v12-nano",
        "repository": "allenai/OlmoEarth-v1_2-Nano",
        "revision": "e1f693ae2a7d5b57871a978e9d09e22d05206747",
        "weights_sha256": "2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378",
        "config_sha256": "4cd2888e79dc543f262cc3d86fcd30d667068fd53a728ca5bd306d5ddb509d1d",
        "architecture": (128, 8, 4, 4.0, 12, 8, 12, 1.0 / 30.0),
        "tensor_count": 86,
        "scalar_count": 1_090_224,
    },
    "tiny": {
        "model_id": "vision-embed-olmoearth-v12-tiny",
        "repository": "allenai/OlmoEarth-v1_2-Tiny",
        "revision": "12a9fdbfeff905d7e147e7497f9f7a95c518eefc",
        "weights_sha256": "835c0b21ab010c4c4515faafa44dc1a41c9bc512d3a30af184803c4f4257697d",
        "config_sha256": "bb11f91f5afbd6138f75feee3f66fc0e272da089d05a6e515713c799057155ac",
        "architecture": (192, 3, 12, 4.0, 12, 8, 64, 1.0 / 30.0),
        "tensor_count": 222,
        "scalar_count": 7_704_592,
    },
    "small": {
        "model_id": "vision-embed-olmoearth-v12-small",
        "repository": "allenai/OlmoEarth-v1_2-Small",
        "revision": "a207c9a789483f95de1e9fb06acadb3da3775863",
        "weights_sha256": "459796ed5680bc85926f9a0e023476d14cb637bc19f826575c43836c909a5fa6",
        "config_sha256": "254703d9b5da4a6679003ff21f2da964a25d903fea70dc0b2cce5d0cd388f70b",
        "architecture": (384, 6, 12, 4.0, 12, 8, 64, 1.0 / 30.0),
        "tensor_count": 222,
        "scalar_count": 26_024_224,
    },
    "base": {
        "model_id": "vision-embed-olmoearth-v12-base",
        "repository": "allenai/OlmoEarth-v1_2-Base",
        "revision": "581aa9baaa7aed4348c0903617eb92ee9f89e2ec",
        "weights_sha256": "57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec",
        "config_sha256": "0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a",
        "architecture": (768, 12, 12, 4.0, 12, 8, 64, 0.0333),
        "tensor_count": 222,
        "scalar_count": 94_513_984,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--weights", type=pathlib.Path)
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


def resolve_sources(args: argparse.Namespace, spec: dict[str, Any]) -> tuple[pathlib.Path, pathlib.Path]:
    if args.weights and args.configuration:
        return args.weights.resolve(), args.configuration.resolve()
    if bool(args.weights) != bool(args.configuration):
        raise SystemExit("--weights and --configuration must be supplied together")
    from huggingface_hub import hf_hub_download

    weights = pathlib.Path(hf_hub_download(
        repo_id=spec["repository"], filename="weights.pth", revision=spec["revision"]
    ))
    configuration = pathlib.Path(hf_hub_download(
        repo_id=spec["repository"], filename="config.json", revision=spec["revision"]
    ))
    return weights.resolve(), configuration.resolve()


def architecture_payload(values: tuple[int, int, int, float, int, int, int, float]) -> dict[str, Any]:
    embedding, heads, depth, mlp_ratio, maximum_sequence, maximum_patch, patch_hidden, temporal_scale = values
    return {
        "embedding_dimension": embedding,
        "head_count": heads,
        "depth": depth,
        "mlp_ratio": mlp_ratio,
        "maximum_sequence_length": maximum_sequence,
        "maximum_patch_size": maximum_patch,
        "patch_hidden_dimension": patch_hidden,
        "position_encoding": "rope_3d_mixed",
        "temporal_coordinate_scale": temporal_scale,
    }


def validate_source_config(path: pathlib.Path, spec: dict[str, Any]) -> None:
    payload = json.loads(path.read_text())
    encoder = payload["model"]["encoder_config"]
    expected = architecture_payload(spec["architecture"])
    actual = {
        "embedding_dimension": int(encoder["embedding_size"]),
        "head_count": int(encoder["num_heads"]),
        "depth": int(encoder["depth"]),
        "mlp_ratio": float(encoder["mlp_ratio"]),
        "maximum_sequence_length": int(encoder["max_sequence_length"]),
        "maximum_patch_size": int(encoder["max_patch_size"]),
        "patch_hidden_dimension": int(encoder["patch_embed_hidden_sizes"][0]),
        "position_encoding": str(encoder["spatial_pos_encoding"]),
        "temporal_coordinate_scale": float(encoder["rope_temporal_coordinate_scale"]),
    }
    if actual != expected:
        raise ValueError(f"encoder architecture mismatch: expected {expected}, found {actual}")
    if not encoder["use_linear_patch_embed"] or encoder["qk_norm"]:
        raise ValueError("unsupported OlmoEarth v1.2 patch embedding or QK normalization")
    overrides = encoder["tokenization_config"]["overrides"]
    expected_bands = {
        "sentinel2_l2a": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"],
        "landsat": ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    }
    for modality, bands in expected_bands.items():
        if overrides[modality]["band_groups"] != [bands]:
            raise ValueError(f"unexpected {modality} tokenization bands")


def keep_encoder_tensor(name: str) -> bool:
    if name.startswith("encoder.blocks.") or name.startswith("encoder.norm."):
        return True
    if name == "encoder.composite_encodings.month_embed.weight":
        return True
    if any(
        name.startswith(f"encoder.composite_encodings.per_modality_channel_embeddings.{modality}")
        for modality in SUPPORTED_MODALITIES
    ):
        return True
    return any(
        name.startswith(f"encoder.patch_embeddings.per_modality_embeddings.{modality}.")
        for modality in SUPPORTED_MODALITIES
    )


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
    spec = VARIANTS[args.variant]
    weights, source_config = resolve_sources(args, spec)
    actual_weights_hash = sha256(weights)
    actual_config_hash = sha256(source_config)
    if actual_weights_hash != spec["weights_sha256"]:
        raise SystemExit(
            f"weights SHA-256 mismatch: expected {spec['weights_sha256']}, found {actual_weights_hash}"
        )
    if actual_config_hash != spec["config_sha256"]:
        raise SystemExit(
            f"configuration SHA-256 mismatch: expected {spec['config_sha256']}, found {actual_config_hash}"
        )
    validate_source_config(source_config, spec)

    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("weights.pth does not contain a weights-only state dictionary")
    tensors = {
        name.removeprefix("encoder."): value.detach().cpu().to(dtype=torch.float32).contiguous()
        for name, value in sorted(state.items())
        if keep_encoder_tensor(name)
    }
    scalar_count = sum(value.numel() for value in tensors.values())
    if len(tensors) != spec["tensor_count"] or scalar_count != spec["scalar_count"]:
        raise ValueError(
            "tensor inventory mismatch: "
            f"expected {spec['tensor_count']}/{spec['scalar_count']}, "
            f"found {len(tensors)}/{scalar_count}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    weights_path = args.output / "model.safetensors"
    write_deterministic_safetensors(
        weights_path,
        tensors,
        metadata={
            "source_weights_sha256": spec["weights_sha256"],
            "dtype": args.dtype,
            "format": FORMAT,
            "model_id": spec["model_id"],
            "source_repository": spec["repository"],
            "source_revision": spec["revision"],
        },
    )
    config = {
        "format": FORMAT,
        "model_id": spec["model_id"],
        "variant": args.variant,
        "source_repository": spec["repository"],
        "source_revision": spec["revision"],
        "source_weights": "weights.pth",
        "source_weights_sha256": spec["weights_sha256"],
        "source_configuration_sha256": spec["config_sha256"],
        "converter": CONVERTER,
        "dtype": args.dtype,
        "tensor_count": len(tensors),
        "scalar_count": scalar_count,
        "supported_modalities": list(SUPPORTED_MODALITIES),
        "architecture": architecture_payload(spec["architecture"]),
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
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

#!/usr/bin/env python3
"""Convert an immutable TESSERA v2 student checkpoint to native MLX safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
from typing import Any

import torch

FORMAT = "mere.run/tessera-v2-mlx-v1"
CONVERTER = "scripts/convert-tessera-v2-mlx.py@v1"

VARIANTS: dict[str, dict[str, Any]] = {
    "nano": {
        "model_id": "vision-embed-tessera-v2-nano",
        "repository": "geotessera/TESSERA-V-2.0-2B-N",
        "revision": "9645033fdcd5c0686bab00720e5553ce307629cf",
        "checkpoint": "ckpt/student_nano.pt",
        "checkpoint_sha256": "bc9929e3643dfab2744c2ce14e7c14f698703a94e9562e4e0982f9d6540b0691",
        "architecture": (128, 36, 2, 4, 384, 256),
        "tensor_count": 66,
        "scalar_count": 1_066_402,
    },
    "small": {
        "model_id": "vision-embed-tessera-v2-small",
        "repository": "geotessera/TESSERA-V-2.0-2B-S",
        "revision": "21760b27ff16ca7aab01986b7b3460e3027b19c6",
        "checkpoint": "ckpt/student_small.pt",
        "checkpoint_sha256": "92619d4ffc2936895f145fdcf710140b58c23c860e02f01873828b10c6958d95",
        "architecture": (128, 64, 4, 4, 1024, 256),
        "tensor_count": 114,
        "scalar_count": 7_112_322,
    },
    "medium": {
        "model_id": "vision-embed-tessera-v2-medium",
        "repository": "geotessera/TESSERA-V-2.0-2B-M",
        "revision": "41db8ee5ddfcf6867f965526c2097d70c3c55c31",
        "checkpoint": "ckpt/student_medium.pt",
        "checkpoint_sha256": "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e",
        "architecture": (128, 110, 4, 4, 1792, 256),
        "tensor_count": 114,
        "scalar_count": 21_031_506,
    },
    "large": {
        "model_id": "vision-embed-tessera-v2-large",
        "repository": "geotessera/TESSERA-V-2.0-2B-L",
        "revision": "b45f24463acf3fcfe030f94735d3e817b24100d0",
        "checkpoint": "ckpt/student_large.pt",
        "checkpoint_sha256": "b5f20239dbb1849c01a3e407b095aafe39b0bf764300206af78cb9b85f9ec1e1",
        "architecture": (128, 160, 4, 4, 2560, 256),
        "tensor_count": 114,
        "scalar_count": 43_831_170,
    },
    "teacher": {
        "model_id": "vision-embed-tessera-v2-teacher",
        "repository": "geotessera/TESSERA-V-2.0-2B-Teacher",
        "revision": "262170691f167085a7f86750066066e3d6ab6e10",
        "checkpoint": "ckpt/tessera_v2_2B_teacher.pt",
        "checkpoint_sha256": "bfca890a9956485edf9d6be61375fde4cd0b4cb4b47db79596f0523a289aa555",
        "architecture": (1024, 1024, 4, 4, 16384, 256),
        "qk_normalization": True,
        "fusion_layer_count": 2,
        "tensor_count": 199,
        "scalar_count": 2_064_266_242,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(args: argparse.Namespace, spec: dict[str, Any]) -> pathlib.Path:
    if args.checkpoint:
        return args.checkpoint.resolve()
    from huggingface_hub import hf_hub_download

    return pathlib.Path(
        hf_hub_download(
            repo_id=spec["repository"],
            filename=spec["checkpoint"],
            revision=spec["revision"],
        )
    ).resolve()


def architecture_payload(spec: dict[str, Any]) -> dict[str, Any]:
    values = spec["architecture"]
    representation, latent, layers, heads, feed_forward, maximum_sequence = values
    payload = {
        "representation_dimension": representation,
        "latent_dimension": latent,
        "layer_count": layers,
        "head_count": heads,
        "feed_forward_dimension": feed_forward,
        "maximum_sequence_length": maximum_sequence,
    }
    if spec.get("qk_normalization"):
        payload["qk_normalization"] = True
        payload["fusion_layer_count"] = spec["fusion_layer_count"]
    return payload


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


def validate_payload(payload: Any, spec: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint does not contain a mapping payload")
    expected_architecture = architecture_payload(spec)
    if spec.get("qk_normalization"):
        state = payload.get("encoder_state_dict")
        args = payload.get("arch") or {}
        if not isinstance(state, dict):
            raise ValueError("teacher checkpoint does not contain encoder_state_dict")
        actual_architecture = {
            "representation_dimension": int(args.get("repr_dim", -1)),
            "latent_dimension": int(args.get("latent_dim", -1)),
            "layer_count": int(args.get("num_layers", -1)),
            "head_count": int(args.get("nhead", -1)),
            "feed_forward_dimension": int(args.get("dim_feedforward", -1)),
            "maximum_sequence_length": 256,
            "qk_normalization": True,
            "fusion_layer_count": 2,
        }
        if not bool(args.get("final_layernorm", False)):
            raise ValueError("teacher checkpoint must enable the final LayerNorm")
        if actual_architecture != expected_architecture:
            raise ValueError(
                f"architecture mismatch: expected {expected_architecture}, found {actual_architecture}"
            )
        return state

    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("student checkpoint does not contain a weights-only model state")
    args = payload.get("args") or {}
    actual_architecture = {
        "representation_dimension": int(args.get("repr_dim", -1)),
        "latent_dimension": int(args.get("latent_dim", -1)),
        "layer_count": int(args.get("num_layers", -1)),
        "head_count": int(args.get("nhead", -1)),
        "feed_forward_dimension": int(args.get("dim_feedforward", -1)),
        "maximum_sequence_length": int(args.get("max_seq_len", -1)),
    }
    if actual_architecture != expected_architecture:
        raise ValueError(
            f"architecture mismatch: expected {expected_architecture}, found {actual_architecture}"
        )
    if bool(args.get("enable_qk_norm", False)):
        raise ValueError("deployable TESSERA v2 students must not enable QK normalization")
    if str(args.get("matryoshka_dims")) != "16,32,64,128":
        raise ValueError("unexpected Matryoshka output dimensions")
    return state


def main() -> int:
    args = parse_args()
    spec = VARIANTS[args.variant]
    checkpoint = resolve_checkpoint(args, spec)
    actual_hash = sha256(checkpoint)
    if actual_hash != spec["checkpoint_sha256"]:
        raise SystemExit(
            f"checkpoint SHA-256 mismatch: expected {spec['checkpoint_sha256']}, found {actual_hash}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_tensors = validate_payload(payload, spec)
    tensors = {
        name: value.detach().cpu().to(dtype=torch.float32).contiguous()
        for name, value in sorted(source_tensors.items())
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
            "source_checkpoint_sha256": spec["checkpoint_sha256"],
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
        "source_checkpoint": spec["checkpoint"],
        "source_checkpoint_sha256": spec["checkpoint_sha256"],
        "converter": CONVERTER,
        "dtype": args.dtype,
        "tensor_count": len(tensors),
        "scalar_count": scalar_count,
        "architecture": architecture_payload(spec),
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

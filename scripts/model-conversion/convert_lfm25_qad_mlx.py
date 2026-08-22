#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = [
#   "gguf==0.19.0",
#   "mlx==0.32.0",
#   "numpy==2.4.3",
#   "safetensors==0.8.0",
# ]
# ///

"""Convert LiquidAI's pinned LFM2.5 QAD Q4_0 GGUFs to native MLX weights."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize
import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file


CONVERTER = "scripts/model-conversion/convert_lfm25_qad_mlx.py"
CONVERTER_VERSION = 1
TOOL_VERSIONS = {
    "gguf": "0.19.0",
    "mlx": "0.32.0",
    "numpy": "2.4.3",
    "safetensors": "0.8.0",
}


@dataclass(frozen=True)
class FilePin:
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class Profile:
    name: str
    model_id: str
    source_repository: str
    source_revision: str
    source_filename: str
    source_pin: FilePin
    metadata_repository: str
    metadata_revision: str
    metadata_pins: dict[str, FilePin]
    expected_source_tensors: int
    expected_q4_tensors: int
    expected_f32_tensors: int
    expected_q6k_tensors: int
    tier: str


COMMON_1_2B_METADATA = {
    "LICENSE": FilePin(10_644, "61d7e939a05911c765b7e98ffaa1ab5ca6c0174a65350766c25cb10197d19fc8"),
    "README.md": FilePin(15_614, "11af6c3a0caaa0d0a6f78c7df5edfe47426bb06cb8bb35b3901d989fb05fdc58"),
    "chat_template.jinja": FilePin(5_487, "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6"),
    "config.json": FilePin(1_224, "15d6157fb6df3f8272e2fe90e18f57727ccf02a125c94469198b0f3281510185"),
    "generation_config.json": FilePin(132, "5ffd97da1dec4308543894569662d96e923ed01f7a9d8c7ff5aea7f800738cbd"),
    "tokenizer.json": FilePin(4_733_389, "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd"),
    "tokenizer_config.json": FilePin(92_225, "2a52ec012d3df831ba434b081bef3726a6ee22501f062ad8353c557a0cfa0d01"),
}


COMMON_2_6B_METADATA = {
    "LICENSE": FilePin(10_574, "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c"),
    "README.md": FilePin(17_816, "321173d06e4d13010fdf8ad9fffb2b4fbb76f39ffcc4c493cdd0895b39d7a37c"),
    "chat_template.jinja": FilePin(5_443, "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef"),
    "config.json": FilePin(1_467, "480f63fa8e1efa534ae8b92774b3b53b8d6812d62a726e9ecfc866933662f273"),
    "generation_config.json": FilePin(327, "7366b93f26f8830e7a94441a3d2b9f344ceb9e1e31a808e0656df40720075874"),
    "tokenizer.json": FilePin(17_905_598, "695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41"),
    "tokenizer_config.json": FilePin(363, "11f1de897317b489dd09199284528382eeb17f566398b16e2039bb2266d26b09"),
}


PROFILES = {
    "1.2b": Profile(
        name="1.2b",
        model_id="text-chat-lfm25-1.2b-qad-4bit",
        source_repository="LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        source_revision="afbd8eaeab5dd94ba0b079ebfb02517d19641e38",
        source_filename="LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf",
        source_pin=FilePin(
            695_755_488,
            "bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1",
        ),
        metadata_repository="LiquidAI/LFM2.5-1.2B-Instruct",
        metadata_revision="df58c174f05ff733f83f8cae10ea9298224c8006",
        metadata_pins=COMMON_1_2B_METADATA,
        expected_source_tensors=148,
        expected_q4_tensors=92,
        expected_f32_tensors=55,
        expected_q6k_tensors=1,
        tier="nano",
    ),
    "2.6b": Profile(
        name="2.6b",
        model_id="text-chat-lfm25-2.6b-qad-4bit",
        source_repository="LiquidAI/LFM2.5-2.6B-GGUF",
        source_revision="f4a289c8a200a5ca71005ba7abc2dad33058a450",
        source_filename="LFM2.5-2.6B-QAD-Q4_0.gguf",
        source_pin=FilePin(
            1_593_894_944,
            "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03",
        ),
        metadata_repository="LiquidAI/LFM2.5-2.6B",
        metadata_revision="a334ee78cd38458bb71eda24109ac42dcec1309d",
        metadata_pins=COMMON_2_6B_METADATA,
        expected_source_tensors=266,
        expected_q4_tensors=166,
        expected_f32_tensors=99,
        expected_q6k_tensors=1,
        tier="small",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, pin: FilePin) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != pin.byte_count:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; expected {pin.byte_count}"
        )
    actual = sha256(path)
    if actual != pin.sha256:
        raise ValueError(f"{path} has SHA-256 {actual}; expected {pin.sha256}")


def verify_environment() -> None:
    for package, expected in TOOL_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} {actual} is installed; expected {expected}")


def mlx_name(gguf_name: str) -> str:
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "token_embd_norm.weight":
        return "model.embedding_norm.weight"

    match = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if match is None:
        raise ValueError(f"Unrecognized LFM2.5 GGUF tensor name: {gguf_name}")
    layer, suffix = match.groups()
    mapped = {
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_k_norm.weight": "self_attn.k_layernorm.weight",
        "attn_norm.weight": "operator_norm.weight",
        "attn_output.weight": "self_attn.out_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_q_norm.weight": "self_attn.q_layernorm.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "ffn_down.weight": "feed_forward.w2.weight",
        "ffn_gate.weight": "feed_forward.w1.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "ffn_up.weight": "feed_forward.w3.weight",
        "shortconv.conv.weight": "conv.conv.weight",
        "shortconv.in_proj.weight": "conv.in_proj.weight",
        "shortconv.out_proj.weight": "conv.out_proj.weight",
    }.get(suffix)
    if mapped is None:
        raise ValueError(f"Unrecognized LFM2.5 GGUF tensor suffix: {suffix}")
    return f"model.layers.{layer}.{mapped}"


def pack_unsigned_nibbles(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint8 or values.shape[-1] % 8:
        raise ValueError("Expected uint8 nibbles with a final dimension divisible by eight")
    lanes = values.reshape(*values.shape[:-1], -1, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * np.uint32(4)).reshape(
        *((1,) * (lanes.ndim - 1)), 8
    )
    return np.bitwise_or.reduce(lanes << shifts, axis=-1)


def unpack_unsigned_nibbles(packed: np.ndarray) -> np.ndarray:
    shifts = (np.arange(8, dtype=np.uint32) * np.uint32(4)).reshape(
        *((1,) * packed.ndim), 8
    )
    return ((packed[..., None] >> shifts) & np.uint32(0xF)).astype(np.uint8).reshape(
        *packed.shape[:-1], -1
    )


def convert_q4_0(raw: np.ndarray, logical_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    rows, columns = logical_shape
    if columns % 32:
        raise ValueError(f"Q4_0 input width must be divisible by 32: {logical_shape}")
    block_count = rows * columns // 32
    blocks = np.asarray(raw, dtype=np.uint8).reshape(block_count, 18)
    scales = blocks[:, :2].copy().view(np.float16).reshape(rows, columns // 32)
    gguf_nibbles = blocks[:, 2:]
    unsigned = np.concatenate(
        [gguf_nibbles & np.uint8(0xF), gguf_nibbles >> np.uint8(4)],
        axis=1,
    ).reshape(rows, columns // 32, 32)
    packed = pack_unsigned_nibbles(unsigned).reshape(rows, columns // 8)
    biases = (scales * np.float16(-8)).astype(np.float16, copy=False)

    # Check the first and last groups of every tensor after the format repack.
    restored = unpack_unsigned_nibbles(packed.reshape(rows, columns // 32, 4))
    for group in (0, restored.shape[1] - 1):
        expected = unsigned[:, group, :]
        actual = restored[:, group, :]
        if not np.array_equal(actual, expected):
            raise ValueError("Q4_0 nibble repack changed quantized values")
    return {"weight": packed, "scales": scales, "biases": biases}


def requantize_q4_0(
    tensor: Any,
) -> tuple[dict[str, np.ndarray], float, float, int]:
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    decoded = dequantize(tensor.data, tensor.tensor_type).reshape(logical_shape)
    source = mx.array(decoded.astype(np.float16, copy=False))
    weight, scales, biases = mx.quantize(
        source,
        group_size=64,
        bits=4,
        mode="affine",
    )
    restored = mx.dequantize(
        weight,
        scales,
        biases,
        group_size=64,
        bits=4,
        mode="affine",
        dtype=mx.float32,
    )
    source_f32 = mx.array(decoded)
    absolute_error = mx.abs(restored - source_f32)
    maximum_error = mx.max(absolute_error)
    mean_error = mx.mean(absolute_error)
    mx.eval(weight, scales, biases, maximum_error, mean_error)
    converted = {
        "weight": np.array(weight, copy=True),
        "scales": np.array(scales, copy=True),
        "biases": np.array(biases, copy=True),
    }
    result = (
        converted,
        float(maximum_error.item()),
        float(mean_error.item()),
        decoded.size,
    )
    del decoded, source, source_f32, restored, absolute_error, weight, scales, biases
    mx.clear_cache()
    return result


def dense_tensor(tensor: Any) -> np.ndarray:
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    if tensor.tensor_type == GGMLQuantizationType.F32:
        value = np.asarray(tensor.data, dtype=np.float32).reshape(logical_shape)
        if tensor.name.endswith("shortconv.conv.weight"):
            value = value.reshape(logical_shape[0], logical_shape[1], 1)
        return value
    raise ValueError(f"Unsupported dense tensor type {tensor.tensor_type} for {tensor.name}")


def convert_q6_k_embedding(
    tensor: Any,
) -> tuple[dict[str, np.ndarray], float, float]:
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    decoded = dequantize(tensor.data, tensor.tensor_type).reshape(logical_shape)
    source = mx.array(decoded.astype(np.float16, copy=False))
    weight, scales, biases = mx.quantize(
        source,
        group_size=64,
        bits=6,
        mode="affine",
    )
    restored = mx.dequantize(
        weight,
        scales,
        biases,
        group_size=64,
        bits=6,
        mode="affine",
        dtype=mx.float32,
    )
    source_f32 = mx.array(decoded)
    maximum_error = mx.max(mx.abs(restored - source_f32))
    mean_error = mx.mean(mx.abs(restored - source_f32))
    mx.eval(weight, scales, biases, maximum_error, mean_error)
    return (
        {
            "weight": np.array(weight, copy=True),
            "scales": np.array(scales, copy=True),
            "biases": np.array(biases, copy=True),
        },
        float(maximum_error.item()),
        float(mean_error.item()),
    )


def converter_self_test() -> None:
    rng = np.random.default_rng(20260821)
    unsigned = rng.integers(0, 16, size=(5, 4, 32), dtype=np.uint8)
    scales = rng.normal(size=(5, 4)).astype(np.float16)
    packed = pack_unsigned_nibbles(unsigned).reshape(5, 16)
    restored = unpack_unsigned_nibbles(packed.reshape(5, 4, 4))
    if not np.array_equal(restored, unsigned):
        raise AssertionError("Synthetic Q4_0 repack failed")
    gguf_values = scales[..., None].astype(np.float32) * (
        unsigned.astype(np.float32) - np.float32(8)
    )
    mlx_values = scales[..., None].astype(np.float32) * unsigned.astype(np.float32) + (
        scales * np.float16(-8)
    )[..., None].astype(np.float32)
    if not np.array_equal(gguf_values, mlx_values):
        raise AssertionError("Q4_0 and MLX affine dequantization differ")


def converted_config(source: Path, q4_layout: str) -> dict[str, Any]:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "lfm2":
        raise ValueError("Pinned metadata is not an LFM2 model")
    group_size = 32 if q4_layout == "exact32" else 64
    quantization = {"group_size": group_size, "bits": 4, "mode": "affine"}
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    config["mlx_conversion"] = {
        "source_format": "gguf-qad-q4_0",
        "q4_0": (
            "bit-exact-repack-to-affine-group-32"
            if q4_layout == "exact32"
            else "decoded-and-requantized-to-affine-4-bit-group-64"
        ),
        "q6_k_embedding": "decoded-and-requantized-to-affine-6-bit-group-64",
    }
    return config


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(output: Path, profile: Profile, q4_layout: str) -> None:
    group_size = 32 if q4_layout == "exact32" else 64
    write_json(
        output / "mererun_model.json",
        {
            "schemaVersion": 3,
            "id": profile.model_id,
            "engine": "lfm2",
            "family": "liquid",
            "tier": profile.tier,
            "variant": "standard",
            "precision": "int4",
            "quantization": {
                "bits": 4,
                "groupSize": group_size,
                "scheme": f"liquid-qad-q4_0-mlx-affine-g{group_size}",
            },
            "supports": ["chat", "code_generation"],
            "components": {
                "tokenizer": {"type": "local", "path": "."},
                "text_encoder": {"type": "local", "path": "."},
            },
            "upstreamRepoId": f"{profile.source_repository}@{profile.source_revision}",
            "sources": [
                {
                    "role": "base-model",
                    "repository": profile.source_repository,
                    "revision": profile.source_revision,
                    "destination_path": ".",
                },
                {
                    "role": "metadata",
                    "repository": profile.metadata_repository,
                    "revision": profile.metadata_revision,
                    "destination_path": ".",
                },
            ],
        },
    )


def write_model_card(source: Path, output: Path, profile: Profile, q4_layout: str) -> None:
    original = (source / "README.md").read_text(encoding="utf-8")
    q4_note = (
        "Every Q4_0 projection nibble and FP16 block scale is retained exactly and "
        "repacked into MLX affine 4-bit/group-32 tensors with `bias = -8 * scale`."
        if q4_layout == "exact32"
        else "Q4_0 projections are decoded and requantized to MLX affine "
        "4-bit/group-64 for the optimized native kernel; maximum and mean "
        "elementwise errors are recorded in `MERERUN_CONVERSION.json`."
    )
    heading = f"""---
library_name: mlx
pipeline_tag: text-generation
license: other
base_model: {profile.source_repository}
tags:
- mlx
- liquid
- lfm2.5
- qad
- 4-bit
---

# {profile.model_id}

This is a deterministic native-MLX conversion of LiquidAI's QAD-trained Q4_0
GGUF `{profile.source_filename}` at immutable revision
`{profile.source_revision}`.

- {q4_note}
- The single Q6_K tied token embedding is decoded and requantized to MLX affine
  6-bit/group-64; its measured maximum and mean elementwise errors are recorded
  in `MERERUN_CONVERSION.json`.
- `MERERUN_CONVERSION.json` records all pinned inputs and output hashes. The
  converter is `{CONVERTER}` in the public mere.run repository.
- Liquid reports that QAD retains Q4_0's compact footprint and native Q4_0
  throughput while recovering accuracy lost by post-training quantization.
  Their throughput points use llama.cpp because the QAD and PTQ checkpoints
  share that GGUF runtime path. This native MLX conversion preserves the
  QAD-trained projection values, but MLX throughput must be measured separately.
- Liquid's original LFM Open License terms remain applicable.

```shell
mere.run model pull {profile.model_id} --accept-model-license
mere.run text chat --model {profile.model_id} --stats --prompt "Hello"
```

## Original model card

"""
    (output / "README.md").write_text(heading + original, encoding="utf-8")


def convert(profile: Profile, source: Path, output: Path, q4_layout: str) -> None:
    source_file = source / profile.source_filename
    verify_file(source_file, profile.source_pin)
    for filename, pin in profile.metadata_pins.items():
        verify_file(source / filename, pin)

    reader = GGUFReader(source_file)
    if profile.expected_source_tensors == 0:
        raise ValueError(f"The immutable {profile.name} tensor inventory has not been pinned")
    if len(reader.tensors) != profile.expected_source_tensors:
        raise ValueError(
            f"GGUF has {len(reader.tensors)} tensors; expected {profile.expected_source_tensors}"
        )

    arrays: dict[str, np.ndarray] = {}
    counts = {"q4_0_repacked": 0, "f32_retained": 0, "q6_k_decoded": 0}
    q6k_maximum_error = 0.0
    q6k_mean_error = 0.0
    q4_maximum_error = 0.0
    q4_weighted_error = 0.0
    q4_element_count = 0
    for tensor in reader.tensors:
        name = mlx_name(tensor.name)
        logical_shape = tuple(int(value) for value in reversed(tensor.shape))
        if tensor.tensor_type == GGMLQuantizationType.Q4_0:
            if q4_layout == "exact32":
                converted = convert_q4_0(tensor.data, logical_shape)
            else:
                converted, maximum_error, mean_error, element_count = requantize_q4_0(tensor)
                q4_maximum_error = max(q4_maximum_error, maximum_error)
                q4_weighted_error += mean_error * element_count
                q4_element_count += element_count
            base = name.removesuffix(".weight")
            arrays[f"{base}.weight"] = converted["weight"]
            arrays[f"{base}.scales"] = converted["scales"]
            arrays[f"{base}.biases"] = converted["biases"]
            counts["q4_0_repacked"] += 1
        elif tensor.tensor_type == GGMLQuantizationType.Q6_K:
            converted, maximum_error, mean_error = convert_q6_k_embedding(tensor)
            base = name.removesuffix(".weight")
            arrays[f"{base}.weight"] = converted["weight"]
            arrays[f"{base}.scales"] = converted["scales"]
            arrays[f"{base}.biases"] = converted["biases"]
            counts["q6_k_decoded"] += 1
            q6k_maximum_error = max(q6k_maximum_error, maximum_error)
            q6k_mean_error = max(q6k_mean_error, mean_error)
        else:
            value = dense_tensor(tensor)
            arrays[name] = value
            counts["f32_retained"] += 1

    expected_counts = {
        "q4_0_repacked": profile.expected_q4_tensors,
        "f32_retained": profile.expected_f32_tensors,
        "q6_k_decoded": profile.expected_q6k_tensors,
    }
    if counts != expected_counts:
        raise ValueError(f"Converted tensor inventory {counts}; expected {expected_counts}")
    expected_output_tensors = (
        profile.expected_q4_tensors * 3
        + profile.expected_f32_tensors
        + profile.expected_q6k_tensors * 3
    )
    if len(arrays) != expected_output_tensors:
        raise ValueError(f"Produced {len(arrays)} tensors; expected {expected_output_tensors}")

    weights_path = output / "model.safetensors"
    save_file(
        dict(sorted(arrays.items())),
        weights_path,
        metadata={
            "format": "mlx",
            "quantization": (
                "QAD Q4_0 exact repack to affine 4-bit group-32"
                if q4_layout == "exact32"
                else "QAD Q4_0 decoded and requantized to affine 4-bit group-64"
            ),
        },
    )
    del arrays

    write_json(output / "config.json", converted_config(source, q4_layout))
    for filename in profile.metadata_pins:
        if filename not in {"config.json", "README.md"}:
            shutil.copyfile(source / filename, output / filename)
    write_model_card(source, output, profile, q4_layout)
    write_manifest(output, profile, q4_layout)

    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "MERERUN_CONVERSION.json":
            artifacts.append(
                {
                    "filename": path.name,
                    "byte_count": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_json(
        output / "MERERUN_CONVERSION.json",
        {
            "converter": CONVERTER,
            "converter_version": CONVERTER_VERSION,
            "source": {
                "repository": profile.source_repository,
                "revision": profile.source_revision,
                "filename": profile.source_filename,
                "byte_count": profile.source_pin.byte_count,
                "sha256": profile.source_pin.sha256,
                "tensor_count": profile.expected_source_tensors,
            },
            "metadata_source": {
                "repository": profile.metadata_repository,
                "revision": profile.metadata_revision,
                "files": {
                    filename: {
                        "byte_count": pin.byte_count,
                        "sha256": pin.sha256,
                    }
                    for filename, pin in profile.metadata_pins.items()
                },
            },
            "conversion": {
                **counts,
                "q4_0_requantized": q4_layout != "exact32",
                "q4_0_group_size": 32 if q4_layout == "exact32" else 64,
                "q4_0_bits": 4,
                "q4_0_maximum_elementwise_error": q4_maximum_error,
                "q4_0_mean_elementwise_error": (
                    q4_weighted_error / q4_element_count if q4_element_count else 0.0
                ),
                "q6_k_requantized": True,
                "q6_k_bits": 6,
                "q6_k_group_size": 64,
                "q6_k_maximum_elementwise_error": q6k_maximum_error,
                "q6_k_mean_elementwise_error": q6k_mean_error,
            },
            "tools": TOOL_VERSIONS,
            "artifacts": artifacts,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--q4-layout",
        choices=["exact32", "affine64"],
        default="exact32",
        help="Exact group-32 repack or measured group-64 requantization.",
    )
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    verify_environment()
    converter_self_test()
    if args.self_test_only:
        print("LFM2.5 QAD converter self-test passed")
        return
    if args.profile is None or args.source is None or args.output is None:
        parser.error("--profile, --source, and --output are required for conversion")

    profile = PROFILES[args.profile]
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        convert(profile, source, temporary, args.q4_layout)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.6",
# ]
# ///
"""Compact an MLX affine INT8 MiniMax-H3 transformer to inference-only INT4.

This is release tooling only. The native Swift runtime never invokes Python.
The source artifact must already use MLX affine 8-bit/group-64 tensors and must
have a compatible AdaLN cache before its cache-covered weights may be omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


GROUP_SIZE = 64
SOURCE_BITS = 8
OUTPUT_BITS = 4
LEVELS = (1 << OUTPUT_BITS) - 1
SAFETENSORS_ALIGNMENT = 8
CHUNK_ROWS = 64
MIN_FREE_HEADROOM = 1 << 30


DTYPES: dict[str, np.dtype] = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "I8": np.dtype("i1"),
    "I32": np.dtype("<i4"),
    "U8": np.dtype("u1"),
    "U32": np.dtype("<u4"),
}


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def byte_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CopyPlan:
    key: str
    source: TensorInfo


@dataclass(frozen=True)
class QuantizePlan:
    weight_key: str
    source_weight: TensorInfo
    source_scales: TensorInfo
    source_biases: TensorInfo | None

    @property
    def scales_key(self) -> str:
        return self.weight_key.removesuffix(".weight") + ".scales"

    @property
    def biases_key(self) -> str:
        return self.weight_key.removesuffix(".weight") + ".biases"

    @property
    def rows(self) -> int:
        return self.source_weight.shape[0]

    @property
    def columns(self) -> int:
        return self.source_weight.shape[1] * (32 // SOURCE_BITS)

    @property
    def output_weight_shape(self) -> tuple[int, int]:
        return (self.rows, self.columns // (32 // OUTPUT_BITS))


Plan = CopyPlan | QuantizePlan


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(handle: BinaryIO) -> tuple[int, dict[str, object]]:
    encoded_length = handle.read(8)
    if len(encoded_length) != 8:
        raise ValueError("Safetensors header length is missing")
    header_length = struct.unpack("<Q", encoded_length)[0]
    if header_length <= 0 or header_length > 64 * 1024 * 1024:
        raise ValueError(f"Invalid safetensors header length: {header_length}")
    encoded_header = handle.read(header_length)
    if len(encoded_header) != header_length:
        raise ValueError("Safetensors header is truncated")
    header = json.loads(encoded_header)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    return header_length, header


def parse_tensors(header: dict[str, object]) -> dict[str, TensorInfo]:
    tensors: dict[str, TensorInfo] = {}
    for key, raw in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"Tensor descriptor for {key} is not an object")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if dtype not in DTYPES:
            raise ValueError(f"Tensor {key} has unsupported dtype {dtype}")
        if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
            raise ValueError(f"Tensor {key} has an invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"Tensor {key} has invalid data offsets")
        info = TensorInfo(dtype, tuple(shape), offsets[0], offsets[1])
        expected = int(np.prod(info.shape, dtype=np.int64)) * DTYPES[dtype].itemsize
        if info.byte_count != expected:
            raise ValueError(
                f"Tensor {key} has {info.byte_count} bytes; expected {expected}"
            )
        tensors[key] = info
    return tensors


def is_cache_covered(key: str) -> bool:
    return ".adaln_proj." in key or key.startswith("time_embedder.")


def make_plan(tensors: dict[str, TensorInfo]) -> list[Plan]:
    active = {key: value for key, value in tensors.items() if not is_cache_covered(key)}
    quantized_weights: dict[str, QuantizePlan] = {}
    quantized_related: set[str] = set()
    for key, weight in active.items():
        if not key.endswith(".weight") or weight.dtype != "U32" or len(weight.shape) != 2:
            continue
        prefix = key.removesuffix(".weight")
        scales_key = prefix + ".scales"
        biases_key = prefix + ".biases"
        scales = active.get(scales_key)
        biases = active.get(biases_key)
        if scales is None or scales.dtype != "BF16" or len(scales.shape) != 2:
            raise ValueError(f"Quantized tensor {key} is missing BF16 scales")
        columns = weight.shape[1] * (32 // SOURCE_BITS)
        expected_scale_shape = (weight.shape[0], columns // GROUP_SIZE)
        if scales.shape != expected_scale_shape:
            raise ValueError(
                f"Tensor {key} scales have shape {scales.shape}; expected {expected_scale_shape}"
            )
        if biases is not None and biases.shape != expected_scale_shape:
            raise ValueError(
                f"Tensor {key} biases have shape {biases.shape}; expected {expected_scale_shape}"
            )
        if columns % GROUP_SIZE or columns % (32 // OUTPUT_BITS):
            raise ValueError(f"Tensor {key} input width {columns} cannot use INT4 group-{GROUP_SIZE}")
        quantized_weights[key] = QuantizePlan(key, weight, scales, biases)
        quantized_related.update((key, scales_key))
        if biases is not None:
            quantized_related.add(biases_key)

    plan: list[Plan] = []
    for key in sorted(active):
        if key in quantized_weights:
            plan.append(quantized_weights[key])
        elif key not in quantized_related:
            plan.append(CopyPlan(key, active[key]))
    if not quantized_weights:
        raise ValueError("No MLX affine INT8 weights were found")
    return plan


def plan_output_tensors(plan: list[Plan]) -> list[tuple[str, str, tuple[int, ...]]]:
    output: list[tuple[str, str, tuple[int, ...]]] = []
    for item in plan:
        if isinstance(item, CopyPlan):
            output.append((item.key, item.source.dtype, item.source.shape))
        else:
            output.extend(
                [
                    (item.weight_key, "U32", item.output_weight_shape),
                    (item.scales_key, "BF16", item.source_scales.shape),
                    (item.biases_key, "BF16", item.source_scales.shape),
                ]
            )
    return output


def output_header(
    tensors: list[tuple[str, str, tuple[int, ...]]],
    metadata: dict[str, str],
) -> tuple[bytes, int]:
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for key, dtype, shape in tensors:
        byte_count = int(np.prod(shape, dtype=np.int64)) * DTYPES[dtype].itemsize
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + byte_count],
        }
        offset += byte_count
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=False).encode("utf-8")
    padding = (-len(encoded)) % SAFETENSORS_ALIGNMENT
    return encoded + b" " * padding, offset


def mapped_array(
    mapping: mmap.mmap,
    data_start: int,
    info: TensorInfo,
) -> np.ndarray:
    dtype = DTYPES[info.dtype]
    return np.ndarray(
        info.shape,
        dtype=dtype,
        buffer=mapping,
        offset=data_start + info.start,
        order="C",
    )


def bf16_to_float32(values: np.ndarray) -> np.ndarray:
    words = values.astype(np.uint32, copy=False)
    return np.left_shift(words, np.uint32(16)).view(np.float32)


def float32_to_bf16(values: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return ((words + rounding) >> np.uint32(16)).astype(np.uint16)


def requantize_group(
    source_codes: np.ndarray,
    source_scales: np.ndarray,
    source_biases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    rows, columns = source_codes.shape
    groups = columns // GROUP_SIZE
    values = (
        source_codes.reshape(rows, groups, GROUP_SIZE).astype(np.float32)
        * source_scales[..., None]
        + source_biases[..., None]
    )
    minimum = values.min(axis=-1)
    maximum = values.max(axis=-1)
    scale = np.maximum((maximum - minimum) / np.float32(LEVELS), np.float32(1e-7))
    use_minimum = np.abs(minimum) > np.abs(maximum)
    scale = np.where(use_minimum, scale, -scale)
    edge = np.where(use_minimum, minimum, maximum)
    q0 = np.rint(edge / scale)
    at_zero = q0 == 0
    scale = np.where(at_zero, scale, edge / np.where(at_zero, np.float32(1), q0))
    bias = np.where(at_zero, np.float32(0), edge)
    codes = np.clip(
        np.rint((values - bias[..., None]) / scale[..., None]),
        0,
        LEVELS,
    ).astype(np.uint32)
    packed_groups = codes.reshape(rows, columns // 8, 8)
    packed = np.zeros((rows, columns // 8), dtype=np.uint32)
    for index in range(8):
        packed |= packed_groups[..., index] << np.uint32(index * OUTPUT_BITS)

    restored = codes.astype(np.float32).reshape(rows, groups, GROUP_SIZE)
    restored = restored * scale[..., None] + bias[..., None]
    difference = values - restored
    absolute = np.abs(difference)
    return (
        packed,
        float32_to_bf16(scale),
        float32_to_bf16(bias),
        float(absolute.sum(dtype=np.float64)),
        float(np.square(difference, dtype=np.float64).sum(dtype=np.float64)),
        difference.size,
    )


def convert_quantized(
    item: QuantizePlan,
    mapping: mmap.mmap,
    data_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    source_weight = mapped_array(mapping, data_start, item.source_weight)
    source_scales_bf16 = mapped_array(mapping, data_start, item.source_scales)
    source_biases_bf16 = (
        mapped_array(mapping, data_start, item.source_biases)
        if item.source_biases is not None
        else np.zeros(item.source_scales.shape, dtype=np.uint16)
    )
    output_weight = np.empty(item.output_weight_shape, dtype=np.uint32)
    output_scales = np.empty(item.source_scales.shape, dtype=np.uint16)
    output_biases = np.empty(item.source_scales.shape, dtype=np.uint16)
    absolute_sum = 0.0
    squared_sum = 0.0
    value_count = 0

    for start in range(0, item.rows, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, item.rows)
        source_codes = source_weight[start:end].view(np.uint8).reshape(end - start, item.columns)
        source_scales = bf16_to_float32(source_scales_bf16[start:end])
        source_biases = bf16_to_float32(source_biases_bf16[start:end])
        packed, scales, biases, chunk_absolute, chunk_squared, chunk_count = requantize_group(
            source_codes,
            source_scales,
            source_biases,
        )
        output_weight[start:end] = packed
        output_scales[start:end] = scales
        output_biases[start:end] = biases
        absolute_sum += chunk_absolute
        squared_sum += chunk_squared
        value_count += chunk_count

    return (
        output_weight,
        output_scales,
        output_biases,
        {
            "mean_absolute_error": absolute_sum / value_count,
            "root_mean_squared_error": (squared_sum / value_count) ** 0.5,
        },
    )


def write_bytes(handle: BinaryIO, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    handle.write(memoryview(contiguous).cast("B"))


def safetensors_metadata(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        _, header = read_header(handle)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError(f"Safetensors metadata is invalid: {path}")
    return metadata


def compact_configuration(document: object) -> dict[str, object]:
    if not isinstance(document, dict):
        raise ValueError("MiniMax-H3 config must be a JSON object")
    quantization = document.get("quantization")
    if quantization != {"group_size": GROUP_SIZE, "bits": SOURCE_BITS, "mode": "affine"}:
        raise ValueError("MiniMax-H3 config must declare affine 8-bit group-64 quantization")
    text_encoder_quantization = document.get("text_encoder_quantization", quantization)
    if text_encoder_quantization != quantization:
        raise ValueError("Source transformer and text encoder quantization must both be affine INT8")
    result = json.loads(json.dumps(document))
    result["quantization"] = {
        "group_size": GROUP_SIZE,
        "bits": OUTPUT_BITS,
        "mode": "affine",
    }
    result["text_encoder_quantization"] = dict(quantization)
    return result


def write_json_atomic(path: Path, document: object) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def convert(
    source: Path,
    ada_ln_cache: Path,
    source_config: Path,
    output: Path,
    output_config: Path,
    receipt: Path,
    source_sha256: str | None,
) -> None:
    if output.exists():
        raise ValueError(f"Output path already exists: {output}")
    if receipt.exists():
        raise ValueError(f"Receipt path already exists: {receipt}")
    if output_config.exists():
        raise ValueError(f"Output config path already exists: {output_config}")
    with source_config.open(encoding="utf-8") as handle:
        converted_config = compact_configuration(json.load(handle))
    with source.open("rb") as source_handle:
        source_header_length, source_header = read_header(source_handle)
    source_metadata = source_header.get("__metadata__", {})
    if not isinstance(source_metadata, dict):
        raise ValueError("Source safetensors metadata is invalid")
    if source_metadata.get("quantization") != "affine 8-bit g64":
        raise ValueError("Source must declare affine 8-bit g64 quantization")
    tensors = parse_tensors(source_header)
    plan = make_plan(tensors)
    output_tensors = plan_output_tensors(plan)
    cache_metadata = safetensors_metadata(ada_ln_cache)
    if cache_metadata.get("format") != "mere.run.minimax-h3-adaln-cache":
        raise ValueError(f"Not a MiniMax-H3 AdaLN cache: {ada_ln_cache}")
    if cache_metadata.get("schema_version") != "2":
        raise ValueError("AdaLN cache schema must be version 2")
    cache_identity = cache_metadata.get("source_identity")
    if cache_identity is None:
        raise ValueError("AdaLN cache source identity is missing")
    source_size_prefix = cache_identity.partition(":")[0]
    if source_size_prefix != str(source.stat().st_size):
        raise ValueError("AdaLN cache does not belong to the source transformer")
    metadata = {
        "format": "mere.run.minimax-h3-inference-transformer",
        "quantization": "affine 4-bit g64",
        "source_quantization": "affine 8-bit g64",
        "adaln_cache_source_identity": cache_identity,
        "cache_covered_weights_omitted": "true",
    }
    encoded_header, output_data_bytes = output_header(output_tensors, metadata)
    required = 8 + len(encoded_header) + output_data_bytes + MIN_FREE_HEADROOM
    free = shutil.disk_usage(output.parent).free
    if free < required:
        raise ValueError(
            f"Conversion needs {required} free bytes including headroom; only {free} are available"
        )

    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    quantization_errors: dict[str, dict[str, float]] = {}
    copied_tensors = 0
    quantized_tensors = 0
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as output_handle:
            mapping = mmap.mmap(source_handle.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                data_start = 8 + source_header_length
                output_handle.write(struct.pack("<Q", len(encoded_header)))
                output_handle.write(encoded_header)
                for index, item in enumerate(plan, start=1):
                    if isinstance(item, CopyPlan):
                        start = data_start + item.source.start
                        end = data_start + item.source.end
                        output_handle.write(mapping[start:end])
                        copied_tensors += 1
                        label = item.key
                    else:
                        weight, scales, biases, errors = convert_quantized(
                            item,
                            mapping,
                            data_start,
                        )
                        write_bytes(output_handle, weight)
                        write_bytes(output_handle, scales)
                        write_bytes(output_handle, biases)
                        quantization_errors[item.weight_key] = errors
                        quantized_tensors += 1
                        label = item.weight_key
                    print(f"[{index}/{len(plan)}] {label}", flush=True)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            finally:
                mapping.close()
        if temporary.stat().st_size != 8 + len(encoded_header) + output_data_bytes:
            raise ValueError("Converted safetensors byte count does not match its header")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    input_digest = source_sha256 or file_sha256(source)
    output_digest = file_sha256(output)
    write_json_atomic(output_config, converted_config)
    aggregate_mae = sum(value["mean_absolute_error"] for value in quantization_errors.values())
    aggregate_rmse = sum(value["root_mean_squared_error"] for value in quantization_errors.values())
    error_count = len(quantization_errors)
    document = {
        "converter": "scripts/model-conversion/requantize_minimax_h3_mlx.py",
        "converter_version": 2,
        "source": {
            "path": str(source),
            "byte_count": source.stat().st_size,
            "sha256": input_digest,
            "identity": cache_identity,
            "quantization": "affine 8-bit g64",
        },
        "output": {
            "path": str(output),
            "byte_count": output.stat().st_size,
            "sha256": output_digest,
            "quantization": "affine 4-bit g64",
        },
        "configuration": {
            "source": str(source_config),
            "output": str(output_config),
            "transformer_quantization": "affine 4-bit g64",
            "text_encoder_quantization": "affine 8-bit g64",
        },
        "cache_covered_weights_omitted": True,
        "copied_tensors": copied_tensors,
        "requantized_tensors": quantized_tensors,
        "mean_layer_mae": aggregate_mae / error_count,
        "mean_layer_rmse": aggregate_rmse / error_count,
        "layer_errors": quantization_errors,
    }
    write_json_atomic(receipt, document)


def unpack_q4(values: np.ndarray, columns: int) -> np.ndarray:
    expanded = np.empty((values.shape[0], values.shape[1], 8), dtype=np.uint32)
    for index in range(8):
        expanded[..., index] = (values >> np.uint32(index * OUTPUT_BITS)) & np.uint32(LEVELS)
    return expanded.reshape(values.shape[0], columns)


def self_test() -> None:
    rng = np.random.default_rng(314159)
    source_codes = rng.integers(0, 256, size=(3, 128), dtype=np.uint8)
    source_scales = rng.uniform(0.0005, 0.05, size=(3, 2)).astype(np.float32)
    source_biases = rng.uniform(-2, 2, size=(3, 2)).astype(np.float32)
    packed, scales_bf16, biases_bf16, _, _, count = requantize_group(
        source_codes,
        source_scales,
        source_biases,
    )
    assert packed.shape == (3, 16)
    assert scales_bf16.shape == (3, 2)
    assert biases_bf16.shape == (3, 2)
    assert count == source_codes.size
    restored_codes = unpack_q4(packed, 128)
    restored = (
        restored_codes.reshape(3, 2, 64).astype(np.float32)
        * bf16_to_float32(scales_bf16)[..., None]
        + bf16_to_float32(biases_bf16)[..., None]
    )
    original = (
        source_codes.reshape(3, 2, 64).astype(np.float32)
        * source_scales[..., None]
        + source_biases[..., None]
    )
    maximum_step = np.abs(bf16_to_float32(scales_bf16)).max()
    assert np.abs(original - restored).max() <= maximum_step * 0.55
    values = np.array([0.0, 1.0, -1.0, 0.1, -0.1], dtype=np.float32)
    round_trip = bf16_to_float32(float32_to_bf16(values))
    assert np.allclose(round_trip, values, rtol=0.004, atol=0.0001)
    config = compact_configuration(
        {"quantization": {"group_size": 64, "bits": 8, "mode": "affine"}}
    )
    assert config["quantization"] == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert config["text_encoder_quantization"] == {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
    }
    print("MiniMax-H3 INT4 requantizer self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact an MLX affine INT8 MiniMax-H3 transformer to cache-backed INT4."
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--adaln-cache", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if (
        args.source is None
        or args.adaln_cache is None
        or args.config is None
        or args.output is None
        or args.output_config is None
        or args.receipt is None
    ):
        parser.error(
            "--source, --adaln-cache, --config, --output, --output-config, and --receipt "
            "are required unless --self-test is used"
        )
    source = args.source.expanduser().resolve()
    ada_ln_cache = args.adaln_cache.expanduser().resolve()
    source_config = args.config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output_config = args.output_config.expanduser().resolve()
    receipt = args.receipt.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Source does not exist: {source}")
    if not ada_ln_cache.is_file():
        raise ValueError(f"AdaLN cache does not exist: {ada_ln_cache}")
    if not source_config.is_file():
        raise ValueError(f"Config does not exist: {source_config}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    convert(
        source,
        ada_ln_cache,
        source_config,
        output,
        output_config,
        receipt,
        args.source_sha256,
    )


if __name__ == "__main__":
    main()

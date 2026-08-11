"""Shared, pinned conversion helpers for Nemotron 3.5 Lightning and DSpark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import mlx.core as mx
from safetensors import safe_open


GROUP_SIZE = 16
BITS = 4
MODE = "nvfp4"
DEFAULT_SHARD_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class FilePin:
    byte_count: int
    sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, pin: FilePin) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned source file is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != pin.byte_count:
        raise ValueError(
            f"{path.name} has {actual_bytes} bytes; expected {pin.byte_count}"
        )
    actual_hash = sha256(path)
    if actual_hash != pin.sha256:
        raise ValueError(
            f"{path.name} has SHA-256 {actual_hash}; expected {pin.sha256}"
        )


def verify_pins(root: Path, pins: dict[str, FilePin]) -> None:
    for filename, pin in pins.items():
        verify_file(root / filename, pin)


def tensor_dtypes(path: Path) -> dict[str, str]:
    with safe_open(path, framework="np") as handle:
        return {key: str(handle.get_slice(key).get_dtype()) for key in handle.keys()}


def pack_modelopt_nvfp4(value: mx.array) -> mx.array:
    """Reinterpret ModelOpt's four bytes as MLX's little-endian uint32 pack.

    ModelOpt and MLX both store each E2M1 value low-nibble first. ModelOpt uses
    uint8 pairs while MLX groups eight nibbles in uint32. This operation changes
    only the container width; the released four-bit values remain bit exact.
    """

    if value.dtype != mx.uint8 or value.shape[-1] % 4 != 0:
        raise ValueError(
            f"Expected uint8 ModelOpt FP4 bytes divisible by four; got "
            f"{value.shape} {value.dtype}"
        )
    grouped = value.reshape(*value.shape[:-1], value.shape[-1] // 4, 4).astype(
        mx.uint32
    )
    return (
        grouped[..., 0]
        | (grouped[..., 1] << 8)
        | (grouped[..., 2] << 16)
        | (grouped[..., 3] << 24)
    )


def _e4m3fn_table() -> mx.array:
    values: list[float] = []
    for raw in range(256):
        sign = -1.0 if raw & 0x80 else 1.0
        exponent = (raw >> 3) & 0xF
        mantissa = raw & 0x7
        if exponent == 0:
            value = sign * (mantissa / 8.0) * (2.0**-6)
        elif exponent == 0xF and mantissa == 0x7:
            value = float("nan")
        else:
            value = sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(value)
    return mx.array(values, dtype=mx.float32)


E4M3FN_TABLE = _e4m3fn_table()


def dequantize_modelopt_fp8(value: mx.array, weight_scale: mx.array) -> mx.array:
    """Materialize a ModelOpt per-tensor E4M3 weight as BF16 for Metal.

    MLX Metal does not expose the static W8A8 ModelOpt linear format. Decoding
    the released E4M3 bytes with their scalar weight scale preserves the
    checkpoint's quantized weight values and avoids a second weight quantizer.
    """

    if value.dtype != mx.uint8 or weight_scale.size != 1:
        raise ValueError("Invalid ModelOpt FP8 weight or scalar weight scale")
    decoded = mx.take(E4M3FN_TABLE, value.astype(mx.int32), axis=0)
    return (decoded * weight_scale.astype(mx.float32)).astype(mx.bfloat16)


class ShardWriter:
    def __init__(self, root: Path, target_bytes: int = DEFAULT_SHARD_BYTES) -> None:
        self.root = root
        self.target_bytes = target_bytes
        self.arrays: dict[str, mx.array] = {}
        self.byte_count = 0
        self.parts: list[tuple[Path, list[str], int]] = []

    def append(self, key: str, value: mx.array) -> None:
        value_bytes = int(value.nbytes)
        if self.arrays and self.byte_count + value_bytes > self.target_bytes:
            self.flush()
        if key in self.arrays:
            raise ValueError(f"Duplicate converted tensor key: {key}")
        self.arrays[key] = value
        self.byte_count += value_bytes

    def append_many(self, arrays: Iterable[tuple[str, mx.array]]) -> None:
        for key, value in arrays:
            self.append(key, value)

    def flush(self) -> None:
        if not self.arrays:
            return
        mx.eval(*self.arrays.values())
        path = self.root / f"part-{len(self.parts) + 1:05d}.safetensors"
        mx.save_safetensors(str(path), self.arrays, metadata={"format": "mlx"})
        self.parts.append((path, sorted(self.arrays), self.byte_count))
        self.arrays = {}
        self.byte_count = 0
        mx.clear_cache()

    def finalize(self) -> tuple[dict[str, str], list[Path], int]:
        self.flush()
        total = len(self.parts)
        weight_map: dict[str, str] = {}
        outputs: list[Path] = []
        total_size = 0
        for index, (temporary, keys, byte_count) in enumerate(self.parts, start=1):
            filename = f"model-{index:05d}-of-{total:05d}.safetensors"
            final = temporary.with_name(filename)
            temporary.rename(final)
            outputs.append(final)
            total_size += byte_count
            for key in keys:
                weight_map[key] = filename
        return weight_map, outputs, total_size


_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>.+)\.experts\.(?P<expert>[0-9]+)\."
    r"(?P<projection>up_proj|down_proj)\."
    r"(?P<suffix>weight|weight_scale|weight_scale_2)$"
)


def stack_modelopt_experts(
    arrays: dict[str, mx.array],
    *,
    expected_experts: int,
) -> tuple[list[tuple[str, mx.array]], set[str]]:
    grouped: dict[tuple[str, str, str], dict[int, mx.array]] = {}
    consumed: set[str] = set()
    for key, value in arrays.items():
        match = _EXPERT_PATTERN.match(key)
        if match is None:
            continue
        group_key = (
            match.group("prefix"),
            match.group("projection"),
            match.group("suffix"),
        )
        grouped.setdefault(group_key, {})[int(match.group("expert"))] = value
        consumed.add(key)

    converted: list[tuple[str, mx.array]] = []
    projections = {(prefix, projection) for prefix, projection, _ in grouped}
    for prefix, projection in sorted(projections):
        expected = list(range(expected_experts))
        values_by_suffix: dict[str, dict[int, mx.array]] = {}
        for suffix in ("weight", "weight_scale", "weight_scale_2"):
            found = grouped.get((prefix, projection, suffix), {})
            if sorted(found) != expected:
                raise ValueError(
                    f"{prefix}.experts.{projection}.{suffix} has "
                    f"{len(found)} experts; expected {expected_experts}"
                )
            values_by_suffix[suffix] = found

        weights = mx.stack(
            [
                pack_modelopt_nvfp4(values_by_suffix["weight"][index])
                for index in expected
            ],
            axis=0,
        )
        scales = mx.stack(
            [values_by_suffix["weight_scale"][index] for index in expected],
            axis=0,
        )
        global_scales = mx.stack(
            [
                values_by_suffix["weight_scale_2"][index].reshape(())
                for index in expected
            ],
            axis=0,
        )
        base = f"{prefix}.experts.{projection}"
        converted.extend(
            [
                (f"{base}.weight", weights),
                (f"{base}.scales", scales),
                (f"{base}.global_scale", global_scales),
            ]
        )
    return converted, consumed


def transform_modelopt_shard(
    arrays: dict[str, mx.array],
    dtypes: dict[str, str],
    *,
    expected_experts: int | None,
    drop_prefixes: tuple[str, ...] = (),
    drop_suffixes: tuple[str, ...] = (),
) -> tuple[list[tuple[str, mx.array]], dict[str, int]]:
    converted: list[tuple[str, mx.array]] = []
    consumed: set[str] = set()
    stats = {
        "dense": 0,
        "fp8_materialized": 0,
        "nvfp4_repacked": 0,
        "expert_matrices_stacked": 0,
        "dropped": 0,
    }

    if expected_experts is not None:
        expert_candidates = {
            key: value
            for key, value in arrays.items()
            if not key.startswith(drop_prefixes) and not key.endswith(drop_suffixes)
        }
        expert_arrays, expert_consumed = stack_modelopt_experts(
            expert_candidates, expected_experts=expected_experts
        )
        converted.extend(expert_arrays)
        consumed.update(expert_consumed)
        stats["expert_matrices_stacked"] += len(expert_arrays) // 3
        stats["nvfp4_repacked"] += len(expert_consumed) // 3

    # ModelOpt names the FP8 input calibration tensor before the weight, so a
    # sorted traversal would otherwise emit it before the weight branch can
    # mark it consumed. Reserve every FP8 projection's auxiliary tensors up
    # front; only the released weight and weight scale define the materialized
    # BF16 value.
    for key in arrays:
        if dtypes[key] != "F8_E4M3" or not key.endswith(".weight"):
            continue
        base = key.removesuffix(".weight")
        consumed.update((f"{base}.weight_scale", f"{base}.input_scale"))

    for key in sorted(arrays):
        if key in consumed:
            continue
        if key.startswith(drop_prefixes) or key.endswith(drop_suffixes):
            stats["dropped"] += 1
            continue

        value = arrays[key]
        if dtypes[key] == "F8_E4M3":
            if not key.endswith(".weight"):
                raise ValueError(f"Unexpected FP8 tensor: {key}")
            base = key.removesuffix(".weight")
            scale_key = f"{base}.weight_scale"
            input_scale_key = f"{base}.input_scale"
            if scale_key not in arrays or input_scale_key not in arrays:
                raise ValueError(f"FP8 projection is missing ModelOpt scales: {base}")
            converted.append(
                (key, dequantize_modelopt_fp8(value, arrays[scale_key]))
            )
            consumed.update((scale_key, input_scale_key))
            stats["fp8_materialized"] += 1
            continue

        if key in consumed:
            continue

        if key.endswith(".weight") and value.dtype == mx.uint8:
            base = key.removesuffix(".weight")
            scale_key = f"{base}.weight_scale"
            global_scale_key = f"{base}.weight_scale_2"
            if scale_key not in arrays or global_scale_key not in arrays:
                raise ValueError(f"NVFP4 projection is missing ModelOpt scales: {base}")
            converted.extend(
                [
                    (key, pack_modelopt_nvfp4(value)),
                    (f"{base}.scales", arrays[scale_key]),
                    (f"{base}.global_scale", arrays[global_scale_key].reshape(())),
                ]
            )
            consumed.update((scale_key, global_scale_key))
            stats["nvfp4_repacked"] += 1
            continue

        if key.endswith(".conv1d.weight") and value.ndim == 3:
            converted.append((key, value.swapaxes(1, 2)))
        else:
            converted.append((key, value))
        stats["dense"] += 1

    return converted, stats


def write_index(output: Path, weight_map: dict[str, str], total_size: int) -> Path:
    path = output / "model.safetensors.index.json"
    path.write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_model_card(source: Path, output: Path, conversion_note: str) -> None:
    """Retain the upstream card while labeling the converted artifact clearly."""

    card = source.read_text()
    if not card.startswith("---\n") or "\n---\n" not in card[4:]:
        raise ValueError(f"Upstream model card has no Hugging Face metadata: {source}")
    card = re.sub(r"(?m)^library_name:.*$", "library_name: mlx", card, count=1)
    card = card.replace("tags:\n", "tags:\n- mlx\n- mere-run\n", 1)
    frontmatter_end = card.index("\n---\n", 4) + len("\n---\n")
    converted_card = card[:frontmatter_end] + "\n" + conversion_note.strip() + "\n\n" + card[frontmatter_end:].lstrip()
    output.write_text(converted_card)

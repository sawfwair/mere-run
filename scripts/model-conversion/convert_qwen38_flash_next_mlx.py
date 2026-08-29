#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "huggingface-hub==1.28.0",
#   "mlx==0.32.2; sys_platform == 'darwin'",
#   "mlx[cuda]==0.32.2; sys_platform == 'linux'",
#   "numpy==2.5.2",
#   "safetensors==0.8.0",
# ]
# ///
"""Build resumable MLX Q4, mixed Q2/Q4, and activation-weighted Q3 artifacts.

This is release tooling, not an inference sidecar. It downloads one immutable
BF16 snapshot, converts one source shard at a time, and writes the requested profiles
without ever instantiating the 180B-parameter model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any, Iterable


SOURCE_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
SOURCE_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
EXPECTED_SOURCE_SHARDS = 131
EXPECTED_SOURCE_TENSORS = 1_658
EXPECTED_SOURCE_WEIGHT_BYTES = 359_999_963_128
EXPECTED_SOURCE_PARAMETERS = 179_999_981_424
EXPECTED_NGRAM_SHARDS = 128
EXPECTED_EXPERT_TENSORS = 98
EXPECTED_OUTPUT_LOGICAL_BYTES = {
    "q4": 104_741_817_208,
    "mixed": 73_094_818_808,
    "q3-activation": 89_642_322_808,
}
EXPECTED_OUTPUT_TENSORS = {"q4": 3_817, "mixed": 3_615, "q3-activation": 3_817}
EXPECTED_QUANTIZED_MODULES = {"q4": 1_055, "mixed": 954, "q3-activation": 1_055}

MLX_VERSION = "0.32.2"
HUGGINGFACE_HUB_VERSION = "1.28.0"
SAFETENSORS_VERSION = "0.8.0"
MODE = "affine"
Q4 = {"bits": 4, "group_size": 64, "mode": MODE}
NGRAM_Q4 = {"bits": 4, "group_size": 32, "mode": MODE}
EXPERT_Q2 = {"bits": 2, "group_size": 128, "mode": MODE}
EXPERT_Q3 = {"bits": 3, "group_size": 64, "mode": MODE}
ACTIVATION_PROFILE_SHA256 = "c4d033b45e939a09e5ab08fb48b66d14262b7e14eef410c9d59c9361e84f89db"
ACTIVATION_PROFILE_CALIBRATION_SHA256 = "467ae6c9002d63a201edbaba050a344fb275a1ae01b4116fdd19efa85cb718eb"
ACTIVATION_PROFILE_SOURCE_REVISION = "6cc9bbc0fae9ce26b7670b3ed1e26d557c154506"
OUTPUT_NAMES = {
    "q4": "Qwen3.8-Flash-Next-MLX-4bit",
    "mixed": "Qwen3.8-Flash-Next-MLX-Mixed-2bit",
    "q3-activation": "Qwen3.8-Flash-Next-MLX-Activation-3bit",
}
HF_REPOSITORIES = {
    "q4": "Sawfwair/Qwen3.8-Flash-Next-MLX-4bit",
    "mixed": "Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit",
    "q3-activation": "Sawfwair/Qwen3.8-Flash-Next-MLX-Activation-3bit",
}
STATE_FILENAME = ".mererun-conversion-state.json"
STATE_VERSION = 1

SOURCE_ALLOW_PATTERNS = [
    "*.json",
    "*.jinja",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "LICENSE*",
    "README.md",
    "merges.txt",
    "vocab.json",
    "model-*.safetensors",
]

SHIFTED_TEXT_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".q_layernorm.weight",
    ".k_layernorm.weight",
    ".norm_key.weight",
    ".norm_query.weight",
    ".norm_conv.weight",
)


@dataclass(frozen=True)
class TensorPlan:
    key: str
    quantization: dict[str, int | str] | None


class ActivationProfile:
    """Lazy access to frozen Q4-teacher expert input second moments."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Activation profile is missing: {self.path}")
        self.sha256 = sha256_file(self.path)
        if self.sha256 != ACTIVATION_PROFILE_SHA256:
            raise ValueError(
                f"Activation profile SHA-256 is {self.sha256}; expected {ACTIVATION_PROFILE_SHA256}"
            )
        self.archive = safe_open(self.path, framework="numpy")
        self.metadata = self.archive.metadata() or {}
        expected_metadata = {
            "method": "q4-expert-input-second-moments-v1",
            "calibration_sha256": ACTIVATION_PROFILE_CALIBRATION_SHA256,
            "source_revision": ACTIVATION_PROFILE_SOURCE_REVISION,
        }
        for key, expected in expected_metadata.items():
            if self.metadata.get(key) != expected:
                raise ValueError(
                    f"Activation profile metadata {key!r} is {self.metadata.get(key)!r}; expected {expected!r}"
                )
        keys = set(self.archive.keys())
        if len(keys) != 576:
            raise ValueError(f"Activation profile has {len(keys)} tensors; expected 576")

    @staticmethod
    def profile_path(module_path: str) -> str:
        prefix = "language_model."
        if not module_path.startswith(prefix):
            raise ValueError(f"Expert module has an unexpected path: {module_path}")
        return module_path.removeprefix(prefix)

    def importance(self, module_path: str, expected_shape: tuple[int, int], mx: Any) -> Any:
        import numpy as np

        path = self.profile_path(module_path)
        combined = np.zeros(expected_shape, dtype=np.float32)
        for modality in ("image", "text"):
            squared_key = f"{path}.{modality}.squared_sum"
            count_key = f"{path}.{modality}.count"
            squared = self.archive.get_tensor(squared_key)
            count = self.archive.get_tensor(count_key)
            if tuple(squared.shape) != expected_shape:
                raise ValueError(
                    f"Activation tensor {squared_key} has shape {squared.shape}; expected {expected_shape}"
                )
            if tuple(count.shape) != (expected_shape[0],):
                raise ValueError(
                    f"Activation tensor {count_key} has shape {count.shape}; expected {(expected_shape[0],)}"
                )
            combined += squared / np.maximum(count[:, None], 1.0)
        return mx.array(combined)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_shard_names() -> list[str]:
    return [
        f"model-{index:05d}-of-{EXPECTED_SOURCE_SHARDS:05d}.safetensors"
        for index in range(1, EXPECTED_SOURCE_SHARDS + 1)
    ]


def remote_source_manifest() -> dict[str, dict[str, int | str]]:
    from huggingface_hub import HfApi

    manifest: dict[str, dict[str, int | str]] = {}
    entries = HfApi().list_repo_tree(
        repo_id=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        recursive=True,
        expand=True,
    )
    for entry in entries:
        path = getattr(entry, "path", None)
        if not isinstance(path, str):
            continue
        record: dict[str, int | str] = {
            "byte_count": int(getattr(entry, "size", 0) or 0)
        }
        lfs = getattr(entry, "lfs", None)
        sha256 = getattr(lfs, "sha256", None)
        if isinstance(sha256, str):
            record["sha256"] = sha256
        manifest[path] = record

    expected = set(source_shard_names())
    actual = {name for name in manifest if name.startswith("model-") and name.endswith(".safetensors")}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Pinned Hub revision has an unexpected shard set: missing={missing}, extra={extra}")
    for name in expected:
        if "sha256" not in manifest[name]:
            raise ValueError(f"Pinned Hub manifest has no SHA-256 for {name}")
    return manifest


def download_source(destination: Path, max_workers: int) -> Path:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=SOURCE_REPOSITORY,
            revision=SOURCE_REVISION,
            local_dir=destination,
            allow_patterns=SOURCE_ALLOW_PATTERNS,
            max_workers=max_workers,
        )
    )


def verify_environment() -> None:
    expected = {
        "mlx": MLX_VERSION,
        "huggingface-hub": HUGGINGFACE_HUB_VERSION,
        "numpy": "2.5.2",
        "safetensors": SAFETENSORS_VERSION,
    }
    for package, version in expected.items():
        actual = importlib.metadata.version(package)
        if actual != version:
            raise RuntimeError(f"{package} {actual} is installed; expected {version}")


def load_source_index(source: Path) -> dict[str, Any]:
    index_path = source / "model.safetensors.index.json"
    value = json.loads(index_path.read_text(encoding="utf-8"))
    metadata = value.get("metadata")
    weight_map = value.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise ValueError("Pinned source index is missing metadata or weight_map")
    if int(metadata.get("total_size", -1)) != EXPECTED_SOURCE_WEIGHT_BYTES:
        raise ValueError("Pinned source index has an unexpected total_size")
    if len(weight_map) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(
            f"Pinned source index has {len(weight_map)} tensors; expected {EXPECTED_SOURCE_TENSORS}"
        )
    if set(weight_map.values()) != set(source_shard_names()):
        raise ValueError("Pinned source index references an unexpected shard set")
    if sum("ngram_embedding.shard_" in key for key in weight_map) != EXPECTED_NGRAM_SHARDS:
        raise ValueError("Pinned source index has an unexpected n-gram shard count")
    if sum(".experts." in key for key in weight_map) != EXPECTED_EXPERT_TENSORS:
        raise ValueError("Pinned source index has an unexpected expert tensor count")
    return value


def sanitize_key(key: str) -> str:
    if key.startswith("model.language_model"):
        return key.replace("model.language_model", "language_model.model", 1)
    if key.startswith("model.visual"):
        return key.replace("model.visual", "vision_tower", 1)
    if key.startswith("lm_head"):
        return key.replace("lm_head", "language_model.lm_head", 1)
    return key


def expert_output_keys(source_key: str) -> tuple[str, ...] | None:
    key = sanitize_key(source_key)
    if key.endswith(".mlp.experts.gate_up_proj"):
        base = key.removesuffix(".experts.gate_up_proj") + ".switch_mlp"
        return (f"{base}.gate_proj.weight", f"{base}.up_proj.weight")
    if key.endswith(".mlp.experts.down_proj"):
        base = key.removesuffix(".experts.down_proj") + ".switch_mlp"
        return (f"{base}.down_proj.weight",)
    return None


def is_router_weight(key: str) -> bool:
    return key.endswith(".mlp.gate.weight") or key.endswith(
        ".mlp.shared_expert_gate.weight"
    )


def is_mixed_sensitive_weight(key: str) -> bool:
    if key.startswith("vision_tower."):
        return True
    if key in {
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    }:
        return True
    if ".self_attn.indexer." in key:
        return True
    if key.startswith("mtp.fc_") or key.startswith("mtp.pre_fc_"):
        return True
    return False


def quantization_for(
    profile: str,
    key: str,
    shape: tuple[int, ...],
    *,
    floating: bool = True,
) -> dict[str, int | str] | None:
    if profile not in OUTPUT_NAMES:
        raise ValueError(f"Unknown profile: {profile}")
    if not floating or not key.endswith(".weight"):
        return None
    if "ngram_embedding.shard_" in key:
        if len(shape) != 2 or shape[-1] % int(NGRAM_Q4["group_size"]):
            raise ValueError(f"N-gram embedding has an incompatible shape: {key} {shape}")
        return dict(NGRAM_Q4)
    if is_router_weight(key):
        return None
    if ".switch_mlp." in key:
        if len(shape) != 3:
            raise ValueError(f"Switch MLP tensor is not rank 3: {key} {shape}")
        if (
            profile == "mixed"
            and key.startswith("language_model.model.layers.")
        ):
            return dict(EXPERT_Q2)
        if (
            profile == "q3-activation"
            and key.startswith("language_model.model.layers.")
        ):
            return dict(EXPERT_Q3)
        return dict(Q4)
    if len(shape) != 2:
        return None
    if profile == "mixed" and is_mixed_sensitive_weight(key):
        return None
    if shape[-1] % int(Q4["group_size"]):
        return None
    return dict(Q4)


def expected_output_plans(
    profile: str,
    source_key: str,
    shape: tuple[int, ...],
    *,
    floating: bool,
) -> tuple[TensorPlan, ...]:
    expert_keys = expert_output_keys(source_key)
    if expert_keys is not None:
        if source_key.endswith("gate_up_proj"):
            if len(shape) != 3 or shape[-2] % 2:
                raise ValueError(f"Fused expert tensor has an incompatible shape: {source_key} {shape}")
            output_shapes = (
                (shape[0], shape[1] // 2, shape[2]),
                (shape[0], shape[1] // 2, shape[2]),
            )
        else:
            output_shapes = (shape,)
        return tuple(
            TensorPlan(key, quantization_for(profile, key, output_shape, floating=floating))
            for key, output_shape in zip(expert_keys, output_shapes)
        )
    key = sanitize_key(source_key)
    return (TensorPlan(key, quantization_for(profile, key, shape, floating=floating)),)


def is_shifted_text_norm(key: str) -> bool:
    if not (key.startswith("language_model.") or key.startswith("mtp.")):
        return False
    return key.endswith(SHIFTED_TEXT_NORM_SUFFIXES)


def transform_dense_value(key: str, value: Any, mx: Any) -> Any:
    if key.endswith("linear_attn.conv1d.weight") and value.ndim == 3 and value.shape[-1] != 1:
        value = mx.moveaxis(value, 2, 1)
    if key == "vision_tower.patch_embed.proj.weight" and value.ndim == 5 and value.shape[-1] != 3:
        value = mx.transpose(value, (0, 2, 3, 4, 1))
    if is_shifted_text_norm(key) and value.ndim == 1:
        value = value + 1.0
    return mx.contiguous(value)


def quantize_value(
    key: str,
    value: Any,
    quantization: dict[str, int | str] | None,
    mx: Any,
    activation_profile: ActivationProfile | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if quantization is None:
        return {key: value}, None
    base = key.removesuffix(".weight")
    if quantization == EXPERT_Q3 and activation_profile is not None:
        packed, scales, biases, metrics = activation_weighted_q3(
            value, base, activation_profile, mx
        )
        return {
            key: packed,
            f"{base}.scales": scales,
            f"{base}.biases": biases,
        }, metrics
    packed = mx.quantize(
        value,
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=str(quantization["mode"]),
    )
    if len(packed) != 3:
        raise RuntimeError(f"MLX affine quantization did not emit biases for {key}")
    return {
        key: packed[0],
        f"{base}.scales": packed[1],
        f"{base}.biases": packed[2],
    }, None


def activation_weighted_q3(
    value: Any,
    module_path: str,
    profile: ActivationProfile,
    mx: Any,
    expert_batch: int = 8,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Fresh BF16->Q3 packing with fixed-code activation-weighted affine refit.

    The packed Q3 codes come directly from the original BF16 checkpoint. Only
    their group scales and biases are refit, using the diagonal input-second-
    moment objective frozen before this candidate existed. Every accepted
    group is rescored through actual MLX dequantization in the exported
    parameter dtype; unobserved or non-improving groups retain stock Q3.
    """

    if value.ndim != 3 or int(value.shape[-1]) % 64:
        raise ValueError(f"Activation-weighted expert has an incompatible shape: {module_path} {value.shape}")
    experts, output_dims, input_dims = (int(dimension) for dimension in value.shape)
    importance = profile.importance(module_path, (experts, input_dims), mx)
    packed_chunks = []
    scale_chunks = []
    bias_chunks = []
    changed_groups = 0
    observed_experts = 0
    q3_error = 0.0
    refit_error = 0.0
    q4_error = 0.0

    for start in range(0, experts, expert_batch):
        end = min(start + expert_batch, experts)
        count = end - start
        source = value[start:end]
        q3 = mx.quantize(source, group_size=64, bits=3, mode="affine")
        if len(q3) != 3:
            raise RuntimeError(f"MLX Q3 affine quantization emitted no biases for {module_path}")
        packed, old_scales, old_biases = q3
        teacher = source.astype(mx.float32).reshape(count, output_dims, -1, 64)
        weights = importance[start:end].reshape(count, 1, -1, 64)
        old_scale_f32 = old_scales.astype(mx.float32).reshape(count, output_dims, -1, 1)
        old_bias_f32 = old_biases.astype(mx.float32).reshape(count, output_dims, -1, 1)
        decoded = mx.dequantize(
            packed,
            old_scales.astype(mx.float32),
            old_biases.astype(mx.float32),
            group_size=64,
            bits=3,
            mode="affine",
        ).reshape(teacher.shape)
        codes = mx.clip(
            mx.round((decoded - old_bias_f32) / mx.where(old_scale_f32 != 0, old_scale_f32, 1)),
            0,
            7,
        )

        mass = mx.sum(weights, axis=-1)
        code_sum = mx.sum(weights * codes, axis=-1)
        code_squared = mx.sum(weights * codes * codes, axis=-1)
        target_sum = mx.sum(weights * teacher, axis=-1)
        product_sum = mx.sum(weights * codes * teacher, axis=-1)
        determinant = mass * code_squared - code_sum * code_sum
        valid = (mass > 0) & (
            determinant > mx.abs(mass * code_squared) * 0.000001
        )
        new_scale = (mass * product_sum - code_sum * target_sum) / mx.where(
            valid, determinant, 1
        )
        new_bias = (target_sum - new_scale * code_sum) / mx.where(mass > 0, mass, 1)
        cast_scale = new_scale.astype(old_scales.dtype)
        cast_bias = new_bias.astype(old_biases.dtype)

        def actual_error(scales: Any, biases: Any) -> Any:
            reconstruction = mx.dequantize(
                packed,
                scales,
                biases,
                group_size=64,
                bits=3,
                mode="affine",
            ).astype(mx.float32).reshape(teacher.shape)
            difference = teacher - reconstruction
            return mx.sum(weights * difference * difference, axis=-1)

        old_error = actual_error(old_scales, old_biases)
        candidate_error = actual_error(cast_scale, cast_bias)
        accept = valid & (candidate_error < old_error)
        accepted_scales = mx.where(accept, cast_scale, old_scales)
        accepted_biases = mx.where(accept, cast_bias, old_biases)
        accepted_error = actual_error(accepted_scales, accepted_biases)

        q4 = mx.quantize(source, group_size=64, bits=4, mode="affine")
        q4_reconstruction = mx.dequantize(
            *q4,
            group_size=64,
            bits=4,
            mode="affine",
        ).astype(mx.float32).reshape(teacher.shape)
        q4_difference = teacher - q4_reconstruction
        q4_chunk_error = mx.sum(weights * q4_difference * q4_difference)
        observed = mx.sum(weights.reshape(count, -1), axis=-1) > 0
        summary = [
            mx.sum(accept),
            mx.sum(observed),
            mx.sum(old_error),
            mx.sum(accepted_error),
            q4_chunk_error,
        ]
        mx.eval(packed, accepted_scales, accepted_biases, *summary)
        packed_chunks.append(packed)
        scale_chunks.append(accepted_scales)
        bias_chunks.append(accepted_biases)
        changed_groups += int(summary[0].item())
        observed_experts += int(summary[1].item())
        q3_error += float(summary[2].item())
        refit_error += float(summary[3].item())
        q4_error += float(summary[4].item())
        del source, teacher, weights, decoded, codes, q3, q4
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    result = (
        mx.concatenate(packed_chunks, axis=0),
        mx.concatenate(scale_chunks, axis=0),
        mx.concatenate(bias_chunks, axis=0),
    )
    mx.eval(*result)
    metrics = {
        "method": "fresh-bf16-fixed-code-diagonal-activation-refit-v1",
        "experts": experts,
        "observed_experts": observed_experts,
        "groups": int(result[1].size),
        "changed_groups": changed_groups,
        "stock_q3_weighted_error": q3_error,
        "refit_q3_weighted_error": refit_error,
        "stock_q4_weighted_error": q4_error,
        "q3_to_q4_error_ratio": refit_error / q4_error if q4_error > 0 else None,
    }
    return result[0], result[1], result[2], metrics


def transform_tensor(
    profile: str,
    source_key: str,
    value: Any,
    mx: Any,
    activation_profile: ActivationProfile | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, int | str]], dict[str, dict[str, Any]]]:
    floating = bool(mx.issubdtype(value.dtype, mx.floating))
    expert_keys = expert_output_keys(source_key)
    if expert_keys is not None:
        if len(expert_keys) == 2:
            values = tuple(mx.contiguous(part) for part in mx.split(value, 2, axis=-2))
        else:
            values = (mx.contiguous(value),)
        outputs: dict[str, Any] = {}
        modules: dict[str, dict[str, int | str]] = {}
        metrics: dict[str, dict[str, Any]] = {}
        for key, expert_value in zip(expert_keys, values):
            quantization = quantization_for(
                profile,
                key,
                tuple(int(item) for item in expert_value.shape),
                floating=floating,
            )
            arrays, module_metrics = quantize_value(
                key, expert_value, quantization, mx, activation_profile
            )
            outputs.update(arrays)
            if quantization is not None:
                modules[key.removesuffix(".weight")] = quantization
            if module_metrics is not None:
                metrics[key.removesuffix(".weight")] = module_metrics
        return outputs, modules, metrics

    key = sanitize_key(source_key)
    value = transform_dense_value(key, value, mx)
    quantization = quantization_for(
        profile,
        key,
        tuple(int(item) for item in value.shape),
        floating=floating,
    )
    outputs, module_metrics = quantize_value(
        key, value, quantization, mx, activation_profile
    )
    modules = (
        {key.removesuffix(".weight"): quantization}
        if quantization is not None
        else {}
    )
    metrics = {key.removesuffix(".weight"): module_metrics} if module_metrics is not None else {}
    return outputs, modules, metrics


def initial_state(
    profile: str,
    activation_profile: ActivationProfile | None = None,
) -> dict[str, Any]:
    state = {
        "schema_version": STATE_VERSION,
        "profile": profile,
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "created_at": utc_now(),
        "completed_shards": {},
    }
    if profile == "q3-activation":
        if activation_profile is None:
            raise ValueError("q3-activation requires --activation-profile")
        state["activation_profile"] = {
            "sha256": activation_profile.sha256,
            "metadata": activation_profile.metadata,
        }
    return state


def load_state(
    output: Path,
    profile: str,
    activation_profile: ActivationProfile | None = None,
) -> dict[str, Any]:
    path = output / STATE_FILENAME
    if not path.exists():
        return initial_state(profile, activation_profile)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_VERSION:
        raise ValueError(f"Unsupported conversion state in {path}")
    if state.get("profile") != profile:
        raise ValueError(f"Conversion state profile mismatch in {path}")
    source = state.get("source", {})
    if source.get("repository") != SOURCE_REPOSITORY or source.get("revision") != SOURCE_REVISION:
        raise ValueError(f"Conversion state source mismatch in {path}")
    if profile == "q3-activation":
        if activation_profile is None:
            raise ValueError("q3-activation requires --activation-profile")
        recorded = state.get("activation_profile", {})
        if recorded.get("sha256") != activation_profile.sha256:
            raise ValueError(f"Conversion state activation profile mismatch in {path}")
    return state


def completed_output_is_usable(output: Path, record: dict[str, Any]) -> bool:
    path = output / str(record.get("filename", ""))
    return path.is_file() and path.stat().st_size == int(record.get("byte_count", -1))


def output_shard_name(index: int) -> str:
    return f"model-{index:05d}-of-{EXPECTED_SOURCE_SHARDS:05d}.safetensors"


def save_output_shard(path: Path, arrays: dict[str, Any], mx: Any) -> tuple[int, str]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    mx.eval(*arrays.values())
    mx.save_safetensors(
        str(temporary),
        arrays,
        metadata={
            "format": "mlx",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
    )
    os.replace(temporary, path)
    return path.stat().st_size, sha256_file(path)


def process_source_shard(
    source_path: Path,
    shard_index: int,
    profiles: Iterable[str],
    outputs: dict[str, Path],
    states: dict[str, dict[str, Any]],
    expected_source: dict[str, int | str],
    mx: Any,
    activation_profile: ActivationProfile | None,
) -> None:
    source_name = source_path.name
    missing_profiles = [
        profile
        for profile in profiles
        if not completed_output_is_usable(
            outputs[profile],
            states[profile]["completed_shards"].get(source_name, {}),
        )
    ]
    if not missing_profiles:
        print(f"[{shard_index}/{EXPECTED_SOURCE_SHARDS}] resume {source_name}", flush=True)
        return

    actual_bytes = source_path.stat().st_size
    if actual_bytes != int(expected_source["byte_count"]):
        raise ValueError(
            f"{source_name} has {actual_bytes} bytes; expected {expected_source['byte_count']}"
        )
    source_sha256 = sha256_file(source_path)
    if source_sha256 != expected_source["sha256"]:
        raise ValueError(
            f"{source_name} has SHA-256 {source_sha256}; expected {expected_source['sha256']}"
        )

    source_arrays = mx.load(str(source_path))
    source_keys = sorted(source_arrays)
    for profile in missing_profiles:
        converted: dict[str, Any] = {}
        modules: dict[str, dict[str, int | str]] = {}
        metrics: dict[str, dict[str, Any]] = {}
        for source_key in source_keys:
            arrays, tensor_modules, tensor_metrics = transform_tensor(
                profile,
                source_key,
                source_arrays[source_key],
                mx,
                activation_profile if profile == "q3-activation" else None,
            )
            duplicate_keys = set(converted).intersection(arrays)
            if duplicate_keys:
                raise ValueError(f"Duplicate converted keys: {sorted(duplicate_keys)}")
            converted.update(arrays)
            modules.update(tensor_modules)
            metrics.update(tensor_metrics)

        filename = output_shard_name(shard_index)
        byte_count, digest = save_output_shard(
            outputs[profile] / filename,
            converted,
            mx,
        )
        states[profile]["completed_shards"][source_name] = {
            "filename": filename,
            "byte_count": byte_count,
            "sha256": digest,
            "source_byte_count": actual_bytes,
            "source_sha256": source_sha256,
            "source_keys": source_keys,
            "output_keys": sorted(converted),
            "quantized_modules": modules,
            "activation_metrics": metrics,
        }
        atomic_json(outputs[profile] / STATE_FILENAME, states[profile])
        del converted, modules, metrics
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    del source_arrays
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    print(
        f"[{shard_index}/{EXPECTED_SOURCE_SHARDS}] converted {source_name} -> {', '.join(missing_profiles)}",
        flush=True,
    )


def aggregate_profile_state(
    state: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, int | str]], int]:
    completed = state["completed_shards"]
    expected = source_shard_names()
    if set(completed) != set(expected):
        missing = sorted(set(expected) - set(completed))
        raise ValueError(f"Conversion is incomplete; missing source shards: {missing}")
    weight_map: dict[str, str] = {}
    modules: dict[str, dict[str, int | str]] = {}
    logical_bytes = 0
    for source_name in expected:
        record = completed[source_name]
        filename = record["filename"]
        for key in record["output_keys"]:
            if key in weight_map:
                raise ValueError(f"Duplicate output key across shards: {key}")
            weight_map[key] = filename
        for path, quantization in record["quantized_modules"].items():
            existing = modules.get(path)
            if existing is not None and existing != quantization:
                raise ValueError(f"Conflicting quantization metadata for {path}")
            modules[path] = quantization
    return dict(sorted(weight_map.items())), dict(sorted(modules.items())), logical_bytes


def output_tensor_inventory(output: Path, weight_map: dict[str, str]) -> tuple[int, int]:
    from safetensors import safe_open

    seen: set[str] = set()
    logical_bytes = 0
    for filename in sorted(set(weight_map.values())):
        with safe_open(output / filename, framework="numpy") as archive:
            keys = set(archive.keys())
            expected = {key for key, shard in weight_map.items() if shard == filename}
            if keys != expected:
                raise ValueError(
                    f"{filename} tensor inventory mismatch: missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
                )
            for key in sorted(keys):
                tensor = archive.get_slice(key)
                shape = tensor.get_shape()
                dtype = tensor.get_dtype()
                item_bytes = {
                    "BOOL": 1,
                    "I8": 1,
                    "U8": 1,
                    "I16": 2,
                    "U16": 2,
                    "F16": 2,
                    "BF16": 2,
                    "I32": 4,
                    "U32": 4,
                    "F32": 4,
                    "I64": 8,
                    "U64": 8,
                    "F64": 8,
                }.get(dtype)
                if item_bytes is None:
                    raise ValueError(f"Unsupported safetensors dtype {dtype} for {key}")
                elements = 1
                for dimension in shape:
                    elements *= int(dimension)
                logical_bytes += elements * item_bytes
            seen.update(keys)
    if seen != set(weight_map):
        raise ValueError("Output tensor inventory did not cover the complete weight map")
    return len(seen), logical_bytes


def converted_config(
    source: Path,
    profile: str,
    modules: dict[str, dict[str, int | str]],
) -> dict[str, Any]:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("Pinned source config is not qwen4_exp")
    quantization: dict[str, Any] = dict(Q4)
    for path, parameters in modules.items():
        if parameters != Q4:
            marker = ".ngram_embedding.shard_"
            if marker in path:
                prefix, shard = path.split(marker, 1)
                path = f"{prefix}.ngram_embedding.shards.{shard}"
            quantization[path] = parameters
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    config["mererun_conversion"] = {
        "profile": profile,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "converter": "scripts/model-conversion/convert_qwen38_flash_next_mlx.py",
        "converter_version": STATE_VERSION,
    }
    return config


def copy_metadata(source: Path, output: Path) -> None:
    excluded = {
        ".gitattributes",
        "config.json",
        "model.safetensors.index.json",
        "README.md",
    }
    for path in source.iterdir():
        if not path.is_file() or path.name in excluded:
            continue
        if path.name.startswith("model-") and path.name.endswith(".safetensors"):
            continue
        shutil.copyfile(path, output / path.name)
    upstream_card = source / "README.md"
    if upstream_card.is_file():
        shutil.copyfile(upstream_card, output / "README.upstream.md")


def model_card(profile: str, artifact_bytes: int, module_counts: Counter[str]) -> str:
    gib = artifact_bytes / (1024**3)
    if profile == "q4":
        title = "Qwen3.8-Flash-Next MLX 4-bit"
        description = (
            "MLX affine Q4/group-64 for eligible language, MTP, and vision matrices; "
            "the 160-wide n-gram table uses Q4/group-32. Routers, norms, biases, "
            "convolutions, and incompatible shapes remain dense."
        )
    elif profile == "mixed":
        title = "Qwen3.8-Flash-Next MLX Mixed 2-bit/4-bit"
        description = (
            "The 48 base routed-expert banks use MLX affine Q2/group-128, the "
            "160-wide n-gram table and remaining eligible core/MTP matrices use "
            "Q4, while token/output embeddings, QSA indexers, routers, vision, "
            "and MTP fusion heads remain BF16. This is the 128 GB Mac profile."
        )
    else:
        title = "Qwen3.8-Flash-Next MLX Activation-Weighted 3-bit"
        description = (
            "The 48 base routed-expert banks use fresh MLX affine Q3/group-64 "
            "codes generated directly from the original BF16 checkpoint. Their "
            "scales and biases are refit against frozen image-and-text expert-input "
            "second moments. Remaining eligible core, MTP, and vision matrices stay "
            "Q4; the 160-wide n-gram table stays Q4/group-32."
        )
    run_section = ""
    if profile == "q3-activation":
        run_section = """
## Run locally with mere.run

This profile uses the managed model ID `vision-chat-q38-flash-next-3bit`.
Review the bundled license and pull the checkpoint:

```bash
mere.run model pull vision-chat-q38-flash-next-3bit \\
    --accept-license-terms
```

Then generate text with the bundled multi-token prediction head:

```bash
mere.run text chat \\
    --model vision-chat-q38-flash-next-3bit \\
    --context-size 32768 \\
    --max-tokens 256 \\
    --temperature 0 \\
    --no-thinking \\
    --stream \\
    --stats \\
    --prompt "Explain sparse attention in three short sentences."
```

"""
    return f"""---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: {SOURCE_REPOSITORY}
pipeline_tag: image-text-to-text
tags:
  - mlx
  - quantized
  - qwen4-exp
---

# {title}

This is a reproducible MLX conversion of
[`{SOURCE_REPOSITORY}`](https://huggingface.co/{SOURCE_REPOSITORY}) at immutable
revision `{SOURCE_REVISION}`.

{description}

- Artifact payload: {gib:.2f} GiB
- Quantized Q2 modules: {module_counts['q2']}
- Quantized Q3/group-64 modules: {module_counts['q3_g64']}
- Quantized Q4/group-32 modules: {module_counts['q4_g32']}
- Quantized Q4/group-64 modules: {module_counts['q4_g64']}
- Source: 180B parameters including 125B main, 51B n-gram embedding, and 4B MTP
- Native context: 262,144 tokens

{run_section}
## Runtime status

The tensor inventory, source hashes, MLX packing, fused-expert split, convolution
layout, and zero-centered RMSNorm conversion are validated by the bundled
`MERERUN_CONVERSION.json`. Use a Qwen4Exp-aware runtime such as mere.run or
mlx-vlm.

## License

This redistribution retains the upstream **Qwen Community License 1.0** in
`LICENSE`. Review it before use. In particular, it contains attribution/display
requirements for very large commercial products and separate-license conditions
for certain commercial Model-as-a-Service and AI Work Assistant uses. The model
is not gated; downloading or using it does not remove those terms.

The upstream model card is preserved as `README.upstream.md`.
"""


def finalize_profile(
    source: Path,
    output: Path,
    profile: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    weight_map, modules, _ = aggregate_profile_state(state)
    tensor_count, logical_bytes = output_tensor_inventory(output, weight_map)
    if tensor_count != EXPECTED_OUTPUT_TENSORS[profile]:
        raise ValueError(
            f"{profile} emitted {tensor_count} tensors; expected {EXPECTED_OUTPUT_TENSORS[profile]}"
        )
    if logical_bytes != EXPECTED_OUTPUT_LOGICAL_BYTES[profile]:
        raise ValueError(
            f"{profile} emitted {logical_bytes} logical bytes; "
            f"expected {EXPECTED_OUTPUT_LOGICAL_BYTES[profile]}"
        )
    if len(modules) != EXPECTED_QUANTIZED_MODULES[profile]:
        raise ValueError(
            f"{profile} quantized {len(modules)} modules; "
            f"expected {EXPECTED_QUANTIZED_MODULES[profile]}"
        )

    index = {
        "metadata": {
            "total_size": logical_bytes,
            "total_parameters": EXPECTED_SOURCE_PARAMETERS,
            "source_total_size": EXPECTED_SOURCE_WEIGHT_BYTES,
        },
        "weight_map": weight_map,
    }
    atomic_json(output / "model.safetensors.index.json", index)
    atomic_json(output / "config.json", converted_config(source, profile, modules))
    copy_metadata(source, output)

    artifacts = []
    artifact_bytes = 0
    for path in sorted(output.glob("model-*.safetensors")):
        byte_count = path.stat().st_size
        artifact_bytes += byte_count
        artifacts.append(
            {
                "filename": path.name,
                "byte_count": byte_count,
                "sha256": sha256_file(path),
            }
        )

    module_counts: Counter[str] = Counter()
    for parameters in modules.values():
        bits = int(parameters["bits"])
        group_size = int(parameters["group_size"])
        if bits == 2:
            module_counts["q2"] += 1
        elif bits == 3:
            module_counts["q3_g64"] += 1
        elif group_size == 32:
            module_counts["q4_g32"] += 1
        else:
            module_counts["q4_g64"] += 1

    activation_metrics = {
        path: metrics
        for source_name in source_shard_names()
        for path, metrics in state["completed_shards"][source_name]
        .get("activation_metrics", {})
        .items()
    }
    receipt = {
        "schema_version": 1,
        "converter": "scripts/model-conversion/convert_qwen38_flash_next_mlx.py",
        "converter_version": STATE_VERSION,
        "completed_at": utc_now(),
        "profile": profile,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "weight_bytes": EXPECTED_SOURCE_WEIGHT_BYTES,
            "parameters": EXPECTED_SOURCE_PARAMETERS,
            "index_sha256": sha256_file(source / "model.safetensors.index.json"),
            "config_sha256": sha256_file(source / "config.json"),
            "shards": [
                {
                    "filename": name,
                    "byte_count": int(state["completed_shards"][name]["source_byte_count"]),
                    "sha256": state["completed_shards"][name]["source_sha256"],
                }
                for name in source_shard_names()
            ],
        },
        "quantization": {
            "default": Q4,
            "ngram_embedding": NGRAM_Q4,
            "base_routed_experts": (
                EXPERT_Q2
                if profile == "mixed"
                else EXPERT_Q3 if profile == "q3-activation" else Q4
            ),
            "module_counts": dict(sorted(module_counts.items())),
            "quantized_module_count": len(modules),
        },
        "validation": {
            "source_tensor_count": EXPECTED_SOURCE_TENSORS,
            "output_tensor_count": tensor_count,
            "output_logical_bytes": logical_bytes,
            "complete_weight_map": True,
            "source_shards_sha256_verified": EXPECTED_SOURCE_SHARDS,
            "output_shards_sha256_verified": len(artifacts),
            "generation_smoke": False,
            "generation_smoke_reason": "Conversion completed before native output-quality qualification.",
        },
        "toolchain": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "numpy": importlib.metadata.version("numpy"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "artifacts": artifacts,
        "hugging_face_repository": HF_REPOSITORIES[profile],
    }
    if profile == "q3-activation":
        receipt["activation_weighting"] = {
            "profile": state["activation_profile"],
            "method": "fresh-bf16-fixed-code-diagonal-activation-refit-v1",
            "module_metrics": dict(sorted(activation_metrics.items())),
        }
    atomic_json(output / "MERERUN_CONVERSION.json", receipt)
    (output / "README.md").write_text(
        model_card(profile, artifact_bytes, module_counts),
        encoding="utf-8",
    )
    return receipt


def validate_source_presence(source: Path, manifest: dict[str, dict[str, int | str]]) -> None:
    for filename in source_shard_names():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Pinned source shard is missing: {path}")
        expected = manifest[filename]
        if path.stat().st_size != int(expected["byte_count"]):
            raise ValueError(f"Pinned source shard has an unexpected size: {path}")


def run_self_test() -> None:
    assert sanitize_key("model.language_model.layers.0.x") == "language_model.model.layers.0.x"
    assert sanitize_key("model.visual.patch_embed.proj.weight") == "vision_tower.patch_embed.proj.weight"
    assert sanitize_key("lm_head.weight") == "language_model.lm_head.weight"
    gate, up = expert_output_keys("model.language_model.layers.0.mlp.experts.gate_up_proj") or ()
    assert gate.endswith("mlp.switch_mlp.gate_proj.weight")
    assert up.endswith("mlp.switch_mlp.up_proj.weight")
    assert quantization_for("mixed", gate, (512, 640, 2560)) == EXPERT_Q2
    assert quantization_for("q3-activation", gate, (512, 640, 2560)) == EXPERT_Q3
    assert quantization_for("q4", gate, (512, 640, 2560)) == Q4
    assert quantization_for("mixed", "mtp.layers.0.mlp.switch_mlp.gate_proj.weight", (512, 640, 2560)) == Q4
    assert quantization_for(
        "mixed",
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        (2_500_012, 160),
    ) == NGRAM_Q4
    assert quantization_for("mixed", "language_model.model.embed_tokens.weight", (248_320, 2_560)) is None
    assert quantization_for("q4", "language_model.model.embed_tokens.weight", (248_320, 2_560)) == Q4
    assert quantization_for("mixed", "vision_tower.blocks.0.attn.qkv.weight", (3_456, 1_152)) is None
    assert quantization_for("q4", "vision_tower.blocks.0.attn.qkv.weight", (3_456, 1_152)) == Q4
    assert quantization_for("q4", "vision_tower.blocks.0.mlp.linear_fc2.weight", (1_152, 4_304)) is None
    assert is_shifted_text_norm("language_model.model.layers.0.attn_hyper_connection.hc_norm.weight")
    assert not is_shifted_text_norm("language_model.model.layers.0.linear_attn.norm.weight")

    import mlx.core as mx

    fixture = mx.arange(4 * 128, dtype=mx.float32).reshape(4, 128) / 97.0 - 2.0
    for parameters in (Q4, EXPERT_Q2, EXPERT_Q3, NGRAM_Q4):
        packed = mx.quantize(
            fixture,
            group_size=int(parameters["group_size"]),
            bits=int(parameters["bits"]),
            mode=str(parameters["mode"]),
        )
        restored = mx.dequantize(
            *packed,
            group_size=int(parameters["group_size"]),
            bits=int(parameters["bits"]),
            mode=str(parameters["mode"]),
        )
        mx.eval(restored)
        maximum = float(mx.max(mx.abs(fixture - restored)))
        if not maximum < 1.5:
            raise AssertionError(f"Unexpected MLX affine round-trip error: {maximum}")

    expert_fixture = mx.arange(2 * 8 * 128, dtype=mx.float32).reshape(2, 8, 128)
    expert_arrays, expert_modules, expert_metrics = transform_tensor(
        "mixed",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        expert_fixture,
        mx,
    )
    expected_expert_modules = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
        "language_model.model.layers.0.mlp.switch_mlp.up_proj",
    }
    assert set(expert_modules) == expected_expert_modules
    assert all(parameters == EXPERT_Q2 for parameters in expert_modules.values())
    assert expert_metrics == {}
    assert set(expert_arrays) == {
        f"{module}.{suffix}"
        for module in expected_expert_modules
        for suffix in ("weight", "scales", "biases")
    }

    class SyntheticProfile:
        def importance(self, module_path, expected_shape, mlx):
            assert module_path.endswith("switch_mlp.gate_proj")
            assert expected_shape == (2, 128)
            return mlx.ones(expected_shape, dtype=mlx.float32)

    q3_source = (
        mx.sin(mx.arange(2 * 8 * 128, dtype=mx.float32) / 17.0)
        .reshape(2, 8, 128)
        .astype(mx.bfloat16)
    )
    q3_packed, q3_scales, q3_biases, q3_metrics = activation_weighted_q3(
        q3_source,
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
        SyntheticProfile(),
        mx,
        expert_batch=1,
    )
    mx.eval(q3_packed, q3_scales, q3_biases)
    assert q3_packed.dtype == mx.uint32
    assert q3_scales.shape == (2, 8, 2)
    assert q3_biases.shape == (2, 8, 2)
    assert q3_metrics["refit_q3_weighted_error"] <= q3_metrics["stock_q3_weighted_error"]
    assert q3_metrics["groups"] == 32

    shifted = transform_dense_value(
        "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight",
        mx.zeros((8,), dtype=mx.bfloat16),
        mx,
    )
    gated = transform_dense_value(
        "language_model.model.layers.0.linear_attn.norm.weight",
        mx.ones((8,), dtype=mx.bfloat16),
        mx,
    )
    text_conv = transform_dense_value(
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        mx.zeros((16, 1, 4), dtype=mx.bfloat16),
        mx,
    )
    vision_conv = transform_dense_value(
        "vision_tower.patch_embed.proj.weight",
        mx.zeros((16, 3, 2, 4, 4), dtype=mx.bfloat16),
        mx,
    )
    mx.eval(shifted, gated, text_conv, vision_conv)
    assert bool(mx.all(shifted == 1.0))
    assert bool(mx.all(gated == 1.0))
    assert text_conv.shape == (16, 4, 1)
    assert vision_conv.shape == (16, 2, 4, 4, 3)

    with tempfile.TemporaryDirectory(prefix="qwen38-converter-self-test-") as directory:
        root = Path(directory)
        output = root / output_shard_name(1)
        byte_count, digest = save_output_shard(output, expert_arrays, mx)
        assert byte_count == output.stat().st_size
        assert digest == sha256_file(output)
        weight_map = {key: output.name for key in expert_arrays}
        tensor_count, logical_bytes = output_tensor_inventory(root, weight_map)
        assert tensor_count == len(expert_arrays)
        assert logical_bytes > 0
    print("Qwen3.8-Flash-Next converter self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(OUTPUT_NAMES),
        default=["mixed", "q4"],
    )
    parser.add_argument("--activation-profile", type=Path)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        verify_environment()
        run_self_test()
        return
    if args.workspace is None:
        raise ValueError("--workspace is required unless --self-test is used")

    verify_environment()
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source = (
        args.source.expanduser().resolve()
        if args.source is not None
        else workspace / "source"
    )
    manifest = remote_source_manifest()
    if not args.skip_download:
        source = download_source(source, args.max_workers)
    validate_source_presence(source, manifest)
    source_index = load_source_index(source)

    activation_profile = None
    if "q3-activation" in args.profiles:
        if args.activation_profile is None:
            raise ValueError("q3-activation requires --activation-profile")
        activation_profile = ActivationProfile(args.activation_profile)
    elif args.activation_profile is not None:
        raise ValueError("--activation-profile is only valid with q3-activation")

    outputs = {profile: workspace / OUTPUT_NAMES[profile] for profile in args.profiles}
    states: dict[str, dict[str, Any]] = {}
    for profile, output in outputs.items():
        output.mkdir(parents=True, exist_ok=True)
        states[profile] = load_state(output, profile, activation_profile)
        (output / ".incomplete").write_text(
            "Conversion is resumable but not publishable until this marker is removed.\n",
            encoding="utf-8",
        )

    if not args.validate_only:
        import mlx.core as mx

        weight_map = source_index["weight_map"]
        keys_by_shard: dict[str, set[str]] = {
            name: {key for key, shard in weight_map.items() if shard == name}
            for name in source_shard_names()
        }
        for index, filename in enumerate(source_shard_names(), start=1):
            process_source_shard(
                source / filename,
                index,
                args.profiles,
                outputs,
                states,
                manifest[filename],
                mx,
                activation_profile,
            )
            for profile in args.profiles:
                record = states[profile]["completed_shards"].get(filename)
                if record is not None and set(record["source_keys"]) != keys_by_shard[filename]:
                    raise ValueError(f"Source tensor inventory mismatch for {filename}")

    for profile in args.profiles:
        receipt = finalize_profile(source, outputs[profile], profile, states[profile])
        (outputs[profile] / ".incomplete").unlink(missing_ok=True)
        print(
            f"finalized {profile}: {outputs[profile]} "
            f"({receipt['validation']['output_logical_bytes']} logical bytes)",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed shards remain resumable.", file=sys.stderr)
        raise SystemExit(130)

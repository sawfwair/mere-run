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
"""Build resumable MLX Q4 and mixed Q2/Q4 Qwen3.8-Flash-Next artifacts.

This is release tooling, not an inference sidecar. It downloads one immutable
BF16 snapshot, converts one source shard at a time, and writes both profiles
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
}
EXPECTED_OUTPUT_TENSORS = {"q4": 3_817, "mixed": 3_615}
EXPECTED_QUANTIZED_MODULES = {"q4": 1_055, "mixed": 954}

MLX_VERSION = "0.32.2"
HUGGINGFACE_HUB_VERSION = "1.28.0"
SAFETENSORS_VERSION = "0.8.0"
MODE = "affine"
Q4 = {"bits": 4, "group_size": 64, "mode": MODE}
NGRAM_Q4 = {"bits": 4, "group_size": 32, "mode": MODE}
EXPERT_Q2 = {"bits": 2, "group_size": 128, "mode": MODE}
OUTPUT_NAMES = {
    "q4": "Qwen3.8-Flash-Next-MLX-4bit",
    "mixed": "Qwen3.8-Flash-Next-MLX-Mixed-2bit",
}
HF_REPOSITORIES = {
    "q4": "Sawfwair/Qwen3.8-Flash-Next-MLX-4bit",
    "mixed": "Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit",
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
) -> dict[str, Any]:
    if quantization is None:
        return {key: value}
    base = key.removesuffix(".weight")
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
    }


def transform_tensor(
    profile: str,
    source_key: str,
    value: Any,
    mx: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, int | str]]]:
    floating = bool(mx.issubdtype(value.dtype, mx.floating))
    expert_keys = expert_output_keys(source_key)
    if expert_keys is not None:
        if len(expert_keys) == 2:
            values = tuple(mx.contiguous(part) for part in mx.split(value, 2, axis=-2))
        else:
            values = (mx.contiguous(value),)
        outputs: dict[str, Any] = {}
        modules: dict[str, dict[str, int | str]] = {}
        for key, expert_value in zip(expert_keys, values):
            quantization = quantization_for(
                profile,
                key,
                tuple(int(item) for item in expert_value.shape),
                floating=floating,
            )
            outputs.update(quantize_value(key, expert_value, quantization, mx))
            if quantization is not None:
                modules[key.removesuffix(".weight")] = quantization
        return outputs, modules

    key = sanitize_key(source_key)
    value = transform_dense_value(key, value, mx)
    quantization = quantization_for(
        profile,
        key,
        tuple(int(item) for item in value.shape),
        floating=floating,
    )
    outputs = quantize_value(key, value, quantization, mx)
    modules = (
        {key.removesuffix(".weight"): quantization}
        if quantization is not None
        else {}
    )
    return outputs, modules


def initial_state(profile: str) -> dict[str, Any]:
    return {
        "schema_version": STATE_VERSION,
        "profile": profile,
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "created_at": utc_now(),
        "completed_shards": {},
    }


def load_state(output: Path, profile: str) -> dict[str, Any]:
    path = output / STATE_FILENAME
    if not path.exists():
        return initial_state(profile)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_VERSION:
        raise ValueError(f"Unsupported conversion state in {path}")
    if state.get("profile") != profile:
        raise ValueError(f"Conversion state profile mismatch in {path}")
    source = state.get("source", {})
    if source.get("repository") != SOURCE_REPOSITORY or source.get("revision") != SOURCE_REVISION:
        raise ValueError(f"Conversion state source mismatch in {path}")
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
        for source_key in source_keys:
            arrays, tensor_modules = transform_tensor(
                profile,
                source_key,
                source_arrays[source_key],
                mx,
            )
            duplicate_keys = set(converted).intersection(arrays)
            if duplicate_keys:
                raise ValueError(f"Duplicate converted keys: {sorted(duplicate_keys)}")
            converted.update(arrays)
            modules.update(tensor_modules)

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
        }
        atomic_json(outputs[profile] / STATE_FILENAME, states[profile])
        del converted, modules
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
    else:
        title = "Qwen3.8-Flash-Next MLX Mixed 2-bit/4-bit"
        description = (
            "The 48 base routed-expert banks use MLX affine Q2/group-128, the "
            "160-wide n-gram table and remaining eligible core/MTP matrices use "
            "Q4, while token/output embeddings, QSA indexers, routers, vision, "
            "and MTP fusion heads remain BF16. This is the 128 GB Mac profile."
        )
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
- Quantized Q4/group-32 modules: {module_counts['q4_g32']}
- Quantized Q4/group-64 modules: {module_counts['q4_g64']}
- Source: 180B parameters including 125B main, 51B n-gram embedding, and 4B MTP
- Native context: 262,144 tokens

## Runtime status

The tensor inventory, source hashes, MLX packing, fused-expert split, convolution
layout, and zero-centered RMSNorm conversion are validated by the bundled
`MERERUN_CONVERSION.json`. A Qwen4Exp-aware MLX runtime is required; do not
expect an older `mlx-lm` or `mlx-vlm` release to dispatch this new architecture.

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
        elif group_size == 32:
            module_counts["q4_g32"] += 1
        else:
            module_counts["q4_g64"] += 1

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
            "base_routed_experts": EXPERT_Q2 if profile == "mixed" else Q4,
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
            "generation_smoke_reason": "Qwen4Exp MLX runtime integration is not yet present in upstream mlx-lm or mlx-vlm.",
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
    for parameters in (Q4, EXPERT_Q2, NGRAM_Q4):
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
    expert_arrays, expert_modules = transform_tensor(
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
    assert set(expert_arrays) == {
        f"{module}.{suffix}"
        for module in expected_expert_modules
        for suffix in ("weight", "scales", "biases")
    }

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
        default=sorted(OUTPUT_NAMES),
    )
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

    outputs = {profile: workspace / OUTPUT_NAMES[profile] for profile in args.profiles}
    states: dict[str, dict[str, Any]] = {}
    for profile, output in outputs.items():
        output.mkdir(parents=True, exist_ok=True)
        states[profile] = load_state(output, profile)
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

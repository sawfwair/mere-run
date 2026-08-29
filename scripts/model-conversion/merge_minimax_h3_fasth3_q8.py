#!/usr/bin/env python3
"""Build a self-contained premerged Q8 FastH3 package from audited local inputs.

This is offline release tooling, not a runtime sidecar. It consumes mere.run's
compact BF16 MiniMax-H3 package and the exact finalized FastVideo adapter. The
208 inference core linears are merged with FP32 accumulation, rounded once to
BF16, and encoded with MLX affine Q8/group-64. The 50 VSA compression gates are
encoded with the same quantizer. Cache-covered AdaLN parameters remain omitted;
the source-bound FastH3 AdaLN table is copied into the output package.

Example::

    python merge_minimax_h3_fasth3_q8.py \
      --base-root /Volumes/MODELS/video-minimax-h3-fl2va-bf16-mlx \
      --adapter /path/to/fastvideo_fasth3_4step_v1_vsa_datafree_rank64.safetensors \
      --fast-h3-cache /path/to/fastvideo_fasth3_v1_vsa_datafree_adaln_cache.safetensors \
      --conversion-location "CA, Canada" \
      --output /Volumes/MODELS/video-minimax-h3-fasth3-vsa-datafree-q8-mlx
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GROUP_SIZE = 64
BITS = 8
BASE_TRANSFORMER_FORMAT = "mere.run.minimax-h3-inference-transformer"
SOURCE_ADAPTER_FORMAT = "fastvideo-lora-v2"
OUTPUT_ADAPTER_FORMAT = "mere.run.minimax-h3-fasth3-premerged-v1"
FAST_H3_MODEL = "FastVideo/FastVideo-FastH3-4-step-v1"
FAST_H3_SOURCE_IDENTITY = (
    "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
    "@b65818d41939b5085451074fe8ca8b799f8d4921:transformer"
)
ADAPTER_FILENAME = "fastvideo_fasth3_4step_v1_vsa_datafree_rank64.safetensors"
FAST_H3_CACHE_FILENAME = "fastvideo_fasth3_v1_vsa_datafree_adaln_cache.safetensors"
EXPECTED_LORA_PAIRS = 362
EXPECTED_CORE_PAIRS = 312
EXPECTED_CORE_TARGETS = 208
EXPECTED_DIFFS = 82
EXPECTED_GATES = 50

QUANTIZED_SUFFIXES = (
    ".attn.qkv_proj.weight",
    ".attn.out_proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
)


@dataclass(frozen=True)
class AdapterTarget:
    module: str
    branch: str | None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_safetensors(path: Path, arrays: dict[str, Any], metadata: dict[str, str]) -> None:
    import mlx.core as mx

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.safetensors")
    try:
        mx.save_safetensors(str(temporary), dict(sorted(arrays.items())), metadata=metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path} has no safetensors header")
        header_length = int.from_bytes(raw_length, "little")
        if not 0 < header_length <= 64 * 1024 * 1024:
            raise ValueError(f"{path} has an invalid safetensors header")
        raw_header = handle.read(header_length)
    if len(raw_header) != header_length:
        raise ValueError(f"{path} has a truncated safetensors header")
    header = json.loads(raw_header)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} metadata is not an object")
    return header, metadata


def adapter_target(source_module: str) -> AdapterTarget:
    if source_module.startswith("transformer_blocks."):
        normalized = "blocks." + source_module.removeprefix("transformer_blocks.")
    elif source_module.startswith("token_refiner.refiner_blocks."):
        normalized = (
            "token_refiner.blocks."
            + source_module.removeprefix("token_refiner.refiner_blocks.")
        )
    else:
        raise ValueError(f"Unsupported FastH3 LoRA module: {source_module}")

    for suffix, branch in (
        (".attn.to_q", "query"),
        (".attn.to_k", "key"),
        (".attn.to_v", "value"),
    ):
        if normalized.endswith(suffix):
            return AdapterTarget(
                normalized.removesuffix(suffix) + ".attn.qkv_proj",
                branch,
            )
    for suffix, replacement in (
        (".attn.to_out.0", ".attn.out_proj"),
        (".ff.net.0.proj", ".mlp.fc1"),
        (".ff.net.2", ".mlp.fc2"),
        (".adaln_proj.linear", ".adaln_proj.linear"),
    ):
        if normalized.endswith(suffix):
            return AdapterTarget(normalized.removesuffix(suffix) + replacement, None)
    raise ValueError(f"Unsupported FastH3 LoRA module: {source_module}")


def difference_target(source_key: str) -> str:
    if source_key.endswith(".diff_b"):
        source = source_key.removesuffix(".diff_b")
        parameter = ".bias"
    elif source_key.endswith(".diff"):
        source = source_key.removesuffix(".diff")
        parameter = ".weight"
    else:
        raise ValueError(f"Unsupported FastH3 difference: {source_key}")
    aliases = {
        "proj_in": "video_patch_proj",
        "audio_proj_in": "audio_patch_proj",
        "context_embedder": "condition_proj",
        "proj_out": "final_layer.video_out",
        "audio_proj_out": "final_layer.audio_out",
        "norm_out.norm": "final_layer.norm",
        "norm_out.linear": "final_layer.adaln_proj.linear",
        "time_embedder.linear_1": "time_embedder.proj_in",
        "time_embedder.linear_2": "time_embedder.proj_out",
    }
    if source in aliases:
        mapped = aliases[source]
    elif source.startswith("transformer_blocks."):
        mapped = "blocks." + source.removeprefix("transformer_blocks.")
    else:
        raise ValueError(f"Unsupported FastH3 difference: {source_key}")
    return mapped + parameter


def is_cache_covered(parameter: str) -> bool:
    return (
        parameter.startswith("time_embedder.")
        or parameter.startswith("final_layer.adaln_proj.")
        or (parameter.startswith("blocks.") and ".adaln_proj." in parameter)
    )


def adapter_inventory(header: dict[str, Any], metadata: dict[str, str]) -> dict[str, Any]:
    if metadata.get("format") != SOURCE_ADAPTER_FORMAT:
        raise ValueError("FastH3 adapter does not have the finalized fastvideo-lora-v2 format")
    if metadata.get("finetuned_model") != FAST_H3_MODEL:
        raise ValueError("FastH3 adapter belongs to another student model")
    down_keys = sorted(key for key in header if key.endswith(".lora_A.weight"))
    up_keys = sorted(key for key in header if key.endswith(".lora_B.weight"))
    diff_keys = sorted(
        key for key in header if key.endswith(".diff") or key.endswith(".diff_b")
    )
    gate_keys = sorted(key for key in header if key.endswith(".set_weight"))
    if (len(down_keys), len(up_keys), len(diff_keys), len(gate_keys)) != (
        EXPECTED_LORA_PAIRS,
        EXPECTED_LORA_PAIRS,
        EXPECTED_DIFFS,
        EXPECTED_GATES,
    ):
        raise ValueError(
            "Unexpected FastH3 adapter closure: "
            f"A={len(down_keys)} B={len(up_keys)} diffs={len(diff_keys)} "
            f"gates={len(gate_keys)}"
        )

    core_pairs = 0
    cache_pairs = 0
    core_targets: set[str] = set()
    for key in down_keys:
        module = key.removesuffix(".lora_A.weight")
        if f"{module}.lora_B.weight" not in header:
            raise ValueError(f"FastH3 adapter is missing the B matrix for {module}")
        target = adapter_target(module)
        if is_cache_covered(target.module):
            cache_pairs += 1
        else:
            core_pairs += 1
            core_targets.add(target.module)
    if (core_pairs, cache_pairs, len(core_targets)) != (
        EXPECTED_CORE_PAIRS,
        EXPECTED_LORA_PAIRS - EXPECTED_CORE_PAIRS,
        EXPECTED_CORE_TARGETS,
    ):
        raise ValueError(
            "Unexpected FastH3 merge closure: "
            f"core_pairs={core_pairs} cache_pairs={cache_pairs} "
            f"core_targets={len(core_targets)}"
        )
    return {
        "lora_pair_count": len(down_keys),
        "core_lora_pair_count": core_pairs,
        "cache_covered_lora_pair_count": cache_pairs,
        "core_target_count": len(core_targets),
        "direct_difference_count": len(diff_keys),
        "compression_gate_count": len(gate_keys),
    }


def quantize_weight(value: Any) -> tuple[Any, Any, Any]:
    import mlx.core as mx

    if value.ndim != 2 or value.shape[-1] % GROUP_SIZE:
        raise ValueError(f"Cannot quantize shape {value.shape} at group size {GROUP_SIZE}")
    packed, scales, biases = mx.quantize(
        value,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode="affine",
    )
    mx.eval(packed, scales, biases)
    if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        raise ValueError("FastH3 exact Metal kernels require BF16 Q8 scales and biases")
    return packed, scales, biases


def build_merge_targets(adapter_header: dict[str, Any]) -> dict[str, list[tuple[str | None, str]]]:
    targets: dict[str, list[tuple[str | None, str]]] = {}
    for key in sorted(adapter_header):
        if not key.endswith(".lora_A.weight"):
            continue
        source = key.removesuffix(".lora_A.weight")
        target = adapter_target(source)
        if is_cache_covered(target.module):
            continue
        targets.setdefault(target.module, []).append((target.branch, source))
    if len(targets) != EXPECTED_CORE_TARGETS:
        raise ValueError(f"Expected {EXPECTED_CORE_TARGETS} core targets; found {len(targets)}")
    return targets


def merge_core_weight(
    base: Any,
    entries: list[tuple[str | None, str]],
    adapter_arrays: dict[str, Any],
) -> Any:
    import mlx.core as mx

    branch_order = {"query": 0, "key": 1, "value": 2}
    if any(branch is not None for branch, _ in entries):
        if {branch for branch, _ in entries} != set(branch_order):
            raise ValueError("Fused QKV target does not contain exactly Q, K, and V")
        if base.shape[0] % 3:
            raise ValueError(f"Fused QKV output width is not divisible by three: {base.shape}")
        rows = base.shape[0] // 3
        merged_branches = []
        for branch, source in sorted(entries, key=lambda entry: branch_order[entry[0]]):
            down = adapter_arrays[f"{source}.lora_A.weight"]
            up = adapter_arrays[f"{source}.lora_B.weight"]
            start = branch_order[branch] * rows
            merged_branches.append(
                base[start : start + rows].astype(mx.float32)
                + mx.matmul(up.astype(mx.float32), down.astype(mx.float32))
            )
        merged = mx.concatenate(merged_branches, axis=0)
    else:
        if len(entries) != 1:
            raise ValueError(f"Dense target has {len(entries)} LoRA pairs instead of one")
        source = entries[0][1]
        down = adapter_arrays[f"{source}.lora_A.weight"]
        up = adapter_arrays[f"{source}.lora_B.weight"]
        merged = base.astype(mx.float32) + mx.matmul(
            up.astype(mx.float32),
            down.astype(mx.float32),
        )
    # The exact kernels require BF16 scale/bias tables. Accumulate in FP32,
    # then make one explicit BF16 rounding before MLX affine quantization.
    return merged.astype(mx.bfloat16)


def convert_transformer(base_path: Path, adapter_path: Path, output: Path) -> dict[str, Any]:
    import mlx.core as mx

    base_header, base_metadata = safetensors_header(base_path)
    adapter_header, adapter_metadata = safetensors_header(adapter_path)
    if base_metadata.get("format") != BASE_TRANSFORMER_FORMAT:
        raise ValueError("Base transformer is not a mere.run compact MiniMax-H3 transformer")
    if base_metadata.get("precision") != "bf16" or base_metadata.get("quantization") != "none":
        raise ValueError("FastH3 Q8 merge requires the compact BF16 base transformer")
    if base_metadata.get("qkv_layout") != "global-qkv-slabs":
        raise ValueError("FastH3 QKV merge requires the global-qkv-slabs base layout")
    if len(base_header) != 428:
        raise ValueError(f"Expected 428 compact BF16 base tensors; found {len(base_header)}")
    inventory = adapter_inventory(adapter_header, adapter_metadata)
    merge_targets = build_merge_targets(adapter_header)
    differences = {
        difference_target(key): key
        for key in adapter_header
        if key.endswith(".diff") or key.endswith(".diff_b")
    }

    base_arrays = mx.load(str(base_path))
    adapter_arrays = mx.load(str(adapter_path))
    converted: dict[str, Any] = {}
    consumed_targets: set[str] = set()
    applied_differences: set[str] = set()
    quantized = copied = changed_dense = 0
    for index, key in enumerate(sorted(base_arrays), start=1):
        value = base_arrays[key]
        if key.endswith(QUANTIZED_SUFFIXES):
            module = key.removesuffix(".weight")
            entries = merge_targets.get(module)
            if entries is None:
                raise ValueError(f"FastH3 adapter has no merge target for {module}")
            merged = merge_core_weight(value, entries, adapter_arrays)
            packed, scales, biases = quantize_weight(merged)
            prefix = key.removesuffix(".weight")
            converted[key] = packed
            converted[f"{prefix}.scales"] = scales
            converted[f"{prefix}.biases"] = biases
            consumed_targets.add(module)
            quantized += 1
            del merged
        elif key in differences:
            difference = adapter_arrays[differences[key]]
            updated = (
                value.astype(mx.float32) + difference.astype(mx.float32)
            ).astype(value.dtype)
            mx.eval(updated)
            converted[key] = updated
            applied_differences.add(key)
            changed_dense += 1
        else:
            mx.eval(value)
            converted[key] = value
            copied += 1
        if index % 10 == 0 or index == len(base_arrays):
            print(
                f"transformer [{index}/{len(base_arrays)}] q8={quantized} "
                f"changed_dense={changed_dense}",
                flush=True,
            )
        mx.clear_cache()

    if consumed_targets != set(merge_targets):
        missing = sorted(set(merge_targets) - consumed_targets)
        raise ValueError(f"Base transformer did not consume FastH3 targets: {missing}")
    expected_applied_differences = {
        target for target in differences if not is_cache_covered(target)
    }
    if applied_differences != expected_applied_differences:
        missing = sorted(expected_applied_differences - applied_differences)
        raise ValueError(f"Base transformer did not consume FastH3 differences: {missing}")
    if (quantized, copied + changed_dense, len(converted)) != (208, 220, 844):
        raise ValueError(
            "Unexpected premerged transformer closure: "
            f"q8={quantized} dense={copied + changed_dense} arrays={len(converted)}"
        )

    output_metadata = dict(base_metadata)
    output_metadata.update(
        {
            "precision": "q8",
            "quantization": "affine 8-bit g64",
            "adaln_cache_source_identity": FAST_H3_SOURCE_IDENTITY,
            "fasth3_premerged": "true",
            "fasth3_adapter_sha256": sha256_file(adapter_path),
            "merge_accumulation": "fp32 then bf16 before quantization",
        }
    )
    atomic_safetensors(output, converted, output_metadata)
    del base_arrays, adapter_arrays
    converted.clear()
    mx.clear_cache()
    return {
        **inventory,
        "input_tensors": len(base_header),
        "output_tensors": 844,
        "quantized_linears": quantized,
        "copied_dense_tensors": copied,
        "updated_dense_tensors": changed_dense,
        "applied_difference_count": len(applied_differences),
        "cache_covered_difference_count": len(differences) - len(applied_differences),
        "precision": "q8",
        "quantization": {"bits": BITS, "group_size": GROUP_SIZE, "mode": "affine"},
        "merge_accumulation": "fp32 then bf16 before quantization",
    }


def convert_gates(adapter_path: Path, output: Path) -> dict[str, Any]:
    import mlx.core as mx

    header, metadata = safetensors_header(adapter_path)
    adapter_inventory(header, metadata)
    arrays = mx.load(str(adapter_path))
    converted: dict[str, Any] = {}
    for block_index in range(EXPECTED_GATES):
        source = f"transformer_blocks.{block_index}.attn.to_gate_compress.set_weight"
        if source not in arrays:
            raise ValueError(f"FastH3 adapter is missing compression gate {block_index}")
        packed, scales, biases = quantize_weight(arrays[source])
        prefix = f"transformer_blocks.{block_index}.attn.to_gate_compress"
        converted[f"{prefix}.weight"] = packed
        converted[f"{prefix}.scales"] = scales
        converted[f"{prefix}.biases"] = biases
        print(f"compression gate [{block_index + 1}/{EXPECTED_GATES}]", flush=True)
        mx.clear_cache()
    if len(converted) != 150:
        raise ValueError(f"Expected 150 quantized gate tensors; found {len(converted)}")
    atomic_safetensors(
        output,
        converted,
        {
            "format": OUTPUT_ADAPTER_FORMAT,
            "finetuned_model": FAST_H3_MODEL,
            "gate_quantization": "affine 8-bit g64",
            "source_adapter_format": SOURCE_ADAPTER_FORMAT,
            "source_adapter_sha256": sha256_file(adapter_path),
            "compression_gate_count": str(EXPECTED_GATES),
            "student_weights": "premerged into transformer.safetensors",
        },
    )
    del arrays
    converted.clear()
    mx.clear_cache()
    return {
        "input_compression_gates": EXPECTED_GATES,
        "output_tensors": 150,
        "quantization": {"bits": BITS, "group_size": GROUP_SIZE, "mode": "affine"},
    }


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare_support_files(
    base_root: Path,
    output: Path,
    adapter_path: Path,
    cache_path: Path,
) -> dict[str, str]:
    excluded = {
        "transformer.safetensors",
        "config.json",
        "mererun_model.json",
        "MODIFICATIONS.md",
        "README.md",
        "SHA256SUMS",
        "transformer.conversion.json",
    }
    methods: dict[str, str] = {}
    for source in sorted(base_root.iterdir()):
        if (
            not source.is_file()
            or source.name in excluded
            or source.name.startswith("adaln_cache")
        ):
            continue
        methods[source.name] = link_or_copy(source, output / source.name)
    methods[FAST_H3_CACHE_FILENAME] = link_or_copy(
        cache_path,
        output / FAST_H3_CACHE_FILENAME,
    )
    if adapter_path.name != ADAPTER_FILENAME:
        print(
            f"warning: source adapter is named {adapter_path.name}; output uses {ADAPTER_FILENAME}",
            flush=True,
        )
    return methods


def write_package_metadata(base_root: Path, output: Path) -> None:
    config = json.loads((base_root / "config.json").read_text(encoding="utf-8"))
    config["quantization"] = {"bits": BITS, "group_size": GROUP_SIZE, "mode": "affine"}
    atomic_json(output / "config.json", config)

    model = json.loads((base_root / "mererun_model.json").read_text(encoding="utf-8"))
    model.update(
        {
            "id": "video-minimax-h3-fasth3-vsa-datafree-mlx",
            "precision": "int8",
            "variant": "distilled",
            "defaults": {"cfg": 1, "sigma_shift": 12, "steps": 5},
            "upstreamRepoId": "Sawfwair/MiniMax-H3-FastH3-VSA-DataFree-MLX-Q8",
            "sources": [
                {
                    "repository": "Sawfwair/MiniMax-H3-FastH3-VSA-DataFree-MLX-Q8",
                    "revision": "local-unpublished",
                    "role": "primary",
                }
            ],
        }
    )
    atomic_json(output / "mererun_model.json", model)

    atomic_text(
        output / "README.md",
        """---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/ec19cc6daf5d8add9417c18e86b6b58cc6c55027/LICENSE
library_name: mlx
pipeline_tag: text-to-video
tags:
  - audio-video
  - apple-silicon
  - fastvideo
  - metal
  - mlx
  - minimax-h3
---

# MiniMax-H3 FastH3 VSA DataFree MLX Q8

This is a self-contained Apple Silicon inference package for FastVideo's
four-evaluation FastH3 VSA DataFree MiniMax-H3 student. It includes the Q8
transformer, Q8 text encoder, both VAEs, tokenizer, VSA compression gates,
and all AdaLN tables required by mere.run. Inference does not download another
base model, adapter, text encoder, or cache.

## License notice

These are modified MiniMax-H3 weights governed by the included MiniMax-H3
Community License Agreement (`LICENSE`) and notice (`NOTICE`). The license
excludes use, distribution, and display in the United States, European Union,
United Kingdom, and Republic of Korea and imposes downstream notice and safety
obligations. Downloading this ungated repository does not waive those terms.

## Conversion

The finalized FastH3 low-rank updates and direct differences were merged into
the compact BF16 student with FP32 accumulation. The 208 inference core linear
weights were rounded once to BF16 and encoded as MLX affine Q8/group-64. The 50
VSA compression gates use the same encoding. Cache-covered AdaLN updates remain
in the source-bound inference table. See `MODIFICATIONS.md`,
`FASTH3_SOURCE_MANIFEST.json`, `FASTH3_CONVERSION.json`, and `SHA256SUMS` for
the complete provenance and integrity receipts.

## Run with mere.run

Use a mere.run build that includes premerged FastH3 Q8 support:

```bash
mere.run video generate \
  "A lighthouse in a winter storm with synchronized wind and surf." \
  --model-root /path/to/this/package \
  --h3-adapter /path/to/this/package/fastvideo_fasth3_4step_v1_vsa_datafree_rank64.safetensors \
  --width 512 --height 320 --num-frames 22 \
  --output lighthouse.mp4
```

The public managed model ID is
`video-minimax-h3-fasth3-vsa-datafree-mlx`. It selects the released schedule,
unit adapter strength, compiled block execution, 64-token VSA tiles, per-head
top-k routing at 90% video-key sparsity, and learned pooled-value compression.
""",
    )

    atomic_text(
        output / "MODIFICATIONS.md",
        """# Modifications to MiniMax H3

These files are MODIFIED versions of the MiniMax H3 Works, redistributed under
the MiniMax H3 Community License Agreement in `LICENSE` and `NOTICE`.

Modified by: Sawfwair (https://github.com/sawfwair/mere-run)

* The finalized FastH3 VSA DataFree low-rank updates and direct differences
  were merged into the compact MiniMax-H3 student with FP32 accumulation.
  Cache-covered timestep and AdaLN updates are represented by the included
  source-bound FastH3 AdaLN inference table.
* The 208 merged transformer and token-refiner core linear weights were rounded
  once to BF16 and quantized to MLX affine 8-bit, group size 64. Dense input,
  normalization, and output precision islands retain their source dtype.
* The 50 VSA compression-gate matrices were quantized to MLX affine 8-bit,
  group size 64 and packaged in the FastH3 adapter filename. That file contains
  no remaining LoRA matrices; the student weights are already merged.
* The Q8 text encoder, video VAE, audio VAE, tokenizer, license, notice, and
  production cache pack are unchanged from the audited base package.

`FASTH3_SOURCE_MANIFEST.json` and `FASTH3_CONVERSION.json` record the exact
inputs, numeric transformations, and output hashes. No additional model
downloads are required at inference time.
""",
    )


def base_sha256sums(base_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (base_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        result[filename] = digest
    return result


def write_sha256sums(output: Path, inherited: dict[str, str]) -> dict[str, str]:
    excluded = {"SHA256SUMS", "README.md"}
    hashes: dict[str, str] = {}
    for path in sorted(value for value in output.iterdir() if value.is_file()):
        if path.name in excluded or path.name.startswith("."):
            continue
        hashes[path.name] = inherited.get(path.name) or sha256_file(path)
    atomic_text(
        output / "SHA256SUMS",
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
    )
    return hashes


def write_transformer_receipt(
    output: Path,
    base_transformer: Path,
    converter_sha256: str,
    completed_at: str,
) -> None:
    transformer = output / "transformer.safetensors"
    atomic_json(
        output / "transformer.conversion.json",
        {
            "schema_version": 1,
            "format": "mere.run.minimax-h3-fasth3-q8-transformer-receipt",
            "converter": "scripts/model-conversion/merge_minimax_h3_fasth3_q8.py",
            "converter_version": 2,
            "converter_sha256": converter_sha256,
            "completed_at": completed_at,
            "input": {
                "filename": base_transformer.name,
                "byte_count": base_transformer.stat().st_size,
                "sha256": sha256_file(base_transformer),
            },
            "output": {
                "filename": transformer.name,
                "byte_count": transformer.stat().st_size,
                "sha256": sha256_file(transformer),
                "precision": "q8",
                "quantization": {
                    "bits": BITS,
                    "group_size": GROUP_SIZE,
                    "mode": "affine",
                },
                "tensor_count": 844,
            },
            "full_receipt": "FASTH3_CONVERSION.json",
        },
    )


def hardware_receipt() -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    if sys.platform == "darwin":
        import subprocess

        for key, command in (
            ("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]),
            ("unified_memory_bytes", ["sysctl", "-n", "hw.memsize"]),
        ):
            try:
                receipt[key] = subprocess.check_output(command, text=True).strip()
            except Exception as error:
                receipt[f"{key}_error"] = str(error)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--fast-h3-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conversion-location", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_root = args.base_root.expanduser().resolve()
    adapter_path = args.adapter.expanduser().resolve()
    cache_path = args.fast_h3_cache.expanduser().resolve()
    output = args.output.expanduser().resolve()
    required = [
        base_root / "transformer.safetensors",
        base_root / "config.json",
        base_root / "mererun_model.json",
        base_root / "SHA256SUMS",
        adapter_path,
        cache_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Missing conversion inputs: {missing}")
    adapter_header, adapter_metadata = safetensors_header(adapter_path)
    inventory = adapter_inventory(adapter_header, adapter_metadata)
    if args.validate_only:
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    base_transformer = base_root / "transformer.safetensors"
    base_source_manifest = json.loads(
        (base_root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    base_transformer_sha = sha256_file(base_transformer)
    adapter_sha = sha256_file(adapter_path)
    cache_sha = sha256_file(cache_path)

    methods = prepare_support_files(
        base_root,
        output,
        adapter_path,
        cache_path,
    )
    write_package_metadata(base_root, output)
    transformer_stats = convert_transformer(
        base_transformer,
        adapter_path,
        output / "transformer.safetensors",
    )
    gate_stats = convert_gates(adapter_path, output / ADAPTER_FILENAME)

    source_manifest = {
        "schema_version": 1,
        "artifact": {
            "format": "mere.run.minimax-h3-fasth3-vsa-datafree-mlx-q8-premerged",
            "repository": "Sawfwair/MiniMax-H3-FastH3-VSA-DataFree-MLX-Q8",
            "partition": "fl2va",
            "task": "text-to-video-with-audio",
        },
        "base": {
            "repository": base_source_manifest.get("repository"),
            "revision": base_source_manifest.get("revision"),
            "transformer_sha256": base_transformer_sha,
            "source_manifest_sha256": sha256_file(base_root / "SOURCE_MANIFEST.json"),
        },
        "adapter": {
            "filename": adapter_path.name,
            "sha256": adapter_sha,
            **inventory,
        },
        "adaln_cache": {
            "path": FAST_H3_CACHE_FILENAME,
            "source_identity": FAST_H3_SOURCE_IDENTITY,
            "sha256": cache_sha,
        },
    }
    atomic_json(output / "FASTH3_SOURCE_MANIFEST.json", source_manifest)

    receipt = {
        "schema_version": 1,
        "converter": "scripts/model-conversion/merge_minimax_h3_fasth3_q8.py",
        "converter_version": 2,
        "converter_sha256": sha256_file(Path(__file__).resolve()),
        "started_at": started_at,
        "completed_at": utc_now(),
        "declared_conversion_location": args.conversion_location,
        "software": {
            "mlx": importlib_metadata.version("mlx"),
            "numpy": importlib_metadata.version("numpy"),
            "safetensors": importlib_metadata.version("safetensors"),
        },
        "hardware": hardware_receipt(),
        "inputs": {
            "base_transformer": {
                "filename": base_transformer.name,
                "byte_count": base_transformer.stat().st_size,
                "sha256": base_transformer_sha,
            },
            "adapter": {
                "filename": adapter_path.name,
                "byte_count": adapter_path.stat().st_size,
                "sha256": adapter_sha,
            },
            "fast_h3_adaln_cache": {
                "filename": cache_path.name,
                "byte_count": cache_path.stat().st_size,
                "sha256": cache_sha,
            },
        },
        "transformer": transformer_stats,
        "compression_gates": gate_stats,
        "support_file_materialization": methods,
        "outputs": {
            "transformer.safetensors": {
                "byte_count": (output / "transformer.safetensors").stat().st_size,
                "sha256": sha256_file(output / "transformer.safetensors"),
            },
            ADAPTER_FILENAME: {
                "byte_count": (output / ADAPTER_FILENAME).stat().st_size,
                "sha256": sha256_file(output / ADAPTER_FILENAME),
            },
        },
    }
    atomic_json(output / "FASTH3_CONVERSION.json", receipt)
    write_transformer_receipt(
        output,
        base_transformer,
        receipt["converter_sha256"],
        receipt["completed_at"],
    )
    inherited_hashes = {
        name: digest
        for name, digest in base_sha256sums(base_root).items()
        if methods.get(name) == "hardlink"
    }
    hashes = write_sha256sums(output, inherited_hashes)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    print(f"complete: {output} ({len(hashes)} checksummed files)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build mere.run's MiniMax-H3 MLX bundle directly from MiniMax's release.

This is audited release tooling, never a runtime sidecar. Every weight input is
downloaded from the exact pinned ``MiniMaxAI/MiniMax-H3`` revision below. No
third-party converted or quantized checkpoint is accepted as an input.

The converter is intentionally usable with MLX's CUDA backend on Linux. That
lets a release operator build the large artifact on an ephemeral GPU host while
preserving the exact MLX affine quantizer used by Apple Silicon at inference.

Remote prerequisites (the official RunPod PyTorch image already supplies
Python and CUDA)::

    python -m pip install 'mlx[cuda]==0.29.3' 'huggingface_hub[hf_xet]==0.34.4' \
        'safetensors==0.6.2' 'numpy==2.2.6'

Example::

    HF_HOME=/workspace/hf-cache HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
      python convert_minimax_h3_official_mlx.py \
        --cache-dir /workspace/hf-cache \
        --conversion-location "CA-MTL-3, Canada" \
        --transformer-precision bf16 \
        --output /workspace/minimax-h3-sawfwair

The output transformer is mixed precision: the 208 transformer/refiner core
linears use the selected affine Q4 or Q8/group-64 width, while the released
BF16/F32 precision islands remain dense. The Qwen3-VL conditioner retains
exactly the 50 language layers H3 reads and uses affine Q8/group-64 for 439
eligible linears. The AdaLN cache is evaluated from the original BF16/F32
projections before those 13B inference-redundant parameters are omitted from
the compact transformer.
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
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SOURCE_REPOSITORY = "MiniMaxAI/MiniMax-H3"
SOURCE_REVISION = "ec19cc6daf5d8add9417c18e86b6b58cc6c55027"
SOURCE_LICENSE_SHA256 = "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44"
SOURCE_LICENSE_BYTES = 17_604
GROUP_SIZE = 64
TEXT_ENCODER_BITS = 8
SAMPLE_POINTS = 31
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
TRANSFORMER_HEAD_COUNT = 56
TRANSFORMER_HEAD_DIMENSION = 128

TRANSFORMER_INDEX = "FL2VA/transformer/model.safetensors.index.json"
TEXT_ENCODER_INDEX = "FL2VA/text_encoder/model.safetensors.index.json"
VIDEO_VAE_WEIGHTS = "FL2VA/video_vae/source/model.safetensors"
AUDIO_VAE_WEIGHTS = "FL2VA/audio_vae/model.safetensors"

COMMON_SOURCE_FILES = {
    "LICENSE",
    "README.md",
    "FL2VA/processor/tokenizer.json",
    "FL2VA/processor/tokenizer_config.json",
}

TRANSFORMER_DENSE_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "condition_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
    "final_layer.adaln_proj.",
)

TRANSFORMER_QUANTIZED_SUFFIXES = (
    ".attn.qkv_proj.weight",
    ".attn.out_proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
)

QUANTIZER_SELF_TEST = {
    4: {
        "weight": "74bebb64dcbfbd2d77e121a4de4f154ce21a5a35f481f699b2df23b135656ec8",
        "scales": "22f7ad21bdcfec89ace95b550404bd3433de39214b81cdc38c775724125499dc",
        "biases": "09ebd55dbdba503aa1b594d6b6a5333be87bb9dd51caaaf4063903e08780d2a5",
    },
    8: {
        "weight": "e7eabb58287772294ce23f376e9f65877545152b2613d172e6d33ec6bcb244c4",
        "scales": "d84bf16b039914838f39df1823f37d9bcd640ea1108a33a2b0e7e4aa42e7ed1c",
        "biases": "09ebd55dbdba503aa1b594d6b6a5333be87bb9dd51caaaf4063903e08780d2a5",
    },
}
QKV_REORDER_SELF_TEST_SHA256 = "11a9143204ee0defe33828d431cca9f051c2a221050ae8543b543cda5c7b7786"


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


def fetch_hub_metadata() -> dict[str, Any]:
    encoded_repo = urllib.parse.quote(SOURCE_REPOSITORY, safe="/")
    url = (
        f"https://huggingface.co/api/models/{encoded_repo}/revision/"
        f"{SOURCE_REVISION}?blobs=true"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        metadata = json.load(response)
    if metadata.get("sha") != SOURCE_REVISION:
        raise ValueError(
            f"Hub resolved {SOURCE_REPOSITORY}@{SOURCE_REVISION} to {metadata.get('sha')}"
        )
    if metadata.get("private") or metadata.get("gated"):
        raise ValueError("The pinned official source unexpectedly became private or gated")
    return metadata


def component_source_files(metadata: dict[str, Any], components: set[str]) -> list[str]:
    available = {entry["rfilename"] for entry in metadata["siblings"]}
    selected = set(COMMON_SOURCE_FILES)
    if "transformer" in components:
        selected.update(name for name in available if name.startswith("FL2VA/transformer/"))
    if "text_encoder" in components:
        selected.update(name for name in available if name.startswith("FL2VA/text_encoder/"))
    if "video_vae" in components:
        selected.update(
            {
                "FL2VA/video_vae/config.json",
                "FL2VA/video_vae/source/config.json",
                VIDEO_VAE_WEIGHTS,
            }
        )
    if "audio_vae" in components:
        selected.update(
            {
                "FL2VA/audio_vae/config.json",
                "FL2VA/audio_vae/metadata.json",
                AUDIO_VAE_WEIGHTS,
            }
        )
    missing = sorted(selected - available)
    if missing:
        raise ValueError(f"Pinned source is missing required files: {missing}")
    return sorted(selected)


def metadata_by_name(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["rfilename"]: entry for entry in metadata["siblings"]}


def print_plan(metadata: dict[str, Any], source_files: list[str]) -> None:
    entries = metadata_by_name(metadata)
    total = sum(int(entries[name].get("size") or 0) for name in source_files)
    print(f"source: {SOURCE_REPOSITORY}@{SOURCE_REVISION}")
    print(f"files: {len(source_files)}")
    print(f"download bytes: {total} ({total / 1e9:.2f} GB)")
    for name in source_files:
        print(f"  {entries[name].get('size', 0):>12}  {name}")


def download_sources(
    metadata: dict[str, Any],
    source_files: list[str],
    cache_dir: Path,
    source_manifest_path: Path,
) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    entries = metadata_by_name(metadata)
    downloaded: dict[str, Path] = {}
    receipt_files: list[dict[str, Any]] = []
    for index, name in enumerate(source_files, start=1):
        print(f"source [{index}/{len(source_files)}] {name}", flush=True)
        local = Path(
            hf_hub_download(
                repo_id=SOURCE_REPOSITORY,
                filename=name,
                revision=SOURCE_REVISION,
                cache_dir=str(cache_dir),
            )
        )
        expected_size = int(entries[name].get("size") or 0)
        actual_size = local.stat().st_size
        if expected_size and actual_size != expected_size:
            raise ValueError(f"{name} has {actual_size} bytes; expected {expected_size}")
        digest = sha256_file(local)
        lfs = entries[name].get("lfs") or {}
        expected_digest = lfs.get("sha256")
        if expected_digest and digest != expected_digest:
            raise ValueError(f"{name} has SHA-256 {digest}; expected {expected_digest}")
        if name == "LICENSE":
            if actual_size != SOURCE_LICENSE_BYTES or digest != SOURCE_LICENSE_SHA256:
                raise ValueError("Pinned MiniMax H3 license bytes changed")
        downloaded[name] = local
        receipt_files.append(
            {
                "path": name,
                "byte_count": actual_size,
                "sha256": digest,
                "hub_lfs_sha256": expected_digest,
            }
        )

    atomic_json(
        source_manifest_path,
        {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "public": True,
            "gated": False,
            "files": receipt_files,
        },
    )
    return downloaded


def load_index(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("weight_map"), dict):
        raise ValueError(f"Invalid safetensors index: {path}")
    return value


def source_shards(
    prefix: str,
    index: dict[str, Any],
    downloaded: dict[str, Path],
) -> list[tuple[str, Path, list[str]]]:
    weight_map: dict[str, str] = index["weight_map"]
    result = []
    for shard in sorted(set(weight_map.values())):
        keys = sorted(key for key, value in weight_map.items() if value == shard)
        name = f"{prefix}/{shard}"
        result.append((name, downloaded[name], keys))
    return result


def release_arrays(arrays: dict[str, Any]) -> None:
    arrays.clear()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def atomic_safetensors(path: Path, arrays: dict[str, Any], metadata: dict[str, str]) -> None:
    import mlx.core as mx

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.safetensors")
    try:
        mx.save_safetensors(str(temporary), arrays, metadata=metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def bf16_bytes(array: Any) -> bytes:
    import numpy as np

    values = np.asarray(array.tolist(), dtype=np.float32)
    words = values.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return ((words + rounding) >> np.uint32(16)).astype("<u2").tobytes()


def quantizer_self_test() -> dict[str, dict[str, str]]:
    import mlx.core as mx
    import numpy as np

    indices = np.arange(17 * 128, dtype=np.int32)
    values = (
        ((indices % 257) - 128).astype(np.float32) / np.float32(32)
        + ((indices * 17) % 31).astype(np.float32) / np.float32(1024)
    ).reshape(17, 128)
    weight = mx.array(values).astype(mx.bfloat16)
    observed: dict[str, dict[str, str]] = {}
    for bits in sorted(QUANTIZER_SELF_TEST):
        packed, scales, biases = mx.quantize(
            weight,
            group_size=GROUP_SIZE,
            bits=bits,
            mode="affine",
        )
        mx.eval(packed, scales, biases)
        packed_bytes = np.asarray(packed.tolist(), dtype="<u4").tobytes()
        values_by_name = {
            "weight": hashlib.sha256(packed_bytes).hexdigest(),
            "scales": hashlib.sha256(bf16_bytes(scales)).hexdigest(),
            "biases": hashlib.sha256(bf16_bytes(biases)).hexdigest(),
        }
        if values_by_name != QUANTIZER_SELF_TEST[bits]:
            raise ValueError(
                f"MLX affine {bits}-bit quantizer differs from the Apple Silicon release contract: "
                f"{values_by_name}"
            )
        observed[str(bits)] = values_by_name
    print("MLX affine Q4/Q8 byte-level self-test passed", flush=True)
    return observed


def quantize_weight(value: Any, bits: int) -> tuple[Any, Any, Any]:
    import mlx.core as mx

    if value.ndim != 2 or value.shape[-1] % GROUP_SIZE:
        raise ValueError(f"Cannot quantize shape {value.shape} at group size {GROUP_SIZE}")
    packed, scales, biases = mx.quantize(
        value,
        group_size=GROUP_SIZE,
        bits=bits,
        mode="affine",
    )
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def reorder_transformer_qkv(value: Any) -> Any:
    """Convert raw per-head QKV rows to the global slabs expected by the runtime.

    MiniMax's checkpoint stores
    ``[head0:q,k,v; head1:q,k,v; ...]``.  The native runtime splits a fused
    projection into ``[all-q; all-k; all-v]`` before reshaping by head, matching
    the official reference model's load-time reorder.  Quantization must happen
    after this permutation so each packed row retains the correct identity.
    """
    import mlx.core as mx

    expected_rows = TRANSFORMER_HEAD_COUNT * 3 * TRANSFORMER_HEAD_DIMENSION
    if value.ndim != 2 or value.shape[0] != expected_rows:
        raise ValueError(
            "Unexpected MiniMax-H3 fused QKV shape: "
            f"{value.shape}; expected [{expected_rows}, hidden_size]"
        )
    trailing_shape = list(value.shape[1:])
    grouped = value.reshape(
        [TRANSFORMER_HEAD_COUNT, 3, TRANSFORMER_HEAD_DIMENSION] + trailing_shape
    )
    query, key, output = (
        part.reshape(
            [TRANSFORMER_HEAD_COUNT * TRANSFORMER_HEAD_DIMENSION] + trailing_shape
        )
        for part in mx.split(grouped, 3, axis=1)
    )
    reordered = mx.concatenate([query, key, output], axis=0)
    mx.eval(reordered)
    return reordered


def qkv_reorder_self_test() -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np

    rows = TRANSFORMER_HEAD_COUNT * 3 * TRANSFORMER_HEAD_DIMENSION
    raw = mx.array(np.arange(rows, dtype=np.int32).reshape(rows, 1))
    reordered = reorder_transformer_qkv(raw)
    observed = np.asarray(reordered.tolist(), dtype="<i4").reshape(-1)
    digest = hashlib.sha256(observed.tobytes()).hexdigest()
    if digest != QKV_REORDER_SELF_TEST_SHA256:
        raise ValueError(f"MiniMax-H3 QKV reorder self-test differs: {digest}")
    receipt = {
        "head_count": TRANSFORMER_HEAD_COUNT,
        "head_dimension": TRANSFORMER_HEAD_DIMENSION,
        "layout": "global-qkv-slabs",
        "sha256": digest,
    }
    print("MiniMax-H3 QKV deinterleave self-test passed", flush=True)
    return receipt


def schedule(point_count: int, shift: float) -> tuple[list[float], list[float]]:
    import numpy as np

    shifted: list[np.float32] = []
    shift32 = np.float32(shift)
    one = np.float32(1)
    for index in range(point_count):
        base = np.float32(point_count - 1 - index) / np.float32(point_count - 1)
        sigma = shift32 * base / (one + (shift32 - one) * base)
        if not shifted or shifted[-1] != sigma:
            shifted.append(sigma)
    sigmas = [float(value) for value in shifted]
    timesteps = [float(np.float32(one - value)) for value in shifted[:-1]]
    return sigmas, timesteps


def load_requested_keys(
    prefix: str,
    index: dict[str, Any],
    downloaded: dict[str, Path],
    requested: Iterable[str],
) -> dict[str, Any]:
    import mlx.core as mx

    weight_map: dict[str, str] = index["weight_map"]
    wanted = set(requested)
    result: dict[str, Any] = {}
    by_shard: dict[str, list[str]] = {}
    for key in wanted:
        shard = weight_map.get(key)
        if shard is None:
            raise ValueError(f"Official transformer is missing {key}")
        by_shard.setdefault(shard, []).append(key)
    for shard, keys in sorted(by_shard.items()):
        raw = mx.load(str(downloaded[f"{prefix}/{shard}"]))
        for key in keys:
            value = raw[key]
            mx.eval(value)
            result[key] = value
        del raw
    return result


def linear(value: Any, weight: Any, bias: Any) -> Any:
    import mlx.core as mx

    return mx.matmul(value.astype(weight.dtype), weight.T) + bias


def build_adaln_cache(
    output: Path,
    transformer_index: dict[str, Any],
    downloaded: dict[str, Path],
    source_identity: str,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    video_sigmas, video_timesteps = schedule(SAMPLE_POINTS, VIDEO_SHIFT)
    audio_sigmas, audio_timesteps = schedule(SAMPLE_POINTS, AUDIO_SHIFT)
    if len(video_timesteps) != 30 or len(audio_timesteps) != 30:
        raise ValueError("Unexpected MiniMax-H3 sampler schedule geometry")

    flat_timesteps: list[np.float32] = []
    condition = np.float32(0.999)
    for video, audio in zip(video_timesteps, audio_timesteps, strict=True):
        video32 = np.float32(video)
        audio32 = np.float32(audio)
        flat_timesteps.extend((video32, audio32, max(video32, condition)))

    time_keys = (
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
    )
    time_weights = load_requested_keys(
        "FL2VA/transformer", transformer_index, downloaded, time_keys
    )
    half_dimension = 128
    frequency_indices = np.arange(half_dimension, dtype=np.float32)
    frequencies = np.exp(
        -np.log(np.float32(10_000)) * frequency_indices / np.float32(half_dimension)
    ).astype(np.float32)
    arguments = mx.array(np.asarray(flat_timesteps, dtype=np.float32)).reshape(-1, 1) \
        * mx.array(frequencies).reshape(1, -1)
    sinusoidal = mx.concatenate([mx.cos(arguments), mx.sin(arguments)], axis=-1)
    time_embeddings = linear(
        nn.silu(linear(
            sinusoidal.astype(mx.float32),
            time_weights["time_embedder.proj_in.weight"],
            time_weights["time_embedder.proj_in.bias"],
        )),
        time_weights["time_embedder.proj_out.weight"],
        time_weights["time_embedder.proj_out.bias"],
    ).reshape(30, 3, 2_688)
    mx.eval(time_embeddings)
    release_arrays(time_weights)

    activated = nn.silu(time_embeddings.reshape(90, 2_688)).astype(mx.bfloat16)
    mx.eval(activated)
    block_modulations: list[Any | None] = [None] * 50
    weight_map: dict[str, str] = transformer_index["weight_map"]
    blocks_by_shard: dict[str, list[int]] = {}
    for block in range(50):
        key = f"blocks.{block}.adaln_proj.linear.weight"
        blocks_by_shard.setdefault(weight_map[key], []).append(block)

    for shard, blocks in sorted(blocks_by_shard.items()):
        raw = mx.load(str(downloaded[f"FL2VA/transformer/{shard}"]))
        for block in blocks:
            prefix = f"blocks.{block}.adaln_proj.linear"
            projected = linear(activated, raw[f"{prefix}.weight"], raw[f"{prefix}.bias"])
            modulation = projected.reshape(30, 9, 32_256).astype(mx.bfloat16)
            mx.eval(modulation)
            block_modulations[block] = modulation
            print(f"AdaLN cache block {block + 1}/50", flush=True)
        del raw
        mx.clear_cache()
    if any(value is None for value in block_modulations):
        raise ValueError("Not every AdaLN block was cached")

    final_keys = (
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
    )
    final_weights = load_requested_keys(
        "FL2VA/transformer", transformer_index, downloaded, final_keys
    )
    final_modulations = linear(
        activated,
        final_weights["final_layer.adaln_proj.linear.weight"],
        final_weights["final_layer.adaln_proj.linear.bias"],
    ).reshape(30, 3, 10_752).astype(mx.bfloat16)
    mx.eval(final_modulations)
    release_arrays(final_weights)

    arrays: dict[str, Any] = {
        "audio_sigmas": mx.array(audio_sigmas, dtype=mx.float32),
        "final_modulations": final_modulations,
        "time_embeddings": time_embeddings,
        "video_sigmas": mx.array(video_sigmas, dtype=mx.float32),
    }
    for block, modulation in enumerate(block_modulations):
        arrays[f"blocks.{block}.modulations"] = modulation
    atomic_safetensors(
        output,
        dict(sorted(arrays.items())),
        {
            "schema_version": "2",
            "format": "mere.run.minimax-h3-adaln-cache",
            "source_identity": source_identity,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_precision": "released mixed BF16/F32",
        },
    )
    stats = {
        "tensor_count": len(arrays),
        "sample_points": SAMPLE_POINTS,
        "steps": 30,
        "video_shift": VIDEO_SHIFT,
        "audio_shift": AUDIO_SHIFT,
        "source_identity": source_identity,
    }
    release_arrays(arrays)
    del activated, time_embeddings, final_modulations, block_modulations
    mx.clear_cache()
    return stats


def is_cache_covered_transformer_key(key: str) -> bool:
    return ".adaln_proj." in key or key.startswith("time_embedder.")


def is_quantized_transformer_key(key: str) -> bool:
    if key.startswith(TRANSFORMER_DENSE_PREFIXES):
        return False
    return key.endswith(TRANSFORMER_QUANTIZED_SUFFIXES)


def convert_transformer(
    output: Path,
    index: dict[str, Any],
    downloaded: dict[str, Path],
    source_identity: str,
    transformer_precision: str,
) -> dict[str, Any]:
    import mlx.core as mx

    transformer_bits = {"q4": 4, "q8": 8, "bf16": None}[transformer_precision]
    converted: dict[str, Any] = {}
    quantized = dense_core = copied = omitted = qkv_reordered = 0
    for shard_index, (name, shard_path, keys) in enumerate(
        source_shards("FL2VA/transformer", index, downloaded), start=1
    ):
        raw = mx.load(str(shard_path))
        print(f"transformer shard {shard_index}/13: {name}", flush=True)
        for key in keys:
            if key == "rope.inv_freq" or is_cache_covered_transformer_key(key):
                omitted += 1
                continue
            value = raw[key]
            if is_quantized_transformer_key(key):
                if key.endswith(".attn.qkv_proj.weight"):
                    value = reorder_transformer_qkv(value)
                    qkv_reordered += 1
                if transformer_bits is None:
                    mx.eval(value)
                    converted[key] = value
                    dense_core += 1
                else:
                    packed, scales, biases = quantize_weight(value, transformer_bits)
                    prefix = key.removesuffix(".weight")
                    converted[key] = packed
                    converted[f"{prefix}.scales"] = scales
                    converted[f"{prefix}.biases"] = biases
                    quantized += 1
            else:
                mx.eval(value)
                converted[key] = value
                copied += 1
        del raw
        mx.clear_cache()

    expected = (0, 208, 220, 107, 52, 428) if transformer_bits is None else (
        208, 0, 220, 107, 52, 844
    )
    if (quantized, dense_core, copied, omitted, qkv_reordered, len(converted)) != expected:
        raise ValueError(
            "Unexpected compact transformer geometry: "
            f"quantized={quantized}, dense_core={dense_core}, copied={copied}, omitted={omitted}, "
            f"qkv_reordered={qkv_reordered}, arrays={len(converted)}"
        )
    precision_metadata = (
        "released mixed BF16/F32"
        if transformer_bits is None
        else f"affine {transformer_bits}-bit g64"
    )
    atomic_safetensors(
        output,
        dict(sorted(converted.items())),
        {
            "format": "mere.run.minimax-h3-inference-transformer",
            "precision": transformer_precision,
            "quantization": "none" if transformer_bits is None else precision_metadata,
            "source_precision": "released mixed BF16/F32",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "cache_covered_weights_omitted": "true",
            "adaln_cache_source_identity": source_identity,
            "qkv_layout": "global-qkv-slabs",
        },
    )
    stats = {
        "input_tensors": len(index["weight_map"]),
        "output_tensors": len(converted),
        "quantized_linears": quantized,
        "dense_core_linears": dense_core,
        "copied_dense_tensors": copied,
        "omitted_cache_or_rope_tensors": omitted,
        "qkv_matrices_deinterleaved": qkv_reordered,
        "qkv_layout": "global-qkv-slabs",
        "precision": transformer_precision,
        "quantization": None if transformer_bits is None else {
            "bits": transformer_bits,
            "group_size": 64,
            "mode": "affine",
        },
        "dense_precision_islands_preserved": list(TRANSFORMER_DENSE_PREFIXES),
    }
    release_arrays(converted)
    return stats


def text_encoder_key(source_key: str) -> str | None:
    if source_key == "lm_head.weight" or source_key == "model.language_model.norm.weight":
        return None
    if source_key.startswith("model.language_model.layers."):
        layer = int(source_key.split(".")[3])
        if layer >= 50:
            return None
    if source_key.startswith("model.language_model."):
        return "model." + source_key.removeprefix("model.language_model.")
    if source_key.startswith("model.visual."):
        return "visual." + source_key.removeprefix("model.visual.")
    raise ValueError(f"Unrecognized official text-encoder tensor: {source_key}")


def is_text_encoder_lookup(key: str) -> bool:
    return key in {"model.embed_tokens.weight", "visual.pos_embed.weight"}


def convert_text_encoder(
    output: Path,
    index: dict[str, Any],
    downloaded: dict[str, Path],
) -> dict[str, Any]:
    import mlx.core as mx

    converted: dict[str, Any] = {}
    quantized = copied = skipped = 0
    shards = source_shards("FL2VA/text_encoder", index, downloaded)
    for shard_index, (name, shard_path, keys) in enumerate(shards, start=1):
        raw = mx.load(str(shard_path))
        print(f"text encoder shard {shard_index}/14: {name}", flush=True)
        for source_key in keys:
            key = text_encoder_key(source_key)
            if key is None:
                skipped += 1
                continue
            value = raw[source_key]
            eligible = (
                key.endswith(".weight")
                and value.ndim == 2
                and value.shape[-1] % GROUP_SIZE == 0
                and not is_text_encoder_lookup(key)
            )
            if eligible:
                packed, scales, biases = quantize_weight(value, TEXT_ENCODER_BITS)
                prefix = key.removesuffix(".weight")
                converted[key] = packed
                converted[f"{prefix}.scales"] = scales
                converted[f"{prefix}.biases"] = biases
                quantized += 1
            else:
                mx.eval(value)
                converted[key] = value
                copied += 1
        del raw
        mx.clear_cache()

    if (quantized, copied, skipped, len(converted)) != (439, 463, 156, 1_780):
        raise ValueError(
            "Unexpected conditioner geometry: "
            f"quantized={quantized}, copied={copied}, skipped={skipped}, arrays={len(converted)}"
        )
    atomic_safetensors(
        output,
        dict(sorted(converted.items())),
        {
            "format": "mere.run.minimax-h3-qwen3-vl-conditioner",
            "quantization": "affine 8-bit g64",
            "source_precision": "released BF16",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "language_layers_retained": "50",
        },
    )
    stats = {
        "input_tensors": len(index["weight_map"]),
        "output_tensors": len(converted),
        "quantized_linears": quantized,
        "copied_dense_tensors": copied,
        "skipped_unused_tensors": skipped,
        "language_layers_retained": 50,
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
    }
    release_arrays(converted)
    return stats


def convert_video_vae(
    output: Path,
    downloaded: dict[str, Path],
) -> dict[str, Any]:
    import mlx.core as mx

    wrapper = json.loads(downloaded["FL2VA/video_vae/config.json"].read_text())
    raw = mx.load(str(downloaded[VIDEO_VAE_WEIGHTS]))
    converted: dict[str, Any] = {}
    for index, key in enumerate(sorted(raw), start=1):
        value = raw[key].astype(mx.float16)
        mx.eval(value)
        converted[key] = value
        if index % 50 == 0:
            print(f"video VAE tensor {index}/{len(raw)}", flush=True)
    converted["latents_mean"] = mx.array(wrapper["latents_mean"], dtype=mx.float16)
    converted["latents_std"] = mx.array(wrapper["latents_std"], dtype=mx.float16)
    del raw
    if len(converted) != 562:
        raise ValueError(f"Unexpected video VAE tensor count: {len(converted)}")
    atomic_safetensors(
        output,
        dict(sorted(converted.items())),
        {
            "format": "mere.run.minimax-h3-video-vae",
            "conversion": "official FP32 tensors cast directly to FP16; latent statistics added",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
    )
    stats = {
        "input_tensors": 560,
        "output_tensors": len(converted),
        "source_precision": "FP32",
        "output_precision": "FP16",
        "layout_transform": "none in artifact; native loader maps convolution/QKV layouts",
    }
    release_arrays(converted)
    return stats


def convert_audio_vae(
    output: Path,
    downloaded: dict[str, Path],
) -> dict[str, Any]:
    import mlx.core as mx

    wrapper = json.loads(downloaded["FL2VA/audio_vae/config.json"].read_text())
    raw = mx.load(str(downloaded[AUDIO_VAE_WEIGHTS]))
    converted: dict[str, Any] = {}
    folded = 0
    for index, key in enumerate(sorted(raw), start=1):
        if key.endswith(".weight_g"):
            continue
        value = raw[key]
        output_key = key
        if key.endswith(".weight_v"):
            output_key = key[: -len("_v")]
            gain = raw[f"{output_key}_g"]
            vector = value
            norm_shape = [vector.shape[0]] + [1] * (vector.ndim - 1)
            norm = mx.sqrt(mx.sum(mx.square(vector.reshape(vector.shape[0], -1)), axis=1)) \
                .reshape(norm_shape)
            value = gain * vector / norm
            folded += 1
        mx.eval(value)
        converted[output_key] = value
        if index % 100 == 0:
            print(f"audio VAE tensor {index}/{len(raw)}", flush=True)
    converted["latents_mean"] = mx.array(wrapper["latents_mean"], dtype=mx.float32)
    converted["latents_std"] = mx.array(wrapper["latents_std"], dtype=mx.float32)
    del raw
    if folded != 172 or len(converted) != 917:
        raise ValueError(
            f"Unexpected audio VAE geometry: folded={folded}, arrays={len(converted)}"
        )
    atomic_safetensors(
        output,
        dict(sorted(converted.items())),
        {
            "format": "mere.run.minimax-h3-audio-vae",
            "conversion": "official FP32 weight-normalization pairs folded; latent statistics added",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
    )
    stats = {
        "input_tensors": 1_087,
        "output_tensors": len(converted),
        "weight_norm_pairs_folded": folded,
        "precision": "FP32",
        "layout_transform": "none in artifact; native loader maps convolution layouts",
    }
    release_arrays(converted)
    return stats


def write_package_support(
    output: Path,
    downloaded: dict[str, Path],
    transformer_precision: str,
) -> None:
    transformer_bits = {"q4": 4, "q8": 8, "bf16": None}[transformer_precision]
    shutil.copyfile(downloaded["LICENSE"], output / "LICENSE")
    shutil.copyfile(
        downloaded["FL2VA/processor/tokenizer.json"], output / "tokenizer.json"
    )
    shutil.copyfile(
        downloaded["FL2VA/processor/tokenizer_config.json"],
        output / "tokenizer_config.json",
    )
    (output / "NOTICE").write_text(
        "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
        "Copyright © 2026 MiniMax. All Rights Reserved.\n",
        encoding="utf-8",
    )
    core_precision_description = (
        "preserved at the released BF16 precision"
        if transformer_bits is None
        else f"quantized directly from the released BF16 tensors to MLX affine {transformer_bits}-bit, group size 64"
    )
    (output / "MODIFICATIONS.md").write_text(
        f"""# Modifications to MiniMax H3

These files are MODIFIED versions of the MiniMax H3 Works, redistributed under
the MiniMax H3 Community License Agreement in `LICENSE` and `NOTICE`.

Modified by: Sawfwair (https://github.com/sawfwair/mere-run)

Every weight input came directly from the public official checkpoint
`MiniMaxAI/MiniMax-H3@ec19cc6daf5d8add9417c18e86b6b58cc6c55027`.
No third-party converted or quantized weights were used.

* The 52 fused transformer and token-refiner QKV matrices were first reordered
  from MiniMax's raw per-head interleave into the official reference model's
  `[all-q; all-k; all-v]` layout. The 208 core linear weights were then
  {core_precision_description}.
  Precision-sensitive BF16/F32 input, timestep, normalization, and output
  tensors remain at their released precision.
* The original BF16/F32 AdaLN projections and timestep MLP were evaluated over
  mere.run's released 31-point schedules. Their resulting inference table is
  stored in `adaln_cache.safetensors`; the 13B cache-covered parameters and the
  recomputed RoPE inverse frequencies are omitted from the compact transformer.
* The Qwen3-VL conditioner keeps the exact 50 language layers MiniMax H3 reads.
  Its 439 eligible linear weights were quantized directly from official BF16 to
  MLX affine 8-bit, group size 64. Unused layers 50-63, the LM head, and final
  language norm are omitted.
* The official video VAE tensors were cast directly from FP32 to FP16 and the
  released latent statistics were added. No weights were trained or inferred.
* The official audio VAE remains FP32. Its 172 weight-normalization pairs were
  folded algebraically into equivalent plain convolution weights, and the
  released latent statistics were added.
* Tokenizer files and the MiniMax H3 license are exact copies from the pinned
  official checkpoint. `SOURCE_MANIFEST.json` and
  `transformer.conversion.json` record the complete input/output provenance.

No weights were retrained, distilled, merged, or pruned beyond the explicit
inference-only omissions and numeric conversions above.
""",
        encoding="utf-8",
    )
    config = {
        "model_type": "minimax_h3",
        "partition": "fl2va",
        "tasks": ["t2va", "fl2va"],
        "sigma_shift_scales": {"video": VIDEO_SHIFT, "audio": AUDIO_SHIFT},
        "fps": 24,
        "text_encoder_quantization": {
            "group_size": 64,
            "bits": 8,
            "mode": "affine",
        },
        "transformer": {
            "hidden_size": 5_376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14_336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "text_dim": 5_120,
            "time_embed_hidden_dim": 5_376,
            "time_embed_dim": 2_688,
            "rope_inv_freq_len": 16,
        },
        "text_encoder": {
            "hidden": 5_120,
            "layers": 50,
            "heads": 64,
            "kv_heads": 8,
            "head_dim": 128,
            "intermediate": 25_600,
            "theta": 5_000_000.0,
        },
    }
    if transformer_bits is not None:
        config["quantization"] = {
            "group_size": 64,
            "bits": transformer_bits,
            "mode": "affine",
        }
    atomic_json(output / "config.json", config)


def hardware_receipt() -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
        ).strip()
        receipt["nvidia_smi"] = query.splitlines()
    except Exception as error:
        receipt["nvidia_smi_error"] = str(error)
    return receipt


def output_receipt(output: Path, filename: str, stats: dict[str, Any]) -> dict[str, Any]:
    path = output / filename
    return {
        "filename": filename,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        **stats,
    }


def write_sha256sums(output: Path) -> None:
    excluded = {"SHA256SUMS", "README.md"}
    lines = []
    for path in sorted(value for value in output.iterdir() if value.is_file()):
        if path.name in excluded or path.name.startswith("."):
            continue
        lines.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument(
        "--conversion-location",
        help="Declared physical conversion location recorded in the release receipt.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("transformer", "text_encoder", "video_vae", "audio_vae"),
        default=("transformer", "text_encoder", "video_vae", "audio_vae"),
    )
    parser.add_argument(
        "--transformer-precision",
        choices=("q4", "q8", "bf16"),
        default="q4",
        help="Storage precision for the 208 transformer core linears.",
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    components = set(args.components)
    if args.self_test_only:
        quantizer_self_test()
        qkv_reorder_self_test()
        return 0

    metadata = fetch_hub_metadata()
    source_files = component_source_files(metadata, components)
    if args.plan:
        print_plan(metadata, source_files)
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --plan or --self-test-only is used")
    if not args.conversion_location:
        raise ValueError("--conversion-location is required for a release conversion")

    output = args.output.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    source_manifest_path = output / "SOURCE_MANIFEST.json"
    downloaded = download_sources(
        metadata, source_files, cache_dir, source_manifest_path
    )
    write_package_support(output, downloaded, args.transformer_precision)
    self_test = quantizer_self_test()
    qkv_self_test = qkv_reorder_self_test()

    import mlx
    import mlx.core as mx
    import numpy
    import safetensors
    import huggingface_hub

    if not mx.cuda.is_available():
        print("warning: MLX CUDA is unavailable; conversion is using the current MLX device", flush=True)

    component_stats: dict[str, dict[str, Any]] = {}
    transformer_index = None
    source_identity = None
    if "transformer" in components:
        transformer_index = load_index(downloaded[TRANSFORMER_INDEX])
        index_sha = sha256_file(downloaded[TRANSFORMER_INDEX])
        source_identity = (
            f"{SOURCE_REPOSITORY}@{SOURCE_REVISION}:FL2VA/transformer:index-sha256:{index_sha}"
        )
        cache_stats = build_adaln_cache(
            output / "adaln_cache.safetensors",
            transformer_index,
            downloaded,
            source_identity,
        )
        component_stats["adaln_cache.safetensors"] = cache_stats
        transformer_stats = convert_transformer(
            output / "transformer.safetensors",
            transformer_index,
            downloaded,
            source_identity,
            args.transformer_precision,
        )
        component_stats["transformer.safetensors"] = transformer_stats

    if "text_encoder" in components:
        text_index = load_index(downloaded[TEXT_ENCODER_INDEX])
        component_stats["text_encoder.safetensors"] = convert_text_encoder(
            output / "text_encoder.safetensors", text_index, downloaded
        )
    if "video_vae" in components:
        component_stats["video_vae.safetensors"] = convert_video_vae(
            output / "video_vae.safetensors", downloaded
        )
    if "audio_vae" in components:
        component_stats["audio_vae.safetensors"] = convert_audio_vae(
            output / "audio_vae.safetensors", downloaded
        )

    outputs = {
        filename: output_receipt(output, filename, stats)
        for filename, stats in component_stats.items()
    }
    conversion_receipt = {
        "converter": "scripts/model-conversion/convert_minimax_h3_official_mlx.py",
        "converter_version": 3,
        "converter_sha256": sha256_file(Path(__file__).resolve()),
        "started_at": started_at,
        "completed_at": utc_now(),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "source_manifest": "SOURCE_MANIFEST.json",
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "third_party_weight_inputs": [],
        },
        "software": {
            "mlx": importlib_metadata.version("mlx"),
            "mlx_cuda": importlib_metadata.version("mlx-cuda")
            if mx.cuda.is_available()
            else None,
            "numpy": numpy.__version__,
            "safetensors": safetensors.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        },
        "hardware": hardware_receipt(),
        "declared_conversion_location": args.conversion_location,
        "mlx_cuda_available": bool(mx.cuda.is_available()),
        "quantizer_self_test": self_test,
        "qkv_reorder_self_test": qkv_self_test,
        "outputs": outputs,
    }
    atomic_json(output / "transformer.conversion.json", conversion_receipt)
    write_sha256sums(output)

    print(json.dumps(conversion_receipt, indent=2, sort_keys=True), flush=True)
    print(f"complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

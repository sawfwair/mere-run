#!/usr/bin/env python3
"""Build the compact FastH3 AdaLN sidecar from the pinned student checkpoint.

The managed MiniMax-H3 BF16 bundle omits the 13B schedule-only AdaLN weights.
FastH3 changes those weights, so its adapter cannot reuse the base bundle's
cache. This tool range-reads only the 106 required tensors from the immutable
FastH3 transformer, evaluates the released four-step schedule with MLX Metal,
and writes the roughly 120 MB source-bound sidecar expected by mere.run.

The remote range reads transfer about 26 GB but never materialize the 70 GB
transformer or the 148 GB complete model on disk. Install requirements:

    python -m pip install 'mlx==0.29.3' 'numpy==2.2.6'

Example:

    python scripts/model-conversion/prepare_minimax_h3_fasth3_vsa.py \
      --adapter "$HOME/Library/Application Support/MereRun/adapters/\
minimax-h3-fasth3-vsa-datafree-4step/bcf40ca6f45/\
fastvideo_fasth3_4step_v1_vsa_datafree_rank64.safetensors"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
SOURCE_REVISION = "b65818d41939b5085451074fe8ca8b799f8d4921"
SOURCE_INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
SOURCE_INDEX_SHA256 = "5be1367650e2faf79edf1f106a8354d1a7ef2e3ff45df40c4514b4ae97e1136b"
SOURCE_IDENTITY = f"{SOURCE_REPOSITORY}@{SOURCE_REVISION}:transformer"
ADAPTER_BYTES = 5_339_117_712
ADAPTER_SHA256 = "42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470"
ADAPTER_FORMAT = "fastvideo-lora-v2"
OUTPUT_FILENAME = "fastvideo_fasth3_v1_vsa_datafree_adaln_cache.safetensors"
BASE_SIGMAS = (0.999, 0.749, 0.5, 0.25, 0.0)
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
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


def verify_adapter(path: Path) -> None:
    if path.stat().st_size != ADAPTER_BYTES:
        raise ValueError(f"FastH3 adapter has {path.stat().st_size} bytes; expected {ADAPTER_BYTES}")
    digest = sha256_file(path)
    if digest != ADAPTER_SHA256:
        raise ValueError(f"FastH3 adapter SHA-256 differs: {digest}")
    header, metadata = read_safetensors_header(path)
    if metadata.get("format") != ADAPTER_FORMAT:
        raise ValueError("FastH3 adapter metadata has an unexpected format")
    pair_count = sum(key.endswith(".lora_A.weight") for key in header)
    difference_count = sum(key.endswith((".diff", ".diff_b")) for key in header)
    gate_count = sum(key.endswith(".attn.to_gate_compress.set_weight") for key in header)
    if (pair_count, difference_count, gate_count) != (362, 82, 50):
        raise ValueError(
            "FastH3 adapter tensor closure differs: "
            f"pairs={pair_count}, differences={difference_count}, gates={gate_count}"
        )


class RangeTensorSource:
    def __init__(self, temporary: Path, token: str | None) -> None:
        self.temporary = temporary
        self.token = token
        self.headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self.transferred_bytes = 0

    def source_url(self, relative_path: str) -> str:
        encoded = urllib.parse.quote(relative_path, safe="/")
        return (
            f"https://huggingface.co/{SOURCE_REPOSITORY}/resolve/"
            f"{SOURCE_REVISION}/{encoded}"
        )

    def request(self, url: str, start: int | None = None, end: int | None = None) -> Any:
        headers = {"Accept-Encoding": "identity"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if start is not None and end is not None:
            headers["Range"] = f"bytes={start}-{end}"
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300)

    def range_bytes(self, url: str, start: int, end: int) -> bytes:
        with self.request(url, start, end) as response:
            if response.status != 206:
                raise ValueError(f"Server ignored byte range {start}-{end} (HTTP {response.status})")
            value = response.read()
        if len(value) != end - start + 1:
            raise ValueError(f"Truncated byte range {start}-{end}")
        self.transferred_bytes += len(value)
        return value

    def shard_header(self, shard: str) -> tuple[int, dict[str, Any]]:
        if shard in self.headers:
            return self.headers[shard]
        url = self.source_url(f"transformer/{shard}")
        header_length = int.from_bytes(self.range_bytes(url, 0, 7), "little")
        if not 0 < header_length <= 64 * 1024 * 1024:
            raise ValueError(f"{shard} has an invalid safetensors header")
        header = json.loads(self.range_bytes(url, 8, 7 + header_length))
        header.pop("__metadata__", None)
        self.headers[shard] = (header_length, header)
        return self.headers[shard]

    def tensor_extent(self, shard: str, key: str) -> int:
        _, header = self.shard_header(shard)
        entry = header.get(key)
        if entry is None:
            raise ValueError(f"{shard} does not contain {key}")
        start, end = map(int, entry["data_offsets"])
        return end - start

    def load_tensor(self, shard: str, key: str) -> tuple[Any, Path]:
        import mlx.core as mx

        header_length, header = self.shard_header(shard)
        entry = header.get(key)
        if entry is None:
            raise ValueError(f"{shard} does not contain {key}")
        start, end = map(int, entry["data_offsets"])
        if start < 0 or end <= start:
            raise ValueError(f"{shard}:{key} has invalid data offsets")
        standalone_entry = {
            key: {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [0, end - start],
            }
        }
        raw_header = json.dumps(standalone_entry, separators=(",", ":")).encode("utf-8")
        raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
        output = self.temporary / f"tensor-{uuid.uuid4().hex}.safetensors"
        with output.open("wb") as handle:
            handle.write(len(raw_header).to_bytes(8, "little"))
            handle.write(raw_header)
            url = self.source_url(f"transformer/{shard}")
            absolute_start = 8 + header_length + start
            absolute_end = 8 + header_length + end - 1
            with self.request(url, absolute_start, absolute_end) as response:
                if response.status != 206:
                    raise ValueError(
                        f"Server ignored {shard}:{key} byte range (HTTP {response.status})"
                    )
                observed = 0
                while chunk := response.read(16 * 1024 * 1024):
                    handle.write(chunk)
                    observed += len(chunk)
            if observed != end - start:
                raise ValueError(f"Truncated tensor {shard}:{key}")
            self.transferred_bytes += observed
        arrays = mx.load(str(output))
        value = arrays[key]
        mx.eval(value)
        return value, output


def fetch_index(source: RangeTensorSource) -> dict[str, Any]:
    with source.request(source.source_url(SOURCE_INDEX)) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SOURCE_INDEX_SHA256:
        raise ValueError("Pinned FastH3 transformer index SHA-256 differs")
    index = json.loads(raw)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 688:
        raise ValueError("Pinned FastH3 transformer index closure differs")
    return index


def shifted_schedule(shift: float) -> tuple[list[float], list[float]]:
    import numpy as np

    shift32 = np.float32(shift)
    one = np.float32(1)
    sigmas = []
    for raw in BASE_SIGMAS:
        base = np.float32(raw)
        sigma = shift32 * base / (one + (shift32 - one) * base)
        sigmas.append(float(sigma))
    timesteps = [float(np.float32(one - np.float32(value))) for value in sigmas[:-1]]
    return sigmas, timesteps


def linear(value: Any, weight: Any, bias: Any) -> Any:
    import mlx.core as mx

    return mx.addmm(bias, value.astype(weight.dtype), weight.T)


def load_group(
    source: RangeTensorSource,
    weight_map: dict[str, str],
    keys: tuple[str, ...],
) -> tuple[dict[str, Any], list[Path]]:
    values: dict[str, Any] = {}
    paths: list[Path] = []
    for key in keys:
        shard = weight_map.get(key)
        if shard is None:
            raise ValueError(f"Pinned FastH3 transformer is missing {key}")
        value, path = source.load_tensor(shard, key)
        values[key] = value
        paths.append(path)
    return values, paths


def release_group(values: dict[str, Any], paths: list[Path]) -> None:
    import mlx.core as mx

    values.clear()
    mx.clear_cache()
    for path in paths:
        path.unlink(missing_ok=True)


def build_cache(source: RangeTensorSource, index: dict[str, Any], output: Path) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    weight_map: dict[str, str] = index["weight_map"]
    video_sigmas, video_timesteps = shifted_schedule(VIDEO_SHIFT)
    audio_sigmas, audio_timesteps = shifted_schedule(AUDIO_SHIFT)
    flat_timesteps: list[np.float32] = []
    condition = np.float32(0.999)
    for video, audio in zip(video_timesteps, audio_timesteps, strict=True):
        video32 = np.float32(video)
        flat_timesteps.extend((video32, np.float32(audio), max(video32, condition)))

    time_keys = (
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
    )
    time_weights, time_paths = load_group(source, weight_map, time_keys)
    frequency_indices = np.arange(128, dtype=np.float32)
    frequencies = np.exp(
        -np.log(np.float32(10_000)) * frequency_indices / np.float32(128)
    ).astype(np.float32)
    arguments = mx.array(np.asarray(flat_timesteps, dtype=np.float32)).reshape(-1, 1) \
        * mx.array(frequencies).reshape(1, -1)
    sinusoidal = mx.concatenate([mx.cos(arguments), mx.sin(arguments)], axis=-1)
    time_embeddings = linear(
        nn.silu(linear(
            sinusoidal.astype(mx.float32),
            time_weights["time_embedder.linear_1.weight"],
            time_weights["time_embedder.linear_1.bias"],
        )),
        time_weights["time_embedder.linear_2.weight"],
        time_weights["time_embedder.linear_2.bias"],
    ).reshape(4, 3, 2_688)
    mx.eval(time_embeddings)
    release_group(time_weights, time_paths)

    activated = nn.silu(time_embeddings.reshape(12, 2_688)).astype(mx.bfloat16)
    mx.eval(activated)
    block_modulations: list[Any] = []
    for block in range(50):
        prefix = f"transformer_blocks.{block}.adaln_proj.linear"
        keys = (f"{prefix}.weight", f"{prefix}.bias")
        values, paths = load_group(source, weight_map, keys)
        projected = linear(activated, values[keys[0]], values[keys[1]])
        modulation = projected.reshape(4, 9, 32_256).astype(mx.bfloat16)
        mx.eval(modulation)
        block_modulations.append(modulation)
        release_group(values, paths)
        transferred = source.transferred_bytes / (1024 ** 3)
        print(f"AdaLN block {block + 1}/50 ({transferred:.2f} GiB transferred)", flush=True)

    final_keys = ("norm_out.linear.weight", "norm_out.linear.bias")
    final_weights, final_paths = load_group(source, weight_map, final_keys)
    final_modulations = linear(
        activated,
        final_weights[final_keys[0]],
        final_weights[final_keys[1]],
    ).reshape(4, 3, 10_752).astype(mx.bfloat16)
    mx.eval(final_modulations)
    release_group(final_weights, final_paths)

    arrays: dict[str, Any] = {
        "audio_sigmas": mx.array(audio_sigmas, dtype=mx.float32),
        "final_modulations": final_modulations,
        "time_embeddings": time_embeddings,
        "video_sigmas": mx.array(video_sigmas, dtype=mx.float32),
    }
    for block, modulation in enumerate(block_modulations):
        arrays[f"blocks.{block}.modulations"] = modulation
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.safetensors")
    try:
        mx.save_safetensors(
            str(temporary),
            dict(sorted(arrays.items())),
            metadata={
                "schema_version": "2",
                "format": "mere.run.minimax-h3-adaln-cache",
                "source_identity": SOURCE_IDENTITY,
            },
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def required_remote_bytes(source: RangeTensorSource, index: dict[str, Any]) -> int:
    weight_map: dict[str, str] = index["weight_map"]
    keys = [
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
        "norm_out.linear.weight",
        "norm_out.linear.bias",
    ]
    for block in range(50):
        prefix = f"transformer_blocks.{block}.adaln_proj.linear"
        keys.extend((f"{prefix}.weight", f"{prefix}.bias"))
    return sum(source.tensor_extent(weight_map[key], key) for key in keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Verified FastH3 VSA DataFree managed adapter path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Output sidecar (default: adapter directory/{OUTPUT_FILENAME}).",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing verified sidecar.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = args.adapter.expanduser().resolve()
    output = (args.output or adapter.with_name(OUTPUT_FILENAME)).expanduser().resolve()
    if output.exists() and not args.force:
        raise ValueError(f"Output already exists: {output} (use --force to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)
    verify_adapter(adapter)
    if shutil.disk_usage(output.parent).free < 2 * 1024 ** 3:
        raise ValueError("At least 2 GiB of free temporary disk space is required")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    with tempfile.TemporaryDirectory(prefix="mere-run-fasth3-adaln-") as raw_temporary:
        source = RangeTensorSource(Path(raw_temporary), token)
        index = fetch_index(source)
        remote_bytes = required_remote_bytes(source, index)
        print(
            f"Verified adapter. The pinned range-read closure is {remote_bytes / (1024 ** 3):.2f} GiB; "
            "the complete checkpoint will not be downloaded.",
            flush=True,
        )
        build_cache(source, index, output)

    header, metadata = read_safetensors_header(output)
    if len(header) != 54 or metadata.get("source_identity") != SOURCE_IDENTITY:
        raise ValueError("Generated FastH3 AdaLN sidecar failed closure validation")
    print(f"Wrote {output}", flush=True)
    print(f"SHA-256 {sha256_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

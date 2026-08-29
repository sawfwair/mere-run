#!/usr/bin/env python3
"""Refresh the exact MiniMax-H3 AdaLN cache pack on Apple Silicon.

The compact MiniMax-H3 packages omit the source AdaLN projections. This tool
reads only those tensors from the immutable official checkpoint by using HTTP
range requests. It processes one source shard at a time, evaluates every
production schedule with MLX Metal, and removes the temporary shard before it
continues.

The refresh is accepted only when every previously published cache reproduces
its exact tensor closure. The output directory also contains a receipt that
binds the new pack to the source revision, tensor ranges, Metal environment,
published baseline file hashes, and reproduced tensor closures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Iterable


CONVERTER_PATH = Path(__file__).with_name("convert_minimax_h3_official_mlx.py")
CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "convert_minimax_h3_official_mlx",
    CONVERTER_PATH,
)
if CONVERTER_SPEC is None or CONVERTER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load {CONVERTER_PATH}")
CONVERTER = importlib.util.module_from_spec(CONVERTER_SPEC)
CONVERTER_SPEC.loader.exec_module(CONVERTER)

TRANSFORMER_INDEX_SHA256 = (
    "fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb"
)
SOURCE_TENSOR_COUNT = 106
SOURCE_TENSOR_BYTES = 26_142_079_488
RANGE_CHUNK_BYTES = 16 * 1024 * 1024
RANGE_ATTEMPTS = 5
CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
EXACT_MLX_VERSION = "0.32.1"

BASELINE_CACHE_SHA256 = {
    "adaln_cache-p5-v6-a3.safetensors":
        "7781b4bf5804884df4b44b099918a9e41ca3e81a883f9cdb8d0668fa2803c567",
    "adaln_cache-p5-v12-a3.safetensors":
        "4d8b5d183f410c15ef35fbbb395ea0c1361bf3b6248a760023e19342529fc74e",
    "adaln_cache-p9-v12-a3.safetensors":
        "ead59f04b1900cd6e34d8995182718869280884936708e8223ac92d16c66a0d2",
    "adaln_cache-p12-v12-a3.safetensors":
        "f420968d64ba4339f215ea782329571b56c7a46f48e82d6e006388e7452ba364",
    "adaln_cache-p16-v12-a3.safetensors":
        "c80297a5c386dab1f5b7c412bd363e8a9eb729b4a32d598ba6e4fd45ff434d42",
    "adaln_cache-p21-v12-a3.safetensors":
        "fd2ba7ba30a3e4a18772616065a81158e3483f8620a1d2a3b952719d014c89b5",
    "adaln_cache.safetensors":
        "aa6cf6facfa4287656767a04746bdfe84ca52c5a3a144a56957f9b42ed109ac4",
}
BASELINE_CACHE_TENSOR_CLOSURE_SHA256 = {
    "adaln_cache-p5-v6-a3.safetensors":
        "ace50ebb5bcc2bac23bef6b94a2f865d4b3a26330645d379903d17707bf49002",
    "adaln_cache-p5-v12-a3.safetensors":
        "f030be9420382b61e7306380238212e0d9a048424a3f42082bd03789cb3a00f5",
    "adaln_cache-p9-v12-a3.safetensors":
        "8234fb9d49b140af2ea99f5691f397e20e3a778752c58443a67f3d3e3870ca26",
    "adaln_cache-p12-v12-a3.safetensors":
        "5ff052ccfb495ff31f629d2652976df92a929906113253466f49564ccdbcbbb9",
    "adaln_cache-p16-v12-a3.safetensors":
        "4200ae29698898790a77f591c20a0dfecc50a58039c050039d587f6e5685a495",
    "adaln_cache-p21-v12-a3.safetensors":
        "65134eb448e60e11f2d615f9a25b406779d0fe88c861c378c72059a947b564a1",
    "adaln_cache.safetensors":
        "54b418c3ce71125c83f12c1ef4786ec450e9d95f1d781d6fe96ee86e935a9959",
}


def required_source_keys() -> tuple[str, ...]:
    keys = [
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
    ]
    for block in range(50):
        prefix = f"blocks.{block}.adaln_proj.linear"
        keys.extend((f"{prefix}.weight", f"{prefix}.bias"))
    keys.extend((
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
    ))
    return tuple(keys)


def resolve_url(path: str) -> str:
    repository = urllib.parse.quote(CONVERTER.SOURCE_REPOSITORY, safe="/")
    filename = urllib.parse.quote(path, safe="/")
    return (
        f"https://huggingface.co/{repository}/resolve/"
        f"{CONVERTER.SOURCE_REVISION}/{filename}?download=true"
    )


def request_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "mere-run-minimax-h3-adaln-refresh/1",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        total = validate_content_range(response.headers.get("Content-Range"), start, end)
        payload = response.read()
    expected = end - start + 1
    if len(payload) != expected:
        raise ValueError(f"Range returned {len(payload)} bytes; expected {expected}")
    return payload, total


def validate_content_range(value: str | None, start: int, end: int) -> int:
    match = CONTENT_RANGE_PATTERN.fullmatch(value or "")
    if match is None:
        raise ValueError(f"Invalid Content-Range header: {value!r}")
    observed_start, observed_end, total = (int(part) for part in match.groups())
    if observed_start != start or observed_end != end or total <= end:
        raise ValueError(
            f"Content-Range {value!r} does not match bytes {start}-{end}"
        )
    return total


def remote_safetensors_header(
    path: str,
    expected_size: int,
) -> tuple[dict[str, Any], dict[str, str], int]:
    url = resolve_url(path)
    raw_length, total = request_range(url, 0, 7)
    if total != expected_size:
        raise ValueError(f"{path} has {total} bytes; expected {expected_size}")
    header_length = int.from_bytes(raw_length, "little")
    if not 0 < header_length <= 64 * 1024 * 1024:
        raise ValueError(f"{path} has an invalid safetensors header length")
    raw_header, repeated_total = request_range(url, 8, 8 + header_length - 1)
    if repeated_total != total:
        raise ValueError(f"{path} changed between range requests")
    header = json.loads(raw_header)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} metadata is not an object")
    payload_bytes = max(int(value["data_offsets"][1]) for value in header.values())
    if 8 + header_length + payload_bytes != total:
        raise ValueError(f"{path} payload extent differs from its file size")
    return header, metadata, header_length


def safetensors_tensor_closure(path: Path) -> str:
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))
        metadata = header.pop("__metadata__", {})
        payload_offset = 8 + header_length
        tensors = []
        for key in sorted(header):
            entry = header[key]
            start, end = (int(value) for value in entry["data_offsets"])
            handle.seek(payload_offset + start)
            remaining = end - start
            digest = hashlib.sha256()
            while remaining:
                chunk = handle.read(min(remaining, RANGE_CHUNK_BYTES))
                if not chunk:
                    raise ValueError(f"{path} ended inside tensor {key}")
                digest.update(chunk)
                remaining -= len(chunk)
            tensors.append({
                "key": key,
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "byte_count": end - start,
                "sha256": digest.hexdigest(),
            })
    closure = {"metadata": metadata, "tensors": tensors}
    encoded = json.dumps(closure, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def subset_header(
    source_header: dict[str, Any],
    keys: Iterable[str],
    metadata: dict[str, str],
) -> tuple[bytes, list[tuple[str, dict[str, Any]]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    cursor = 0
    output_header: dict[str, Any] = {"__metadata__": metadata}
    for key in sorted(keys):
        source = source_header[key]
        start, end = (int(value) for value in source["data_offsets"])
        byte_count = end - start
        entry = {
            "dtype": source["dtype"],
            "shape": source["shape"],
            "data_offsets": [cursor, cursor + byte_count],
        }
        output_header[key] = entry
        selected.append((key, source))
        cursor += byte_count
    encoded = json.dumps(
        output_header,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return len(encoded).to_bytes(8, "little") + encoded, selected


def stream_range(
    url: str,
    start: int,
    end: int,
    destination: BinaryIO,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    next_byte = start
    attempts = 0
    while next_byte <= end:
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={next_byte}-{end}",
                "User-Agent": "mere-run-minimax-h3-adaln-refresh/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                validate_content_range(
                    response.headers.get("Content-Range"),
                    next_byte,
                    end,
                )
                while chunk := response.read(RANGE_CHUNK_BYTES):
                    destination.write(chunk)
                    digest.update(chunk)
                    next_byte += len(chunk)
            if next_byte <= end:
                raise OSError(f"Range ended at byte {next_byte - 1}; expected {end}")
        except Exception:
            attempts += 1
            if attempts >= RANGE_ATTEMPTS:
                raise
            time.sleep(min(2 ** attempts, 16))
    return digest.hexdigest(), end - start + 1


def materialize_subset_shard(
    source_path: str,
    source_header: dict[str, Any],
    source_metadata: dict[str, str],
    source_header_length: int,
    keys: Iterable[str],
    destination: Path,
) -> list[dict[str, Any]]:
    encoded_header, selected = subset_header(source_header, keys, source_metadata)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    records: list[dict[str, Any]] = []
    url = resolve_url(source_path)
    try:
        with temporary.open("wb") as output:
            output.write(encoded_header)
            for key, source in selected:
                source_start, source_end = (int(value) for value in source["data_offsets"])
                absolute_start = 8 + source_header_length + source_start
                absolute_end = 8 + source_header_length + source_end - 1
                digest, byte_count = stream_range(
                    url,
                    absolute_start,
                    absolute_end,
                    output,
                )
                records.append({
                    "key": key,
                    "shard": source_path.rsplit("/", 1)[-1],
                    "dtype": source["dtype"],
                    "shape": source["shape"],
                    "byte_count": byte_count,
                    "sha256": digest,
                })
                print(f"source tensor {key} ({byte_count / 1e6:.1f} MB)", flush=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return records


def hardware_receipt() -> dict[str, Any]:
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory = int(subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip())
    return {"chip": chip, "unified_memory_bytes": memory}


def source_closure_digest(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        sorted(records, key=lambda value: value["key"]),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def schedule_inputs() -> list[dict[str, Any]]:
    import numpy as np

    schedule_values: list[dict[str, Any]] = []
    condition = np.float32(0.999)
    for point_count, video_shift, audio_shift in CONVERTER.CACHE_SCHEDULES:
        video_sigmas, video_timesteps = CONVERTER.schedule(point_count, video_shift)
        audio_sigmas, audio_timesteps = CONVERTER.schedule(point_count, audio_shift)
        timesteps: list[list[float]] = []
        for video, audio in zip(video_timesteps, audio_timesteps, strict=True):
            video32 = np.float32(video)
            audio32 = np.float32(audio)
            timesteps.append([
                float(video32),
                float(audio32),
                float(max(video32, condition)),
            ])
        schedule_values.append({
            "point_count": point_count,
            "video_shift": video_shift,
            "audio_shift": audio_shift,
            "video_sigmas": video_sigmas,
            "audio_sigmas": audio_sigmas,
            "timesteps": timesteps,
            "step_count": point_count - 1,
        })
    return schedule_values


def build_pack(output: Path, work_directory: Path) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    if platform.system() != "Darwin" or "gpu" not in str(mx.default_device()).lower():
        raise ValueError("The exact production cache refresh requires MLX Metal")
    mlx_version = importlib_metadata.version("mlx")
    if mlx_version != EXACT_MLX_VERSION:
        raise ValueError(
            f"The exact production cache refresh requires MLX {EXACT_MLX_VERSION}; "
            f"found {mlx_version}"
        )

    metadata = CONVERTER.fetch_hub_metadata()
    entries = CONVERTER.metadata_by_name(metadata)
    index_path = CONVERTER.TRANSFORMER_INDEX
    index_bytes = urllib.request.urlopen(resolve_url(index_path), timeout=180).read()
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    if index_digest != TRANSFORMER_INDEX_SHA256:
        raise ValueError(f"Transformer index SHA-256 changed: {index_digest}")
    transformer_index = json.loads(index_bytes)
    weight_map: dict[str, str] = transformer_index["weight_map"]
    source_identity = (
        f"{CONVERTER.SOURCE_REPOSITORY}@{CONVERTER.SOURCE_REVISION}:"
        f"FL2VA/transformer:index-sha256:{index_digest}"
    )

    required = required_source_keys()
    missing = sorted(set(required) - set(weight_map))
    if missing:
        raise ValueError(f"Pinned transformer is missing source tensors: {missing}")
    by_shard: dict[str, list[str]] = {}
    for key in required:
        by_shard.setdefault(weight_map[key], []).append(key)

    schedule_values = schedule_inputs()
    frequency_indices = np.arange(128, dtype=np.float32)
    frequencies = np.exp(
        -np.log(np.float32(10_000)) * frequency_indices / np.float32(128)
    ).astype(np.float32)

    time_embeddings: list[Any] | None = None
    step_time_embeddings: list[list[Any]] | None = None
    block_modulations: list[list[Any] | None] = [None] * 50
    final_modulations: list[Any] | None = None
    source_records: list[dict[str, Any]] = []
    headers: dict[str, tuple[dict[str, Any], dict[str, str], int]] = {}

    logical_bytes = 0
    for shard, keys in sorted(by_shard.items()):
        source_path = f"FL2VA/transformer/{shard}"
        source_entry = entries[source_path]
        header = remote_safetensors_header(source_path, int(source_entry["size"]))
        headers[shard] = header
        logical_bytes += sum(
            int(header[0][key]["data_offsets"][1])
            - int(header[0][key]["data_offsets"][0])
            for key in keys
        )
    if len(required) != SOURCE_TENSOR_COUNT or logical_bytes != SOURCE_TENSOR_BYTES:
        raise ValueError(
            f"Source closure is {len(required)} tensors and {logical_bytes} bytes; "
            f"expected {SOURCE_TENSOR_COUNT} tensors and {SOURCE_TENSOR_BYTES} bytes"
        )

    for shard, keys in sorted(by_shard.items()):
        source_path = f"FL2VA/transformer/{shard}"
        subset_path = work_directory / shard
        header, source_metadata, header_length = headers[shard]
        print(f"source shard {shard}", flush=True)
        source_records.extend(materialize_subset_shard(
            source_path,
            header,
            source_metadata,
            header_length,
            keys,
            subset_path,
        ))
        raw = mx.load(str(subset_path))

        if "time_embedder.proj_in.weight" in raw:
            frequency_array = mx.array(frequencies).reshape(1, -1)
            time_embeddings = []
            step_time_embeddings = []
            for values in schedule_values:
                schedule_steps = []
                for timesteps in values["timesteps"]:
                    arguments = mx.array(timesteps).reshape(-1, 1) * frequency_array
                    sinusoidal = mx.concatenate(
                        [mx.cos(arguments), mx.sin(arguments)],
                        axis=-1,
                    )
                    time_embedding = CONVERTER.linear(
                        nn.silu(CONVERTER.linear(
                            sinusoidal.astype(mx.float32),
                            raw["time_embedder.proj_in.weight"],
                            raw["time_embedder.proj_in.bias"],
                        )),
                        raw["time_embedder.proj_out.weight"],
                        raw["time_embedder.proj_out.bias"],
                    )
                    mx.eval(time_embedding)
                    schedule_steps.append(time_embedding)
                stacked = mx.stack(schedule_steps, axis=0)
                mx.eval(stacked)
                step_time_embeddings.append(schedule_steps)
                time_embeddings.append(stacked)

        if step_time_embeddings is None:
            raise ValueError("The first source shard did not provide the timestep embedding")
        for block in range(50):
            prefix = f"blocks.{block}.adaln_proj.linear"
            if f"{prefix}.weight" not in raw:
                continue
            schedule_modulations = []
            for schedule_steps in step_time_embeddings:
                step_modulations = []
                for time_embedding in schedule_steps:
                    projected = CONVERTER.linear(
                        nn.silu(time_embedding),
                        raw[f"{prefix}.weight"],
                        raw[f"{prefix}.bias"],
                    )
                    modulation = projected.reshape(9, 32_256).astype(mx.bfloat16)
                    mx.eval(modulation)
                    step_modulations.append(modulation)
                stacked = mx.stack(step_modulations, axis=0)
                mx.eval(stacked)
                schedule_modulations.append(stacked)
            block_modulations[block] = schedule_modulations
            print(f"AdaLN cache block {block + 1}/50", flush=True)

        if "final_layer.adaln_proj.linear.weight" in raw:
            final_modulations = []
            for schedule_steps in step_time_embeddings:
                step_modulations = []
                for time_embedding in schedule_steps:
                    projected = CONVERTER.linear(
                        nn.silu(time_embedding),
                        raw["final_layer.adaln_proj.linear.weight"],
                        raw["final_layer.adaln_proj.linear.bias"],
                    )
                    modulation = projected.reshape(3, 10_752).astype(mx.bfloat16)
                    mx.eval(modulation)
                    step_modulations.append(modulation)
                stacked = mx.stack(step_modulations, axis=0)
                mx.eval(stacked)
                final_modulations.append(stacked)

        del raw
        mx.clear_cache()
        subset_path.unlink()

    if time_embeddings is None or final_modulations is None:
        raise ValueError("The source closure did not produce final AdaLN tensors")
    if any(value is None for value in block_modulations):
        raise ValueError("The source closure did not produce every AdaLN block")

    outputs: dict[str, dict[str, Any]] = {}
    index_entries: list[dict[str, Any]] = []
    for schedule_index, values in enumerate(schedule_values):
        filename = CONVERTER.cache_filename(
            values["point_count"],
            values["video_shift"],
            values["audio_shift"],
        )
        arrays: dict[str, Any] = {
            "audio_sigmas": mx.array(values["audio_sigmas"], dtype=mx.float32),
            "final_modulations": final_modulations[schedule_index],
            "time_embeddings": time_embeddings[schedule_index],
            "video_sigmas": mx.array(values["video_sigmas"], dtype=mx.float32),
        }
        for block, modulation in enumerate(block_modulations):
            arrays[f"blocks.{block}.modulations"] = modulation[schedule_index]
        cache_path = output / filename
        CONVERTER.atomic_safetensors(
            cache_path,
            dict(sorted(arrays.items())),
            {
                "schema_version": CONVERTER.CACHE_SCHEMA_VERSION,
                "format": CONVERTER.CACHE_FORMAT,
                "source_identity": source_identity,
            },
        )
        digest = CONVERTER.sha256_file(cache_path)
        tensor_closure_digest = safetensors_tensor_closure(cache_path)
        byte_count = cache_path.stat().st_size
        outputs[filename] = {
            "sha256": digest,
            "tensor_closure_sha256": tensor_closure_digest,
            "byte_count": byte_count,
            "point_count": values["point_count"],
            "video_shift": values["video_shift"],
            "audio_shift": values["audio_shift"],
            "evaluation_backend": "mlx-metal",
        }
        index_entries.append({
            "schedule": {
                "point_count": values["point_count"],
                "video_flow_shift": values["video_shift"],
                "audio_flow_shift": values["audio_shift"],
            },
            "filename": filename,
            "byte_count": byte_count,
            "sha256": digest,
        })
        CONVERTER.release_arrays(arrays)

    for filename, expected in BASELINE_CACHE_TENSOR_CLOSURE_SHA256.items():
        observed = outputs[filename]["tensor_closure_sha256"]
        if observed != expected:
            raise ValueError(
                f"Baseline cache {filename} has tensor closure {observed}; "
                f"expected {expected}"
            )

    index_output = output / "adaln_cache.index.json"
    CONVERTER.atomic_json(index_output, {
        "schema_version": CONVERTER.CACHE_PACK_SCHEMA_VERSION,
        "format": CONVERTER.CACHE_PACK_FORMAT,
        "source_identity": source_identity,
        "entries": sorted(
            index_entries,
            key=lambda entry: (
                entry["schedule"]["point_count"],
                entry["schedule"]["video_flow_shift"],
                entry["schedule"]["audio_flow_shift"],
            ),
        ),
    })
    outputs[index_output.name] = {
        "sha256": CONVERTER.sha256_file(index_output),
        "byte_count": index_output.stat().st_size,
        "entry_count": len(index_entries),
        "evaluation_backend": "mlx-metal",
    }

    receipt = {
        "schema_version": 1,
        "format": "mere.run.minimax-h3-adaln-cache-pack-refresh",
        "source_identity": source_identity,
        "source_repository": CONVERTER.SOURCE_REPOSITORY,
        "source_revision": CONVERTER.SOURCE_REVISION,
        "source_index_sha256": index_digest,
        "source_tensor_count": len(source_records),
        "source_tensor_bytes": sum(value["byte_count"] for value in source_records),
        "source_tensor_closure_sha256": source_closure_digest(source_records),
        "source_tensors": sorted(source_records, key=lambda value: value["key"]),
        "evaluation_backend": "mlx-metal",
        "hardware": hardware_receipt(),
        "software": {
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "mlx": mlx_version,
        },
        "published_baseline_cache_sha256": BASELINE_CACHE_SHA256,
        "baseline_cache_tensor_closure_sha256": BASELINE_CACHE_TENSOR_CLOSURE_SHA256,
        "baseline_tensor_closures_matched": True,
        "outputs": outputs,
    }
    receipt_path = output / "adaln_cache.refresh.json"
    CONVERTER.atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--work-directory",
        type=Path,
        help="Directory for one temporary source shard. The default uses the system temp root.",
    )
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.plan:
        print(f"source: {CONVERTER.SOURCE_REPOSITORY}@{CONVERTER.SOURCE_REVISION}")
        print(f"source tensors: {SOURCE_TENSOR_COUNT}")
        print(f"range bytes: {SOURCE_TENSOR_BYTES}")
        print(f"production schedules: {len(CONVERTER.CACHE_SCHEDULES)}")
        return 0

    work_parent = args.work_directory.expanduser().resolve() \
        if args.work_directory is not None else None
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="mere-run-h3-adaln-",
        dir=work_parent,
    ) as directory:
        build_pack(output, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

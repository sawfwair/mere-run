#!/usr/bin/env python3
"""Verify a native Nemotron Omni checkpoint against its pinned BF16 source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
from typing import BinaryIO, Iterable


SOURCE_REPOSITORY = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
SOURCE_REVISION = "24e67ea000b7c2837fc8f9488aa2008524fac8ba"
FORMAT = "mere-run-nemotron-omni-native-v1"
EXPERT_FORMAT = "mere-run-nemotron-omni-experts-v1"
TOTAL_WEIGHT_BYTES = 66_031_270_520
EXPERT_WEIGHT_BYTES = 58_749_616_128
NON_EXPERT_WEIGHT_BYTES = 7_281_654_392
SOURCE_EXPERT_RE = re.compile(
    r"^language_model\.backbone\.layers\.(?P<layer>\d+)\.mixer\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>up_proj|down_proj)\.weight$"
)
PACKED_EXPERT_RE = re.compile(
    r"^backbone\.layers\.(?P<layer>\d+)\.mixer\.experts\."
    r"(?P<projection>up_proj|down_proj)\.weight$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_index(root: Path) -> dict:
    return json.loads((root / "model.safetensors.index.json").read_text())


def load_safetensors_header(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    with path.open("rb") as handle:
        length_data = handle.read(8)
        if len(length_data) != 8:
            raise ValueError(f"Safetensors file is too small: {path}")
        header_length = struct.unpack("<Q", length_data)[0]
        header_data = handle.read(header_length)
        if len(header_data) != header_length:
            raise ValueError(f"Safetensors header is truncated: {path}")
    header = json.loads(header_data)
    file_metadata = header.pop("__metadata__", {})
    data_start = 8 + header_length
    tensors = {}
    for key, value in header.items():
        start, end = value["data_offsets"]
        tensors[key] = {
            "dtype": value["dtype"],
            "shape": value["shape"],
            "start": data_start + start,
            "end": data_start + end,
        }
    return file_metadata, tensors


def update_range(
    digest: hashlib._Hash,
    handle: BinaryIO,
    start: int,
    end: int,
) -> int:
    handle.seek(start)
    remaining = end - start
    total = remaining
    while remaining:
        chunk = handle.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise ValueError("Safetensors payload ended unexpectedly")
        digest.update(chunk)
        remaining -= len(chunk)
    return total


def digest_ranges(
    ranges: Iterable[tuple[Path, int, int]],
    handles: dict[Path, BinaryIO],
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for path, start, end in ranges:
        handle = handles.get(path)
        if handle is None:
            handle = path.open("rb")
            handles[path] = handle
        total += update_range(digest, handle, start, end)
    return digest.hexdigest(), total


def metadata_cache(root: Path, filenames: Iterable[str]) -> dict[str, dict[str, dict]]:
    return {
        filename: load_safetensors_header(root / filename)[1]
        for filename in sorted(set(filenames))
    }


def verify_non_experts(
    source: Path,
    native: Path,
    source_index: dict,
    native_index: dict,
    source_headers: dict[str, dict[str, dict]],
    native_headers: dict[str, dict[str, dict]],
    handles: dict[Path, BinaryIO],
) -> tuple[list[dict], int]:
    source_map = source_index["weight_map"]
    native_map = native_index["weight_map"]
    expected_keys = {key for key in source_map if SOURCE_EXPERT_RE.match(key) is None}
    if set(native_map) != expected_keys:
        raise ValueError("Native non-expert index does not exactly partition source keys")
    results = []
    total = 0
    for filename in sorted(set(native_map.values())):
        source_keys = sorted(
            (key for key in expected_keys if source_map[key] == filename),
            key=lambda key: source_headers[filename][key]["start"],
        )
        native_keys = sorted(
            (key for key, shard in native_map.items() if shard == filename),
            key=lambda key: native_headers[filename][key]["start"],
        )
        if native_keys != source_keys:
            raise ValueError(f"Tensor order changed in {filename}")
        source_ranges = [
            (
                source / filename,
                source_headers[filename][key]["start"],
                source_headers[filename][key]["end"],
            )
            for key in source_keys
        ]
        native_ranges = [
            (
                native / filename,
                native_headers[filename][key]["start"],
                native_headers[filename][key]["end"],
            )
            for key in native_keys
        ]
        source_digest, source_bytes = digest_ranges(source_ranges, handles)
        native_digest, native_bytes = digest_ranges(native_ranges, handles)
        if source_bytes != native_bytes or source_digest != native_digest:
            raise ValueError(f"Non-expert payload mismatch in {filename}")
        total += native_bytes
        results.append(
            {
                "filename": filename,
                "tensor_count": len(native_keys),
                "payload_bytes": native_bytes,
                "sha256": native_digest,
            }
        )
        print(f"verified non-expert shard {len(results)}/17: {filename}", flush=True)
    return results, total


def verify_experts(
    source: Path,
    native: Path,
    source_index: dict,
    source_headers: dict[str, dict[str, dict]],
    handles: dict[Path, BinaryIO],
) -> tuple[list[dict], int, int]:
    source_map = source_index["weight_map"]
    pack_path = native / "experts-bf16.safetensors"
    pack_metadata, pack_header = load_safetensors_header(pack_path)
    expected_metadata = {
        "format": EXPERT_FORMAT,
        "source_repo": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "payload_bytes": str(EXPERT_WEIGHT_BYTES),
    }
    if pack_metadata != expected_metadata:
        raise ValueError("Packed expert metadata does not match the pinned source")
    groups: dict[tuple[int, str], list[tuple[int, str]]] = {}
    for key in source_map:
        match = SOURCE_EXPERT_RE.match(key)
        if match is None:
            continue
        group = (int(match["layer"]), match["projection"])
        groups.setdefault(group, []).append((int(match["expert"]), key))
    if len(groups) != 46:
        raise ValueError(f"Expected 46 expert groups; found {len(groups)}")

    results = []
    total = 0
    tensor_count = 0
    for layer, projection in sorted(groups):
        source_keys = sorted(groups[(layer, projection)])
        if [expert for expert, _key in source_keys] != list(range(128)):
            raise ValueError(f"Expert inventory is incomplete for layer {layer} {projection}")
        output_key = f"backbone.layers.{layer}.mixer.experts.{projection}.weight"
        match = PACKED_EXPERT_RE.match(output_key)
        if match is None or output_key not in pack_header:
            raise ValueError(f"Packed expert tensor is missing: {output_key}")
        source_ranges = []
        for _expert, key in source_keys:
            filename = source_map[key]
            metadata = source_headers[filename][key]
            source_ranges.append(
                (source / filename, metadata["start"], metadata["end"])
            )
        output_metadata = pack_header[output_key]
        source_digest, source_bytes = digest_ranges(source_ranges, handles)
        output_digest, output_bytes = digest_ranges(
            [(pack_path, output_metadata["start"], output_metadata["end"])],
            handles,
        )
        if source_bytes != output_bytes or source_digest != output_digest:
            raise ValueError(f"Expert payload mismatch for {output_key}")
        total += output_bytes
        tensor_count += len(source_keys)
        results.append(
            {
                "tensor": output_key,
                "source_tensor_count": len(source_keys),
                "payload_bytes": output_bytes,
                "sha256": output_digest,
            }
        )
        print(f"verified expert group {len(results)}/46: {output_key}", flush=True)
    return results, total, tensor_count


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    native = args.native.resolve()
    source_index = load_index(source)
    native_index = load_index(native)
    source_headers = metadata_cache(source, source_index["weight_map"].values())
    native_headers = metadata_cache(native, native_index["weight_map"].values())
    handles: dict[Path, BinaryIO] = {}
    try:
        non_expert_results, non_expert_bytes = verify_non_experts(
            source,
            native,
            source_index,
            native_index,
            source_headers,
            native_headers,
            handles,
        )
        expert_results, expert_bytes, expert_source_tensors = verify_experts(
            source,
            native,
            source_index,
            source_headers,
            handles,
        )
    finally:
        for handle in handles.values():
            handle.close()
    if non_expert_bytes != NON_EXPERT_WEIGHT_BYTES:
        raise ValueError(f"Wrong non-expert byte count: {non_expert_bytes}")
    if expert_bytes != EXPERT_WEIGHT_BYTES:
        raise ValueError(f"Wrong expert byte count: {expert_bytes}")
    if non_expert_bytes + expert_bytes != TOTAL_WEIGHT_BYTES:
        raise ValueError("Native payload does not exactly partition the source checkpoint")

    receipt = {
        "schema_version": 1,
        "status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "format": FORMAT,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_tensor_count": len(source_index["weight_map"]),
        "native_non_expert_tensor_count": len(native_index["weight_map"]),
        "source_expert_tensor_count": expert_source_tensors,
        "native_expert_tensor_count": len(expert_results),
        "total_weight_bytes": TOTAL_WEIGHT_BYTES,
        "non_expert_weight_bytes": non_expert_bytes,
        "expert_weight_bytes": expert_bytes,
        "non_expert_shards": non_expert_results,
        "expert_groups": expert_results,
    }
    write_json_atomic(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in [
        "status",
        "source_tensor_count",
        "native_non_expert_tensor_count",
        "source_expert_tensor_count",
        "native_expert_tensor_count",
        "total_weight_bytes",
    ]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

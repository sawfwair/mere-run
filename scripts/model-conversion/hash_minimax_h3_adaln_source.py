#!/usr/bin/env python3
"""Hash the exact MiniMax-H3 tensors that produce the AdaLN cache pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))
    header.pop("__metadata__", None)
    return header, 8 + header_length


def tensor_sha256(path: Path, offset: int, byte_count: int) -> str:
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{path} ended inside a tensor payload")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Preserve the snapshot directory when the index itself is a Hub-cache
    # symlink; sibling shard links live beside that logical path.
    index_path = args.index.expanduser().absolute()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    keys = sorted(
        key
        for key in weight_map
        if ".adaln_proj." in key or key.startswith("time_embedder.")
    )
    if len(keys) != 106:
        raise ValueError(f"AdaLN source closure has {len(keys)} tensors; expected 106")

    headers: dict[Path, tuple[dict[str, Any], int]] = {}
    tensors: dict[str, dict[str, Any]] = {}
    for key in keys:
        shard = index_path.parent / weight_map[key]
        if shard not in headers:
            headers[shard] = safetensors_header(shard)
        header, payload_offset = headers[shard]
        entry = header[key]
        start, end = (int(value) for value in entry["data_offsets"])
        tensors[key] = {
            "byte_count": end - start,
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "sha256": tensor_sha256(shard, payload_offset + start, end - start),
        }

    canonical = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "schema_version": 1,
        "format": "mere.run.minimax-h3-adaln-source-closure",
        "index_sha256": sha256_file(index_path),
        "tensor_count": len(tensors),
        "tensor_bytes": sum(entry["byte_count"] for entry in tensors.values()),
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
        "tensors": tensors,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

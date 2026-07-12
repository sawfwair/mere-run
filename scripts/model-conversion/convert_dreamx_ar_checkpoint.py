#!/usr/bin/env python3
"""Stream and convert the released DreamX AR checkpoint to native BF16."""

import argparse
import json
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np


DEFAULT_REPO = "GD-ML/DreamX-World-5B"
DEFAULT_REVISION = "67487c4a61466bb7166d30b7187dd465e0ac9f6c"
DEFAULT_FILENAME = "model.safetensors"


def request(url, byte_range=None):
    headers = {"User-Agent": "mere.run-dreamx-ar-converter/1"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def read_range(url, start, end):
    with request(url, (start, end)) as response:
        if response.status != 206:
            raise RuntimeError(f"Server ignored byte range {start}-{end}: HTTP {response.status}")
        payload = response.read()
    expected = end - start + 1
    if len(payload) != expected:
        raise RuntimeError(f"Short range response: expected {expected}, received {len(payload)}")
    return payload


def read_header(url):
    header_size = struct.unpack("<Q", read_range(url, 0, 7))[0]
    header = json.loads(read_range(url, 8, 7 + header_size).decode("utf-8"))
    return header, 8 + header_size


def native_key_and_shape(key, shape):
    if key == "patch_embedding.weight":
        return "patch_embedding_proj.weight", [shape[0], int(np.prod(shape[1:]))]
    exact = {
        "patch_embedding.bias": "patch_embedding_proj.bias",
        "text_embedding.0.weight": "text_embedding_0.weight",
        "text_embedding.0.bias": "text_embedding_0.bias",
        "text_embedding.2.weight": "text_embedding_1.weight",
        "text_embedding.2.bias": "text_embedding_1.bias",
        "time_embedding.0.weight": "time_embedding_0.weight",
        "time_embedding.0.bias": "time_embedding_0.bias",
        "time_embedding.2.weight": "time_embedding_1.weight",
        "time_embedding.2.bias": "time_embedding_1.bias",
        "time_projection.1.weight": "time_projection.weight",
        "time_projection.1.bias": "time_projection.bias",
    }
    if key in exact:
        return exact[key], shape
    if ".ffn.0." in key:
        return key.replace(".ffn.0.", ".ffn.fc1."), shape
    if ".ffn.2." in key:
        return key.replace(".ffn.2.", ".ffn.fc2."), shape
    return key, shape


def f32_to_bf16(payload):
    values = np.frombuffer(payload, dtype="<u4")
    special = (values & np.uint32(0x7F80_0000)) == np.uint32(0x7F80_0000)
    bias = np.uint32(0x0000_7FFF) + ((values >> np.uint32(16)) & np.uint32(1))
    rounded = np.where(special, values, values + bias)
    return (rounded >> np.uint32(16)).astype("<u2", copy=False).tobytes()


def build_manifest(source_header, repo, revision):
    tensors = []
    output_offset = 0
    for source_key, entry in source_header.items():
        if source_key == "__metadata__":
            continue
        if entry["dtype"] != "F32":
            raise RuntimeError(f"Expected F32 source tensor, got {entry['dtype']} for {source_key}")
        source_start, source_end = entry["data_offsets"]
        native_key, native_shape = native_key_and_shape(source_key, entry["shape"])
        output_size = (source_end - source_start) // 2
        tensors.append({
            "source_key": source_key,
            "native_key": native_key,
            "shape": native_shape,
            "source_start": source_start,
            "source_end": source_end,
            "output_start": output_offset,
            "output_end": output_offset + output_size,
        })
        output_offset += output_size

    output_header = {
        "__metadata__": {
            "format": "pt",
            "source_repo": repo,
            "source_revision": revision,
            "conversion": "F32-to-BF16-rne",
            "runtime": "mere.run-native-causal-wan2",
        }
    }
    for tensor in tensors:
        output_header[tensor["native_key"]] = {
            "dtype": "BF16",
            "shape": tensor["shape"],
            "data_offsets": [tensor["output_start"], tensor["output_end"]],
        }
    encoded = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return tensors, encoded, output_offset


def convert(url, destination, tensors, header, source_data_start, output_bytes, chunk_bytes):
    temporary = destination.with_suffix(destination.suffix + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_data_start = 8 + len(header)
    expected_prefix = struct.pack("<Q", len(header)) + header

    if temporary.exists():
        with temporary.open("rb") as existing:
            if existing.read(len(expected_prefix)) != expected_prefix:
                raise RuntimeError(f"Partial file has a different header: {temporary}")
    else:
        with temporary.open("wb") as output:
            output.write(expected_prefix)

    with temporary.open("r+b") as output:
        for index, tensor in enumerate(tensors, start=1):
            output_start = output_data_start + tensor["output_start"]
            output_end = output_data_start + tensor["output_end"]
            current_size = output.seek(0, os.SEEK_END)
            if current_size >= output_end:
                continue
            if current_size != output_start:
                output.truncate(output_start)
                output.seek(output_start)

            source_start = source_data_start + tensor["source_start"]
            source_end = source_data_start + tensor["source_end"]
            cursor = source_start
            while cursor < source_end:
                end = min(cursor + chunk_bytes, source_end) - 1
                payload = read_range(url, cursor, end)
                if len(payload) % 4:
                    raise RuntimeError(f"Unaligned F32 range for {tensor['source_key']}")
                output.write(f32_to_bf16(payload))
                output.flush()
                cursor = end + 1

            completed = tensor["output_end"]
            if index == len(tensors) or index % 10 == 0:
                print(
                    f"[{index}/{len(tensors)}] {completed / (1024 ** 3):.2f}/"
                    f"{output_bytes / (1024 ** 3):.2f} GiB {tensor['native_key']}",
                    flush=True,
                )

        final_size = output.seek(0, os.SEEK_END)
        expected_size = output_data_start + output_bytes
        if final_size != expected_size:
            raise RuntimeError(f"Output size mismatch: expected {expected_size}, got {final_size}")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{args.filename}"
    source_header, source_data_start = read_header(url)
    tensors, output_header, output_bytes = build_manifest(source_header, args.repo, args.revision)
    summary = {
        "repo": args.repo,
        "revision": args.revision,
        "tensor_count": len(tensors),
        "source_bytes": sum(t["source_end"] - t["source_start"] for t in tensors),
        "output_bytes": output_bytes,
        "output_gibibytes": output_bytes / (1024 ** 3),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.dry_run:
        convert(
            url,
            args.output,
            tensors,
            output_header,
            source_data_start,
            output_bytes,
            args.chunk_mib * 1024 * 1024,
        )


if __name__ == "__main__":
    main()

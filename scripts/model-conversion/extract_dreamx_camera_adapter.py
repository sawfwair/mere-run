#!/usr/bin/env python3
"""Range-extract DreamX camera-attention tensors without downloading the 24.5 GB model."""

import argparse
import json
import os
import re
import struct
import sys
import urllib.request
from pathlib import Path


DEFAULT_REPO = "GD-ML/DreamX-World-5B-Cam"
DEFAULT_REVISION = "a4379c7723f6ebd02139e2e8fd62d6ef523e86e3"


def request(url, byte_range=None):
    headers = {"User-Agent": "mere.run-dreamx-camera-extractor/1"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def read_json(url):
    with request(url) as response:
        return json.load(response)


def read_range(url, start, end):
    with request(url, (start, end)) as response:
        if response.status != 206:
            raise RuntimeError(f"Server ignored byte range {start}-{end}: HTTP {response.status}")
        payload = response.read()
    expected = end - start + 1
    if len(payload) != expected:
        raise RuntimeError(f"Short range response: expected {expected}, received {len(payload)}")
    return payload


def read_safetensors_header(url):
    header_size = struct.unpack("<Q", read_range(url, 0, 7))[0]
    header = json.loads(read_range(url, 8, 7 + header_size).decode("utf-8"))
    return header, 8 + header_size


def selected_tensors(repo, revision, pattern):
    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    index = read_json(f"{base}/diffusion_pytorch_model.safetensors.index.json")
    selected_keys = sorted(key for key in index["weight_map"] if pattern.search(key))
    if not selected_keys:
        raise RuntimeError(f"No tensors matched {pattern.pattern!r}")
    shard_names = sorted({index["weight_map"][key] for key in selected_keys})
    tensors = []
    for shard_name in shard_names:
        shard_url = f"{base}/{shard_name}"
        header, data_start = read_safetensors_header(shard_url)
        for key in selected_keys:
            if index["weight_map"][key] != shard_name:
                continue
            entry = header.get(key)
            if entry is None:
                raise RuntimeError(f"Index key missing from shard header: {key}")
            start, end = entry["data_offsets"]
            tensors.append({
                "key": key,
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "url": shard_url,
                "shard": shard_name,
                "start": data_start + start,
                "end": data_start + end,
                "size": end - start,
            })
    tensors.sort(key=lambda item: (item["shard"], item["start"]))
    return tensors


def contiguous_groups(tensors):
    groups = []
    for tensor in tensors:
        if groups and groups[-1][-1]["shard"] == tensor["shard"] and groups[-1][-1]["end"] == tensor["start"]:
            groups[-1].append(tensor)
        else:
            groups.append([tensor])
    return groups


def output_header(tensors, repo, revision):
    header = {
        "__metadata__": {
            "format": "pt",
            "source_repo": repo,
            "source_revision": revision,
            "selection": ".cam_self_attn.",
        }
    }
    offset = 0
    for tensor in tensors:
        header[tensor["key"]] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + tensor["size"]],
        }
        offset += tensor["size"]
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return encoded


def extract(tensors, destination, repo, revision):
    header = output_header(tensors, repo, revision)
    groups = contiguous_groups(tensors)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    total = sum(tensor["size"] for tensor in tensors)
    with temporary.open("wb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        for index, group in enumerate(groups, start=1):
            start = group[0]["start"]
            end = group[-1]["end"] - 1
            payload = read_range(group[0]["url"], start, end)
            output.write(payload)
            completed += len(payload)
            print(
                f"[{index}/{len(groups)}] {group[0]['shard']} "
                f"{completed / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GiB",
                file=sys.stderr,
                flush=True,
            )
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--match", default=r"\.cam_self_attn\.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tensors = selected_tensors(args.repo, args.revision, re.compile(args.match))
    groups = contiguous_groups(tensors)
    total = sum(tensor["size"] for tensor in tensors)
    summary = {
        "repo": args.repo,
        "revision": args.revision,
        "tensor_count": len(tensors),
        "range_count": len(groups),
        "bytes": total,
        "gibibytes": total / (1024 ** 3),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        extract(tensors, args.output, args.repo, args.revision)


if __name__ == "__main__":
    main()

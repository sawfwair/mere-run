#!/usr/bin/env python3
"""Create a complete Qwen3.8 Flash Next pack with automatic PLE placement metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any


FORMAT = "mere-run-q38-ple-safetensors-placement-v1"
MANIFEST = "MERERUN_PLE_STORE.json"
PLE_MARKER = ".ple.ple_embedding.ngram_embedding."
PACK_METADATA_OVERRIDES = {"README.md"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, output: Path) -> None:
    try:
        os.link(source, output)
    except OSError:
        shutil.copy2(source, output)


def validate_pack(source: Path, pack: Path) -> dict[str, Any]:
    source_files = sorted(path.name for path in source.iterdir() if path.is_file())
    pack_files = sorted(
        path.name
        for path in pack.iterdir()
        if path.is_file() and path.name not in {MANIFEST, "MERERUN_PLE_PACK.json"}
    )
    if source_files != pack_files:
        raise ValueError("Complete pack does not contain exactly the source checkpoint files")
    for filename in source_files:
        if filename in PACK_METADATA_OVERRIDES:
            continue
        source_path = source / filename
        pack_path = pack / filename
        source_stat = source_path.stat()
        pack_stat = pack_path.stat()
        if source_stat.st_size != pack_stat.st_size:
            raise ValueError(f"Pack file size differs: {filename}")
        if (source_stat.st_dev, source_stat.st_ino) != (pack_stat.st_dev, pack_stat.st_ino):
            if sha256_file(source_path) != sha256_file(pack_path):
                raise ValueError(f"Pack file digest differs: {filename}")

    index = json.loads((pack / "model.safetensors.index.json").read_text())
    manifest = json.loads((pack / MANIFEST).read_text())
    indexed_ple_files = sorted(
        {
            filename
            for key, filename in index["weight_map"].items()
            if PLE_MARKER in key
        }
    )
    manifest_files = sorted(value["path"] for value in manifest["files"])
    if manifest_files != indexed_ple_files:
        raise ValueError("Manifest does not exactly cover the indexed PLE files")
    for value in manifest["files"]:
        path = pack / value["path"]
        if path.stat().st_size != int(value["byte_count"]):
            raise ValueError(f"PLE file size differs: {value['path']}")
        if sha256_file(path) != value["sha256"]:
            raise ValueError(f"PLE file digest differs: {value['path']}")
    return {
        "status": "verified",
        "source_files": len(source_files),
        "weight_tensors": len(index["weight_map"]),
        "ple_files": len(manifest_files),
        "ple_bytes": sum(int(value["byte_count"]) for value in manifest["files"]),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        for source_file in source.iterdir():
            if source_file.is_file():
                link_or_copy(source_file, staging / source_file.name)

        index_path = source / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        ple_files = sorted(
            {
                filename
                for key, filename in index["weight_map"].items()
                if PLE_MARKER in key
            }
        )
        files = []
        for position, filename in enumerate(ple_files, start=1):
            path = source / filename
            files.append(
                {
                    "path": filename,
                    "byte_count": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            print(f"hashed PLE file {position}/{len(ple_files)}: {filename}", flush=True)
        manifest = {
            "version": 1,
            "format": FORMAT,
            "artifact_id": args.artifact_id,
            "index": "model.safetensors.index.json",
            "preferred_placement": "internal_cache",
            "files": files,
            "source": {
                "repo_id": args.source_repo,
                "revision": args.source_revision,
                "index_sha256": sha256_file(index_path),
            },
        }
        (staging / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt = {
            "format": FORMAT,
            "generator": "scripts/model-conversion/prepare_qwen38_flash_next_ple_pack.py",
            "source_repo": args.source_repo,
            "source_revision": args.source_revision,
            "source_index_sha256": manifest["source"]["index_sha256"],
            "source_file_count": sum(1 for path in source.iterdir() if path.is_file()),
            "weight_tensor_count": len(index["weight_map"]),
            "ple_file_count": len(files),
            "ple_bytes": sum(value["byte_count"] for value in files),
            "manifest_sha256": sha256_file(staging / MANIFEST),
        }
        (staging / "MERERUN_PLE_PACK.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validation = validate_pack(source, staging)
        receipt["validation"] = validation
        (staging / "MERERUN_PLE_PACK.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-repo")
    parser.add_argument("--source-revision")
    parser.add_argument("--artifact-id")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate is None:
        missing = [
            option
            for option, value in [
                ("--output", args.output),
                ("--source-repo", args.source_repo),
                ("--source-revision", args.source_revision),
                ("--artifact-id", args.artifact_id),
            ]
            if value is None
        ]
        if missing:
            parser.error(f"build mode requires: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    if args.validate is not None:
        result = validate_pack(
            args.source.expanduser().resolve(),
            args.validate.expanduser().resolve(),
        )
    else:
        result = build(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "jinja2==3.1.6",
#   "mlx==0.32.2; sys_platform == 'darwin'",
#   "mlx[cuda]==0.32.2; sys_platform == 'linux'",
#   "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb",
# ]
# ///
"""Fail-fast text and vision qualification for a local Qwen4-Exp artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time
from typing import Any

import mlx.core as mx
from mlx_vlm import apply_chat_template, generate, load
from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model


MLX_VLM_REVISION = "6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb"
RAW_PLE_RE = re.compile(
    r"^(?P<prefix>.+\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<index>\d+))\.weight$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.png")):
        digest = sha256_file(path)
        if digest in result:
            raise ValueError(f"Duplicate fixture SHA-256 {digest}: {result[digest]} and {path}")
        result[digest] = path
    return result


def load_cases(manifests: list[Path], fixtures: dict[str, Path]) -> list[dict[str, Any]]:
    cases = []
    seen = set()
    for manifest in manifests:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        for source in value["cases"]:
            case = dict(source)
            identity = (manifest.name, case["id"])
            if identity in seen:
                raise ValueError(f"Duplicate qualification case: {identity}")
            seen.add(identity)
            if "image_sha256" in case:
                try:
                    case["resolved_image"] = str(fixtures[case["image_sha256"]])
                except KeyError as error:
                    raise FileNotFoundError(
                        f"Fixture {case['image_sha256']} for {identity} is missing"
                    ) from error
            case["manifest"] = str(manifest)
            cases.append(case)
    return cases


def prepare_model_view(model: Path, view: Path) -> Path:
    marker = view / "EXTERNAL_PLE.json"
    if marker.is_file():
        return view
    if view.exists() and any(view.iterdir()):
        raise ValueError(f"External-PLE view is incomplete or unrecognized: {view}")
    try:
        prepare_external_ple_model(model, view, cache_rows=0)
    except ValueError as error:
        if str(error) != "checkpoint contains no Qwen4-Exp PLE shards":
            raise
        prepare_raw_key_external_ple_model(model, view)
    return view


def prepare_raw_key_external_ple_model(model: Path, view: Path) -> None:
    """Build upstream's mmap view for raw Qwen ``shard_N`` tensor keys.

    Current mlx-vlm sanitizes these keys only after reading the checkpoint, but
    its external-PLE preparer scans the pre-sanitized index for ``shards.N``.
    This compatibility path emits the same manifest schema from the original
    safetensors byte ranges and does not copy the PLE payload.
    """

    view.mkdir(parents=True, exist_ok=True)
    if any(view.iterdir()):
        raise ValueError(f"External-PLE fallback target is not empty: {view}")
    if model.stat().st_dev != view.stat().st_dev:
        raise ValueError("Model and external-PLE view must share a filesystem")

    source_index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    weight_map = source_index["weight_map"]
    prefixes = []
    for key in weight_map:
        match = RAW_PLE_RE.match(key)
        if match:
            prefixes.append((int(match.group("index")), match.group("prefix")))
    prefixes.sort()
    if len(prefixes) != 128 or [index for index, _prefix in prefixes] != list(range(128)):
        raise ValueError("Raw-key checkpoint must contain PLE shards 0 through 127")

    headers: dict[str, tuple[int, dict[str, Any]]] = {}

    def tensor_descriptor(key: str) -> dict[str, Any]:
        filename = weight_map[key]
        if filename not in headers:
            with (model / filename).open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                headers[filename] = (
                    8 + header_size,
                    json.loads(stream.read(header_size)),
                )
        payload_start, header = headers[filename]
        info = header[key]
        return {
            "file": filename,
            "offset": payload_start + int(info["data_offsets"][0]),
            "dtype": info["dtype"],
            "shape": info["shape"],
        }

    quantization = config.get("quantization_config") or config.get("quantization", {})
    manifest_quantization = {"bits": 4, "group_size": 32, "mode": "affine"}
    shards = []
    row_start = 0
    row_width = None
    for _index, prefix in prefixes:
        base, shard = prefix.split(".ngram_embedding.shard_", 1)
        canonical_prefix = f"{base}.ngram_embedding.shards.{shard}"
        parameters = quantization.get(
            prefix,
            quantization.get(canonical_prefix, quantization),
        )
        current = {
            key: parameters.get(key) for key in ("bits", "group_size", "mode")
        }
        if current != manifest_quantization:
            raise ValueError(f"PLE tensor {prefix!r} has unsupported quantization {current}")
        tensors = {
            name: tensor_descriptor(f"{prefix}.{name}")
            for name in ("weight", "scales", "biases")
        }
        if [tensors[name]["dtype"] for name in ("weight", "scales", "biases")] != [
            "U32",
            "BF16",
            "BF16",
        ]:
            raise ValueError(f"PLE tensor {prefix!r} has an invalid dtype layout")
        weight_shape = tensors["weight"]["shape"]
        current_width = int(weight_shape[1]) * 8
        row_count = int(weight_shape[0])
        groups = current_width // manifest_quantization["group_size"]
        if len(weight_shape) != 2 or current_width != 160:
            raise ValueError(f"PLE tensor {prefix!r} has an invalid weight shape")
        if any(tensors[name]["shape"] != [row_count, groups] for name in ("scales", "biases")):
            raise ValueError(f"PLE tensor {prefix!r} has an invalid scale/bias shape")
        if row_width is None:
            row_width = current_width
        elif row_width != current_width:
            raise ValueError("PLE shards do not share one row width")
        shards.append({"row_start": row_start, "row_count": row_count, **tensors})
        row_start += row_count

    manifest_path = view / "ple-store.json"
    manifest = {
        "version": 2,
        "source_root": os.path.relpath(model, view),
        "layout": "safetensors_ranges",
        "row_width": row_width,
        "row_count": row_start,
        "quantization": manifest_quantization,
        "cache_rows": 0,
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    raw_marker = ".ple.ple_embedding.ngram_embedding.shard_"
    resident_weight_map = {
        key: filename for key, filename in weight_map.items() if raw_marker not in key
    }
    resident_files = sorted(set(resident_weight_map.values()))
    for filename in resident_files:
        source = model / filename
        os.symlink(os.path.relpath(source, view), view / filename)
    target_index = dict(source_index)
    target_index["weight_map"] = resident_weight_map
    target_index.setdefault("metadata", {})["external_ple_bytes"] = str(
        sum(
            descriptor["shape"][0]
            * descriptor["shape"][1]
            * (4 if name == "weight" else 2)
            for shard in shards
            for name, descriptor in shard.items()
            if name in {"weight", "scales", "biases"}
        )
    )
    (view / "model.safetensors.index.json").write_text(
        json.dumps(target_index, indent=2) + "\n", encoding="utf-8"
    )

    for source_file in model.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name == "config.json" or source_file.suffix == ".safetensors":
            continue
        if source_file.name in {
            "model.safetensors.index.json",
            "MANIFEST.sha256",
            "MEMORY_CONTEXT_SWEEP.jsonl",
        }:
            continue
        shutil.copy2(source_file, view / source_file.name)

    config["text_config"]["ple_storage"] = {
        "manifest": manifest_path.name,
        "cache_rows": 0,
    }
    for field in ("quantization", "quantization_config"):
        parameters = config.get(field)
        if isinstance(parameters, dict):
            config[field] = {
                key: value for key, value in parameters.items() if raw_marker not in key
            }
    (view / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (view / "EXTERNAL_PLE.json").write_text(
        json.dumps(
            {
                "format": "qwen4_exp_external_ple_raw_keys_v1",
                "source_model": str(model),
                "source_index": "model.safetensors.index.json",
                "external_ple_manifest": manifest_path.name,
                "symlinked_weight_files": resident_files,
                "mlx_vlm_revision": MLX_VLM_REVISION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_case(model, processor, case: dict[str, Any]) -> dict[str, Any]:
    image = [case["resolved_image"]] if "resolved_image" in case else None
    prompt = apply_chat_template(
        processor,
        model.config,
        case["prompt"],
        num_images=1 if image else 0,
    )
    started = time.monotonic()
    response = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        max_tokens=int(case.get("maxTokens", 96)),
        temperature=0.0,
        verbose=False,
    )
    actual = response.text.strip()
    expected = case["expected"]
    return {
        "id": case["id"],
        "manifest": case["manifest"],
        "modality": "image" if image else "text",
        "prompt": case["prompt"],
        "image_sha256": case.get("image_sha256"),
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
        "elapsed_seconds": time.monotonic() - started,
        "prompt_tokens": response.prompt_tokens,
        "generation_tokens": response.generation_tokens,
        "prompt_tps": response.prompt_tps,
        "generation_tps": response.generation_tps,
        "peak_memory_gb": response.peak_memory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    ple = parser.add_mutually_exclusive_group(required=True)
    ple.add_argument("--external-ple-view", type=Path)
    ple.add_argument("--resident-ple", action="store_true")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--modality", choices=("image", "text"))
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    view_path = (
        model_path
        if args.resident_ple
        else prepare_model_view(
            model_path,
            args.external_ple_view.expanduser().resolve(),
        )
    )
    fixtures = fixture_index(args.fixtures.expanduser().resolve())
    manifests = [path.expanduser().resolve() for path in args.manifests]
    cases = load_cases(manifests, fixtures)
    available_cases = len(cases)
    if args.modality is not None:
        cases = [
            case
            for case in cases
            if ("image" if "resolved_image" in case else "text") == args.modality
        ]
    if args.case_limit is not None:
        if args.case_limit <= 0:
            raise ValueError("--case-limit must be positive")
        cases = cases[: args.case_limit]

    loaded_at = datetime.now(timezone.utc).isoformat()
    model, processor = load(str(view_path), lazy=False, strict=True)
    results = []
    for case in cases:
        result = run_case(model, processor, case)
        results.append(result)
        print(
            f"[{len(results)}/{len(cases)}] {result['id']} "
            f"{'PASS' if result['passed'] else 'FAIL'} actual={result['actual']!r}",
            flush=True,
        )
        if not result["passed"]:
            break

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "loaded_at": loaded_at,
        "model": str(model_path),
        "ple_mode": "resident" if args.resident_ple else "external",
        "external_ple_view": None if args.resident_ple else str(view_path),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in manifests
        ],
        "cases_available": available_cases,
        "cases_requested": len(cases),
        "cases_run": len(results),
        "passed": len(results) == len(cases) and all(result["passed"] for result in results),
        "peak_active_memory_bytes": mx.get_active_memory(),
        "peak_memory_bytes": mx.get_peak_memory(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

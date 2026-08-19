#!/usr/bin/env python3
"""Compose a complete MiniMax-H3 release from validated official-source pieces.

The large conditioner and VAEs are byte-identical between compact revisions.
This tool keeps their previously audited official-source bytes, overlays a new
CUDA-reproduced transformer plus Metal-exact AdaLN pack, and emits one complete
conversion receipt for the independent artifact validator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import validate_minimax_h3_official_artifact as validator


OVERLAY_OUTPUTS = validator.CACHE_FILES | {
    "adaln_cache.index.json",
    "transformer.safetensors",
}
REUSED_OUTPUTS = {
    "audio_vae.safetensors",
    "text_encoder.safetensors",
    "video_vae.safetensors",
}
OVERLAY_SUPPORT_FILES = {
    "config.json",
    "MODIFICATIONS.md",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = source.resolve()
    try:
        os.link(resolved, destination)
    except OSError:
        shutil.copy2(resolved, destination)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_base_root(base: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for source in base.iterdir():
        if not source.is_file():
            continue
        hardlink_or_copy(source, output / source.name)


def validated_output_entry(
    root: Path,
    outputs: dict[str, Any],
    filename: str,
    known_sha256: str | None = None,
) -> dict[str, Any]:
    entry = outputs.get(filename)
    require(isinstance(entry, dict), f"missing receipt output: {filename}")
    path = root / filename
    require(path.is_file(), f"missing output: {path}")
    require(entry.get("byte_count") == path.stat().st_size,
            f"receipt byte count differs for {filename}")
    actual_sha256 = known_sha256 or validator.sha256_file(path)
    require(entry.get("sha256") == actual_sha256,
            f"receipt hash differs for {filename}")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-repository", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument(
        "--base-transformer-precision",
        choices=("bf16", "q8"),
        help="Precision used only to validate the complete base bundle.",
    )
    parser.add_argument(
        "--transformer-precision",
        choices=("bf16", "q8"),
        required=True,
    )
    args = parser.parse_args()

    base = args.base.expanduser().resolve()
    overlay = args.overlay.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(base.is_dir(), f"missing base root: {base}")
    require(overlay.is_dir(), f"missing overlay root: {overlay}")
    require(not output.exists(), f"output already exists: {output}")

    base_hashes = validator.verify_sha256sums(base)
    validator.verify_source_manifest(base)
    validator.verify_transformer(
        base,
        args.base_transformer_precision or args.transformer_precision,
    )
    validator.verify_text_encoder(base)
    validator.verify_vaes_and_cache(base)

    base_receipt = load_json(base / "transformer.conversion.json")
    overlay_receipt = load_json(overlay / "transformer.conversion.json")
    require(base_receipt.get("source", {}).get("repository") ==
            validator.SOURCE_REPOSITORY, "base receipt has the wrong source")
    require(base_receipt.get("source", {}).get("revision") ==
            validator.SOURCE_REVISION, "base receipt has the wrong revision")
    require(base_receipt.get("source", {}).get("third_party_weight_inputs") == [],
            "base receipt contains third-party weight inputs")
    require(overlay_receipt.get("converter_version") == 5,
            "overlay was not built by converter version 5")
    require(overlay_receipt.get("source", {}).get("repository") ==
            validator.SOURCE_REPOSITORY, "overlay receipt has the wrong source")
    require(overlay_receipt.get("source", {}).get("revision") ==
            validator.SOURCE_REVISION, "overlay receipt has the wrong revision")
    require(overlay_receipt.get("source", {}).get("third_party_weight_inputs") == [],
            "overlay receipt contains third-party weight inputs")
    require(overlay_receipt.get("adaln_cache_pack", {}).get("evaluation_backend") ==
            "mlx-metal", "overlay does not contain a Metal-exact cache pack")

    base_outputs = base_receipt.get("outputs", {})
    overlay_outputs = overlay_receipt.get("outputs", {})
    require(isinstance(base_outputs, dict), "base receipt outputs must be an object")
    require(isinstance(overlay_outputs, dict), "overlay receipt outputs must be an object")
    require(set(overlay_outputs) == OVERLAY_OUTPUTS,
            "overlay output closure must contain only transformer and AdaLN files")

    validated_overlay_outputs = {
        filename: validated_output_entry(overlay, overlay_outputs, filename)
        for filename in sorted(OVERLAY_OUTPUTS)
    }

    copy_base_root(base, output)
    for filename in sorted(OVERLAY_OUTPUTS | OVERLAY_SUPPORT_FILES):
        if filename not in OVERLAY_OUTPUTS:
            require((overlay / filename).is_file(), f"missing overlay support file: {filename}")
        destination = output / filename
        destination.unlink(missing_ok=True)
        hardlink_or_copy(overlay / filename, destination)

    merged_outputs = {
        filename: validated_output_entry(
            base,
            base_outputs,
            filename,
            known_sha256=base_hashes[filename],
        )
        for filename in sorted(REUSED_OUTPUTS)
    }
    merged_outputs.update(validated_overlay_outputs)

    source_manifest_sha256 = validator.sha256_file(output / "SOURCE_MANIFEST.json")
    merged_receipt = dict(overlay_receipt)
    merged_receipt["source"] = dict(overlay_receipt["source"])
    merged_receipt["source"]["source_manifest_sha256"] = source_manifest_sha256
    merged_receipt["outputs"] = merged_outputs
    merged_receipt["reused_official_source_components"] = {
        "repository": args.base_repository,
        "revision": args.base_revision,
        "source_manifest_sha256": source_manifest_sha256,
        "conversion_receipt_sha256": base_hashes["transformer.conversion.json"],
        "outputs": sorted(REUSED_OUTPUTS),
    }
    validator.require(
        set(merged_outputs) ==
        (validator.RUNTIME_FILES - {
            "LICENSE",
            "MODIFICATIONS.md",
            "NOTICE",
            "SHA256SUMS",
            "SOURCE_MANIFEST.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "transformer.conversion.json",
        }),
        "composed output receipt closure differs",
    )
    validator.require(
        merged_outputs["transformer.safetensors"].get("precision") ==
        args.transformer_precision,
        "overlay transformer precision differs",
    )

    conversion_path = output / "transformer.conversion.json"
    # The complete root is initially materialized with hard links. Break the
    # inherited link before rewriting either generated receipt so composing a
    # new immutable revision cannot mutate the audited base bundle.
    atomic_text(
        conversion_path,
        json.dumps(merged_receipt, indent=2, sort_keys=True) + "\n",
    )
    sums_path = output / "SHA256SUMS"
    lines = []
    for filename in sorted(validator.RUNTIME_FILES - {"SHA256SUMS"}):
        digest = merged_outputs.get(filename, {}).get("sha256")
        if digest is None:
            digest = validator.sha256_file(output / filename)
        lines.append(f"{digest}  {filename}")
    atomic_text(sums_path, "\n".join(lines) + "\n")
    print(json.dumps({
        "status": "composed",
        "files": len(validator.RUNTIME_FILES),
        "bytes": sum((output / name).stat().st_size for name in validator.RUNTIME_FILES),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

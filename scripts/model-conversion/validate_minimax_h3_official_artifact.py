#!/usr/bin/env python3
"""Validate a MiniMax-H3 FL2VA bundle built by the official-source converter."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "MiniMaxAI/MiniMax-H3"
SOURCE_REVISION = "ec19cc6daf5d8add9417c18e86b6b58cc6c55027"
SOURCE_BYTES = 144_035_116_604
SOURCE_FILE_COUNT = 48
LICENSE_SHA256 = "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44"
LICENSE_BYTES = 17_604
EXPECTED_QUANTIZER_SELF_TEST = {
    "4": {
        "weight": "74bebb64dcbfbd2d77e121a4de4f154ce21a5a35f481f699b2df23b135656ec8",
        "scales": "22f7ad21bdcfec89ace95b550404bd3433de39214b81cdc38c775724125499dc",
        "biases": "09ebd55dbdba503aa1b594d6b6a5333be87bb9dd51caaaf4063903e08780d2a5",
    },
    "8": {
        "weight": "e7eabb58287772294ce23f376e9f65877545152b2613d172e6d33ec6bcb244c4",
        "scales": "d84bf16b039914838f39df1823f37d9bcd640ea1108a33a2b0e7e4aa42e7ed1c",
        "biases": "09ebd55dbdba503aa1b594d6b6a5333be87bb9dd51caaaf4063903e08780d2a5",
    },
}
EXPECTED_QKV_REORDER_SELF_TEST = {
    "head_count": 56,
    "head_dimension": 128,
    "layout": "global-qkv-slabs",
    "sha256": "11a9143204ee0defe33828d431cca9f051c2a221050ae8543b543cda5c7b7786",
}
RUNTIME_FILES = {
    "LICENSE",
    "MODIFICATIONS.md",
    "NOTICE",
    "SHA256SUMS",
    "SOURCE_MANIFEST.json",
    "adaln_cache.safetensors",
    "audio_vae.safetensors",
    "config.json",
    "text_encoder.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "transformer.conversion.json",
    "transformer.safetensors",
    "video_vae.safetensors",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        require(len(raw_length) == 8, f"{path.name} has no safetensors header")
        header_length = int.from_bytes(raw_length, "little")
        require(0 < header_length <= 64 * 1024 * 1024, f"{path.name} header is invalid")
        raw_header = handle.read(header_length)
    require(len(raw_header) == header_length, f"{path.name} header is truncated")
    header = json.loads(raw_header)
    metadata = header.pop("__metadata__", {})
    require(isinstance(metadata, dict), f"{path.name} metadata must be an object")
    spans = sorted(
        (int(entry["data_offsets"][0]), int(entry["data_offsets"][1]), key)
        for key, entry in header.items()
    )
    cursor = 0
    for start, end, key in spans:
        require(start == cursor and end >= start, f"{path.name}:{key} has invalid offsets")
        cursor = end
    require(
        8 + header_length + cursor == path.stat().st_size,
        f"{path.name} payload extent does not match file size",
    )
    return header, metadata


def verify_sha256sums(root: Path) -> dict[str, str]:
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    observed: dict[str, str] = {}
    for line in lines:
        digest, filename = line.split("  ", maxsplit=1)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"bad hash for {filename}")
        require(filename not in observed, f"duplicate SHA256SUMS entry: {filename}")
        observed[filename] = digest
    expected = RUNTIME_FILES - {"SHA256SUMS"}
    require(set(observed) == expected, "SHA256SUMS file set does not match the release contract")
    for filename, expected_digest in observed.items():
        actual = sha256_file(root / filename)
        require(actual == expected_digest, f"SHA-256 mismatch for {filename}")
    return observed


def verify_source_manifest(root: Path) -> None:
    manifest = load_json(root / "SOURCE_MANIFEST.json")
    require(manifest.get("repository") == SOURCE_REPOSITORY, "wrong source repository")
    require(manifest.get("revision") == SOURCE_REVISION, "wrong source revision")
    require(manifest.get("public") is True, "official source was not recorded as public")
    require(manifest.get("gated") is False, "official source was not recorded as ungated")
    files = manifest.get("files")
    require(isinstance(files, list) and len(files) == SOURCE_FILE_COUNT, "wrong source file count")
    require(sum(entry["byte_count"] for entry in files) == SOURCE_BYTES, "wrong source byte total")
    paths = [entry["path"] for entry in files]
    require(paths == sorted(set(paths)), "source paths must be unique and sorted")
    for entry in files:
        require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None, "bad source hash")
        lfs_hash = entry.get("hub_lfs_sha256")
        require(lfs_hash is None or lfs_hash == entry["sha256"], "source LFS hash disagrees")


def verify_transformer(root: Path) -> None:
    tensors, metadata = safetensors_header(root / "transformer.safetensors")
    require(len(tensors) == 844, "wrong compact transformer tensor count")
    require(metadata.get("source_repository") == SOURCE_REPOSITORY, "wrong transformer source")
    require(metadata.get("source_revision") == SOURCE_REVISION, "wrong transformer revision")
    require(metadata.get("quantization") == "affine 4-bit g64", "wrong transformer quantization")
    require(metadata.get("cache_covered_weights_omitted") == "true", "cache omission not declared")
    require(metadata.get("qkv_layout") == "global-qkv-slabs", "wrong transformer QKV layout")
    require(len([key for key in tensors if key.endswith(".scales")]) == 208, "wrong Q4 scale count")
    require(len([key for key in tensors if key.endswith(".biases")]) == 208, "wrong Q4 bias count")
    require("condition_proj.weight" in tensors, "dense condition projection is missing")
    require(tensors["condition_proj.weight"]["dtype"] == "BF16", "condition projection lost BF16")
    require(not any("adaln_proj" in key for key in tensors), "AdaLN weight survived compaction")
    require(not any(key.startswith("time_embedder.") for key in tensors), "time MLP survived compaction")
    require("rope.inv_freq" not in tensors, "recomputed RoPE tensor survived compaction")


def verify_text_encoder(root: Path) -> None:
    tensors, metadata = safetensors_header(root / "text_encoder.safetensors")
    require(len(tensors) == 1_780, "wrong conditioner tensor count")
    require(metadata.get("source_repository") == SOURCE_REPOSITORY, "wrong conditioner source")
    require(metadata.get("source_revision") == SOURCE_REVISION, "wrong conditioner revision")
    require(metadata.get("quantization") == "affine 8-bit g64", "wrong conditioner quantization")
    require(metadata.get("language_layers_retained") == "50", "wrong conditioner layer declaration")
    require(len([key for key in tensors if key.endswith(".scales")]) == 439, "wrong Q8 scale count")
    require(len([key for key in tensors if key.endswith(".biases")]) == 439, "wrong Q8 bias count")
    for key in tensors:
        match = re.match(r"model\.layers\.(\d+)\.", key)
        require(match is None or int(match.group(1)) < 50, f"unused language layer in {key}")
    require(not any(key.startswith("lm_head.") for key in tensors), "unused LM head survived")


def verify_vaes_and_cache(root: Path) -> None:
    video, video_metadata = safetensors_header(root / "video_vae.safetensors")
    require(len(video) == 562, "wrong video VAE tensor count")
    require({entry["dtype"] for entry in video.values()} == {"F16"}, "video VAE is not all FP16")
    require(video_metadata.get("source_repository") == SOURCE_REPOSITORY, "wrong video VAE source")

    audio, audio_metadata = safetensors_header(root / "audio_vae.safetensors")
    require(len(audio) == 917, "wrong audio VAE tensor count")
    require({entry["dtype"] for entry in audio.values()} == {"F32"}, "audio VAE is not all FP32")
    require(audio_metadata.get("source_repository") == SOURCE_REPOSITORY, "wrong audio VAE source")
    require(not any(key.endswith(".weight_g") or key.endswith(".weight_v") for key in audio),
            "audio weight normalization was not fully folded")

    cache, cache_metadata = safetensors_header(root / "adaln_cache.safetensors")
    require(len(cache) == 54, "wrong AdaLN cache tensor count")
    require(cache_metadata.get("schema_version") == "2", "wrong AdaLN cache schema")
    require(cache_metadata.get("source_repository") == SOURCE_REPOSITORY, "wrong cache source")
    require(len([key for key in cache if re.fullmatch(r"blocks\.\d+\.modulations", key)]) == 50,
            "wrong cached block count")


def verify_receipts(root: Path, hashes: dict[str, str], location: str) -> None:
    require((root / "LICENSE").stat().st_size == LICENSE_BYTES, "wrong license byte count")
    require(hashes["LICENSE"] == LICENSE_SHA256, "wrong license hash")
    config = load_json(root / "config.json")
    require(config.get("model_type") == "minimax_h3", "wrong config model type")
    require(config.get("partition") == "fl2va", "wrong config partition")
    require(config.get("quantization") == {"bits": 4, "group_size": 64, "mode": "affine"},
            "wrong transformer config quantization")
    require(config.get("text_encoder_quantization") ==
            {"bits": 8, "group_size": 64, "mode": "affine"},
            "wrong conditioner config quantization")

    receipt = load_json(root / "transformer.conversion.json")
    source = receipt.get("source", {})
    require(source.get("repository") == SOURCE_REPOSITORY, "wrong receipt source repository")
    require(source.get("revision") == SOURCE_REVISION, "wrong receipt source revision")
    require(source.get("third_party_weight_inputs") == [], "third-party weight input recorded")
    require(receipt.get("mlx_cuda_available") is True, "release conversion did not use MLX CUDA")
    require(receipt.get("declared_conversion_location") == location, "wrong conversion location")
    require(receipt.get("quantizer_self_test") == EXPECTED_QUANTIZER_SELF_TEST,
            "quantizer self-test receipt differs")
    require(receipt.get("qkv_reorder_self_test") == EXPECTED_QKV_REORDER_SELF_TEST,
            "QKV reorder self-test receipt differs")
    software = receipt.get("software", {})
    require(software.get("mlx") == "0.29.3", "wrong MLX version")
    require(software.get("mlx_cuda") == "0.29.3", "wrong MLX CUDA version")
    for filename, output in receipt.get("outputs", {}).items():
        require(filename in hashes, f"unhashed receipt output: {filename}")
        require(output.get("sha256") == hashes[filename], f"receipt hash differs for {filename}")
        require(output.get("byte_count") == (root / filename).stat().st_size,
                f"receipt byte count differs for {filename}")
    transformer = receipt.get("outputs", {}).get("transformer.safetensors", {})
    require(transformer.get("qkv_matrices_deinterleaved") == 52,
            "wrong transformer QKV reorder count")
    require(transformer.get("qkv_layout") == "global-qkv-slabs",
            "wrong transformer QKV receipt layout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--conversion-location", required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    missing = sorted(RUNTIME_FILES - {path.name for path in root.iterdir() if path.is_file()})
    require(not missing, f"missing release files: {missing}")

    hashes = verify_sha256sums(root)
    verify_source_manifest(root)
    verify_transformer(root)
    verify_text_encoder(root)
    verify_vaes_and_cache(root)
    verify_receipts(root, hashes, args.conversion_location)
    total = sum((root / filename).stat().st_size for filename in RUNTIME_FILES)
    print(json.dumps({"status": "verified", "files": len(RUNTIME_FILES), "bytes": total}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

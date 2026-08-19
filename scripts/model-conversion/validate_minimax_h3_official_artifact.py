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
ADALN_SOURCE_TENSOR_COUNT = 106
ADALN_SOURCE_TENSOR_BYTES = 26_142_079_488
ADALN_SOURCE_TENSOR_CLOSURE_SHA256 = (
    "e2ccc0cab72b9183a0347e3999f4559cdc315b7b363a5fe9196890dd315f5a40"
)
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
CACHE_SCHEDULES = (
    *((point_count, 12.0, 3.0) for point_count in (5, 9, 12, 16, 21, 31)),
    (5, 6.0, 3.0),
)


def cache_filename(point_count: int, video_shift: float, audio_shift: float) -> str:
    if (point_count, video_shift, audio_shift) == (31, 12.0, 3.0):
        return "adaln_cache.safetensors"

    def label(value: float) -> str:
        return str(int(value)) if value.is_integer() else str(value)

    return f"adaln_cache-p{point_count}-v{label(video_shift)}-a{label(audio_shift)}.safetensors"


CACHE_FILES = {
    cache_filename(point_count, video_shift, audio_shift)
    for point_count, video_shift, audio_shift in CACHE_SCHEDULES
}
RUNTIME_FILES = {
    "LICENSE",
    "MODIFICATIONS.md",
    "NOTICE",
    "SHA256SUMS",
    "SOURCE_MANIFEST.json",
    "adaln_cache.index.json",
    "audio_vae.safetensors",
    "config.json",
    "text_encoder.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "transformer.conversion.json",
    "transformer.safetensors",
    "video_vae.safetensors",
} | CACHE_FILES


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


def verify_transformer(root: Path, transformer_precision: str) -> None:
    transformer_bits = {"q4": 4, "q8": 8, "bf16": None}[transformer_precision]
    tensors, metadata = safetensors_header(root / "transformer.safetensors")
    expected_tensor_count = 428 if transformer_bits is None else 844
    require(len(tensors) == expected_tensor_count, "wrong compact transformer tensor count")
    require(metadata.get("source_repository") == SOURCE_REPOSITORY, "wrong transformer source")
    require(metadata.get("source_revision") == SOURCE_REVISION, "wrong transformer revision")
    expected_quantization = "none" if transformer_bits is None else f"affine {transformer_bits}-bit g64"
    require(metadata.get("precision") == transformer_precision, "wrong transformer precision")
    require(metadata.get("quantization") == expected_quantization, "wrong transformer quantization")
    require(metadata.get("cache_covered_weights_omitted") == "true", "cache omission not declared")
    require(metadata.get("qkv_layout") == "global-qkv-slabs", "wrong transformer QKV layout")
    expected_quantized_linears = 0 if transformer_bits is None else 208
    require(len([key for key in tensors if key.endswith(".scales")]) == expected_quantized_linears,
            "wrong transformer scale count")
    require(len([key for key in tensors if key.endswith(".biases")]) == expected_quantized_linears,
            "wrong transformer bias count")
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

    pack = load_json(root / "adaln_cache.index.json")
    require(pack.get("schema_version") == 1, "wrong AdaLN cache-pack schema")
    require(pack.get("format") == "mere.run.minimax-h3-adaln-cache-pack",
            "wrong AdaLN cache-pack format")
    source_identity = pack.get("source_identity")
    require(isinstance(source_identity, str) and SOURCE_REVISION in source_identity,
            "wrong AdaLN cache-pack source identity")
    entries = pack.get("entries")
    require(isinstance(entries, list) and len(entries) == len(CACHE_SCHEDULES),
            "wrong AdaLN cache-pack entry count")
    expected_schedules = {
        (point_count, video_shift, audio_shift): cache_filename(
            point_count, video_shift, audio_shift
        )
        for point_count, video_shift, audio_shift in CACHE_SCHEDULES
    }
    observed_schedules: dict[tuple[int, float, float], str] = {}
    for entry in entries:
        schedule_value = entry.get("schedule", {})
        geometry = (
            schedule_value.get("point_count"),
            schedule_value.get("video_flow_shift"),
            schedule_value.get("audio_flow_shift"),
        )
        filename = entry.get("filename")
        require(geometry in expected_schedules, f"unexpected AdaLN schedule: {geometry}")
        require(filename == expected_schedules[geometry], f"wrong filename for {geometry}")
        require(geometry not in observed_schedules, f"duplicate AdaLN schedule: {geometry}")
        observed_schedules[geometry] = filename
        cache_path = root / filename
        require(entry.get("byte_count") == cache_path.stat().st_size,
                f"cache-pack byte count differs for {filename}")
        require(entry.get("sha256") == sha256_file(cache_path),
                f"cache-pack SHA-256 differs for {filename}")

        cache, cache_metadata = safetensors_header(cache_path)
        steps = geometry[0] - 1
        require(len(cache) == 54, f"wrong AdaLN tensor count for {filename}")
        require(cache_metadata.get("schema_version") == "2",
                f"wrong AdaLN schema for {filename}")
        require(cache_metadata.get("format") == "mere.run.minimax-h3-adaln-cache",
                f"wrong AdaLN format for {filename}")
        require(cache_metadata.get("source_identity") == source_identity,
                f"wrong AdaLN source identity for {filename}")
        require(cache_metadata.get("source_repository") in (None, SOURCE_REPOSITORY),
                f"wrong AdaLN source for {filename}")
        require(cache["time_embeddings"]["shape"] == [steps, 3, 2_688],
                f"wrong time embedding geometry for {filename}")
        require(cache["final_modulations"]["shape"] == [steps, 3, 10_752],
                f"wrong final modulation geometry for {filename}")
        require(cache["video_sigmas"]["shape"] == [geometry[0]],
                f"wrong video sigma geometry for {filename}")
        require(cache["audio_sigmas"]["shape"] == [geometry[0]],
                f"wrong audio sigma geometry for {filename}")
        block_keys = [key for key in cache if re.fullmatch(r"blocks\.\d+\.modulations", key)]
        require(len(block_keys) == 50, f"wrong block count for {filename}")
        require(all(cache[key]["shape"] == [steps, 9, 32_256] for key in block_keys),
                f"wrong block modulation geometry for {filename}")
    require(observed_schedules == expected_schedules, "AdaLN production schedule closure differs")


def verify_receipts(
    root: Path,
    hashes: dict[str, str],
    location: str,
    transformer_precision: str,
) -> None:
    transformer_bits = {"q4": 4, "q8": 8, "bf16": None}[transformer_precision]
    require((root / "LICENSE").stat().st_size == LICENSE_BYTES, "wrong license byte count")
    require(hashes["LICENSE"] == LICENSE_SHA256, "wrong license hash")
    config = load_json(root / "config.json")
    require(config.get("model_type") == "minimax_h3", "wrong config model type")
    require(config.get("partition") == "fl2va", "wrong config partition")
    expected_quantization = None if transformer_bits is None else {
        "bits": transformer_bits,
        "group_size": 64,
        "mode": "affine",
    }
    require(config.get("quantization") == expected_quantization,
            "wrong transformer config quantization")
    require(config.get("text_encoder_quantization") ==
            {"bits": 8, "group_size": 64, "mode": "affine"},
            "wrong conditioner config quantization")

    receipt = load_json(root / "transformer.conversion.json")
    require(receipt.get("converter_version") == 5,
            "wrong converter version; CUDA builds must import a Metal-exact AdaLN pack")
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
    cache_receipt = receipt.get("adaln_cache_pack", {})
    require(cache_receipt.get("schema_version") == 1,
            "wrong Metal AdaLN receipt schema")
    require(cache_receipt.get("format") ==
            "mere.run.minimax-h3-adaln-cache-pack-receipt",
            "wrong Metal AdaLN receipt format")
    require(cache_receipt.get("evaluation_backend") == "mlx-metal",
            "production AdaLN cache was not evaluated on MLX Metal")
    require(cache_receipt.get("generator") == "mere.run model optimize",
            "production AdaLN cache did not come from the Mere optimizer")
    require(cache_receipt.get("source_identity") ==
            "MiniMaxAI/MiniMax-H3@ec19cc6daf5d8add9417c18e86b6b58cc6c55027:"
            "FL2VA/transformer:index-sha256:"
            "fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb",
            "Metal AdaLN receipt has the wrong source identity")
    require(cache_receipt.get("source_tensor_count") == ADALN_SOURCE_TENSOR_COUNT,
            "Metal AdaLN receipt has the wrong source tensor count")
    require(cache_receipt.get("source_tensor_bytes") == ADALN_SOURCE_TENSOR_BYTES,
            "Metal AdaLN receipt has the wrong source tensor byte count")
    require(cache_receipt.get("source_tensor_closure_sha256") ==
            ADALN_SOURCE_TENSOR_CLOSURE_SHA256,
            "Metal AdaLN receipt has the wrong official-source tensor closure")
    require(cache_receipt.get("hardware", {}).get("chip") == "Apple M4 Max",
            "Metal AdaLN receipt has the wrong evaluation hardware")
    parity = cache_receipt.get("real_generation_parity", [])
    require({item.get("point_count") for item in parity} == {9, 21},
            "Metal AdaLN receipt lacks 9- and 21-point parity")
    require(all(item.get("full_mp4_sha256") == item.get("compact_mp4_sha256")
                for item in parity),
            "Metal AdaLN receipt records non-identical output")
    software = receipt.get("software", {})
    require(software.get("mlx") == "0.29.3", "wrong MLX version")
    require(software.get("mlx_cuda") == "0.29.3", "wrong MLX CUDA version")
    for filename, output in receipt.get("outputs", {}).items():
        require(filename in hashes, f"unhashed receipt output: {filename}")
        require(output.get("sha256") == hashes[filename], f"receipt hash differs for {filename}")
        require(output.get("byte_count") == (root / filename).stat().st_size,
                f"receipt byte count differs for {filename}")
        if filename in CACHE_FILES or filename == "adaln_cache.index.json":
            require(output.get("evaluation_backend") == "mlx-metal",
                    f"{filename} is not marked as a Metal-evaluated output")
    expected_outputs = RUNTIME_FILES - {
        "LICENSE",
        "MODIFICATIONS.md",
        "NOTICE",
        "SHA256SUMS",
        "SOURCE_MANIFEST.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "transformer.conversion.json",
    }
    require(set(receipt.get("outputs", {})) == expected_outputs,
            "conversion output receipt closure differs")
    transformer = receipt.get("outputs", {}).get("transformer.safetensors", {})
    require(transformer.get("precision") == transformer_precision,
            "wrong transformer receipt precision")
    require(transformer.get("quantization") == expected_quantization,
            "wrong transformer receipt quantization")
    require(transformer.get("qkv_matrices_deinterleaved") == 52,
            "wrong transformer QKV reorder count")
    require(transformer.get("qkv_layout") == "global-qkv-slabs",
            "wrong transformer QKV receipt layout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--conversion-location", required=True)
    parser.add_argument(
        "--transformer-precision",
        choices=("q4", "q8", "bf16"),
        default="q4",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    missing = sorted(RUNTIME_FILES - {path.name for path in root.iterdir() if path.is_file()})
    require(not missing, f"missing release files: {missing}")

    hashes = verify_sha256sums(root)
    verify_source_manifest(root)
    verify_transformer(root, args.transformer_precision)
    verify_text_encoder(root)
    verify_vaes_and_cache(root)
    verify_receipts(root, hashes, args.conversion_location, args.transformer_precision)
    total = sum((root / filename).stat().st_size for filename in RUNTIME_FILES)
    print(json.dumps({"status": "verified", "files": len(RUNTIME_FILES), "bytes": total}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

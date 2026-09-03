#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "huggingface-hub==1.28.0",
#   "mlx==0.32.2; sys_platform == 'darwin'",
#   "numpy==2.5.2",
#   "safetensors==0.8.0",
# ]
# ///
"""Build the native MLX Q4 Cosmos3-Super text-to-image artifact.

The conversion is resumable and processes one transformer shard at a time.
The source revision and tensor inventory are immutable release inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any
from urllib.request import urlopen


SOURCE_REPOSITORY = "nvidia/Cosmos3-Super-Text2Image-4Step"
SOURCE_REVISION = "aa0d5a57b7b045d68daa60fbacd84ec723c7cb7b"
TARGET_REPOSITORY = "Sawfwair/Cosmos3-Super-Text2Image-4Step-MLX-4bit"
SOURCE_TRANSFORMER_BYTES = 127_997_188_608
SOURCE_TENSOR_COUNT = 1_425
SOURCE_SHARD_COUNT = 27
BITS = 4
GROUP_SIZE = 64
MODE = "affine"
STATE_VERSION = 1
STATE_FILENAME = ".mererun-conversion-state.json"
LICENSE_URL = "https://openmdw.ai/license/1-1/"

SOURCE_PATTERNS = [
    "BIAS.md",
    "EXPLAINABILITY.md",
    "PRIVACY.md",
    "README.md",
    "SAFETY.md",
    "model_index.json",
    "modular_model_index.json",
    "scheduler/*",
    "text_tokenizer/*",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model*.safetensors*",
    "vae/*",
]

COPY_PATHS = [
    "BIAS.md",
    "EXPLAINABILITY.md",
    "PRIVACY.md",
    "SAFETY.md",
    "model_index.json",
    "modular_model_index.json",
    "scheduler",
    "text_tokenizer",
    "vae",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_environment() -> None:
    expected = {
        "huggingface-hub": "1.28.0",
        "mlx": "0.32.2",
        "numpy": "2.5.2",
        "safetensors": "0.8.0",
    }
    for package, version in expected.items():
        actual = importlib.metadata.version(package)
        if actual != version:
            raise RuntimeError(f"{package} {actual} is installed; expected {version}")


def download_source(destination: Path, workers: int) -> Path:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(
        repo_id=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        local_dir=destination,
        allow_patterns=SOURCE_PATTERNS,
        max_workers=workers,
    ))


def remote_manifest() -> dict[str, dict[str, int | str]]:
    from huggingface_hub import HfApi

    result: dict[str, dict[str, int | str]] = {}
    for entry in HfApi().list_repo_tree(
        repo_id=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        recursive=True,
        expand=True,
    ):
        path = getattr(entry, "path", None)
        if not isinstance(path, str):
            continue
        record: dict[str, int | str] = {"byte_count": int(getattr(entry, "size", 0) or 0)}
        lfs = getattr(entry, "lfs", None)
        digest = getattr(lfs, "sha256", None)
        if isinstance(digest, str):
            record["sha256"] = digest
        result[path] = record
    return result


def load_source_index(source: Path) -> dict[str, Any]:
    path = source / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise ValueError("Pinned transformer index is malformed")
    if len(weight_map) != SOURCE_TENSOR_COUNT:
        raise ValueError(f"Pinned transformer has {len(weight_map)} tensors; expected {SOURCE_TENSOR_COUNT}")
    if int(metadata.get("total_size", -1)) != SOURCE_TRANSFORMER_BYTES:
        raise ValueError("Pinned transformer total_size does not match the audited revision")
    shards = set(weight_map.values())
    if len(shards) != SOURCE_SHARD_COUNT:
        raise ValueError(f"Pinned transformer has {len(shards)} shards; expected {SOURCE_SHARD_COUNT}")
    return index


def omitted_tensor(key: str) -> bool:
    return key == "lm_head.weight" or key == "audio_modality_embed" or key.startswith("audio_proj_")


def should_quantize(key: str, value: Any) -> bool:
    if not key.endswith(".weight") or value.ndim != 2:
        return False
    if int(value.shape[-1]) % GROUP_SIZE:
        return False
    return key == "embed_tokens.weight" or key.startswith("layers.")


def transform_tensor(key: str, value: Any, mx: Any) -> dict[str, Any]:
    if omitted_tensor(key):
        return {}
    if not should_quantize(key, value):
        return {key: value}
    base = key.removesuffix(".weight")
    packed = mx.quantize(value, group_size=GROUP_SIZE, bits=BITS, mode=MODE)
    if len(packed) != 3:
        raise RuntimeError(f"MLX affine quantization emitted no biases for {key}")
    return {
        key: packed[0],
        f"{base}.scales": packed[1],
        f"{base}.biases": packed[2],
    }


def initial_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_VERSION,
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "quantization": {"bits": BITS, "group_size": GROUP_SIZE, "mode": MODE},
        "started_at": utc_now(),
        "completed_shards": {},
    }


def load_state(output: Path) -> dict[str, Any]:
    path = output / STATE_FILENAME
    if not path.exists():
        return initial_state()
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_VERSION:
        raise ValueError(f"Unsupported conversion state: {path}")
    if state.get("source") != initial_state()["source"]:
        raise ValueError(f"Conversion source mismatch: {path}")
    return state


def save_shard(path: Path, arrays: dict[str, Any], mx: Any) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    mx.eval(*arrays.values())
    mx.save_safetensors(str(temporary), arrays, metadata={
        "format": "mlx",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
    })
    os.replace(temporary, path)


def convert_shards(
    source: Path,
    output: Path,
    source_index: dict[str, Any],
    manifest: dict[str, dict[str, int | str]],
) -> tuple[dict[str, str], int]:
    import mlx.core as mx

    output_transformer = output / "transformer"
    output_transformer.mkdir(parents=True, exist_ok=True)
    state = load_state(output)
    source_weight_map: dict[str, str] = source_index["weight_map"]
    source_shards = sorted(set(source_weight_map.values()))

    for position, shard_name in enumerate(source_shards, start=1):
        destination = output_transformer / shard_name
        prior = state["completed_shards"].get(shard_name)
        if prior and destination.is_file() and destination.stat().st_size == prior.get("byte_count"):
            print(f"[{position}/{len(source_shards)}] resume {shard_name}", flush=True)
            continue

        source_path = source / "transformer" / shard_name
        remote = manifest.get(f"transformer/{shard_name}")
        if not remote:
            raise ValueError(f"Remote manifest is missing transformer/{shard_name}")
        if source_path.stat().st_size != int(remote["byte_count"]):
            raise ValueError(f"Downloaded shard size mismatch: {source_path}")
        source_digest = sha256_file(source_path)
        if remote.get("sha256") and source_digest != remote["sha256"]:
            raise ValueError(f"Downloaded shard SHA-256 mismatch: {source_path}")

        source_arrays = mx.load(str(source_path))
        converted: dict[str, Any] = {}
        quantized_modules = 0
        for key in sorted(source_arrays):
            arrays = transform_tensor(key, source_arrays[key], mx)
            if f"{key.removesuffix('.weight')}.scales" in arrays:
                quantized_modules += 1
            overlap = set(converted).intersection(arrays)
            if overlap:
                raise ValueError(f"Duplicate converted keys: {sorted(overlap)}")
            converted.update(arrays)
        save_shard(destination, converted, mx)
        state["completed_shards"][shard_name] = {
            "byte_count": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "source_byte_count": source_path.stat().st_size,
            "source_sha256": source_digest,
            "output_keys": sorted(converted),
            "quantized_modules": quantized_modules,
        }
        atomic_json(output / STATE_FILENAME, state)
        del converted, source_arrays
        mx.clear_cache()
        print(f"[{position}/{len(source_shards)}] converted {shard_name}", flush=True)

    weight_map: dict[str, str] = {}
    module_count = 0
    for shard_name in source_shards:
        record = state["completed_shards"].get(shard_name)
        if not record:
            raise ValueError(f"Conversion is incomplete: {shard_name}")
        module_count += int(record["quantized_modules"])
        for key in record["output_keys"]:
            if key in weight_map:
                raise ValueError(f"Duplicate output tensor: {key}")
            weight_map[key] = shard_name
    return dict(sorted(weight_map.items())), module_count


def logical_bytes(output: Path, weight_map: dict[str, str]) -> int:
    from safetensors import safe_open

    item_bytes = {
        "BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
    }
    total = 0
    for filename in sorted(set(weight_map.values())):
        with safe_open(output / "transformer" / filename, framework="numpy") as archive:
            actual = set(archive.keys())
            expected = {key for key, shard in weight_map.items() if shard == filename}
            if actual != expected:
                raise ValueError(f"Output inventory mismatch in {filename}")
            for key in actual:
                view = archive.get_slice(key)
                elements = 1
                for dimension in view.get_shape():
                    elements *= int(dimension)
                total += elements * item_bytes[view.get_dtype()]
    return total


def copy_metadata(source: Path, output: Path) -> None:
    for relative in COPY_PATHS:
        source_path = source / relative
        destination = output / relative
        if source_path.is_dir():
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
    shutil.copy2(source / "README.md", output / "UPSTREAM_MODEL_CARD.md")


def write_transformed_config(source: Path, output: Path) -> None:
    config = json.loads((source / "transformer" / "config.json").read_text(encoding="utf-8"))
    config.update({
        "hidden_act": "silu",
        "sound_gen": False,
        "mererun_text_to_image_only": True,
        "quantization": {"bits": BITS, "group_size": GROUP_SIZE, "mode": MODE},
        "quantization_config": {"bits": BITS, "group_size": GROUP_SIZE, "mode": MODE},
        "mererun_conversion": {
            "converter": "scripts/model-conversion/convert_cosmos3_super_t2i_mlx.py",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
    })
    atomic_json(output / "transformer" / "config.json", config)


def write_license(output: Path) -> None:
    with urlopen(LICENSE_URL, timeout=60) as response:
        contents = response.read()
    (output / "OPENMDW-1.1-LICENSE.html").write_bytes(contents)


def write_checksums(output: Path) -> None:
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", STATE_FILENAME}
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in files]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def remove_platform_metadata(output: Path) -> None:
    for path in output.rglob("*"):
        if path.is_file() and (path.name.startswith("._") or path.name == ".DS_Store"):
            path.unlink()


def model_card_path() -> Path:
    return Path(__file__).resolve().parent / "model-cards" / "cosmos3-super-t2i-4step-mlx-4bit.md"


def finalize(
    source: Path,
    output: Path,
    source_index: dict[str, Any],
    source_manifest: dict[str, dict[str, int | str]],
    weight_map: dict[str, str],
    module_count: int,
) -> None:
    copy_metadata(source, output)
    write_transformed_config(source, output)
    total_size = logical_bytes(output, weight_map)
    atomic_json(output / "transformer" / "diffusion_pytorch_model.safetensors.index.json", {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    })
    atomic_json(output / "SOURCE_MANIFEST.json", {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "transformer_index_metadata": source_index["metadata"],
        "files": source_manifest,
    })
    shutil.copy2(model_card_path(), output / "README.md")
    write_license(output)
    receipt = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "converter": "scripts/model-conversion/convert_cosmos3_super_t2i_mlx.py",
        "converter_sha256": sha256_file(Path(__file__).resolve()),
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "target_repository": TARGET_REPOSITORY,
        "quantization": {"bits": BITS, "group_size": GROUP_SIZE, "mode": MODE},
        "omitted_weights": ["lm_head.weight", "audio_modality_embed", "audio_proj_in.*", "audio_proj_out.*"],
        "transformer": {
            "logical_bytes": total_size,
            "tensor_count": len(weight_map),
            "quantized_module_count": module_count,
            "shard_count": len(set(weight_map.values())),
        },
        "software": {
            package: importlib.metadata.version(package)
            for package in ["huggingface-hub", "mlx", "numpy", "safetensors"]
        },
        "hardware": {"machine": platform.machine(), "platform": platform.platform()},
    }
    atomic_json(output / "CONVERSION.json", receipt)
    remove_platform_metadata(output)
    write_checksums(output)
    state_path = output / STATE_FILENAME
    if state_path.exists():
        state_path.unlink()
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def upload(output: Path, private: bool) -> None:
    from huggingface_hub import HfApi

    remove_platform_metadata(output)
    write_checksums(output)
    api = HfApi()
    api.create_repo(TARGET_REPOSITORY, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=TARGET_REPOSITORY,
        repo_type="model",
        folder_path=output,
        commit_message="Add native MLX Q4 Cosmos3-Super text-to-image checkpoint",
        ignore_patterns=["._*", "**/._*", ".DS_Store", "**/.DS_Store", STATE_FILENAME],
    )


def self_test() -> None:
    import mlx.core as mx

    source = mx.arange(256, dtype=mx.float32).reshape(4, 64) / 255
    converted = transform_tensor("layers.0.mlp.up_proj.weight", source, mx)
    expected = {
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.up_proj.scales",
        "layers.0.mlp.up_proj.biases",
    }
    if set(converted) != expected:
        raise AssertionError(f"Q4 self-test emitted {sorted(converted)}")
    restored = mx.dequantize(
        converted["layers.0.mlp.up_proj.weight"],
        converted["layers.0.mlp.up_proj.scales"],
        converted["layers.0.mlp.up_proj.biases"],
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
    )
    error = float(mx.max(mx.abs(restored - source)).item())
    if error <= 0 or error >= 0.05:
        raise AssertionError(f"Q4 self-test error is outside the expected range: {error}")
    if transform_tensor("lm_head.weight", source, mx):
        raise AssertionError("Language-model head was not omitted")
    if transform_tensor("audio_proj_in.weight", source, mx):
        raise AssertionError("Sound projection was not omitted")
    print(json.dumps({"q4_max_absolute_error": error, "status": "passed"}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pinned Cosmos3-Super Text2Image 4-Step weights to native MLX Q4."
    )
    parser.add_argument("--workspace", type=Path, help="Large-volume conversion workspace.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent Hugging Face downloads.")
    parser.add_argument("--upload", action="store_true", help=f"Upload to {TARGET_REPOSITORY} after validation.")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help=f"Upload an already completed artifact to {TARGET_REPOSITORY} without reconverting it.",
    )
    parser.add_argument("--public", action="store_true", help="Create the target repository as public; private is default.")
    parser.add_argument("--self-test", action="store_true", help="Run a small MLX quantizer contract test and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    verify_environment()
    if args.self_test:
        self_test()
        return 0
    if args.workspace is None:
        raise ValueError("--workspace is required unless --self-test is used")
    workspace = args.workspace.expanduser().resolve()
    source = workspace / "source"
    output = workspace / "Cosmos3-Super-Text2Image-4Step-MLX-4bit"
    if args.upload_only:
        required = [output / "CONVERSION.json", output / "SHA256SUMS", output / "README.md"]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"Completed artifact is missing required files: {missing}")
        upload(output, private=not args.public)
        print(f"uploaded: https://huggingface.co/{TARGET_REPOSITORY}", flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest = remote_manifest()
    source = download_source(source, args.workers)
    source_index = load_source_index(source)
    weight_map, module_count = convert_shards(source, output, source_index, manifest)
    finalize(source, output, source_index, manifest, weight_map, module_count)
    if args.upload:
        upload(output, private=not args.public)
    print(f"complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

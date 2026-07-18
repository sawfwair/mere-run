#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.14.4"
# dependencies = [
#   "huggingface-hub==1.24.0",
#   "mlx==0.32.0",
#   "mlx-lm==0.31.3",
#   "mlx-vlm==0.6.5",
#   "mlx-audio==0.4.4",
#   "numpy==2.5.1",
#   "safetensors==0.8.0",
#   "transformers==5.14.1",
# ]
# ///

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


SOURCE_REPO = "google/gemma-4-12B-it"
SOURCE_REVISION = "12ace6d648d72bd41519e140f1185f34d38c7e3d"
SOURCE_WEIGHT = {
    "filename": "model.safetensors",
    "byte_count": 23_919_549_408,
    "sha256": "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d",
}
PINNED_METADATA = {
    "config.json": {
        "byte_count": 4_423,
        "sha256": "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9",
    },
    "chat_template.jinja": {
        "byte_count": 18_683,
        "sha256": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    },
    "generation_config.json": {
        "byte_count": 260,
        "sha256": "a8349d9bd64cc5841297fcb5002f0fdc4749c473c8f1b10ea337f9ce4ee7014e",
    },
    "processor_config.json": {
        "byte_count": 1_382,
        "sha256": "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9",
    },
    "tokenizer_config.json": {
        "byte_count": 2_102,
        "sha256": "09e6222fd7049dae603d0f61a8b10af5a813f22dca10e6cdc6918f76d7661104",
    },
    "tokenizer.json": {
        "byte_count": 32_169_626,
        "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    },
}
TOOL_VERSIONS = {
    "huggingface-hub": "1.24.0",
    "mlx": "0.32.0",
    "mlx-lm": "0.31.3",
    "mlx-vlm": "0.6.5",
    "mlx-audio": "0.4.4",
    "numpy": "2.5.1",
    "safetensors": "0.8.0",
    "transformers": "5.14.1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, pin: dict[str, int | str]) -> None:
    byte_count = path.stat().st_size
    if byte_count != pin["byte_count"]:
        raise ValueError(
            f"{path} has {byte_count} bytes; expected {pin['byte_count']}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != pin["sha256"]:
        raise ValueError(
            f"{path} has SHA-256 {actual_sha256}; expected {pin['sha256']}"
        )


def verify_environment() -> None:
    for package, expected in TOOL_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} {actual} is installed; expected {expected}")


def download_pinned_metadata(destination: Path) -> None:
    for filename, pin in PINNED_METADATA.items():
        url = (
            f"https://huggingface.co/{SOURCE_REPO}/resolve/"
            f"{SOURCE_REVISION}/{filename}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "mere.run-converter/1"})
        with urllib.request.urlopen(request) as response:
            payload = response.read()
        path = destination / filename
        path.write_bytes(payload)
        verify_file(path, pin)


def stage_source(source: Path, destination: Path) -> None:
    excluded = {".cache", "mererun_model.json", *PINNED_METADATA}
    for path in source.iterdir():
        if path.name in excluded:
            continue
        os.symlink(path.resolve(), destination / path.name)
    download_pinned_metadata(destination)


def restore_converted_metadata(source: Path, output: Path) -> None:
    for filename in PINNED_METADATA:
        if filename != "config.json":
            shutil.copyfile(source / filename, output / filename)


def verify_output(output: Path) -> list[dict[str, int | str]]:
    for filename, pin in PINNED_METADATA.items():
        if filename != "config.json":
            verify_file(output / filename, pin)

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config["text_config"]["max_position_embeddings"] != 262_144:
        raise ValueError("Converted config lost the pinned 262144-token context")
    quantization = config.get("quantization")
    if quantization != {"group_size": 64, "bits": 4, "mode": "affine"}:
        raise ValueError(f"Unexpected converted quantization metadata: {quantization}")

    weights = sorted(output.glob("*.safetensors"))
    if not weights:
        raise ValueError("Conversion emitted no safetensors weights")
    return [
        {
            "filename": path.name,
            "byte_count": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in weights
    ]


def write_provenance(output: Path, artifacts: list[dict[str, int | str]]) -> None:
    provenance = {
        "converter": "scripts/model-conversion/convert_gemma4_12b_mlx.py",
        "converter_version": 1,
        "source": {
            "repository": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            **SOURCE_WEIGHT,
        },
        "metadata": PINNED_METADATA,
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "dtype": "bfloat16",
        },
        "tools": TOOL_VERSIONS,
        "artifacts": artifacts,
    }
    path = output / "MERERUN_CONVERSION.json"
    path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the pinned Google Gemma 4 12B-it checkpoint to MLX affine 4-bit."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output path already exists: {output}")

    verify_environment()
    verify_file(source / SOURCE_WEIGHT["filename"], SOURCE_WEIGHT)

    from mlx_vlm.convert import convert

    with tempfile.TemporaryDirectory(prefix="mererun-gemma4-convert-") as temporary:
        staged_source = Path(temporary) / "source"
        staged_source.mkdir()
        stage_source(source, staged_source)
        convert(
            hf_path=str(staged_source),
            mlx_path=str(output),
            quantize=True,
            q_group_size=64,
            q_bits=4,
            q_mode="affine",
            dtype="bfloat16",
        )
        restore_converted_metadata(staged_source, output)

    artifacts = verify_output(output)
    write_provenance(output, artifacts)
    print(output)


if __name__ == "__main__":
    main()

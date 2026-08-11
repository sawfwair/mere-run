#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = [
#   "mlx==0.32.0",
#   "safetensors==0.8.0",
# ]
# ///

"""Convert NVIDIA's pinned Nemotron 3.5 Lightning DSpark companion to MLX."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

import mlx.core as mx

from nemotron35_conversion import (
    FilePin,
    sha256,
    tensor_dtypes,
    transform_modelopt_shard,
    verify_pins,
    write_json,
    write_model_card,
)


SOURCE_REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"
SOURCE_REVISION = "e3af76fbff445ef795958bee96bc1126af70fd57"
MODEL_ID = "text-chat-nemotron-35-lightning-dspark"
EXPECTED_SOURCE_TENSORS = 118
EXPECTED_NVFP4_PROJECTIONS = 20

PINS = {
    "model.safetensors": FilePin(1_349_057_504, "27e3225376a7d65e2b602a3909c8526217bdc7ad29a2310f5f8b66f7d5343d13"),
    "config.json": FilePin(1_995, "e390ae49aaf2137b4bf3009c229cba852c44032163bb151b20bbc2376d3a6125"),
    "hf_quant_config.json": FilePin(280, "5b4eea93e73f8b14ac5a8ec98369c8100fb9911e25fe030afe87a82d44e68887"),
    "LICENSE": FilePin(2_618, "8a0c5e232551abcb102d1bd39a34866ef4520361b613bd6405d55b36562e4d88"),
    "README.md": FilePin(13_786, "5d74007da17d68602c25b9ceed3d6cb434550233eba7722ec6fa11461aeeda2d"),
}


def verify_environment() -> None:
    actual = importlib.metadata.version("mlx")
    if actual != "0.32.0":
        raise RuntimeError(f"mlx {actual} is installed; expected 0.32.0")


def converted_config(source: Path) -> dict:
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["Qwen3DSparkModel"]:
        raise ValueError("Pinned companion is not Qwen3DSparkModel")
    if config.get("target_layer_ids") != [1, 5, 19, 29, 41, 51]:
        raise ValueError("Pinned companion target-layer contract changed")
    if config.get("block_size") != 8 or config.get("dspark_markov_rank") != 512:
        raise ValueError("Pinned DSpark block or Markov contract changed")
    config.pop("quantization_config", None)
    config["quantization"] = {
        "bits": 4,
        "group_size": 16,
        "mode": "nvfp4",
        "global_scale": True,
    }
    config["mlx_conversion"] = {
        "source_format": "modelopt-nvfp4",
        "nvfp4": "bit-exact-repack-with-retained-global-scale",
    }
    return config


def convert(source: Path, output: Path) -> None:
    verify_pins(source, PINS)
    source_weights = source / "model.safetensors"
    arrays = mx.load(str(source_weights))
    dtypes = tensor_dtypes(source_weights)
    if len(arrays) != EXPECTED_SOURCE_TENSORS or set(arrays) != set(dtypes):
        raise ValueError("Pinned DSpark tensor inventory changed")
    converted, stats = transform_modelopt_shard(
        arrays,
        dtypes,
        expected_experts=None,
    )
    if stats["nvfp4_repacked"] != EXPECTED_NVFP4_PROJECTIONS:
        raise ValueError(f"Repacked {stats['nvfp4_repacked']} DSpark projections")
    if stats["fp8_materialized"] != 0 or len(converted) != EXPECTED_SOURCE_TENSORS:
        raise ValueError("Unexpected DSpark conversion inventory")

    output_arrays = dict(converted)
    mx.eval(*output_arrays.values())
    weights = output / "model.safetensors"
    mx.save_safetensors(str(weights), output_arrays, metadata={"format": "mlx"})
    write_json(output / "config.json", converted_config(source))
    for filename in ("LICENSE", "hf_quant_config.json"):
        shutil.copyfile(source / filename, output / filename)
    write_model_card(
        source / "README.md",
        output / "README.md",
        f"""
# MLX DSpark companion for mere.run

This repository is a deterministic Apple Silicon MLX conversion of
[`{SOURCE_REPOSITORY}`](https://huggingface.co/{SOURCE_REPOSITORY}) at revision
`{SOURCE_REVISION}`. It is a speculative-decoding companion for
[`text-chat-nemotron-35-lightning`](https://huggingface.co/Sawfwair/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-MLX), not a standalone chat model.

The released ModelOpt NVFP4 nibbles are repacked bit-for-bit into MLX's native
NVFP4 container while retaining block and global scales. The native mere.run
verifier uses NVIDIA's recommended three-token proposal width and falls back to
serial target decoding when measured draft acceptance is below break-even.
`MERERUN_CONVERSION.json` records pinned source and artifact hashes; the
reproducible converter is
`scripts/model-conversion/convert_nemotron35_dspark_mlx.py` in mere.run.

The original NVIDIA model card follows unchanged below. The bundled
OpenMDW-1.1 license remains applicable.
""",
    )
    write_json(
        output / "mererun_model.json",
        {
            "schemaVersion": 3,
            "id": MODEL_ID,
            "engine": "nemotron-h",
            "family": "nemotron",
            "tier": "small",
            "variant": "standard",
            "precision": "int4",
            "quantization": {
                "bits": 4,
                "groupSize": 16,
                "scheme": "modelopt-nvfp4-mlx",
            },
            "supports": [],
            "components": {"text_encoder": {"type": "local", "path": "."}},
            "upstreamRepoId": f"{SOURCE_REPOSITORY}@{SOURCE_REVISION}",
            "sources": [
                {
                    "role": "draft-model",
                    "repository": SOURCE_REPOSITORY,
                    "revision": SOURCE_REVISION,
                    "destination_path": ".",
                }
            ],
        },
    )
    write_json(
        output / "MERERUN_CONVERSION.json",
        {
            "converter": "scripts/model-conversion/convert_nemotron35_dspark_mlx.py",
            "converter_version": 1,
            "source": {
                "repository": SOURCE_REPOSITORY,
                "revision": SOURCE_REVISION,
                "files": {
                    name: {"byte_count": pin.byte_count, "sha256": pin.sha256}
                    for name, pin in PINS.items()
                },
                "tensor_count": EXPECTED_SOURCE_TENSORS,
            },
            "conversion": stats,
            "quantization": {
                "bits": 4,
                "group_size": 16,
                "mode": "nvfp4",
                "global_scale_retained": True,
                "requantized": False,
            },
            "tools": {"mlx": "0.32.0"},
            "artifacts": [
                {
                    "filename": weights.name,
                    "byte_count": weights.stat().st_size,
                    "sha256": sha256(weights),
                },
                {
                    "filename": "config.json",
                    "byte_count": (output / "config.json").stat().st_size,
                    "sha256": sha256(output / "config.json"),
                },
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    verify_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        convert(source, temporary)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

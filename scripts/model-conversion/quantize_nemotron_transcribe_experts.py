#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = ["mlx==0.32.0", "safetensors==0.8.0"]
# ///
"""Convert the pinned BF16 Transcribe checkpoint to MLX affine W4A16 routed experts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import struct
import tempfile

import mlx.core as mx

from nemotron35_conversion import ShardWriter, sha256, write_index, write_json

REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
REVISION = "ef905575a106fdad904f3b0ce661dccad654607d"
SOURCE_BYTES = 67_496_013_734
SOURCE_SHA256 = "03142dde5176c1d4df21ee829642745a1d670ea4d79e1a4684c64d53195acff2"
FORMAT = {"version": 1, "bits": 4, "groupSize": 64, "mode": "affine",
          "scope": "routed-experts", "layout": "expert-output-input"}


def is_expert(key: str) -> bool:
    return key.startswith("llm.model.layers.") and ".mixer.experts." in key and key.endswith(
        (".gate_and_up_projs", ".down_projs"))


def convert(source: Path, output: Path) -> None:
    path = source / "model.safetensors"
    print("Verifying pinned BF16 source SHA-256...", flush=True)
    if path.stat().st_size != SOURCE_BYTES or sha256(path) != SOURCE_SHA256:
        raise ValueError("Source weights do not match the pinned NVIDIA BF16 checkpoint")
    inventory_path = Path(__file__).resolve().parents[2] / "Sources/AudioNemotronTranscribeModel/Resources/tensor-inventory.json"
    inventory = json.loads(inventory_path.read_text())
    arrays = mx.load(str(path))
    with path.open("rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    for key, spec in inventory.items():
        if list(arrays[key].shape) != spec["shape"] or header[key]["dtype"] != spec["dtype"]:
            raise ValueError(f"Source tensor does not match the pinned inventory: {key}")
    unexpected = set(arrays) - set(inventory)
    if any(not key.startswith("ctc_timestamp.") and not key.endswith("._extra_state") for key in unexpected):
        raise ValueError("Unexpected source inference tensors")
    for key in unexpected:
        del arrays[key]

    writer = ShardWriter(output)
    count = 0
    for key in sorted(inventory):
        value = arrays.pop(key)
        if is_expert(key):
            # NeMo uses [expert, input, output]; MLX QMM packs the last (input) dimension.
            packed, scales, biases = mx.quantize(value.transpose(0, 2, 1), group_size=64, bits=4, mode="affine")
            mx.eval(packed, scales, biases)
            writer.append(key + ".qweight", packed)
            writer.append(key + ".scales", scales)
            writer.append(key + ".biases", biases)
            count += 1
            print(f"Quantized routed projection {count}/46: {key}", flush=True)
        else:
            writer.append(key, value)
        del value
    if count != 46:
        raise ValueError(f"Expected 46 routed expert matrices, converted {count}")
    weight_map, shards, total_size = writer.finalize()
    write_index(output, weight_map, total_size)
    write_json(output / "expert-quantization.json", FORMAT)
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "THIRD_PARTY_NOTICES.md"]:
        shutil.copyfile(source / name, output / name)
    shutil.copytree(source / "licenses", output / "licenses")
    manifest = json.loads((source / "mererun_model.json").read_text())
    manifest.update(id=output.name, precision="int4",
                    quantization={"bits": 4, "groupSize": 64, "scheme": "mlx-affine-routed-experts"})
    write_json(output / "mererun_model.json", manifest)
    write_json(output / "MERERUN_CONVERSION.json", {
        "converter": "scripts/model-conversion/quantize_nemotron_transcribe_experts.py",
        "converter_version": 1, "mlx_version": importlib.metadata.version("mlx"),
        "source": {"repository": REPOSITORY, "revision": REVISION,
                   "weights_bytes": SOURCE_BYTES, "weights_sha256": SOURCE_SHA256},
        "quantization": FORMAT, "routed_matrices": count, "inference_tensors": len(inventory),
        "output_tensors": len(weight_map), "tensor_bytes": total_size,
        "artifacts": [{"filename": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in shards],
    })
    (output / "README.md").write_text(
        "# Nemotron Transcribe with 4-bit routed experts\n\n"
        f"Derived from {REPOSITORY}@{REVISION}. MLX affine 4-bit weights, group size 64, "
        "BF16 scales/biases, routed experts only. All other inference tensors retain their source dtype. "
        "MTP and CTC heads are not executed. The original OpenMDW-1.1 license applies.\n\n"
        "This is a local MLX conversion, distinct from NVIDIA's NVFP4 format. "
        "See MERERUN_CONVERSION.json for source and output hashes.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if importlib.metadata.version("mlx") != "0.32.0":
        raise RuntimeError("Use the pinned mlx==0.32.0 conversion environment")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        convert(source, temporary)
        # The manifest names the final artifact, rather than its atomic staging directory.
        manifest_path = temporary / "mererun_model.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["id"] = output.name
        write_json(manifest_path, manifest)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()

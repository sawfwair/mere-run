#!/usr/bin/env python3
"""Convert the pinned official SCAIL-2 checkpoint into the native MLX layout.

Requires Python 3.11+, PyTorch 2.4+, NumPy 1.26+, and safetensors. The
transformer checkpoint is opened with ``mmap=True`` and emitted in bounded
shards so conversion does not materialize both the 65 GB source checkpoint and
the complete BF16 result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file


MODEL_REPO = "zai-org/SCAIL-2"
MODEL_REVISION = "150cc0ca4e98e50e60b9295dacde39442fdccab2"
CODE_REPO = "zai-org/SCAIL-2"
CODE_REVISION = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
TRANSFORMER_PATH = Path("model/1/fsdp2_rank_0000_checkpoint.pt")
TEXT_ENCODER_PATH = Path("umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth")
CLIP_PATH = Path("models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth")
VAE_PATH = Path("Wan2.1_VAE.pth")
TOKENIZER_PATH = Path("umt5-xxl/tokenizer.json")
PINNED_SOURCE_FILES = {
    Path("README.md"): (
        3_310,
        "cade35f0cf5dfec66510b9ec0e99eaf5bb4424fcdeeeb344d937516756cae1d3",
    ),
    TRANSFORMER_PATH: (
        65_582_563_395,
        "d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f",
    ),
    TEXT_ENCODER_PATH: (
        11_361_920_418,
        "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
    ),
    CLIP_PATH: (
        2_528_485_611,
        "020daacf4c0ce94284df584d243cefb6ddfadefff8772226a8e85431df5de2da",
    ),
    VAE_PATH: (
        507_609_880,
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    ),
    TOKENIZER_PATH: (
        16_837_417,
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
    ),
}
MINIMUM_OUTPUT_FREE_BYTES = 86_000_000_000
PINNED_CODE_FILE_SHA256 = {
    Path("convert.py"): "0519f688643b0a2c3f13660f196e34280704ddb3e2575aa226e2d2f8f2b4397f",
    Path("LICENSE"): "7cfe2db2477ee1e98b57ea1d8655aa8c39416022bab61aa4303e09af05495797",
}
NUMPY_SAFE_GLOBALS = [
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    type(np.dtype(np.uint32)),
]


class ShardedSafetensorsWriter:
    def __init__(self, output_root: Path, prefix: str, maximum_bytes: int) -> None:
        self.output_root = output_root
        self.prefix = prefix
        self.maximum_bytes = maximum_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.temporary_files: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self.seen_keys: set[str] = set()
        self.total_bytes = 0

    def add(self, key: str, value: torch.Tensor) -> None:
        if key in self.seen_keys:
            raise ValueError(f"Duplicate converted tensor key: {key}")
        self.seen_keys.add(key)
        tensor = value.detach().cpu().contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + tensor_bytes > self.maximum_bytes:
            self._flush()
        self.pending[key] = tensor
        self.pending_bytes += tensor_bytes
        self.total_bytes += tensor_bytes

    def finish(self) -> Path:
        self._flush()
        shard_count = len(self.temporary_files)
        if shard_count == 0:
            raise ValueError(f"No tensors were written for {self.prefix}")
        renamed: dict[str, str] = {}
        for index, temporary in enumerate(self.temporary_files, start=1):
            final_name = f"{self.prefix}-{index:05d}-of-{shard_count:05d}.safetensors"
            final_path = self.output_root / final_name
            temporary.replace(final_path)
            renamed[temporary.name] = final_name
        final_weight_map = {
            key: renamed[filename] for key, filename in self.weight_map.items()
        }
        index_path = self.output_root / f"{self.prefix}.safetensors.index.json"
        index_path.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": self.total_bytes},
                    "weight_map": final_weight_map,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return index_path

    def _flush(self) -> None:
        if not self.pending:
            return
        shard_index = len(self.temporary_files) + 1
        filename = f"{self.prefix}-{shard_index:05d}.safetensors"
        path = self.output_root / filename
        save_file(self.pending, path)
        for key in self.pending:
            self.weight_map[key] = filename
        self.temporary_files.append(path)
        self.pending = {}
        self.pending_bytes = 0


def load_state_dict(path: Path, nested_key: str | None = None) -> dict[str, torch.Tensor]:
    with torch.serialization.safe_globals(NUMPY_SAFE_GLOBALS):
        loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    state_dict = loaded[nested_key] if nested_key is not None else loaded
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dictionary in {path}")
    return state_dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_snapshot(source_root: Path) -> None:
    for relative_path, (expected_size, expected_sha256) in PINNED_SOURCE_FILES.items():
        path = source_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned SCAIL-2 source file: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Unexpected byte count for {relative_path}: "
                f"expected {expected_size}, found {actual_size}"
            )
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Unexpected SHA-256 for {relative_path}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )


def load_upstream_transformer_mapper(code_root: Path) -> Callable[[str, torch.Tensor], dict[str, torch.Tensor]]:
    convert_path = code_root / "convert.py"
    spec = importlib.util.spec_from_file_location("scail2_upstream_convert", convert_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {convert_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_new_mappings


def native_transformer_tensors(key: str, value: torch.Tensor) -> Iterable[tuple[str, torch.Tensor]]:
    mapped = key
    replacements = {
        "patch_embedding.weight": "patch_embedding_proj.weight",
        "patch_embedding.bias": "patch_embedding_proj.bias",
        "patch_embedding_pose.weight": "patch_embedding_pose_proj.weight",
        "patch_embedding_pose.bias": "patch_embedding_pose_proj.bias",
        "patch_embedding_mask.weight": "patch_embedding_mask_proj.weight",
        "patch_embedding_mask.bias": "patch_embedding_mask_proj.bias",
        "text_embedding.0.": "text_embedding_0.",
        "text_embedding.2.": "text_embedding_1.",
        "time_embedding.0.": "time_embedding_0.",
        "time_embedding.2.": "time_embedding_1.",
        "time_projection.1.": "time_projection.",
        "img_emb.proj.0.": "img_emb.layer_0.",
        "img_emb.proj.1.": "img_emb.layer_1.",
        "img_emb.proj.3.": "img_emb.layer_3.",
        "img_emb.proj.4.": "img_emb.layer_4.",
    }
    for source, target in replacements.items():
        if source in mapped:
            mapped = mapped.replace(source, target)
            break
    mapped = re.sub(r"^(blocks\.\d+\.ffn)\.0\.", r"\1.fc1.", mapped)
    mapped = re.sub(r"^(blocks\.\d+\.ffn)\.2\.", r"\1.fc2.", mapped)
    if mapped.endswith("_proj.weight"):
        value = value.reshape(value.shape[0], -1)
    target_dtype = torch.float32 if is_float32_transformer_tensor(mapped) else torch.bfloat16
    yield mapped, value.to(target_dtype)


def is_float32_transformer_tensor(key: str) -> bool:
    """Match the FP32 autocast islands in upstream SCAIL-2 inference."""
    return (
        key.startswith("time_embedding_")
        or key.startswith("time_projection.")
        or key.startswith("head.")
        or key.endswith("modulation")
    )


def native_t5_tensor(key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
    mapped = re.sub(r"^(blocks\.\d+\.ffn)\.gate\.0\.", r"\1.gate_proj.", key)
    return mapped, value.to(torch.bfloat16)


def native_clip_tensor(key: str, value: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    if key == "log_scale":
        return None
    if not key.startswith("visual."):
        return None
    mapped = key.removeprefix("visual.")
    block_match = re.match(r"transformer\.(\d+)\.", mapped)
    if block_match is not None and int(block_match.group(1)) >= 31:
        return None
    if mapped.startswith("post_norm.") or mapped == "head":
        return None
    mapped = re.sub(r"^(transformer\.\d+\.mlp)\.0\.", r"\1.layer_0.", mapped)
    mapped = re.sub(r"^(transformer\.\d+\.mlp)\.2\.", r"\1.layer_2.", mapped)
    if mapped == "patch_embedding.weight":
        value = value.permute(0, 2, 3, 1)
    return mapped, value.to(torch.float16)


def grouped_vae_path(path: str) -> str:
    match = re.match(r"^(encoder\.downsamples)\.(\d+)\.(.*)$", path)
    if match is not None:
        index = int(match.group(2))
        encoder_layout = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
            (3, 0), (3, 1),
        ]
        stage, layer = encoder_layout[index]
        return f"encoder.downsamples.{stage}.downsamples.{layer}.{match.group(3)}"
    match = re.match(r"^(decoder\.upsamples)\.(\d+)\.(.*)$", path)
    if match is not None:
        index = int(match.group(2))
        decoder_layout = [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 0), (3, 1), (3, 2),
        ]
        stage, layer = decoder_layout[index]
        return f"decoder.upsamples.{stage}.upsamples.{layer}.{match.group(3)}"
    return path


def native_vae_tensor(key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
    mapped = grouped_vae_path(key)
    residual_layers = {"0": "layer_0", "2": "layer_2", "3": "layer_3", "6": "layer_6"}
    match = re.search(r"\.residual\.(0|2|3|6)\.", mapped)
    if match is not None:
        mapped = mapped.replace(
            f".residual.{match.group(1)}.",
            f".residual.{residual_layers[match.group(1)]}.",
        )
    mapped = re.sub(r"\.head\.0\.", ".head.layer_0.", mapped)
    mapped = re.sub(r"\.head\.2\.", ".head.layer_2.", mapped)
    mapped = mapped.replace(".resample.1.weight", ".resample_weight")
    mapped = mapped.replace(".resample.1.bias", ".resample_bias")
    mapped = mapped.replace(".to_qkv.weight", ".to_qkv_weight")
    mapped = mapped.replace(".to_qkv.bias", ".to_qkv_bias")
    mapped = mapped.replace(".proj.weight", ".proj_weight")
    mapped = mapped.replace(".proj.bias", ".proj_bias")
    if mapped.endswith(".gamma"):
        value = value.reshape(-1)
    elif value.ndim == 5:
        value = value.permute(0, 2, 3, 4, 1)
    elif value.ndim == 4:
        value = value.permute(0, 2, 3, 1)
    return mapped, value.to(torch.float32)


def write_sharded_component(
    state_dict: dict[str, torch.Tensor],
    writer: ShardedSafetensorsWriter,
    mapper: Callable[[str, torch.Tensor], Iterable[tuple[str, torch.Tensor]]],
) -> None:
    for source_key, source_value in state_dict.items():
        for target_key, target_value in mapper(source_key, source_value):
            writer.add(target_key, target_value)
    writer.finish()


def write_single_component(
    state_dict: dict[str, torch.Tensor],
    destination: Path,
    mapper: Callable[[str, torch.Tensor], tuple[str, torch.Tensor] | None],
) -> None:
    converted: dict[str, torch.Tensor] = {}
    for source_key, source_value in state_dict.items():
        mapped = mapper(source_key, source_value)
        if mapped is not None:
            if mapped[0] in converted:
                raise ValueError(f"Duplicate converted tensor key: {mapped[0]}")
            converted[mapped[0]] = mapped[1].detach().cpu().contiguous()
    if not converted:
        raise ValueError(f"No tensors were written to {destination}")
    save_file(converted, destination)


def write_config(output_root: Path) -> None:
    configuration: dict[str, Any] = {
        "_class_name": "WanSCAILModel",
        "model_type": "scail2",
        "patch_size": [1, 2, 2],
        "text_len": 512,
        "in_dim": 20,
        "mask_dim": 28,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "eps": 1e-6,
        "vae_stride": [4, 8, 8],
        "vae_z_dim": 16,
        "sample_steps": 40,
        "sample_shift": 3.0,
        "sample_guide_scale": 5.0,
        "sample_fps": 16,
        "segment_len": 81,
        "segment_overlap": 5,
    }
    (output_root / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(output_root: Path) -> None:
    manifest: dict[str, Any] = {
        "schemaVersion": 3,
        "id": "video-scail2-14b-mlx",
        "engine": "wan-video",
        "family": "video",
        "tier": "latest",
        "variant": "base",
        "precision": "bf16",
        "defaults": {
            "steps": 40,
            "cfg": 5.0,
            "sigma_shift": 3.0,
        },
        "supports": ["video_generation"],
        "components": {
            "tokenizer": {"type": "local", "path": "."},
            "text_encoder": {"type": "local", "path": "."},
            "transformer": {"type": "local", "path": "."},
            "vae": {"type": "local", "path": "."},
        },
        "upstreamRepoId": f"{MODEL_REPO}@{MODEL_REVISION}",
        "createdAt": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
    }
    (output_root / "mererun_model.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_code_revision(code_root: Path) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != CODE_REVISION:
        raise ValueError(f"Expected SCAIL-2 code revision {CODE_REVISION}, found {actual}")
    for relative_path, expected_sha256 in PINNED_CODE_FILE_SHA256.items():
        path = code_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned SCAIL-2 code file: {path}")
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Unexpected SHA-256 for code file {relative_path}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-shard-gib", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    code_root = args.code_root.resolve()
    output_root = args.output_root.resolve()
    verify_code_revision(code_root)
    verify_model_snapshot(source_root)
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f"Output root is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise ValueError(f"Output root must be empty: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root.parent).free
    if free_bytes < MINIMUM_OUTPUT_FREE_BYTES:
        raise ValueError(
            f"SCAIL-2 conversion needs at least {MINIMUM_OUTPUT_FREE_BYTES} free bytes "
            f"at {output_root.parent}; found {free_bytes}"
        )
    maximum_bytes = math.floor(args.maximum_shard_gib * 1024**3)
    if maximum_bytes <= 0:
        raise ValueError("--maximum-shard-gib must be positive")
    staging_root = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging_root.mkdir()
    try:
        transformer_mapper = load_upstream_transformer_mapper(code_root)
        transformer_state = load_state_dict(source_root / TRANSFORMER_PATH, nested_key="module")
        transformer_writer = ShardedSafetensorsWriter(staging_root, "model", maximum_bytes)

        def map_transformer(key: str, value: torch.Tensor) -> Iterable[tuple[str, torch.Tensor]]:
            for upstream_key, upstream_value in transformer_mapper(key, value).items():
                yield from native_transformer_tensors(upstream_key, upstream_value)

        write_sharded_component(transformer_state, transformer_writer, map_transformer)
        del transformer_state

        text_state = load_state_dict(source_root / TEXT_ENCODER_PATH)
        text_writer = ShardedSafetensorsWriter(staging_root, "t5_encoder", maximum_bytes)
        write_sharded_component(
            text_state,
            text_writer,
            lambda key, value: [native_t5_tensor(key, value)],
        )
        del text_state

        clip_state = load_state_dict(source_root / CLIP_PATH)
        write_single_component(clip_state, staging_root / "clip.safetensors", native_clip_tensor)
        del clip_state

        vae_state = load_state_dict(source_root / VAE_PATH)
        write_single_component(vae_state, staging_root / "vae.safetensors", native_vae_tensor)
        del vae_state

        shutil.copy2(source_root / TOKENIZER_PATH, staging_root / "tokenizer.json")
        shutil.copy2(source_root / "README.md", staging_root / "README-SCAIL-2.md")
        shutil.copy2(code_root / "LICENSE", staging_root / "LICENSE-SCAIL-2")
        write_config(staging_root)
        write_manifest(staging_root)
        provenance = {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "code_repo": CODE_REPO,
            "code_revision": CODE_REVISION,
            "source_sha256": {
                str(path): digest for path, (_, digest) in PINNED_SOURCE_FILES.items()
            },
            "transformer_source": str(TRANSFORMER_PATH),
            "text_encoder_source": str(TEXT_ENCODER_PATH),
            "clip_source": str(CLIP_PATH),
            "vae_source": str(VAE_PATH),
        }
        (staging_root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_root.exists():
            output_root.rmdir()
        staging_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

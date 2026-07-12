#!/usr/bin/env python3
"""Export exact PyTorch parity tensors for InstantMesh Base reconstruction.

Only the Apache-2.0 DINO/camera encoder and triplane transformer modules are
loaded from the pinned upstream checkout. The neural field heads are expressed
with ordinary PyTorch layers directly from the checkpoint tensor contract.
NVIDIA's proprietary renderer/FlexiCubes files and Zero123++ are never imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import ViTConfig


CHECKPOINT_SHA256 = "22701cd25201d624ebb1568b93cf91b43a2c32006835c08fe73e1f3c9f6c44b5"
CHECKPOINT_BYTES = 1_253_574_354
UPSTREAM_REVISION = "08822c52fdc399b93ea00e4fa9e596344ed52ccc"
TENSOR_COUNT = 455
SCALAR_COUNT = 313_352_516
PREFIX = "lrm_generator."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(checkpoint: Path, upstream: Path) -> None:
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint does not match the pinned InstantMesh Base artifact")
    if not (upstream / "src/models/encoder/dino.py").is_file():
        raise ValueError("--upstream must be a TencentARC/InstantMesh checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"upstream revision mismatch: expected {UPSTREAM_REVISION}, found {revision}")


def load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    root = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(root, dict) or set(root) != {"state_dict"}:
        raise ValueError("unexpected Lightning checkpoint root")
    state = {
        key.removeprefix(PREFIX): value
        for key, value in root["state_dict"].items()
        if key.startswith(PREFIX) and "source_camera" not in key
    }
    if len(state) != TENSOR_COUNT or sum(value.numel() for value in state.values()) != SCALAR_COUNT:
        raise ValueError("reconstruction tensor inventory mismatch")
    return state


class ReferenceEncoder(nn.Module):
    def __init__(self, vit_model: type[nn.Module]) -> None:
        super().__init__()
        configuration = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-12,
            qkv_bias=True,
        )
        self.model = vit_model(configuration, add_pooling_layer=False)
        self.camera_embedder = nn.Sequential(
            nn.Linear(16, 768, bias=True),
            nn.SiLU(),
            nn.Linear(768, 768, bias=True),
        )

    def forward(self, images: torch.Tensor, cameras: torch.Tensor) -> torch.Tensor:
        batch, views = images.shape[:2]
        flat_images = images.reshape(batch * views, *images.shape[2:])
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        flat_images = (flat_images - mean) / std
        flat_cameras = cameras.reshape(batch * views, 16)
        conditioning = self.camera_embedder(flat_cameras)
        tokens = self.model(
            pixel_values=flat_images,
            adaln_input=conditioning,
            interpolate_pos_encoding=True,
        ).last_hidden_state
        return tokens.reshape(batch, views * tokens.shape[1], tokens.shape[2])


def make_mlp(input_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, output_size),
    )


class ReferenceDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net_sdf = make_mlp(120, 1)
        self.net_rgb = make_mlp(120, 3)
        self.net_deformation = make_mlp(120, 3)
        self.net_weight = make_mlp(960, 21)


class ReferenceSynthesizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = ReferenceDecoder()


class ReferenceModel(nn.Module):
    def __init__(self, vit_model: type[nn.Module], transformer: nn.Module) -> None:
        super().__init__()
        self.encoder = ReferenceEncoder(vit_model)
        self.transformer = transformer
        self.synthesizer = ReferenceSynthesizer()


def cameras(device: torch.device) -> torch.Tensor:
    azimuths = np.deg2rad(np.array([30, 90, 150, 210, 270, 330], dtype=np.float32))
    elevations = np.deg2rad(np.array([20, -10, 20, -10, 20, -10], dtype=np.float32))
    radius = 4.0
    positions = torch.from_numpy(np.stack([
        radius * np.cos(elevations) * np.cos(azimuths),
        radius * np.cos(elevations) * np.sin(azimuths),
        radius * np.sin(elevations),
    ], axis=-1).astype(np.float32)).float()
    z_axis = F.normalize(positions, dim=-1)
    up = torch.tensor([0, 0, 1], dtype=torch.float32).expand_as(z_axis)
    x_axis = F.normalize(torch.linalg.cross(up, z_axis, dim=-1), dim=-1)
    y_axis = F.normalize(torch.linalg.cross(z_axis, x_axis, dim=-1), dim=-1)
    c2w = torch.stack([x_axis, y_axis, z_axis, positions], dim=-1)
    focal = np.float32(0.5 / np.tan(np.deg2rad(30.0) * 0.5))
    intrinsics = torch.tensor([focal, focal, 0.5, 0.5]).repeat(6, 1)
    return torch.cat([c2w.reshape(6, 12), intrinsics], dim=-1).unsqueeze(0).to(device)


def input_images(device: torch.device) -> torch.Tensor:
    y = torch.linspace(0, 1, 320, device=device).view(1, 1, 320, 1, 1)
    x = torch.linspace(0, 1, 320, device=device).view(1, 1, 1, 320, 1)
    view = torch.arange(6, device=device, dtype=torch.float32).view(1, 6, 1, 1, 1) / 5
    red = (0.55 * x + 0.25 * y + 0.20 * view).expand(1, 6, 320, 320, 1)
    green = (0.15 * x + 0.65 * y + 0.20 * (1 - view)).expand(1, 6, 320, 320, 1)
    blue = (0.5 + 0.25 * torch.sin((x + y + view) * torch.pi)).expand(1, 6, 320, 320, 1)
    return torch.cat([red, green, blue], dim=-1).clamp(0, 1)


def sample_features(planes: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    coordinates = torch.stack([
        positions[:, [0, 1]],
        positions[:, [0, 2]],
        positions[:, [2, 1]],
    ], dim=0).unsqueeze(1)
    sampled = F.grid_sample(
        planes[0], coordinates,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return sampled.permute(0, 3, 2, 1).reshape(3, positions.shape[0], 40).permute(1, 0, 2).reshape(-1, 120)


def save(path: Path, value: torch.Tensor) -> None:
    np.save(path, value.detach().float().cpu().numpy(), allow_pickle=False)


def export(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    output = args.output.resolve()
    validate_inputs(checkpoint, upstream)
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(upstream))
    from src.models.encoder.dino import ViTModel  # noqa: PLC0415
    from src.models.decoder.transformer import TriplaneTransformer  # noqa: PLC0415

    device = torch.device(args.device)
    transformer = TriplaneTransformer(
        inner_dim=1024,
        image_feat_dim=768,
        triplane_low_res=32,
        triplane_high_res=64,
        triplane_dim=40,
        num_layers=12,
        num_heads=16,
    )
    model = ReferenceModel(ViTModel, transformer)
    state = load_state(checkpoint)
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)

    with torch.inference_mode():
        images_nhwc = input_images(device)
        camera_values = cameras(device)
        images_nchw = images_nhwc.permute(0, 1, 4, 2, 3).contiguous()
        image_tokens = model.encoder(images_nchw, camera_values)
        initial_tokens = model.transformer.pos_embed.expand(images_nhwc.shape[0], -1, -1)
        hidden = initial_tokens
        for layer in model.transformer.layers:
            hidden = layer(hidden, image_tokens)
        final_tokens = model.transformer.norm(hidden)
        batch = final_tokens.shape[0]
        planar = final_tokens.reshape(batch, 3, 32, 32, 1024).permute(1, 0, 4, 2, 3)
        planar = planar.contiguous().reshape(3 * batch, 1024, 32, 32)
        planes = model.transformer.deconv(planar).reshape(3, batch, 40, 64, 64).permute(1, 0, 2, 3, 4)

        query_positions = torch.tensor([
            [-1.05, -1.05, -1.05], [-0.75, 0.20, 0.40], [-0.10, -0.20, 0.30],
            [0.0, 0.0, 0.0], [0.40, -0.50, 0.60], [0.85, 0.75, -0.25],
            [1.05, 1.05, 1.05], [1.20, 0.0, 0.0],
        ], dtype=torch.float32, device=device)
        features = sample_features(planes, query_positions)
        decoder = model.synthesizer.decoder
        signed_distance = decoder.net_sdf(features)
        raw_deformation = decoder.net_deformation(features)
        deformation = torch.tanh(raw_deformation) / (128 * 4)
        color = torch.sigmoid(decoder.net_rgb(features)) * 1.002 - 0.001
        cell_weight = decoder.net_weight(features.reshape(1, 8 * 120)) * 0.1

    save(output / "input.npy", images_nhwc)
    save(output / "cameras.npy", camera_values)
    save(output / "image_tokens.npy", image_tokens)
    save(output / "tokens_initial.npy", initial_tokens)
    save(output / "tokens_final.npy", final_tokens)
    save(output / "scene_code_bpc_hw.npy", planes)
    save(output / "query_positions.npy", query_positions)
    save(output / "query_sdf.npy", signed_distance)
    save(output / "query_deformation_raw.npy", raw_deformation)
    save(output / "query_deformation.npy", deformation)
    save(output / "query_color.npy", color)
    save(output / "query_cell_weight.npy", cell_weight)
    (output / "fixture.json").write_text(json.dumps({
        "checkpoint": {
            "byteCount": CHECKPOINT_BYTES,
            "sha256": CHECKPOINT_SHA256,
        },
        "device": str(device),
        "licenseBoundary": {
            "importedUpstreamModules": [
                "src.models.encoder.dino",
                "src.models.decoder.transformer",
            ],
            "nvidiaProprietaryRendererImported": False,
            "viewGenerationIncluded": False,
            "zero123PlusIncluded": False,
        },
        "sourceCodeRevision": UPSTREAM_REVISION,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument("--upstream", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return result


if __name__ == "__main__":
    export(parser().parse_args())

#!/usr/bin/env python3
"""Export a learned Wan 2.1 VAE trace from the pinned SCAIL-2 source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


VAE_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("scail2_upstream_vae", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def deterministic(shape: tuple[int, ...], scale: float, phase: float) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    values = torch.arange(count, dtype=torch.float32)
    return torch.sin(values * scale + phase).reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if checkpoint.stat().st_size != 507_609_880 or sha256(checkpoint) != VAE_SHA256:
        raise RuntimeError("VAE checkpoint does not match the pinned SCAIL-2 source")
    module = load_module(args.upstream.resolve() / "wan/modules/vae.py")
    device = torch.device(args.device)
    model = module.WanVAE(
        z_dim=16,
        vae_pth=str(checkpoint),
        dtype=torch.float32,
        device=device,
    )
    video_nhwc = deterministic((1, 5, 32, 32, 3), 0.0031, 0.2)
    video_cthw = video_nhwc.permute(0, 4, 1, 2, 3).squeeze(0).to(device)
    with torch.inference_mode():
        encoded_cthw = model.encode([video_cthw])[0]
        decoded_cthw = model.decode([encoded_cthw])[0]
    traces = {
        "vae_video_nthwc": video_nhwc.contiguous(),
        "vae_encoded_nthwc": encoded_cthw.unsqueeze(0).permute(0, 2, 3, 4, 1).cpu().contiguous(),
        "vae_decoded_nthwc": decoded_cthw.unsqueeze(0).permute(0, 2, 3, 4, 1).cpu().contiguous(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        traces,
        str(args.output),
        metadata={"checkpoint_sha256": VAE_SHA256},
    )
    print(f"Wrote {len(traces)} tensors to {args.output}")


if __name__ == "__main__":
    main()

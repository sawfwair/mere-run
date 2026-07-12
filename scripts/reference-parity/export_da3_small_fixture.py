#!/usr/bin/env python3
"""Export frozen DA3-Small raw-output fixtures from the official graph.

This is release-time verification tooling only. It never participates in the
mere.run runtime. The script intentionally requires an already checked-out
authoritative source tree and a local checkpoint, then verifies both pins
before importing PyTorch code.

Example:
  PYTHONPATH=/path/to/Depth-Anything-3/src python \
    scripts/reference-parity/export_da3_small_fixture.py \
    --source /path/to/Depth-Anything-3 \
    --checkpoint /path/to/model.safetensors \
    --config /path/to/config.json \
    --output /tmp/da3-small-parity
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

EXPECTED_SOURCE_REVISION = "41736238f5bced4debf3f2a12375d2466874866d"
EXPECTED_CHECKPOINT_SHA256 = "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
EXPECTED_CHECKPOINT_BYTES = 137_248_940
EXPECTED_MODEL_REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
EXPECTED_CONFIG_SHA256 = "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2"
EXPECTED_CONFIG_BYTES = 1_202


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--height", type=int, default=28)
    parser.add_argument("--width", type=int, default=28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.views < 1:
        raise SystemExit("--views must be positive")
    if args.height % 14 or args.width % 14:
        raise SystemExit("--height and --width must be divisible by 14")
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != EXPECTED_SOURCE_REVISION:
        raise SystemExit(
            f"official source revision mismatch: expected {EXPECTED_SOURCE_REVISION}, got {source_revision}"
        )
    if args.checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise SystemExit("checkpoint byte count does not match the pinned DA3-Small artifact")
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            f"checkpoint SHA-256 mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    if args.config.stat().st_size != EXPECTED_CONFIG_BYTES:
        raise SystemExit("config byte count does not match the pinned DA3-Small artifact")
    config_sha = sha256(args.config)
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise SystemExit(
            f"config SHA-256 mismatch: expected {EXPECTED_CONFIG_SHA256}, got {config_sha}"
        )

    source_package = args.source / "src"
    if str(source_package) not in sys.path:
        sys.path.insert(0, str(source_package))

    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_model
    from depth_anything_3.cfg import create_object

    config_document = json.loads(args.config.read_text())
    model = create_object(OmegaConf.create(config_document["config"])).eval()

    # The Hub serializer owns the network below a `model` module. Wrapping the
    # authoritative network also lets safetensors restore its deliberately
    # shared auxiliary LayerNorm without inventing duplicate tensors.
    class HubWrapper(torch.nn.Module):
        def __init__(self, network: torch.nn.Module):
            super().__init__()
            self.model = network

    missing, unexpected = load_model(
        HubWrapper(model), str(args.checkpoint), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint load failed: missing={missing}, unexpected={unexpected}")

    element_count = args.views * args.height * args.width * 3
    input_nhwc = (
        torch.arange(element_count, dtype=torch.float32)
        .reshape(1, args.views, args.height, args.width, 3)
        .div(float(element_count))
        .sub(0.5)
    )
    input_nchw = input_nhwc.permute(0, 1, 4, 2, 3).contiguous()
    with torch.inference_mode():
        output = model(input_nchw, ref_view_strategy="first")
        conditioning_extrinsics = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(
            1, args.views, 1, 1
        )
        if args.views > 1:
            conditioning_extrinsics[0, 1:, 0, 3] = -0.1 * torch.arange(
                1, args.views, dtype=torch.float32
            )
        conditioning_intrinsics = torch.zeros(1, args.views, 3, 3, dtype=torch.float32)
        conditioning_intrinsics[..., 0, 0] = float(args.width)
        conditioning_intrinsics[..., 1, 1] = float(args.height)
        conditioning_intrinsics[..., 0, 2] = float(args.width) / 2
        conditioning_intrinsics[..., 1, 2] = float(args.height) / 2
        conditioning_intrinsics[..., 2, 2] = 1
        conditioned_output = model(
            input_nchw,
            extrinsics=conditioning_extrinsics,
            intrinsics=conditioning_intrinsics,
            ref_view_strategy="first",
        )

    args.output.mkdir(parents=True, exist_ok=True)
    input_nhwc.numpy().astype("<f4", copy=False).tofile(args.output / "input.f32")
    output.depth.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output / "depth.f32"
    )
    output.depth_conf.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output / "confidence.f32"
    )
    output.extrinsics.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output / "extrinsics.f32"
    )
    output.intrinsics.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output / "intrinsics.f32"
    )
    conditioning_extrinsics.numpy().astype("<f4", copy=False).tofile(
        args.output / "conditioning-extrinsics.f32"
    )
    conditioning_intrinsics.numpy().astype("<f4", copy=False).tofile(
        args.output / "conditioning-intrinsics.f32"
    )
    conditioned_output.depth.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output / "conditioned-depth.f32"
    )
    conditioned_output.depth_conf.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output / "conditioned-confidence.f32"
    )
    conditioned_output.extrinsics.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output / "conditioned-extrinsics.f32"
    )
    conditioned_output.intrinsics.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(
        args.output / "conditioned-intrinsics.f32"
    )
    (args.output / "shape.json").write_text(
        json.dumps(
            {"views": args.views, "height": args.height, "width": args.width},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "checkpoint_repository": "depth-anything/DA3-SMALL",
                "checkpoint_revision": EXPECTED_MODEL_REVISION,
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "source_repository": "ByteDance-Seed/Depth-Anything-3",
                "source_revision": source_revision,
                "license": "Apache-2.0",
                "reference_view_strategy": "first",
                "torch_version": torch.__version__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

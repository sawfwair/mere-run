#!/usr/bin/env python3
"""Export a deterministic upstream Video Depth Anything Small parity fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


SOURCE_REVISION = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
CHECKPOINTS = {
    "relative": "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
    "metric": "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", choices=sorted(CHECKPOINTS), default="relative")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    args = parser.parse_args()

    if args.frames <= 0 or args.frames > 32:
        raise ValueError("--frames must be between 1 and 32")
    if args.height <= 0 or args.width <= 0 or args.height % 14 or args.width % 14:
        raise ValueError("--height and --width must be positive multiples of 14")
    actual_revision = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != SOURCE_REVISION:
        raise ValueError(f"source revision mismatch: {actual_revision}")
    expected_checkpoint_hash = CHECKPOINTS[args.variant]
    actual_checkpoint_hash = sha256(args.checkpoint)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError(f"checkpoint SHA-256 mismatch: {actual_checkpoint_hash}")

    sys.path.insert(0, str(args.source_root))
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe="ape",
        metric=args.variant == "metric",
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    count = args.frames * args.height * args.width * 3
    raw = (np.arange(count, dtype=np.float32) % 251) / np.float32(250)
    raw = raw.reshape(1, args.frames, args.height, args.width, 3)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    torch_input = torch.from_numpy(normalized).permute(0, 1, 4, 2, 3).contiguous()
    with torch.no_grad():
        depth = model(torch_input).cpu().numpy().astype("<f4", copy=False)

    args.output.mkdir(parents=True, exist_ok=False)
    normalized.astype("<f4", copy=False).tofile(args.output / "input.f32")
    depth.tofile(args.output / "depth.f32")
    manifest = {
        "checkpointSHA256": expected_checkpoint_hash,
        "depthShape": list(depth.shape),
        "inputLayout": "BTHWC",
        "inputShape": list(normalized.shape),
        "sourceRevision": SOURCE_REVISION,
        "torchVersion": torch.__version__,
        "variant": args.variant,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

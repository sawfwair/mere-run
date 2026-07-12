#!/usr/bin/env python3
"""Freeze InstantMesh's pinned torchvision bicubic preprocessing output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tv_functional
from torchvision.transforms import InterpolationMode


SOURCE_REVISION = "08822c52fdc399b93ea00e4fa9e596344ed52ccc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_images(view_count: int, size: int) -> np.ndarray:
    """Build deterministic opaque RGB views without depending on a codec."""
    views = []
    for view in range(view_count):
        y, x, channel = np.indices((size, size, 3), dtype=np.uint32)
        values = (
            x * (view * 7 + 3)
            + y * (view * 5 + 11)
            + channel * 53
            + view * 29
            + (x * y) % 97
        ) % 256
        views.append(values.astype(np.uint8))
    return np.stack(views)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-size", type=int, default=257)
    parser.add_argument("--target-size", type=int, default=320)
    parser.add_argument("--views", type=int, choices=(4, 6), default=4)
    args = parser.parse_args()

    if args.source_size <= 0 or args.target_size <= 0:
        raise ValueError("source and target sizes must be positive")
    if args.source_size == args.target_size:
        raise ValueError("source size must differ from target size to exercise resize parity")

    source = source_images(args.views, args.source_size)
    tensor = torch.from_numpy(source).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    resized = tv_functional.resize(
        tensor,
        args.target_size,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).clamp(0, 1)
    resized_nhwc = resized.permute(0, 2, 3, 1).contiguous().numpy().astype("<f4")

    args.output.mkdir(parents=True, exist_ok=False)
    source.tofile(args.output / "source-rgb.u8")
    resized_nhwc.tofile(args.output / "resized.f32")
    manifest = {
        "sourceRevision": SOURCE_REVISION,
        "sourceSize": args.source_size,
        "targetSize": args.target_size,
        "viewCount": args.views,
        "sourceShape": list(source.shape),
        "outputShape": list(resized_nhwc.shape),
        "interpolation": "torchvision InterpolationMode.BICUBIC (3)",
        "antialias": True,
        "clamp": [0.0, 1.0],
        "sourceRGBSHA256": sha256(args.output / "source-rgb.u8"),
        "resizedSHA256": sha256(args.output / "resized.f32"),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

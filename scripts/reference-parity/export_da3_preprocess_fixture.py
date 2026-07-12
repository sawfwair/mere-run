#!/usr/bin/env python3
"""Freeze the pinned DA3 upper-bound image/camera preprocessing path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


SOURCE_REVISION = "41736238f5bced4debf3f2a12375d2466874866d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_image(width: int, height: int, seed: int) -> np.ndarray:
    values = (np.arange(width * height * 3, dtype=np.uint32) * (seed * 2 + 1) + seed) % 251
    return values.astype(np.uint8).reshape(height, width, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--process-resolution", type=int, default=56)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"source revision mismatch: {revision}")
    sys.path.insert(0, str(args.source_root / "src"))
    from depth_anything_3.utils.io.input_processor import InputProcessor

    specs = [(67, 37, 3), (37, 67, 7), (23, 17, 11)]
    images = [source_image(width, height, seed) for width, height, seed in specs]
    intrinsics = []
    extrinsics = []
    for index, (width, height, _) in enumerate(specs):
        intrinsics.append(
            np.array(
                [
                    [width * 0.8, 0, width * 0.47],
                    [0, height * 0.9, height * 0.53],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        )
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = -0.2 * index
        extrinsics.append(transform)

    normalized, output_extrinsics, output_intrinsics = InputProcessor()(
        images,
        np.stack(extrinsics),
        np.stack(intrinsics),
        process_res=args.process_resolution,
        process_res_method="upper_bound_resize",
        num_workers=1,
        sequential=True,
        print_progress=False,
    )
    normalized_nhwc = normalized.permute(0, 2, 3, 1).contiguous().numpy().astype("<f4")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    processed_rgb = np.rint(np.clip((normalized_nhwc * std + mean) * 255, 0, 255)).astype(np.uint8)

    args.output.mkdir(parents=True, exist_ok=False)
    normalized_nhwc.tofile(args.output / "normalized.f32")
    processed_rgb.tofile(args.output / "processed-rgb.u8")
    output_intrinsics.numpy().astype("<f4").tofile(args.output / "intrinsics.f32")
    output_extrinsics.numpy().astype("<f4").tofile(args.output / "extrinsics.f32")
    manifest = {
        "sourceRevision": revision,
        "processResolution": args.process_resolution,
        "sourceSpecs": [list(item) for item in specs],
        "normalizedShape": list(normalized_nhwc.shape),
        "normalizedSHA256": sha256(args.output / "normalized.f32"),
        "processedRGBSHA256": sha256(args.output / "processed-rgb.u8"),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze the pinned VDA OpenCV cubic preprocessing contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


SOURCE_REVISION = "4f5ae23172ba60fd7bc11ef671cca678842c7072"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--source-height", type=int, default=37)
    parser.add_argument("--source-width", type=int, default=61)
    parser.add_argument("--input-size", type=int, default=56)
    args = parser.parse_args()
    actual_revision = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != SOURCE_REVISION:
        raise ValueError(f"source revision mismatch: {actual_revision}")
    sys.path.insert(0, str(args.source_root))
    import cv2
    from video_depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

    resize = Resize(
        width=args.input_size,
        height=args.input_size,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    prepare = PrepareForNet()
    count = args.frames * args.source_height * args.source_width * 3
    source = (np.arange(count, dtype=np.uint32) * 37 % 256).astype(np.uint8)
    source = source.reshape(args.frames, args.source_height, args.source_width, 3)
    processed = []
    for frame in source:
        sample = resize({"image": frame.astype(np.float32) / np.float32(255)})
        sample = normalize(sample)
        sample = prepare(sample)
        processed.append(np.transpose(sample["image"], (1, 2, 0)))
    normalized = np.ascontiguousarray(np.stack(processed), dtype="<f4")

    args.output.mkdir(parents=True, exist_ok=False)
    source.tofile(args.output / "source.rgb8")
    normalized.tofile(args.output / "normalized.f32")
    manifest = {
        "inputSize": args.input_size,
        "normalizedLayout": "THWC",
        "normalizedShape": list(normalized.shape),
        "sourceLayout": "THWC",
        "sourceRevision": SOURCE_REVISION,
        "sourceShape": list(source.shape),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

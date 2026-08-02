#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.6",
#   "pillow==11.3.0",
#   "scikit-image==0.25.2",
# ]
# ///
"""Add deterministic PSNR/SSIM revisit scores to a captured DreamX report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


REMAINING_METRICS = {
    "LPIPS": "requires the pinned learned perceptual metric lane",
    "DINO-Sim": "requires the pinned DINO image encoder lane",
    "VPR-Sim": "requires the pinned visual-place-recognition lane",
    "SP-Match": "requires the pinned SuperPoint matcher lane",
    "CLIP-Video": "requires the pinned full-video encoder lane",
}


def normalized_rgb(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        return np.asarray(image, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    width = report["geometry"]["width"]
    height = report["geometry"]["height"]
    reference = normalized_rgb(Path(report["source"]), width, height)

    for scenario in report["scenarios"]:
        if scenario["kind"] not in {"revisit", "loop"} or not scenario["steps"]:
            continue
        terminal_path = Path(scenario["steps"][-1]["terminal_frame"])
        terminal = normalized_rgb(terminal_path, width, height)
        psnr = float(peak_signal_noise_ratio(reference, terminal, data_range=255))
        ssim = float(structural_similarity(reference, terminal, channel_axis=2, data_range=255))
        scenario["visual_metrics"] = {
            "status": "partial",
            "reference": str(Path(report["source"]).resolve()),
            "candidate": str(terminal_path.resolve()),
            "normalization": "Pillow RGB center-crop LANCZOS to runtime geometry",
            "PSNR": psnr,
            "SSIM": ssim,
            "unscored": REMAINING_METRICS,
            "note": "PSNR/SSIM are real image scores; remaining learned metrics are not inferred.",
        }

    output = args.output or args.report
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

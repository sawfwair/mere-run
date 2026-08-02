#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.6",
#   "pillow==11.3.0",
#   "scikit-image==0.25.2",
# ]
# ///
"""Add matched-baseline PSNR/SSIM revisit gains to a DreamX report."""

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

    scenarios = {scenario["id"]: scenario for scenario in report["scenarios"]}
    for scenario in report["scenarios"]:
        baseline_id = scenario.get("baseline_scenario_id")
        if not baseline_id or not scenario["steps"]:
            continue
        baseline_scenario = scenarios.get(baseline_id)
        if not baseline_scenario or not baseline_scenario["steps"]:
            raise SystemExit(f"Missing matched baseline {baseline_id} for {scenario['id']}")
        terminal_path = Path(scenario["steps"][-1]["terminal_frame"])
        baseline_path = Path(baseline_scenario["steps"][-1]["terminal_frame"])
        terminal = normalized_rgb(terminal_path, width, height)
        baseline = normalized_rgb(baseline_path, width, height)
        revisit_psnr = float(peak_signal_noise_ratio(reference, terminal, data_range=255))
        baseline_psnr = float(peak_signal_noise_ratio(reference, baseline, data_range=255))
        revisit_ssim = float(structural_similarity(reference, terminal, channel_axis=2, data_range=255))
        baseline_ssim = float(structural_similarity(reference, baseline, channel_axis=2, data_range=255))
        scenario["visual_metrics"] = {
            "status": "partial",
            "reference": str(Path(report["source"]).resolve()),
            "candidate": str(terminal_path.resolve()),
            "baseline": str(baseline_path.resolve()),
            "normalization": "Pillow RGB center-crop LANCZOS to runtime geometry",
            "absolute": {
                "revisit": {"PSNR": revisit_psnr, "SSIM": revisit_ssim},
                "baseline": {"PSNR": baseline_psnr, "SSIM": baseline_ssim},
            },
            "gains": {
                "delta_PSNR": revisit_psnr - baseline_psnr,
                "delta_SSIM": revisit_ssim - baseline_ssim,
            },
            "unscored": REMAINING_METRICS,
            "note": "PSNR/SSIM are matched-baseline gains; remaining learned metrics are not inferred.",
        }

    output = args.output or args.report
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

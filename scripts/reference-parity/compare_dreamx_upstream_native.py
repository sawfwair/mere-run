#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.6",
#   "Pillow==11.3.0",
#   "scikit-image==0.25.2",
# ]
# ///
"""Build an evidence receipt for the exact DreamX upstream/native scenario.

This gates identical input and media contracts, not byte-identical pixels.
PyTorch and MLX use different seeded random-number implementations, so paired
PSNR/SSIM samples are diagnostic and are never promoted to a numeric parity
claim. Learned semantic/video metrics remain an explicit separate lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


EXPECTED = {
    "action_seq": ["w", "wj", "wl"],
    "action_speed_list": [4, 6, 6],
    "num_output_frames": 63,
    "width": 1280,
    "height": 704,
    "seed": 42,
    "fps": 16,
    "speed": 1.5,
    "pixel_frame_count": 249,
    "duration_seconds": 15.5625,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def normalized_probe(raw: dict) -> dict:
    stream = raw["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream["avg_frame_rate"])),
        "frames": int(stream["nb_read_frames"]),
        "duration_seconds": float(raw["format"]["duration"]),
    }


def extract_frame(video: Path, index: int, output: Path) -> None:
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", "-update", "1", str(output),
    ], check=True)


def paired_diagnostics(native: Path, upstream: Path) -> list[dict]:
    samples = []
    with tempfile.TemporaryDirectory(prefix="dreamx-parity-") as directory:
        root = Path(directory)
        for index in (0, 124, 248):
            native_frame = root / f"native-{index}.png"
            upstream_frame = root / f"upstream-{index}.png"
            extract_frame(native, index, native_frame)
            extract_frame(upstream, index, upstream_frame)
            left = np.asarray(Image.open(native_frame).convert("RGB"), dtype=np.uint8)
            right = np.asarray(Image.open(upstream_frame).convert("RGB"), dtype=np.uint8)
            samples.append({
                "frame_index": index,
                "psnr": float(peak_signal_noise_ratio(left, right, data_range=255)),
                "ssim": float(structural_similarity(left, right, channel_axis=2, data_range=255)),
            })
    return samples


def contract_errors(
    request: dict,
    upstream_input: dict,
    upstream_invocation: dict,
    native_probe: dict,
    upstream_probe: dict,
) -> list[str]:
    errors = []
    for name in ("action_seq", "action_speed_list"):
        if request.get(name) != EXPECTED[name]:
            errors.append(f"native {name} differs from the pinned scenario")
        if upstream_input.get(name) != EXPECTED[name]:
            errors.append(f"upstream {name} differs from the pinned scenario")
    if upstream_input.get("caption", upstream_input.get("prompt")) != request.get("prompt"):
        errors.append("prompt differs between upstream and native")
    for name in ("num_output_frames", "seed", "fps"):
        if upstream_invocation.get(name) != EXPECTED[name]:
            errors.append(
                f"upstream {name}={upstream_invocation.get(name)!r}, expected {EXPECTED[name]!r}"
            )
    if not math.isclose(upstream_invocation["color_correction_strength"], 0.3, abs_tol=1e-9):
        errors.append(
            "upstream color_correction_strength="
            f"{upstream_invocation['color_correction_strength']!r}, expected 0.3"
        )
    if not upstream_invocation["chunk_relative"]:
        errors.append("upstream run did not declare --chunk_relative")
    if not math.isclose(upstream_invocation["trajectory_speed"], EXPECTED["speed"], abs_tol=1e-9):
        errors.append(
            f"upstream trajectory speed={upstream_invocation['trajectory_speed']!r}, "
            f"expected {EXPECTED['speed']!r}"
        )
    for name in ("num_output_frames", "width", "height", "seed", "fps", "speed"):
        if request.get(name) != EXPECTED[name]:
            errors.append(f"native {name}={request.get(name)!r}, expected {EXPECTED[name]!r}")
    for label, media in (("native", native_probe), ("upstream", upstream_probe)):
        for name, expected_name in (("width", "width"), ("height", "height"), ("frames", "pixel_frame_count")):
            if media[name] != EXPECTED[expected_name]:
                errors.append(f"{label} {name}={media[name]!r}, expected {EXPECTED[expected_name]!r}")
        if not math.isclose(media["fps"], EXPECTED["fps"], abs_tol=1e-6):
            errors.append(f"{label} fps={media['fps']!r}, expected {EXPECTED['fps']!r}")
        if not math.isclose(media["duration_seconds"], EXPECTED["duration_seconds"], abs_tol=1e-3):
            errors.append(
                f"{label} duration={media['duration_seconds']!r}, expected {EXPECTED['duration_seconds']!r}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--upstream-video", type=Path, required=True)
    parser.add_argument("--upstream-input", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--upstream-container-digest", required=True)
    parser.add_argument("--upstream-num-output-frames", type=int, default=63)
    parser.add_argument("--upstream-seed", type=int, default=42)
    parser.add_argument("--upstream-fps", type=int, default=16)
    parser.add_argument("--upstream-color-correction-strength", type=float, default=0.3)
    parser.add_argument("--upstream-trajectory-speed", type=float, default=1.5)
    parser.add_argument("--upstream-chunk-relative", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_report = json.loads(args.native_report.read_text())
    scenario = next(item for item in native_report["scenarios"] if item["id"] == "upstream_native_15s")
    step = scenario["steps"][0]
    native_video = Path(step["video"])
    upstream_items = json.loads(args.upstream_input.read_text())
    if len(upstream_items) != 1:
        raise SystemExit("The parity input must contain exactly one upstream item.")

    native_media = normalized_probe(probe(native_video))
    upstream_media = normalized_probe(probe(args.upstream_video))
    upstream_invocation = {
        "program": "inference_ar_forcing.py",
        "num_output_frames": args.upstream_num_output_frames,
        "seed": args.upstream_seed,
        "fps": args.upstream_fps,
        "color_correction_strength": args.upstream_color_correction_strength,
        "trajectory_speed": args.upstream_trajectory_speed,
        "chunk_relative": args.upstream_chunk_relative,
    }
    errors = contract_errors(
        step["request"], upstream_items[0], upstream_invocation, native_media, upstream_media
    )
    receipt = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "parity_scope": "exact input, trajectory, block schedule, and encoded-media contract",
        "pixel_equivalence_expected": False,
        "pixel_equivalence_note": (
            "PyTorch/CUDA and Swift/MLX seed independent random-number implementations; "
            "paired pixel scores are diagnostics, not the parity gate."
        ),
        "source": {"path": str(args.source.resolve()), "sha256": sha256(args.source)},
        "upstream": {
            "commit": args.upstream_commit,
            "model_revision": args.model_revision,
            "base_model_revision": args.base_model_revision,
            "model_sha256": args.model_sha256,
            "container_digest": args.upstream_container_digest,
            "invocation": upstream_invocation,
            "input_sha256": sha256(args.upstream_input),
            "input": upstream_items[0],
            "video": str(args.upstream_video.resolve()),
            "video_sha256": sha256(args.upstream_video),
            "media": upstream_media,
        },
        "native": {
            "request": step["request"],
            "receipt": step["receipt"],
            "video": str(native_video.resolve()),
            "video_sha256": sha256(native_video),
            "media": native_media,
        },
        "paired_pixel_diagnostics": paired_diagnostics(native_video, args.upstream_video),
        "learned_metrics": {
            "status": "not_scored",
            "required": ["LPIPS", "DINO-Sim", "VPR-Sim", "SP-Match", "CLIP-Video"],
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

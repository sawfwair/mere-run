#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "einops==0.8.1",
#   "kornia==0.8.1",
#   "lpips==0.1.4",
#   "numpy==2.2.3",
#   "opencv-python==4.11.0.86",
#   "pillow==11.1.0",
#   "scikit-image==0.25.2",
#   "timm==1.0.15",
#   "torch==2.6.0",
#   "torchvision==0.21.0",
#   "transformers==4.49.0",
# ]
# ///
"""Score DreamX revisit memory with the paper's learned metric families.

This is an evaluation-only adapter around independently pinned public metric
implementations. It does not enter the mere.run binary or product dependency
graph. Revisit gains use a matched-duration non-revisit scenario as required by
DreamX section 5.3; geometry success never substitutes for a visual score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


DINO_MODEL = "facebook/dino-vitb16"
DINO_REVISION = "f205d5d8e640a89a2b8ef0369670dfc37cc07fc2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MUTUAL_VPR_REVISION = "bd373c6e734556f33d9b4cbc5396862592e624a7"
MUTUAL_VPR_WEIGHTS_SHA256 = "e8ca2129256bee963d7a8f0e9db7ef0763a16b520bb06811511ea1992ab2f1f0"
LIGHTGLUE_REVISION = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
PAPER_REFERENCE = {
    "delta_PSNR": 3.92,
    "delta_SSIM": 0.098,
    "delta_LPIPS": 0.232,
    "delta_DINO_Sim": 0.246,
    "delta_VPR_Sim": 0.142,
    "delta_SP_Match": 0.216,
    "CLIP_Video": 0.991,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_git_revision(root: Path, expected: str, label: str) -> None:
    if not root.is_dir():
        raise SystemExit(f"{label} source root not found: {root}")
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Could not verify {label} source revision: {error}") from error
    if actual != expected:
        raise SystemExit(f"{label} revision {actual} does not match pinned {expected}")


def rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def normalized_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = ImageOps.fit(
        rgb(path),
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    return np.asarray(image, dtype=np.uint8)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right).mean().detach().cpu())


def extract_video_frames(videos: list[Path], output: Path) -> list[Path]:
    frames: list[Path] = []
    for video_index, video in enumerate(videos):
        step_dir = output / f"step-{video_index:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(video),
                "-vsync", "0", str(step_dir / "%06d.png"),
            ],
            check=True,
        )
        frames.extend(sorted(step_dir.glob("*.png")))
    if len(frames) < 2:
        raise RuntimeError("CLIP-Video requires at least two decoded frames")
    return frames


class MetricModels:
    def __init__(
        self,
        device: torch.device,
        mutual_vpr_root: Path,
        mutual_vpr_weights: Path,
        lightglue_root: Path,
    ) -> None:
        self.device = device
        self.lpips = lpips.LPIPS(net="alex").eval().to(device)

        self.dino_processor = AutoImageProcessor.from_pretrained(
            DINO_MODEL, revision=DINO_REVISION
        )
        self.dino = AutoModel.from_pretrained(
            DINO_MODEL, revision=DINO_REVISION
        ).eval().to(device)

        self.clip_processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL, revision=CLIP_REVISION
        )
        self.clip = CLIPModel.from_pretrained(
            CLIP_MODEL, revision=CLIP_REVISION
        ).eval().to(device)

        sys.path.insert(0, str(mutual_vpr_root))
        from cosplace_model.cosplace_network import MutualVPR

        self.vpr = MutualVPR(pretrained_foundation=False, output_dim=512)
        state = torch.load(mutual_vpr_weights, map_location="cpu", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.vpr.load_state_dict(state, strict=True)
        self.vpr.eval().to(device)
        self.vpr_transform = transforms.Compose([
            transforms.Resize((504, 504), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        sys.path.insert(0, str(lightglue_root))
        from lightglue import LightGlue, SuperPoint

        self.superpoint = SuperPoint(max_num_keypoints=1024).eval().to(device)
        self.lightglue = LightGlue(features="superpoint").eval().to(device)

    @torch.inference_mode()
    def lpips_distance(self, left: Image.Image, right: Image.Image) -> float:
        transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        return float(self.lpips(
            transform(left).unsqueeze(0).to(self.device),
            transform(right).unsqueeze(0).to(self.device),
        ).squeeze().detach().cpu())

    @torch.inference_mode()
    def dino_descriptor(self, image: Image.Image) -> torch.Tensor:
        values = self.dino_processor(images=image, return_tensors="pt")["pixel_values"]
        hidden = self.dino(pixel_values=values.to(self.device)).last_hidden_state[:, 0]
        return F.normalize(hidden, dim=-1)

    @torch.inference_mode()
    def vpr_descriptor(self, image: Image.Image) -> torch.Tensor:
        return F.normalize(
            self.vpr(self.vpr_transform(image).unsqueeze(0).to(self.device)), dim=-1
        )

    @torch.inference_mode()
    def sp_match_ratio(self, left: Image.Image, right: Image.Image) -> float:
        from lightglue.utils import rbd

        to_tensor = transforms.ToTensor()
        features0 = self.superpoint.extract(to_tensor(left).to(self.device))
        features1 = self.superpoint.extract(to_tensor(right).to(self.device))
        matches = self.lightglue({"image0": features0, "image1": features1})
        features0, features1, matches = [rbd(item) for item in (features0, features1, matches)]
        denominator = min(len(features0["keypoints"]), len(features1["keypoints"]))
        return float(len(matches["matches"]) / denominator) if denominator else 0.0

    @torch.inference_mode()
    def clip_descriptors(self, images: list[Image.Image]) -> torch.Tensor:
        outputs = []
        for start in range(0, len(images), 16):
            values = self.clip_processor(
                images=images[start:start + 16], return_tensors="pt"
            )["pixel_values"].to(self.device)
            outputs.append(F.normalize(self.clip.get_image_features(pixel_values=values), dim=-1))
        return torch.cat(outputs)


def absolute_scores(
    models: MetricModels,
    reference_path: Path,
    candidate_path: Path,
    width: int,
    height: int,
) -> dict:
    reference = rgb(reference_path)
    candidate = rgb(candidate_path)
    reference_pixels = normalized_rgb(reference_path, width, height)
    candidate_pixels = normalized_rgb(candidate_path, width, height)
    reference_dino = models.dino_descriptor(reference)
    reference_vpr = models.vpr_descriptor(reference)
    return {
        "PSNR": float(peak_signal_noise_ratio(
            reference_pixels, candidate_pixels, data_range=255
        )),
        "SSIM": float(structural_similarity(
            reference_pixels, candidate_pixels, channel_axis=2, data_range=255
        )),
        "LPIPS": models.lpips_distance(reference, candidate),
        "DINO-Sim": cosine(reference_dino, models.dino_descriptor(candidate)),
        "VPR-Sim": cosine(reference_vpr, models.vpr_descriptor(candidate)),
        "SP-Match": models.sp_match_ratio(reference, candidate),
    }


def pair_scores(
    models: MetricModels,
    reference_path: Path,
    revisit_path: Path,
    baseline_path: Path,
    width: int,
    height: int,
) -> dict:
    revisit = absolute_scores(models, reference_path, revisit_path, width, height)
    baseline = absolute_scores(models, reference_path, baseline_path, width, height)

    return {
        "absolute": {
            "revisit": revisit,
            "baseline": baseline,
        },
        "gains": {
            "delta_PSNR": revisit["PSNR"] - baseline["PSNR"],
            "delta_SSIM": revisit["SSIM"] - baseline["SSIM"],
            "delta_LPIPS": baseline["LPIPS"] - revisit["LPIPS"],
            "delta_DINO_Sim": revisit["DINO-Sim"] - baseline["DINO-Sim"],
            "delta_VPR_Sim": revisit["VPR-Sim"] - baseline["VPR-Sim"],
            "delta_SP_Match": revisit["SP-Match"] - baseline["SP-Match"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mutual-vpr-root", type=Path, required=True)
    parser.add_argument("--mutual-vpr-weights", type=Path, required=True)
    parser.add_argument("--lightglue-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--minimum-gain", type=float, default=0.0)
    parser.add_argument("--minimum-clip-video", type=float, default=0.97)
    parser.add_argument("--minimum-periodic-psnr", type=float, default=11.0)
    parser.add_argument("--minimum-periodic-ssim", type=float, default=0.10)
    parser.add_argument("--maximum-periodic-lpips", type=float, default=0.70)
    parser.add_argument("--minimum-periodic-dino", type=float, default=0.60)
    parser.add_argument("--minimum-periodic-vpr", type=float, default=0.20)
    parser.add_argument("--minimum-periodic-sp-match", type=float, default=0.04)
    args = parser.parse_args()

    require_git_revision(args.mutual_vpr_root, MUTUAL_VPR_REVISION, "MutualVPR")
    require_git_revision(args.lightglue_root, LIGHTGLUE_REVISION, "LightGlue")
    if not args.mutual_vpr_weights.is_file():
        raise SystemExit(f"MutualVPR weights not found: {args.mutual_vpr_weights}")
    actual_weights_hash = sha256(args.mutual_vpr_weights)
    if actual_weights_hash != MUTUAL_VPR_WEIGHTS_SHA256:
        raise SystemExit(
            f"MutualVPR weights SHA-256 {actual_weights_hash} does not match pinned "
            f"{MUTUAL_VPR_WEIGHTS_SHA256}"
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")

    report = json.loads(args.report.read_text())
    scenarios = {scenario["id"]: scenario for scenario in report["scenarios"]}
    paired = [scenario for scenario in report["scenarios"] if scenario.get("baseline_scenario_id")]
    periodic = [scenario for scenario in report["scenarios"] if scenario.get("periodic_revisit_stride")]
    if not paired and not periodic:
        raise SystemExit("Report has no matched-baseline or periodic-revisit scenario")
    device = torch.device(args.device)
    models = MetricModels(
        device, args.mutual_vpr_root, args.mutual_vpr_weights, args.lightglue_root
    )
    failures = []

    with tempfile.TemporaryDirectory(prefix="mere-run-dreamx-metrics-") as temporary:
        temporary_root = Path(temporary)
        for scenario in paired:
            baseline = scenarios.get(scenario["baseline_scenario_id"])
            if baseline is None or not baseline.get("steps") or not scenario.get("steps"):
                raise SystemExit(f"Missing captured baseline for {scenario['id']}")
            scores = pair_scores(
                models,
                Path(report["source"]),
                Path(scenario["steps"][-1]["terminal_frame"]),
                Path(baseline["steps"][-1]["terminal_frame"]),
                report["geometry"]["width"],
                report["geometry"]["height"],
            )
            frame_paths = extract_video_frames(
                [Path(step["video"]) for step in scenario["steps"]],
                temporary_root / scenario["id"],
            )
            clip_features = models.clip_descriptors([rgb(path) for path in frame_paths])
            clip_video = float(
                F.cosine_similarity(clip_features[:-1], clip_features[1:]).mean().cpu()
            )
            scores["CLIP-Video"] = clip_video
            gain_failures = [
                name for name, value in scores["gains"].items() if value < args.minimum_gain
            ]
            if clip_video < args.minimum_clip_video:
                gain_failures.append("CLIP-Video")
            scores.update({
                "status": "passed" if not gain_failures else "failed",
                "failed_metrics": gain_failures,
                "thresholds": {
                    "minimum_gain": args.minimum_gain,
                    "minimum_CLIP_Video": args.minimum_clip_video,
                },
                "paper_reference": PAPER_REFERENCE,
                "protocol": (
                    "DreamX section 5.3 matched-duration baseline gains; CLIP-Video is the "
                    "absolute mean cosine similarity of consecutive generated frames."
                ),
                "model_receipts": {
                    "DINO": f"{DINO_MODEL}@{DINO_REVISION}",
                    "CLIP": f"{CLIP_MODEL}@{CLIP_REVISION}",
                    "MutualVPR": f"{MUTUAL_VPR_REVISION}:{actual_weights_hash}",
                    "SuperPoint-LightGlue": LIGHTGLUE_REVISION,
                    "LPIPS": "lpips==0.1.4 alex",
                },
            })
            scenario["visual_metrics"] = scores
            if gain_failures:
                failures.append({"scenario": scenario["id"], "metrics": gain_failures})

        for scenario in periodic:
            stride = scenario["periodic_revisit_stride"]
            revisit_steps = scenario["steps"][stride - 1::stride]
            minimum_count = scenario.get("minimum_periodic_revisits", 1)
            if len(revisit_steps) < minimum_count:
                raise SystemExit(
                    f"{scenario['id']} captured {len(revisit_steps)} periodic revisits, "
                    f"requires {minimum_count}"
                )
            samples = [
                absolute_scores(
                    models,
                    Path(report["source"]),
                    Path(step["terminal_frame"]),
                    report["geometry"]["width"],
                    report["geometry"]["height"],
                )
                for step in revisit_steps
            ]
            frame_paths = extract_video_frames(
                [Path(step["video"]) for step in scenario["steps"]],
                temporary_root / scenario["id"],
            )
            clip_features = models.clip_descriptors([rgb(path) for path in frame_paths])
            clip_video = float(
                F.cosine_similarity(clip_features[:-1], clip_features[1:]).mean().cpu()
            )
            aggregate = {
                "minimum_PSNR": min(sample["PSNR"] for sample in samples),
                "minimum_SSIM": min(sample["SSIM"] for sample in samples),
                "maximum_LPIPS": max(sample["LPIPS"] for sample in samples),
                "minimum_DINO_Sim": min(sample["DINO-Sim"] for sample in samples),
                "minimum_VPR_Sim": min(sample["VPR-Sim"] for sample in samples),
                "minimum_SP_Match": min(sample["SP-Match"] for sample in samples),
                "CLIP-Video": clip_video,
            }
            thresholds = {
                "minimum_PSNR": args.minimum_periodic_psnr,
                "minimum_SSIM": args.minimum_periodic_ssim,
                "maximum_LPIPS": args.maximum_periodic_lpips,
                "minimum_DINO_Sim": args.minimum_periodic_dino,
                "minimum_VPR_Sim": args.minimum_periodic_vpr,
                "minimum_SP_Match": args.minimum_periodic_sp_match,
                "minimum_CLIP_Video": args.minimum_clip_video,
            }
            periodic_failures = []
            for name in ("minimum_PSNR", "minimum_SSIM", "minimum_DINO_Sim", "minimum_VPR_Sim", "minimum_SP_Match"):
                if aggregate[name] < thresholds[name]:
                    periodic_failures.append(name)
            if aggregate["maximum_LPIPS"] > thresholds["maximum_LPIPS"]:
                periodic_failures.append("maximum_LPIPS")
            if clip_video < thresholds["minimum_CLIP_Video"]:
                periodic_failures.append("CLIP-Video")
            scenario["visual_metrics"] = {
                "status": "passed" if not periodic_failures else "failed",
                "protocol": (
                    f"{len(samples)} exact-pose periodic revisits sampled every {stride} "
                    "actions across the complete soak; all floors use the worst revisit."
                ),
                "periodic_revisits": samples,
                "aggregate": aggregate,
                "thresholds": thresholds,
                "failed_metrics": periodic_failures,
                "model_receipts": {
                    "DINO": f"{DINO_MODEL}@{DINO_REVISION}",
                    "CLIP": f"{CLIP_MODEL}@{CLIP_REVISION}",
                    "MutualVPR": f"{MUTUAL_VPR_REVISION}:{actual_weights_hash}",
                    "SuperPoint-LightGlue": LIGHTGLUE_REVISION,
                    "LPIPS": "lpips==0.1.4 alex",
                },
            }
            if periodic_failures:
                failures.append({"scenario": scenario["id"], "metrics": periodic_failures})

    report["learned_visual_gate"] = {
        "status": "passed" if not failures else "failed",
        "evaluated_scenarios": [scenario["id"] for scenario in paired + periodic],
        "failures": failures,
    }
    if failures:
        report["status"] = "failed"
    output = args.output or args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["learned_visual_gate"]["status"],
        "output": str(output.resolve()),
        "learned_visual_gate": report["learned_visual_gate"],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the real source timeline and noise before motion preparation.

This prepares an input package, not completed skeleton conditioning.
Dependencies: av==15.1.0 numpy==1.26.4 torch==2.8.0 safetensors==0.6.2
packaging==25.0. Use a clean pinned upstream checkout.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = "8cd60c40d90882de07645cc435dcf24bc9b4fbd1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4196)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    if revision != REVISION or dirty:
        raise SystemExit("Use a clean upstream checkout at " + REVISION)
    if args.seed < 0:
        raise SystemExit("The seed must be nonnegative.")
    if args.output.exists():
        raise SystemExit("Choose a new output directory to preserve frozen input packages.")
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    from safetensors.torch import save_file
    from fdanyone.video import decode_canonical_clip, write_lossless_video, write_gvhmr_video
    from fdanyone.views import resolve_view_plan
    from fdanyone.model.conditioning import load_prompt_context

    context = load_prompt_context(args.prompt)
    clip = decode_canonical_clip(args.source)
    plan = resolve_view_plan(views_per_layer=4)
    args.output.mkdir(parents=True)
    shutil.copy2(args.source, args.output / args.source.name)
    shutil.copy2(args.prompt, args.output / "prompt_context.safetensors")
    clip.write_metadata(args.output / "source-timeline.json")
    write_lossless_video(clip, args.output / "source-canonical.mkv")
    write_gvhmr_video(clip, args.output / "source-gvhmr.mp4")
    # Pinned model.inference._noise uses CPU FP32 samples and then casts BF16.
    noise = torch.randn((4, 48, 31, 80, 44), generator=torch.Generator("cpu").manual_seed(args.seed)).bfloat16()
    save_file({"initial_latents": noise, "prompt_context": context}, args.output / "initial-inputs.safetensors")
    paths = sorted(path for path in args.output.iterdir() if path.is_file())
    manifest = dict(
        format="mere.run.4danyone.source-package", version=1, status="awaiting_motion_conditioning",
        upstreamRevision=revision, seed=args.seed, source=args.source.name, viewPlan=plan.to_dict(),
        frames=len(clip.frames), fpsNumerator=clip.fps_num, fpsDenominator=clip.fps_den,
        originalWidth=clip.width, originalHeight=clip.height,
        canonicalPixelsVerified=True,
        files={path.name: dict(sha256=digest(path), bytes=path.stat().st_size) for path in paths},
        missing=["Recovered motion", "Source foreground crop", "Four target skeleton videos", "Solved cameras",
                 "VAE source latents and encoded pose features"],
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

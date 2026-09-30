#!/usr/bin/env python3
"""Export all 24 Base steps with trained weights and frozen small-grid inputs.

Dependencies: torch==2.8.0 numpy==1.26.4 einops==0.8.1 safetensors==0.6.2
packaging==25.0. Runs the unmodified upstream denoising math on CPU in FP32.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from replay_4danyone_bf16 import CHECKPOINT_SHA256, REVISION, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    if revision != REVISION or dirty:
        raise SystemExit("Use a clean upstream checkout at " + REVISION)
    if digest(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("Released checkpoint checksum mismatch")
    if args.output.exists():
        raise SystemExit("Choose a new output directory to preserve earlier traces.")
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    from safetensors.torch import load_file, save_file
    from fdanyone.model.loader import _load_dit
    from fdanyone.model.denoise import denoise_group
    from fdanyone.model.routing import routing_steps
    from fdanyone.views import resolve_view_plan
    from fdanyone.vendor.diffsynth.schedulers.flow_match import FlowMatchScheduler

    torch.set_num_threads(8)
    trace = load_file(args.reference)
    model, _ = _load_dit(args.checkpoint, "sdpa")
    model.float()
    scheduler = FlowMatchScheduler(num_inference_steps=24, shift=5, sigma_min=0, extra_one_step=True)
    denoiser = SimpleNamespace(model=model, scheduler=scheduler, dtype=torch.float32)
    routes = routing_steps(view_plan=resolve_view_plan(views_per_layer=4), num_steps=24)
    latents = trace["latents"].clone()
    outputs = {"initial": latents.clone(), "timesteps": scheduler.timesteps.clone()}
    args.output.mkdir(parents=True)
    started = time.monotonic()
    with torch.inference_mode():
        for step, groups in enumerate(routes):
            for group in groups:
                indices = torch.tensor(group)
                updated = denoise_group(
                    denoiser, latents[indices], trace["sources"][:1], trace["context"],
                    trace["pose.features"][indices], trace["pose.null"], step,
                )
                latents.index_copy_(0, indices, updated)
            if not torch.isfinite(latents).all():
                raise SystemExit("Nonfinite output at step " + str(step + 1))
            outputs["step" + str(step + 1)] = latents.clone().contiguous()
            (args.output / "progress.json").write_text(json.dumps(dict(
                completedSteps=step + 1, seconds=time.monotonic() - started)) + "\n")
            print(f"Base step {step + 1}/24 complete", flush=True)
    output = args.output / "reference.safetensors"
    save_file(outputs, output)
    receipt = dict(
        upstreamRevision=revision, checkpointSHA256=CHECKPOINT_SHA256,
        inputReferenceSHA256=digest(args.reference), outputSHA256=digest(output),
        torchVersion=torch.__version__, device="cpu", dtype="float32", steps=24,
        views=4, sourceViews=1, initialShape=list(latents.shape),
        elapsedSeconds=time.monotonic() - started,
        scope="Full Base trajectory on a small diagnostic grid; not full-resolution video or recovered motion.",
    )
    (args.output / "reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay the unmodified trained BF16 graph on frozen small-grid inputs.

Dependencies: torch==2.8.0 numpy==1.26.4 einops==0.8.1 safetensors==0.6.2
packaging==25.0. CUDA is required by default; CPU must be selected explicitly
and is a diagnostic, not evidence of CUDA parity. No motion recovery runs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REVISION = "8cd60c40d90882de07645cc435dcf24bc9b4fbd1"
CHECKPOINT_SHA256 = "aff60b0db2d333bd9e960a9cf3333cc8dd40fe76614a22f75a0da72be4e8289f"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; no CPU fallback is allowed.")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    if revision != REVISION or dirty:
        raise SystemExit("Use a clean upstream checkout at " + REVISION)
    if digest(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("Released checkpoint checksum mismatch")
    if args.output.exists():
        raise SystemExit("Choose a new output directory to preserve earlier traces.")
    sys.path.insert(0, str(args.upstream.resolve()))
    from fdanyone.model.loader import _load_dit

    torch.set_num_threads(8)
    trace = load_file(args.reference)
    model, _ = _load_dit(args.checkpoint, "sdpa")
    model.to(device=args.device, dtype=torch.bfloat16)
    tensors = {}
    metrics = {}

    def record(key, value):
        tensors[key] = value.detach().cpu().clone().contiguous()

    def frozen(key):
        return trace[key].to(device=args.device, dtype=torch.bfloat16)

    started = time.monotonic()
    with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        for name, count in [("direct", 1), ("packed", 5)]:
            hooks = [
                model.text_embedding.register_forward_hook(
                    lambda module, inputs, output, label=name + ".context": record(label, output)),
                model.time_embedding.register_forward_hook(
                    lambda module, inputs, output, label=name + ".time": record(label, output)),
                model.blocks[0].register_forward_pre_hook(
                    lambda module, inputs, label=name + ".assembly": record(label, inputs[0])),
            ]
            for index in [0, 14, 29]:
                hooks.append(model.blocks[index].register_forward_hook(
                    lambda module, inputs, output, label=f"{name}.block{index}": record(label, output)))
            prediction = model(
                x=frozen("latents"), x_src=frozen("sources")[:count],
                timestep=torch.full((4,), 625, device=args.device, dtype=torch.bfloat16),
                context=frozen("context"), pose_features=frozen("pose.features"),
                null_pose_feature=frozen("pose.null").repeat(1 if count == 1 else 2, 1, 1, 1, 1),
            )
            record(name + ".prediction", prediction)
            for hook in hooks:
                hook.remove()
            actual = tensors[name + ".prediction"].float()
            expected = trace[name + ".bf16_input_control"].float()
            if not torch.isfinite(actual).all():
                raise SystemExit("Nonfinite " + name + " prediction")
            error = actual - expected
            metrics[name] = dict(
                normalizedRMSE=float(torch.sqrt(error.square().mean() / expected.square().mean())),
                maximumError=float(error.abs().max()),
            )
            print(name + " prediction complete", flush=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
    args.output.mkdir(parents=True)
    output = args.output / (args.device + "-bf16-trace.safetensors")
    save_file(tensors, output)
    receipt = dict(
        upstreamRevision=revision, checkpointSHA256=CHECKPOINT_SHA256,
        inputReferenceSHA256=digest(args.reference), traceSHA256=digest(output),
        torchVersion=torch.__version__, device=args.device, weightDtype="bfloat16",
        autocastDtype="bfloat16", attentionBackend="sdpa", elapsedSeconds=time.monotonic() - started,
        metricsAgainstAlignedFP32=metrics,
        scope="Two trained 30-block forwards on frozen small-grid inputs; not a generation trajectory or visual qualification.",
    )
    if args.device == "cuda":
        receipt.update(gpu=torch.cuda.get_device_name(), cudaVersion=torch.version.cuda)
    (args.output / (args.device + "-bf16-trace.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

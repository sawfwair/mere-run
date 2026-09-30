#!/usr/bin/env python3
"""Export trained-checkpoint 4DAnyone and Wan VAE reference tensors on CPU.

Dependencies: torch==2.8.0 numpy==1.26.4 einops==0.8.1 safetensors==0.6.2
packaging==25.0. ffmpeg must be available. No automatic motion recovery runs.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

REVISION = "8cd60c40d90882de07645cc435dcf24bc9b4fbd1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def native_vae_weights(state):
    result = {}
    for original, value in state.items():
        key = original.removeprefix("model.")
        key = re.sub(r"\.(residual|head)\.(\d+)\.", r".\1.layer_\2.", key)
        key = key.replace(".resample.1.weight", ".resample_weight")
        key = key.replace(".resample.1.bias", ".resample_bias")
        for child in ["to_qkv", "proj"]:
            key = key.replace("." + child + ".weight", "." + child + "_weight")
            key = key.replace("." + child + ".bias", "." + child + "_bias")
        if value.ndim == 5:
            value = value.permute(0, 2, 3, 4, 1)
        elif value.ndim == 4 and not key.endswith("gamma"):
            value = value.permute(0, 2, 3, 1)
        if key.endswith("gamma"):
            value = value.reshape(-1)
        result[key] = value.contiguous()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=["vae", "transformer"], required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if revision != REVISION or dirty:
        raise SystemExit("Use a clean upstream checkout at " + REVISION)
    sys.path.insert(0, str(upstream))
    import numpy as np
    import torch
    from safetensors.torch import save_file, load_file
    from fdanyone.model.loader import _load_dit, _load_vae, load_pose_encoder
    from fdanyone.model.conditioning import load_prompt_context

    torch.set_num_threads(8)
    torch.manual_seed(4172)
    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {}
    receipt = dict(upstreamRevision=revision, torchVersion=torch.__version__, device="cpu", dtype="float32",
                   stage=args.stage, conditioning="Diagnostic inputs; no recovered human motion or visual-quality claim.")
    started = time.monotonic()

    def record(key, value):
        tensors[key] = value.detach().clone().cpu().contiguous()

    with torch.inference_mode():
        if args.stage == "vae":
            checkpoint = args.assets / "Wan2.2_VAE.pth"
            expected = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
            if digest(checkpoint) != expected:
                raise SystemExit("VAE checkpoint checksum mismatch")
            video_path = args.assets / "7017803-hd_1080_1920_30fps.mp4"
            decoded = subprocess.check_output([
                "ffmpeg", "-v", "error", "-i", str(video_path), "-vf",
                "fps=8,scale=96:64:force_original_aspect_ratio=increase,crop=96:64",
                "-frames:v", "5", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ])
            video = torch.from_numpy(np.frombuffer(decoded, dtype=np.uint8).copy())
            video = video.reshape(5, 64, 96, 3).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1
            record("video", video)
            model = _load_vae(checkpoint, torch.float32)
            save_file(native_vae_weights(model.state_dict()), args.output / "vae.safetensors")
            print("VAE: native weights exported", flush=True)
            for frames in [1, 5]:
                sample = video[:, :, :frames]
                latent = model.encode_view(sample)
                record(f"encode{frames}", latent)
                record(f"decode{frames}", model.decode_view(latent))
                print(f"VAE: {frames} frames encoded and decoded", flush=True)
            receipt["checkpointSHA256"] = expected
            receipt["sourceSHA256"] = digest(video_path)
        else:
            checkpoint = args.assets / "model.safetensors"
            expected = "aff60b0db2d333bd9e960a9cf3333cc8dd40fe76614a22f75a0da72be4e8289f"
            if digest(checkpoint) != expected:
                raise SystemExit("Transformer checkpoint checksum mismatch")
            vae_trace = load_file(args.output / "vae-reference.safetensors")
            source = vae_trace["encode5"]
            context = load_prompt_context(args.assets / "prompt_context.safetensors").float()
            # Fixed diagnostic RGB inputs exercise learned pose weights. These
            # are explicitly not a replacement for rendered motion skeletons.
            pose_video = torch.randn(4, 3, 5, 64, 96) * 0.4
            pose_model = load_pose_encoder(checkpoint, "cpu").float()
            poses = pose_model(pose_video)
            null = pose_model(torch.full_like(pose_video[:1], -1))
            record("pose.video", pose_video)
            record("pose.features", poses)
            record("pose.null", null)
            del pose_model
            gc.collect()
            latents = torch.randn(4, 48, 2, 4, 6)
            # Five distinct frozen source tensors cover reference packing.
            sources = torch.cat([source] + [source * (0.9 + i * 0.04) for i in range(4)])
            for key, value in dict(latents=latents, sources=sources, context=context).items():
                record(key, value)
            model, _ = _load_dit(checkpoint, "sdpa")
            model = model.float()
            print("Transformer: all trained weights loaded", flush=True)
            for name, count in [("direct", 1), ("packed", 5)]:
                hooks = []
                for index in [0, 14, 29]:
                    hooks.append(model.blocks[index].register_forward_hook(
                        lambda module, inputs, output, label=f"{name}.block{index}": record(label, output)))
                prediction = model(
                    x=latents, x_src=sources[:count], timestep=torch.full((4,), 625.0),
                    context=context, pose_features=poses,
                    null_pose_feature=null.repeat(1 if count == 1 else 2, 1, 1, 1, 1),
                )
                record(name + ".prediction", prediction)
                for hook in hooks:
                    hook.remove()
                print(f"Transformer: {name} prediction complete", flush=True)
            # Separate arithmetic drift from input quantization. The released
            # denoiser casts its timestep to BF16: 625 becomes 624. Feed the FP32
            # control exactly the values the native BF16 model will receive.
            def quantized(value):
                return value.bfloat16().float()

            for name, count in [("direct", 1), ("packed", 5)]:
                prediction = model(
                    x=quantized(latents), x_src=quantized(sources[:count]),
                    timestep=quantized(torch.full((4,), 625.0)), context=quantized(context),
                    pose_features=quantized(poses),
                    null_pose_feature=quantized(null.repeat(1 if count == 1 else 2, 1, 1, 1, 1)),
                )
                record(name + ".bf16_input_control", prediction)
            receipt["bf16Control"] = "FP32 graph with BF16-rounded inputs, including timestep 625 -> 624."
            receipt["checkpointSHA256"] = expected

    filename = args.stage + "-reference.safetensors"
    save_file(tensors, args.output / filename)
    receipt.update(elapsedSeconds=time.monotonic() - started, tensors=len(tensors),
                   referenceSHA256=digest(args.output / filename))
    (args.output / (args.stage + "-reference.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

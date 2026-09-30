#!/usr/bin/env python3
"""Export small CPU/FP32 fixtures by executing pinned, unmodified 4DAnyone code.

uv run --python 3.11 --with torch==2.8.0 --with numpy==1.26.4 \
  --with einops==0.8.1 --with safetensors==0.6.2 --with packaging==25.0 python \
  scripts/reference-parity/export_4danyone_fixture.py --upstream CHECKOUT \
  --checkpoint-header HEADER_JSON --output Tests/MereRunCoreTests/Fixtures/FourDAnyone
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REVISION = "8cd60c40d90882de07645cc435dcf24bc9b4fbd1"
MODEL_REVISION = "4c80e87b805a5f8461cf339cdbe2fb4249e585aa"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint-header", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if revision != REVISION or dirty:
        raise SystemExit("Use a clean upstream checkout at " + REVISION)
    sys.path.insert(0, str(root))
    import torch
    from safetensors.torch import save_file
    from fdanyone.vendor.diffsynth.models import wan_video_dit as dit
    from fdanyone.vendor.diffsynth.models.wan_video_pose_encoder import PoseEncoder
    from fdanyone.vendor.diffsynth.schedulers.flow_match import FlowMatchScheduler
    from fdanyone.model.denoise import denoise_group
    from fdanyone.model.routing import denoising_camera_order, routing_steps
    from fdanyone.views import resolve_view_plan

    torch.set_num_threads(2)
    torch.manual_seed(4162)
    # Change dimensions only. Every forward, routing, packing and scheduler
    # operation is imported from the pinned upstream checkout.
    dimensions = dict(LATENT_CHANNELS=4, MODEL_DIM=24, FFN_DIM=40,
                      FREQUENCY_DIM=8, TEXT_DIM=16, NUM_HEADS=2, NUM_LAYERS=2)
    for key, value in dimensions.items():
        setattr(dit, key, value)
    model = dit.FourDAnyoneDiT(attention_backend="sdpa").eval()
    model.freqs = dit.precompute_freqs_cis_3d(12)
    tensors = {}
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()},
              args.output / "transformer.safetensors")

    def record(name, value):
        tensors[name] = value.detach().clone().contiguous()

    with torch.inference_mode():
        # 6x10 is deliberately not divisible by four: ViewPack must pad and crop.
        latents = torch.randn(4, 4, 2, 6, 10) * 0.7
        sources = torch.randn(5, 4, 2, 6, 10) * 0.5
        poses = torch.randn(4, 24, 2, 3, 5) * 0.2
        null = torch.randn(2, 24, 2, 3, 5) * 0.1 + 0.03
        context = torch.randn(1, 7, 16) * 0.4
        for key, value in dict(latents=latents, sources=sources, poses=poses,
                               null=null, context=context).items():
            record(key, value)
        for name, count in [("direct", 1), ("packed", 5)]:
            packed_count = 1 if count == 1 else 2
            tokens, grid = model._patchify(latents)
            tokens, _ = model._pack_sources(tokens, sources[:count], grid)
            tokens = model._add_pose_features(tokens, poses, null[:packed_count], 4, packed_count, grid)
            record(name + ".assembly", tokens)
            times = torch.tensor([625.0] * 4 + [0.0] * packed_count)
            time = model.time_embedding(dit.sinusoidal_embedding_1d(8, times))
            record(name + ".time", time)
            handle = model.blocks[0].register_forward_hook(
                lambda module, inputs, output, prefix=name: record(prefix + ".block0", output))
            output = model(x=latents, x_src=sources[:count], timestep=times[:4], context=context,
                           pose_features=poses, null_pose_feature=null[:packed_count])
            handle.remove()
            record(name + ".prediction", output)

        plans = []
        for views, pitches, enabled in [(6, [15], True), (8, [30, -15, 15], True), (8, [15], False)]:
            plan = resolve_view_plan(views_per_layer=views, layer_pitches=pitches, enable_tcr=enabled)
            plans.append(dict(viewsPerLayer=views, pitches=pitches, routing=enabled,
                              order=denoising_camera_order(plan), routes=routing_steps(view_plan=plan, num_steps=5)))
        plan = resolve_view_plan(views_per_layer=8)
        scheduler = FlowMatchScheduler(num_inference_steps=4, shift=5, sigma_min=0, extra_one_step=True)
        denoiser = SimpleNamespace(model=model, scheduler=scheduler, dtype=torch.float32)
        initial = torch.randn(8, 4, 2, 6, 10) * 0.5
        all_poses = torch.randn(8, 24, 2, 3, 5) * 0.2
        record("generation.initial", initial)
        record("generation.poses", all_poses)
        generated = initial.clone()
        for step, groups in enumerate(routing_steps(view_plan=plan, num_steps=4)):
            for group in groups:
                ids = torch.tensor(group)
                updated = denoise_group(denoiser, generated[ids], sources, context,
                                        all_poses[ids], null, step)
                generated.index_copy_(0, ids, updated)
            record("generation.step" + str(step + 1), generated)
        record("schedule4", torch.cat([scheduler.sigmas, torch.zeros(1)]))
        scheduler.set_timesteps(24)
        record("schedule24", torch.cat([scheduler.sigmas, torch.zeros(1)]))

        # Full production pose channels; deterministic weights avoid committing
        # 14 MB of generated parameters. The Swift test implements this weight
        # recipe, while outputs and all convolutions come from upstream.
        pose = PoseEncoder(out_dim=3072).eval()
        recipe = []
        for index, (key, value) in enumerate(sorted(pose.state_dict().items())):
            fan = math.prod(value.shape[1:]) if value.ndim > 1 else 1
            amplitude = math.sqrt(6 / fan) if value.ndim > 1 else 0.01
            offset = 0.0 if value.ndim > 1 else 0.02
            if key == "scale":
                amplitude, offset = 0.0, 2.0
            pattern = ((torch.arange(value.numel(), dtype=torch.int64) * 17 + index * 13) % 101 - 50).float()
            value.copy_((pattern / 50 * amplitude + offset).reshape(value.shape))
            recipe.append(dict(key=key, shape=list(value.shape), index=index,
                               amplitude=amplitude, offset=offset))
        video = torch.randn(1, 3, 5, 32, 64) * 0.6
        record("pose.video", video)
        record("pose.output", pose(video))
        record("pose.nullOutput", pose(torch.full_like(video, -1)))

    save_file(tensors, args.output / "reference.safetensors")
    header = json.loads(args.checkpoint_header.read_text())
    schema = {key: value["shape"] for key, value in header.items() if key != "__metadata__"}
    if len(schema) != 1182 or any(value["dtype"] != "BF16" for key, value in header.items() if key != "__metadata__"):
        raise SystemExit("Unexpected release header; require the recorded 1182 BF16 tensors")
    (args.output / "checkpoint-schema.json").write_text(json.dumps(schema, sort_keys=True, indent=2) + "\n")
    source_files = ["fdanyone/vendor/diffsynth/models/wan_video_dit.py",
                    "fdanyone/vendor/diffsynth/models/wan_video_pose_encoder.py",
                    "fdanyone/vendor/diffsynth/schedulers/flow_match.py",
                    "fdanyone/model/routing.py", "fdanyone/model/denoise.py", "fdanyone/views.py"]
    manifest = dict(upstreamRevision=REVISION, modelRevision=MODEL_REVISION, torchVersion=torch.__version__,
                    device="cpu", dtype="float32", dimensions=dimensions, plans=plans, poseWeights=recipe,
                    sourceHashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
                    files={name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in
                           ["transformer.safetensors", "reference.safetensors", "checkpoint-schema.json"]},
                    scope="Synthetic weights and prepared tensors; not trained-checkpoint or CUDA/BF16 parity.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tensors": len(tensors), "schemaKeys": len(schema)}))


if __name__ == "__main__":
    main()

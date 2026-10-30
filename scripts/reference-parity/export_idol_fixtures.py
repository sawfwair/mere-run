"""Generate tiny CPU oracle fixtures from speridlabs/idol-sdk at 4702a7f.

Usage: python export_idol_fixtures.py --source /path/to/idol-sdk --output Tests/ImageRuntimeTests/Fixtures/Idol
Requires torch, safetensors, tensordict, numpy, pillow, plyfile, tqdm.
The reference SDK is loaded verbatim; no checkpoint assets are downloaded.
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import torch
from safetensors.torch import save_file
from tensordict import TensorDict

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
revision = "4702a7f00850e34b6e83a2170cc123c3e9570744"
assert subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip() == revision
# Avoid pipeline imports (which pull in the unrelated background-removal dependency).
for name, relative in [("idol", "idol"), ("idol.models", "idol/models")]:
    package = types.ModuleType(name)
    package.__path__ = [str(args.source / relative)]
    sys.modules[name] = package
for name, relative in [("idol.gaussian", "idol/gaussian.py"), ("idol.models.layers", "idol/models/layers.py"),
                       ("idol.models.dit", "idol/models/dit.py"), ("idol.models.decoder", "idol/models/decoder.py"),
                       ("idol.sampler", "idol/sampler.py")]:
    spec = importlib.util.spec_from_file_location(name, args.source / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from idol.models.dit import LatentSeqMMFlowModel
from idol.models.decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder
from idol.sampler import sample_flow

args.output.mkdir(parents=True, exist_ok=True)
torch.manual_seed(20261030)
representation = dict(num_gaussians=32, lr=dict(_xyz=1., _features_dc=1., _opacity=1., _scaling=1., _rotation=.1),
                      perturbe_size=1.5, scaling_bias=.004, opacity_bias=.1, **{"3d_filter_kernel_size": .0009})
den_config = dict(q_token_length=4, in_channels=4, cam_channels=5, model_channels=32, cond_channels=8,
                  cond2_channels=4, num_blocks=2, num_refiner_blocks=2, num_heads=4, num_head_channels=8, mlp_ratio=2.)
dec_config = dict(model_channels=32, cond_channels=4, num_blocks=2, num_heads=4, mlp_ratio=2.)
gs_config = dict(**dec_config, in_channels=3, representation_config=representation)
configuration = dict(denoiser=dict(name="LatentSeqMMFlowModel", args=den_config),
                     decoder=dict(name="OctreeProbabilityFixedlenDecoder", args=dec_config),
                     decoder_gs=dict(name="GaussianFixedlenDecoder", args=gs_config), image_size=16, max_voxel_level=3,
                     image_encoders=dict(dinov3="facebook/dinov3-vith16plus-pretrain-lvd1689m", flux2_vae="black-forest-labs/FLUX.2-dev"))
(args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
models = [("denoiser", LatentSeqMMFlowModel(**den_config)), ("decoder", OctreeProbabilityFixedlenDecoder(**dec_config)),
          ("decoder_gs", GaussianFixedlenDecoder(**gs_config))]
for name, model in models:
    with torch.no_grad():
        for key, value in model.named_parameters():
            value.copy_(torch.randn_like(value.float()) * .05)
            if key.endswith(("gamma", "norm.weight", "norm1.weight", "norm2.weight")):
                value.add_(1)
        if name == "denoiser": model.pos_pe.copy_(torch.rand_like(model.pos_pe))
        if name == "decoder_gs":
            model.points_offset_perturbation.copy_(torch.randn_like(model.points_offset_perturbation) * .1)
            model.base_offset_scale.fill_(-.75)
    save_file({key: value.float().contiguous() for key, value in model.state_dict().items()}, args.output / f"{name}.safetensors")
    model.eval()
den, dec, gs = [model for _, model in models]
latent = torch.randn(1, 4, 4)
camera = torch.randn(1, 1, 5)
dino = torch.randn(1, 3, 8)
vae = torch.randn(1, 3, 4)
points = torch.rand(1, 3, 3)
posterior = torch.arange(32 * 4 * 6, dtype=torch.float32).reshape(1, 32, 4, 6) / 700
mean = torch.linspace(-.2, .2, 128)
variance = torch.linspace(.1, 1., 128)
inputs = dict(latent=latent, camera=camera, dino=dino, vae=vae, points=points, posterior=posterior,
              running_mean=mean, running_variance=variance)
save_file(inputs, args.output / "inputs.safetensors")
state = TensorDict(dict(latent=latent, camera=camera), batch_size=1)
condition = TensorDict(dict(feature1=dino, feature2=vae), batch_size=1)
with torch.no_grad():
    velocity = den(state, torch.tensor([413.725]), condition)
    trajectory = sample_flow(den, state, condition, condition.apply(torch.zeros_like), steps=3, cfg_strength=3., rescale_t=3., verbose=False)
    logits = dec(points, torch.tensor([8]), latent)
    features = gs(points, latent)
    gaussian = gs.to_representation(points, features)[0]
    results = dict(velocity_latent=velocity["latent"], velocity_camera=velocity["camera"], trajectory_latent=trajectory["latent"],
                   trajectory_camera=trajectory["camera"], octree_logits=logits, gaussian_features=features,
                   xyz=gaussian.xyz, dc=gaussian.features_dc.reshape(-1, 3), scale=gaussian.scaling, rotation=gaussian.rotation,
                   opacity=gaussian.opacity)
    for dtype, label in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        packed = posterior.to(dtype).view(1, 32, 2, 2, 3, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 128, 2, 3)
        std = (variance.to(dtype) + 1e-4).sqrt().view(1, 128, 1, 1)
        normalized = ((packed - mean.to(dtype).view(1, 128, 1, 1)) / std).float().flatten(2).transpose(1, 2)
        results[f"packed_{label}"] = torch.cat([torch.zeros(1, 5, 128), normalized], dim=1).contiguous()
    save_file({key: value.contiguous() for key, value in results.items()}, args.output / "results.safetensors")
    gaussian.save_ply(str(args.output / "expected.ply"))

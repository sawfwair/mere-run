#!/usr/bin/env python3
"""Compare native AuK trained-weight receipts with upstream PyTorch on CPU.

Run after qualify-auk-reference.py and AuKQualificationTests. This checks VAE
encode/decode and a guided or unguided diffusion forward pass, not a CUDA run.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--receipts', type=Path, required=True)
p.add_argument('--variant', choices=['base', 'flash'], required=True)
a = p.parse_args()
revision = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if revision != '6943a1e967409e8c73139a7a345f2a611cfb3dd6':
    raise SystemExit('Unexpected upstream revision')
sys.path.insert(0, str(a.upstream / 'src'))
from auk.model import Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig

torch.set_num_threads(8)
inputs = mx.load(str(a.receipts / 'inputs.safetensors'))
conditioning = mx.load(str(a.receipts / 'conditioning.safetensors'))
diffusion = mx.load(str(a.receipts / 'diffusion.safetensors'))
reference = mx.load(str(a.receipts / 'reference.safetensors'))
native = mx.load(str(a.receipts / 'native-parity.safetensors'))
config = OmegaConf.load(a.model / 'config.yaml')
metrics = {}
def tensor(value):
    return torch.from_numpy(np.array(value, dtype=np.float32))
def check(name, actual, tolerance=0.001):
    expected = tensor(native[name])
    difference = actual - expected
    relative = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected).clamp_min(1e-20))
    metrics[name] = {'relative_l2': relative, 'maximum_absolute': float(difference.abs().max()), 'tolerance': tolerance}
    print(name, metrics[name], flush=True)
    (a.receipts / 'torch-parity.json').write_text(json.dumps(metrics, indent=2))
    if not np.isfinite(relative) or relative >= tolerance:
        raise RuntimeError('PyTorch parity failed: ' + name)

with torch.no_grad():
    kwargs = OmegaConf.to_container(config.model.vae.model_init_kwargs, resolve=True)
    vae = load_vae_model('BigVGANFlowVAE', BigVGANFlowVAEConfig.from_dict(kwargs),
                         str(a.model / 'vae.safetensors'), map_location='cpu').float().eval()
    stats = vae.audio_encoder(tensor(inputs['wave24']).permute(0, 2, 1))
    mean = stats[:, :64].transpose(1, 2)
    latent = (mean - vae.global_mean.float()) / torch.sqrt(vae.global_log_std.float())
    check('reference_latent', latent)
    for name in ['text', 'audio']:
        latent = tensor(diffusion[name + '_latent'])
        wave = vae.inference_from_latents(vae.denormalize(latent).permute(0, 2, 1)).permute(0, 2, 1)
        check(name + '_decoder', wave)
    del vae, stats, mean, latent, wave
    gc.collect()
    arch = OmegaConf.to_container(config.model.arch, resolve=True)
    arch.update(attn_backend='torch', checkpoint_activations=False, latent_dim=64)
    dit = Flux2Edit(**arch).float().eval()
    raw = load_file(str(a.model / ('auk_' + a.variant + '.safetensors')))
    dit.load_state_dict({k.removeprefix('transformer.'): v for k, v in raw.items() if k.startswith('transformer.')}, strict=False)
    del raw
    gc.collect()
    for name in ['text', 'audio']:
        dit.clear_cache()
        ref = tensor(reference['latent']) if name == 'audio' else torch.zeros(1, 0, 64)
        pred = dit(x=tensor(diffusion['initial']), text=tensor(conditioning[name]), time=torch.tensor([0.0]),
                   ref=ref, cfg_infer=a.variant == 'base', cache=False)
        velocity = pred[:1] + 2 * (pred[:1] - pred[1:2]) if a.variant == 'base' else pred
        check(name + '_velocity', velocity)
print('PyTorch CPU component parity passed', flush=True)

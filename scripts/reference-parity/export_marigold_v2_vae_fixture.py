# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.10.0", "diffusers==0.38.0", "transformers==5.17.0",
#                 "accelerate==1.15.0", "pillow==12.3.0", "numpy==2.5.3"]
# ///
"""Capture the pinned Diffusers VAE on one shared 256 by 176 input tensor."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLQwenImage
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--image", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", default="mps")
p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
dtype = getattr(torch, a.dtype)
im = Image.open(a.image).convert("RGB").resize((256, 176), Image.Resampling.BICUBIC)
x = torch.from_numpy(np.array(im).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].contiguous()
save_file({"input": x}, a.output / "input.safetensors")
vae = AutoencoderKLQwenImage.from_pretrained(a.root / "vae", torch_dtype=dtype, local_files_only=True).to(a.device).eval()
out = {"input": x}
def put(name, t):
    out[name] = t.detach().cpu().float().contiguous()
    print(name, list(t.shape), str(t.dtype), float(t.float().std()), flush=True)
with torch.no_grad():
    posterior = vae.encode(x[:, :, None].to(device=a.device, dtype=dtype)).latent_dist
    raw = posterior.mode()
    put("raw_mode", raw.squeeze(2))
    put("posterior_std", posterior.std.squeeze(2))
    mean = torch.tensor(vae.config.latents_mean, device=a.device, dtype=dtype).view(1, 16, 1, 1, 1)
    inv_std = 1 / torch.tensor(vae.config.latents_std, device=a.device, dtype=dtype).view(1, 16, 1, 1, 1)
    put("normalized", ((raw - mean) * inv_std).squeeze(2))
    decoded = vae.decode(raw).sample
    put("base_decoded", decoded.squeeze(2))
    state = load_file(a.root / "marigold/depth/Log-stage2/trainables.safetensors")
    decoder = {k.removeprefix("VAE."): v for k, v in state.items() if k.startswith("VAE.")}
    result = vae.load_state_dict(decoder, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    print("decoder tensors", len(decoder), flush=True)
    put("tuned_decoded", vae.decode(raw).sample.squeeze(2))
save_file(out, a.output / f"reference-{a.dtype}.safetensors")
for key in ["base_decoded", "tuned_decoded"]:
    px = ((out[key][0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(px).save(a.output / f"reference-{a.dtype}-{key}.png")
(a.output / f"reference-{a.dtype}.json").write_text(json.dumps({"torch": torch.__version__, "diffusers": "0.38.0", "device": a.device, "dtype": a.dtype}, indent=2))

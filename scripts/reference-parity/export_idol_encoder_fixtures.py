"""Tiny shared-encoder oracles for IDOL. No pretrained weights or network calls.

Requires torch 2.15.0, transformers 5.22.1, diffusers 0.42.1, safetensors.
Usage: python export_idol_encoder_fixtures.py --output Tests/ImageRuntimeTests/Fixtures/Idol
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)
config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
    image_size=32, patch_size=16, num_register_tokens=4, use_gated_mlp=True, hidden_act="silu",
    query_bias=True, key_bias=False, value_bias=True)
dino = DINOv3ViTModel(config).eval()
image = torch.randn(1, 3, 32, 48)
vae = AutoencoderKLFlux2(block_out_channels=(32, 32), down_block_types=("DownEncoderBlock2D",) * 2,
    up_block_types=("UpDecoderBlock2D",) * 2, latent_channels=32, norm_num_groups=32, layers_per_block=1).eval()
vae_image = torch.randn(1, 3, 16, 16)
save_file({key.removeprefix("model."): value.contiguous() for key, value in dino.state_dict().items()}, args.output / "dino.safetensors")
save_file({key: value.contiguous() for key, value in vae.state_dict().items()}, args.output / "vae.safetensors")
outputs = {"dino_input": image, "vae_input": vae_image}
with torch.no_grad():
    outputs["dino_fp32"] = F.layer_norm(dino(image).last_hidden_state.float(), (32,))
    outputs["vae_fp32"] = vae.encode(vae_image).latent_dist.parameters
    dino.to(torch.bfloat16)
    vae.to(torch.bfloat16)
    outputs["dino_bf16"] = F.layer_norm(dino(image.to(torch.bfloat16)).last_hidden_state.float(), (32,))
    outputs["vae_bf16"] = vae.encode(vae_image.to(torch.bfloat16)).latent_dist.parameters
save_file({key: value.contiguous() for key, value in outputs.items()}, args.output / "encoder-results.safetensors")

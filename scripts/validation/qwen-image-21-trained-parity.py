#!/usr/bin/env python3
"""Compare opt-in Swift checkpoint exports against pinned Diffusers/Transformers.

Use the environment documented in qwen-image-21-reference.py. This script
loads only local checkpoint files, never downloads, and is not a runtime dependency.
Generate native-components.safetensors with QwenImage21QualificationTests first.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from diffusers import QwenImage21Transformer2DModel, AutoencoderKLQwenImage21
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
parser.add_argument('--native-file', default='native-components.safetensors')
parser.add_argument('--report-name', default='trained-parity')
parser.add_argument('--transformer-only', action='store_true')
parser.add_argument('--tolerance', type=float, default=0.05)
parser.add_argument('--prepare-tokenizer', action='store_true', help='Write tokenizer cases before the native tests.')
args = parser.parse_args()
if args.prepare_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'processor', local_files_only=True)
    system = '<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n'
    prompts = ['A red ceramic teapot on a wooden table.', '一只红色茶壶，透明背景。',
               'Café — naïve typography: “mere.run” 🌻', ' ',
               '<image1><|vision_start|>' + '<|image_pad|>' * 16 + '<|vision_end|>Change the teapot to blue.',
               '<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>Combine both objects.']
    texts = [system] + [system + '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n' for prompt in prompts]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'tokenizer-reference.json').write_text(json.dumps(
        [dict(text=text, ids=tokenizer.encode(text)) for text in texts], ensure_ascii=False, indent=2) + '\n')
    raise SystemExit(0)
torch.set_num_threads(8)
torch.set_grad_enabled(False)
native = load_file(str(args.output / args.native_file))
target_grid = int(native.get('target_grid', torch.tensor(2)))
target_count = target_grid * target_grid
dtype = native['text_latents'].dtype
metrics, reference = {}, {}


def compare(name, actual):
    tolerance = args.tolerance
    actual = actual.detach().cpu().float()
    expected = native[name].float()
    if actual.shape != expected.shape:
        raise ValueError(f'{name}: shape {actual.shape} != {expected.shape}')
    error = actual - expected
    relative = (error.norm() / actual.norm().clamp_min(1e-20)).item()
    metrics[name] = dict(shape=list(actual.shape), relative_l2=relative,
                         maximum_absolute=error.abs().max().item(), tolerance=tolerance,
                         passed=bool(torch.isfinite(actual).all()) and relative < tolerance)
    reference[name] = actual.contiguous()
    print(name, metrics[name], flush=True)
    (args.output / (args.report_name + '.json')).write_text(json.dumps(metrics, indent=2) + '\n')


def release():
    gc.collect()
    if args.device == 'mps':
        torch.mps.empty_cache()


def compare_conditioner_and_vae():
    print('Loading local Qwen3-VL checkpoint', flush=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model / 'text_encoder', dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model / 'processor', local_files_only=True)
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    system = '<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n'
    drop = len(tokenizer.encode(system))
    assert drop == int(native['drop_count'])
    handle = encoder.model.language_model.norm.register_forward_hook(lambda module, values, result: values[0])
    for mode in ['text', 'image']:
        prefix = '' if mode == 'text' else '<image1><|vision_start|><|image_pad|><|vision_end|>'
        presentation = system + '<|im_start|>user\n' + prefix + 'A red ceramic teapot on a wooden table.<|im_end|>\n<|im_start|>assistant\n'
        ids = torch.tensor([tokenizer.encode(presentation)], device=args.device)
        assert torch.equal(ids.cpu(), native[mode + '_ids'].long()), 'Tokenizer differs'
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True,
                      mm_token_type_ids=(ids == image_id).long())
        if mode == 'image':
            pixels = native['vision'].to(args.device)
            patches = pixels.unsqueeze(2).repeat(1, 1, 2, 1, 1).reshape(1, 3, 2, 1, 2, 16, 1, 2, 16)
            kwargs.update(pixel_values=patches.permute(0, 3, 6, 4, 7, 1, 2, 5, 8).reshape(4, -1),
                          image_grid_thw=torch.tensor([[1, 2, 2]], device=args.device))
        result = encoder(**kwargs).hidden_states[-1][:, drop:]
        compare(mode + '_conditioning', result)
    handle.remove()
    del encoder, result, kwargs
    release()

    print('Loading local RGBA VAE checkpoint', flush=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(args.model / 'vae', torch_dtype=torch.bfloat16,
                                                 local_files_only=True).eval().to(args.device)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.bfloat16, device=args.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.bfloat16, device=args.device).view(1, -1, 1, 1, 1)
    pixels = native['rgba'].to(args.device).permute(0, 3, 1, 2).unsqueeze(2)
    encoded = (vae.encode(pixels).latent_dist.mode() - mean) / std
    compare('encoded', encoded[:, :, 0].permute(0, 2, 3, 1))
    latent = native['decode_latent'].to(args.device).permute(0, 3, 1, 2).unsqueeze(2)
    decoded = (vae.decode(latent * std + mean).sample + 1) / 2
    compare('decoded', decoded[:, :, 0].permute(0, 2, 3, 1))
    del vae, encoded, decoded
    release()


if not args.transformer_only:
    compare_conditioner_and_vae()

print('Loading local DiT checkpoint', flush=True)
dit = QwenImage21Transformer2DModel.from_pretrained(args.model / 'transformer',
    torch_dtype=dtype, local_files_only=True).eval().to(args.device)
for mode in ['text', 'image']:
    cache = QwenImage21KVCache(dit.config.num_layers)
    common = dict(encoder_hidden_states=native[mode + '_conditioning'].to(device=args.device, dtype=dtype),
                  img_mask=native[mode + '_mask'].to(args.device),
                  img_shapes=[([] if mode == 'text' else [(1, 2, 2)]) + [(1, target_grid, target_grid)]])
    full = dit(hidden_states=native[mode + '_latents'].to(args.device),
               timestep=torch.tensor([0.75], device=args.device),
               kv_cache=cache, kv_cache_mode='extract', **common).sample[:, -target_count:]
    compare(mode + '_velocity', full)
    cached = dit(hidden_states=native[mode + '_changed'].to(args.device),
                 timestep=torch.tensor([0.4], device=args.device),
                 kv_cache=cache, kv_cache_mode='cached', **common).sample[:, -target_count:]
    compare(mode + '_cached', cached)
    uncached = dit(hidden_states=native[mode + '_changed'].to(args.device),
                   timestep=torch.tensor([0.4], device=args.device), **common).sample[:, -target_count:]
    compare(mode + '_uncached', uncached)
save_file(reference, str(args.output / (args.report_name + '-reference.safetensors')))
raise SystemExit(0 if all(metric['passed'] for metric in metrics.values()) else 1)

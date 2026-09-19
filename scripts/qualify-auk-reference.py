#!/usr/bin/env python3
"""Create trained-weight parity receipts without duplicating converted checkpoints.

Requires the pinned AuK checkout, original AuK/Qwen weights, MLX 0.32.2,
PyTorch 2.7, transformers 4.57, soundfile, soxr, and omegaconf.
The upstream converter runs unchanged; its save callback retains tensors in RAM.
"""
import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
import soxr

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--thinker', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--variant', choices=['base', 'flash'], required=True)
p.add_argument('--frontend-only', action='store_true')
a = p.parse_args()
revision = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if revision != '6943a1e967409e8c73139a7a345f2a611cfb3dd6':
    raise SystemExit('Unexpected upstream revision')
sys.path.insert(0, str(a.upstream / 'src'))
from auk_mlx import convert
from auk_mlx.dit import DiTConfig, Flux2Edit
from auk_mlx.qwen_thinker import AudioConfig, TextConfig, ThinkerEncoder
from auk_mlx.vae import BigVGANFlowVAE, VAEConfig
from transformers import Qwen2_5OmniProcessor
from omegaconf import OmegaConf

a.out.mkdir(parents=True, exist_ok=True)
captured = {}
def retain(tensors, destination):
    captured[Path(destination).name] = {k: mx.array(v) for k, v in tensors.items()}
convert.save_file = retain

def release():
    gc.collect()
    mx.clear_cache()

def save(name, tensors):
    mx.eval(tensors)
    mx.save_safetensors(str(a.out / (name + '.safetensors')), tensors)
    print('Saved', name, flush=True)

cfg = OmegaConf.load(a.model / 'config.yaml')
wav, rate = sf.read(a.upstream / 'assets/demo-input-audio/zero-shot-tts/ref.wav', dtype='float32', always_2d=True)
# An off-hop boundary exercises audio chunk padding and placeholder rounding.
wav = wav.mean(axis=1)[:rate * 2 + 137]
wave24 = soxr.resample(wav, rate, 24000, quality='VHQ') if rate != 24000 else wav
wave16 = soxr.resample(wav, rate, 16000, quality='VHQ') if rate != 16000 else wav
instruction = 'Say the following in a calm voice: "Welcome home, how was work today?"'
processor = Qwen2_5OmniProcessor.from_pretrained(str(a.thinker))
inputs = {}
for name, audio in [('text', None), ('audio', wave16)]:
    content = [{'type': 'text', 'text': instruction + ('|<no_prompt_audio>|' if audio is None else '')}]
    if audio is not None:
        content.append({'type': 'audio', 'audio': 'reference.wav'})
    prompt = processor.apply_chat_template([[{'role': 'user', 'content': content}]], tokenize=False, add_generation_prompt=True)
    kwargs = {'audio': [audio]} if audio is not None else {}
    batch = processor(text=prompt, padding=True, return_tensors='np', **kwargs)
    values = {'ids': mx.array(batch['input_ids'])}
    if audio is not None:
        length = int(batch['feature_attention_mask'].sum())
        values['mel'] = mx.array(batch['input_features'][:, :, :length].transpose(0, 2, 1).copy())
    inputs[name] = values
save('inputs', {'wave24': mx.array(wave24.reshape(1, -1, 1)), 'wave16': mx.array(wave16),
                **{name + '_' + key: value for name, values in inputs.items() for key, value in values.items()}})
(a.out / 'request.json').write_text(json.dumps({'instruction': instruction, 'variant': a.variant, 'upstream_revision': revision}, indent=2))

if a.frontend_only:
    raise SystemExit(0)

convert.convert_dit(str(a.model / ('auk_' + a.variant + '.safetensors')), str(a.out / ('dit_' + a.variant + '.safetensors')))
fusion = captured.pop('fusion_' + a.variant + '.safetensors')
dit_weights = captured.pop('dit_' + a.variant + '.safetensors')
# Release the large DiT mapping until its stage; reconverting avoids a second disk copy.
del dit_weights
release()

convert.convert_thinker(str(a.thinker), str(a.out / 'metadata'))
meta = json.loads((a.out / 'metadata/thinker_config.json').read_text())
th = ThinkerEncoder(TextConfig(**meta['text']), AudioConfig(**meta['audio']))
th.load_weights(list(captured.pop('thinker.safetensors').items()), strict=False)
th.eval()
conditioning = {}
for name, values in inputs.items():
    ids = values['ids']
    kwargs = {}
    if 'mel' in values:
        token_id = json.loads((a.thinker / 'config.json').read_text())['thinker_config']['audio_token_index']
        kwargs = {'audio_features': values['mel'], 'audio_feature_len': values['mel'].shape[1], 'audio_token_mask': ids == token_id}
    hidden = th(ids, **kwargs)
    stacked = mx.stack([mx.fast.layer_norm(h, None, None, 1e-5) for h in hidden[1:]])
    conditioning[name] = (stacked * mx.softmax(fusion['layer_weights'])[:, None, None, None]).sum(axis=0) * fusion['layer_scale']
    mx.eval(conditioning[name])
    del hidden, stacked
save('conditioning', conditioning)
del th
release()

convert.convert_vae(str(a.model / 'vae.safetensors'), str(a.out / 'vae.safetensors'))
vae = BigVGANFlowVAE(VAEConfig.from_dict(OmegaConf.to_container(cfg.model.vae.model_init_kwargs, resolve=True)))
vae.load_weights(list(captured.pop('vae.safetensors').items()))
vae.eval()
reference = vae.encode(mx.array(wave24.reshape(1, -1, 1)))
mx.eval(reference)
save('reference', {'latent': reference})
del vae
release()

convert.convert_dit(str(a.model / ('auk_' + a.variant + '.safetensors')), str(a.out / ('dit_' + a.variant + '.safetensors')))
dit = Flux2Edit(DiTConfig.from_dict(dict(cfg.model.arch, latent_dim=64)), inv_freq=np.array(fusion['inv_freq']))
dit.load_weights(list(captured.pop('dit_' + a.variant + '.safetensors').items()))
captured.clear()
dit.eval()
initial = mx.random.normal((1, 100, 64), key=mx.random.key(42))
schedule = mx.array([0, .07612049579620361, .2928932309150696, .6173166036605835, 1]) if a.variant == 'flash' else 1 - mx.cos(mx.array(np.linspace(0, 1, 33, dtype=np.float32)) * (np.pi / 2))
outputs = {'initial': initial, 'schedule': schedule}
for name in ['text', 'audio']:
    ref = reference if name == 'audio' else mx.zeros((1, 0, 64))
    y = initial
    dit.clear_cache()
    for i in range(len(schedule) - 1):
        pred = dit(y, conditioning[name], schedule[i:i+1], ref=ref, cfg_infer=a.variant == 'base', cache=True)
        velocity = pred[:1] + 2 * (pred[:1] - pred[1:2]) if a.variant == 'base' else pred
        if i == 0:
            outputs[name + '_velocity'] = velocity
            mx.eval(velocity)
        y = y + velocity * (schedule[i+1] - schedule[i])
        mx.eval(y)
    outputs[name + '_latent'] = y
save('diffusion', outputs)
del dit
release()
convert.convert_vae(str(a.model / 'vae.safetensors'), str(a.out / 'vae.safetensors'))
vae = BigVGANFlowVAE(VAEConfig.from_dict(OmegaConf.to_container(cfg.model.vae.model_init_kwargs, resolve=True)))
vae.load_weights(list(captured.pop('vae.safetensors').items()))
vae.eval()
decoded = {name: vae.decode(outputs[name + '_latent']) for name in ['text', 'audio']}
save('waveforms', decoded)
print('Reference receipts complete', flush=True)

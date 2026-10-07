#!/usr/bin/env python3
"""Export small independent fixtures from the original, pinned LiquidAI D1 Python files.

Requires torch, transformers, safetensors. Pass directories containing the released
files from the revisions in docs/runtime/d1.md; this script does not run in inference.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from safetensors.torch import save_file
from transformers import Lfm2Config


def package(name, root):
    module = types.ModuleType(name)
    module.__path__ = [str(root)]
    sys.modules[name] = module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--causal-reference', type=Path, required=True)
    parser.add_argument('--omni-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    package('d1_causal_reference', args.causal_reference)
    package('d1_omni_reference', args.omni_reference)
    causal = importlib.import_module('d1_causal_reference.lfm2_vl')
    encoder = importlib.import_module('d1_omni_reference.encoder')
    audio = importlib.import_module('d1_omni_reference.audio')
    vision = importlib.import_module('d1_omni_reference.vision')
    torch.manual_seed(20261007)
    text = dict(vocab_size=128, hidden_size=64, intermediate_size=96,
                num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                layer_types=['conv', 'full_attention', 'conv'], norm_eps=1e-5,
                conv_L_cache=3, max_position_embeddings=256, rope_theta=10000.0,
                block_auto_adjust_ff_dim=False, block_ffn_dim_multiplier=1.0,
                block_multiple_of=16)
    vis = dict(model_type='siglip2_vision_model', hidden_size=8, intermediate_size=16,
               num_hidden_layers=2, num_attention_heads=2, num_channels=3,
               num_patches=256, patch_size=16, layer_norm_eps=1e-6, vision_use_head=False)
    aud = dict(feat_in=128, n_layers=2, d_model=16, subsampling_conv_channels=4,
               ff_expansion_factor=2, n_heads=2, conv_kernel_size=3, residual_width=16)
    config = dict(model_type='d1_omni', bos_token_id=1, text_config=text, vision_config=vis,
                  audio_config=aud, projector_hidden_size=16, head_layers=2, max_length=256,
                  image_text_length=128, audio_text_length=128, temperatures={'choice:2': 1.5})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    import torchvision.transforms.v2.functional as tvf
    arrays = {}
    expected = {}
    ids = torch.tensor([[1, 3, 5, 8, 13, 21, 34]])
    c = causal.language_model(Lfm2Config(**text)).eval()
    trunk = encoder.Trunk(text).eval()
    head = encoder.DecisionHead(64, 2).eval()
    tower = vision.Vision(vis, 16, 64).eval()
    sound = audio.Audio(aud, 64).eval()
    for prefix, module in [('causal', c), ('encoder', trunk), ('head', head), ('vision', tower), ('audio', sound)]:
        arrays.update({prefix + '.' + k: v.contiguous() for k, v in module.state_dict().items()})
    media = torch.randn(1, 4, 64)
    pixels = torch.randn(1, 16, 768)
    wave = 0.3 * torch.sin(torch.arange(9600) * 0.073) + 0.07 * torch.cos(torch.arange(9600) * 0.031)
    raster = torch.arange(17 * 31 * 3, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(17, 31, 3)
    arrays.update(ids=ids, media=media, pixels=pixels, waveform=wave, raster=raster)
    with torch.no_grad():
        expected['resize_up'] = tvf.resize(raster.permute(2, 0, 1), [32, 64], interpolation=tvf.InterpolationMode.BILINEAR, antialias=True).permute(1, 2, 0).contiguous()
        expected['resize_down'] = tvf.resize(raster.permute(2, 0, 1), [8, 12], interpolation=tvf.InterpolationMode.BILINEAR, antialias=True).permute(1, 2, 0).contiguous()
        expected['resize_cubic_up'] = tvf.resize(raster.permute(2, 0, 1), [32, 64], interpolation=tvf.InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).contiguous()
        expected['resize_cubic_down'] = tvf.resize(raster.permute(2, 0, 1), [8, 12], interpolation=tvf.InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).contiguous()
        expected['causal'] = c(ids).last_hidden_state
        expected['causal_logits'] = expected['causal'][:, -1] @ c.embed_tokens.weight.T
        expected['omni'] = trunk(trunk.embed_tokens(ids), torch.ones_like(ids).bool(), torch.tensor([0]))
        combined = torch.cat([media, trunk.embed_tokens(ids)], dim=1)
        expected['omni_media'] = trunk(combined, torch.ones(combined.shape[:2]).bool(), torch.tensor([4]))
        changed = combined.clone(); changed[:, 4:] += 10
        expected['omni_media_changed'] = trunk(changed, torch.ones(combined.shape[:2]).bool(), torch.tensor([4]))
        expected['head'] = head(expected['omni'], torch.ones_like(ids).bool(), torch.tensor([[2, 5]]),
                                torch.tensor([[True, True]]), torch.tensor([0]))
        inputs = dict(pixel_values=pixels, pixel_attention_mask=torch.ones(1, 16).bool(), spatial_shapes=torch.tensor([[4, 4]]))
        hidden = tower.tower(**inputs).last_hidden_state
        expected['vision'] = tower.projector(hidden.reshape(1, 4, 4, 8))
        expected['mel'], _ = sound.frontend(audio.waveform(wave))
        expected['audio'] = sound(wave)
    arrays.update({'expected.' + k: v.contiguous() for k, v in expected.items()})
    save_file(arrays, str(args.output / 'reference.safetensors'))
    # Tokenizer parity fixtures are optional when only arithmetic source files are available.
    if all((root / 'tokenizer.json').is_file() for root in [args.causal_reference, args.omni_reference]):
        from transformers import AutoTokenizer
        requests = [
            {'state': {'z': 'café 😀', 'a': True}, 'questions': {'route': {'type': 'choice', 'instructions': 'Choose a route.', 'criteria': {'last': 'Damage', 'first': 'Shipping'}}}},
            {'state': 'It arrived.', 'questions': {'arrived': {'type': 'noul', 'instructions': 'Did it arrive?'}}},
            {'state': None, 'questions': {'quality': {'type': 'score', 'instructions': 'Rate quality.', 'criteria': ['low', 'medium', 'high']}}},
            {'state': '<|mask|> ' + ('long state ' * 1500), 'questions': {'pick': {'type': 'choice', 'instructions': 'Choose <|reserved_9|>.', 'criteria': {'A': 'one', 'B': 'two'}}}},
        ]
        groups = []
        for family, namespace, root in [('causal', 'd1_causal_reference', args.causal_reference), ('omni', 'd1_omni_reference', args.omni_reference)]:
            prompt = importlib.import_module(namespace + '.prompt')
            tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
            cases = []
            for request in requests:
                question = prompt.as_question(next(iter(request['questions'].values())))
                if family == 'causal':
                    ids = tokenizer.encode(prompt.render(tokenizer, request['state'], question, bos=tokenizer.bos_token), add_special_tokens=False)
                    markers = []
                else:
                    ids, markers = prompt.encode(tokenizer, '' if request['state'] is None else request['state'], question, 16384)
                cases.append(dict(request=json.dumps(request, ensure_ascii=False), ids=ids, markers=markers))
            groups.append(dict(family=family, cases=cases))
        (args.output / 'prompts.json').write_text(json.dumps(groups, ensure_ascii=False, indent=2) + '\n')
    provenance = dict(causal_repository='LiquidAI/d1-3B', causal_revision='da1fe36a861f24690f27f622dca1d8688503d113',
                      omni_repository='LiquidAI/d1-omni-600M', omni_revision='414f8d6438174f5b2133a9c21a478fc42625e308',
                      torch=torch.__version__, transformers=importlib.import_module('transformers').__version__, torchvision=importlib.import_module('torchvision').__version__, seed=20261007, source_sha256={})
    for root in [args.causal_reference, args.omni_reference]:
        for path in sorted(root.glob('*.py')):
            provenance['source_sha256'][root.name + '/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()

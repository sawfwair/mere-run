#!/usr/bin/env python3
"""Export tiny, independently computed native Clef Omni parity fixtures.

Requires torch, transformers==5.10.2, numpy, safetensors, pillow. Pass a local
snapshot's joint_schema_model.py and config.json. No trained weights are copied.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType

import torch
import numpy as np
import transformers
from transformers import WhisperFeatureExtractor
from safetensors.torch import save_file
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTextConfig, Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel, Qwen3OmniMoeVisionEncoder, Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoePreTrainedModelForConditionalGeneration,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.reference_root / 'joint_schema_model.py'
    expected = '21d05ebb8cbea26a26af65eca3d16cf4b69bf80a320e2d5b203e86f0d632a670'
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected or transformers.__version__ != '5.10.2':
        raise ValueError('Reference source or Transformers version differs from the pinned exporter.')
    spec = importlib.util.spec_from_file_location('clef_omni_reference', source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    torch.manual_seed(719)
    torch.set_num_threads(1)
    root = json.loads((args.reference_root / 'config.json').read_text())
    thinker = root['thinker_config']
    text = thinker['text_config']
    text.update(hidden_size=8, vocab_size=256, num_hidden_layers=3, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, num_experts=4, num_experts_per_tok=2,
                moe_intermediate_size=12, intermediate_size=12)
    text['rope_scaling']['mrope_section'] = [2, 1, 1]
    vision = thinker['vision_config']
    vision.update(depth=3, hidden_size=8, intermediate_size=16, num_heads=2,
                  out_hidden_size=8, image_size=32, num_position_embeddings=4,
                  deepstack_visual_indexes=[0, 1, 2])
    audio = thinker['audio_config']
    audio.update(d_model=8, encoder_layers=2, num_hidden_layers=2, encoder_attention_heads=2,
                 encoder_ffn_dim=16, downsample_hidden_size=2, output_dim=8)
    text_model = Qwen3OmniMoeThinkerTextModel(Qwen3OmniMoeTextConfig(**text)).eval()
    vision_model = Qwen3OmniMoeVisionEncoder(Qwen3OmniMoeVisionEncoderConfig(**vision)).eval()
    audio_model = Qwen3OmniMoeAudioEncoder(Qwen3OmniMoeAudioEncoderConfig(**audio)).eval()
    # Avoid nearly-zero default-initialized experts hiding an incorrect router.
    for model in [text_model, vision_model, audio_model]:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.12)
                if 'norm' in name and name.endswith('weight'):
                    parameter.add_(1)
    weights = {}
    for key, value in text_model.state_dict().items():
        if key.endswith('mlp.experts.gate_up_proj'):
            prefix = key.removesuffix('gate_up_proj')
            for expert in range(4):
                gate, up = value[expert].chunk(2, dim=0)
                weights[f'thinker.model.{prefix}{expert}.gate_proj.weight'] = gate.clone()
                weights[f'thinker.model.{prefix}{expert}.up_proj.weight'] = up.clone()
        elif key.endswith('mlp.experts.down_proj'):
            prefix = key.removesuffix('down_proj')
            for expert in range(4):
                weights[f'thinker.model.{prefix}{expert}.down_proj.weight'] = value[expert].clone()
        else:
            weights['thinker.model.' + key] = value.clone()
    weights['thinker.lm_head.weight'] = torch.randn(256, 8) * 0.2
    for prefix, model in [('thinker.visual.', vision_model), ('thinker.audio_tower.', audio_model)]:
        weights.update({prefix + key: value.clone() for key, value in model.state_dict().items()})
    ids = torch.tensor([[5, 7, 3, 9, 12, 4, 15, 1, 19, 23, 26]])
    positions = torch.tensor([[[0, 1, 2, 2, 15, 16, 17, 18, 19, 20, 21]],
                              [[0, 1, 2, 3, 2, 16, 17, 18, 19, 20, 21]],
                              [[0, 1, 3, 2, 3, 16, 17, 18, 19, 20, 21]]], dtype=torch.float32)
    deepstack = [torch.randn(2, 8) * 0.1 for _ in range(3)]
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, [2, 4]] = True
    patches = torch.randn(8, 1536) * 0.3
    mel = torch.randn(128, 121) * 0.2
    with torch.inference_mode():
        hidden = text_model(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state
        visual_hidden = text_model(input_ids=ids, position_ids=positions, use_cache=False,
                                   visual_pos_masks=mask, deepstack_visual_embeds=deepstack).last_hidden_state
        vision_output = vision_model(patches, grid_thw=torch.tensor([[2, 2, 2]]))
        audio_output = audio_model(mel, feature_lens=torch.tensor([121])).last_hidden_state
    reference_arrays = {'ids': ids.int(), 'positions': positions, 'hidden': hidden,
                        'visual_hidden': visual_hidden, 'patches': patches,
                        'vision': vision_output.pooler_output, 'mel': mel, 'audio': audio_output}
    for index, values in enumerate(deepstack): reference_arrays[f'deepstack.{index}'] = values
    for index, values in enumerate(vision_output.deepstack_features): reference_arrays[f'vision_deepstack.{index}'] = values
    head_config = dict(hidden_size=8, width=8, routing_layers=1, layers=1, heads=2, feedforward=16)
    head = reference.JointSchemaHead(**head_config).eval()
    questions = (reference.EncodedQuestion('route', 1, (1, 3), ((3, 5), (5, 7)), ('a', 'b')),
                 reference.EncodedQuestion('truth', 0, (7, 8), ((8, 9), (9, 11)), ('true', 'false')))
    record = reference.EncodedRecord(tuple(ids[0].tolist()), questions, 'tiny', None)
    with torch.inference_mode():
        logits = head(hidden, ids, torch.ones_like(ids), [record], weights['thinker.lm_head.weight'])[0]
    for index, values in enumerate(logits): reference_arrays[f'head_logits.{index}'] = values
    # Independent official rotary layout for image + standalone audio + heard video.
    tc = SimpleNamespace(**thinker)
    layout_owner = SimpleNamespace(config=tc, spatial_merge_size=2)
    layout_owner.get_llm_pos_ids_for_vision = MethodType(
        Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision, layout_owner)
    image_ids = [tc.vision_start_token_id] + [tc.image_token_id] * 2 + [tc.vision_end_token_id, 78]
    audio_ids = [tc.audio_start_token_id] + [tc.audio_token_id] * 13 + [tc.audio_end_token_id, 78]
    video_ids = ([tc.vision_start_token_id, tc.audio_start_token_id] + [tc.video_token_id] * 2
                 + [tc.audio_token_id] * 13 + [tc.video_token_id] * 2
                 + [tc.audio_end_token_id, tc.vision_end_token_id, 78])
    layout_ids = torch.tensor([[77] + image_ids + audio_ids + video_ids + [77]])
    layout_positions, _ = Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index(
        layout_owner, layout_ids, attention_mask=torch.ones_like(layout_ids), image_grid_thw=torch.tensor([[1, 2, 4]]),
        video_grid_thw=torch.tensor([[2, 2, 4]]), use_audio_in_video=True,
        audio_seqlens=torch.tensor([100, 100]), second_per_grids=torch.tensor([1.0]))
    reference_arrays['layout_ids'] = layout_ids.int()
    reference_arrays['layout_positions'] = layout_positions
    clips = [np.sin(np.arange(n, dtype=np.float64) * .03).astype(np.float32) * .1 for n in [16001, 20000]]
    features = WhisperFeatureExtractor(feature_size=128)(clips, sampling_rate=16000, padding=True,
                    truncation=False, return_attention_mask=True, return_tensors='pt')
    for index, clip in enumerate(clips):
        reference_arrays[f'waveform.{index}'] = torch.from_numpy(clip)
        frames = int(features.attention_mask[index].sum())
        reference_arrays[f'waveform_mel.{index}'] = features.input_features[index, :, :frames].unsqueeze(0)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in weights.items()}, output / 'model.safetensors')
    save_file({k: v.contiguous() for k, v in reference_arrays.items()}, output / 'reference.safetensors')
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, output / 'joint_head.safetensors')
    (output / 'config.json').write_text(json.dumps(root, indent=2) + '\n')
    (output / 'joint_head_config.json').write_text(json.dumps(head_config, indent=2) + '\n')
    (output / 'processor_config.json').write_bytes((args.reference_root / 'processor_config.json').read_bytes())
    (output / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'model.safetensors' for k in weights}}, indent=2) + '\n')
    provenance = {'upstream': 'Cloudflare/clef-omni', 'revision': '0db1cd2607d76a7bdb2a382f659e7b313079f84b',
                  'reference_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'transformers': '5.10.2', 'torch': torch.__version__, 'seed': 719,
                  'scope': 'tiny independent untrained FP32 modules; not full-checkpoint qualification'}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()

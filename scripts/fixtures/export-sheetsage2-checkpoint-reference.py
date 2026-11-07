#!/usr/bin/env python3
"""Export a bounded FP32 oracle using the pinned, separately supplied checkpoints.

Use torch/torchaudio 2.8.0, transformers 4.45.2, safetensors 0.5.3,
numpy 2.2.6, mir_eval 0.8.2, and pretty_midi 0.2.10. No files are downloaded.
The shortened audio window measures trained-weight arithmetic, not whole-song
transcription accuracy. Run the native CLI separately for production windows.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
import transformers
from safetensors.torch import load_file, save_file


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--recording', type=Path, help='Also generate reference tokens with production 300-second padding.')
args = parser.parse_args()
source_hashes = {
    'modeling_mert2.py': 'b1a3174e5649c4b26b0c90d8626f0adacfbbba111a58ed3bb72ad651945a2f5c',
    'modeling_sheetsage2.py': 'f1c45346dd76329fcbeadc5c721312836509326310c034634bc12d75263688bb',
    'generation_sheetsage2.py': 'b58dbbfaebfba2a441a79b8c36183f0e4956acdd3aaf7c655c559933b4c20afd',
    'tokenization_sheetsage2.py': '2ad9f7734e3d9f750fe575f22fd9ec03a9a4ab52497a6314e76f53ed9024fa75',
}
for name, expected in source_hashes.items():
    if sha256(args.upstream / name) != expected:
        raise SystemExit(f'Unexpected pinned source: {name}')
parent_path = args.model / 'mert2/model.safetensors'
parent_hash = sha256(parent_path)
if parent_hash != 'e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d':
    raise SystemExit('Unexpected MERT2 parent checkpoint')
sys.path.insert(0, str(args.upstream.parent))
package = args.upstream.name
mert = importlib.import_module(package + '.modeling_mert2')
ss = importlib.import_module(package + '.modeling_sheetsage2')
configuration = importlib.import_module(package + '.configuration_sheetsage2')
torch.set_num_threads(4)
torch.manual_seed(831001)
config = configuration.SheetSage2Config(**json.loads((args.model / 'config.json').read_text()))
model = ss.SheetSage2Model(config).eval()
missing, unexpected = model.load_state_dict(load_file(str(args.model / 'model.safetensors')), strict=False)
if set(missing) != {'decoder.embed_tokens.weight', 'output_projection.weight'} or unexpected:
    raise SystemExit(f'Unexpected adapter coverage: missing={missing}, extra={unexpected}')
encoder_config = mert.MERT2Config(**config.backbone_config)
encoder_config._attn_implementation = 'sdpa'
model.encoder = mert.MERT2Model(encoder_config).eval()
model.encoder.load_state_dict(load_file(str(parent_path)), strict=True)
model.merge_lora()
# Preserve all release dimensions and weights, shortening only audio padding.
model.config.input_audio_length = 2
audio = torch.sin(torch.arange(48000) * 0.037) * 0.2 + torch.randn(48000) * 0.03
tokens = torch.tensor([model.tokenizer.prompt_prefix(['timestamp', 'downbeat_meter', 'structure', 'key', 'chord_full', 'melody_full'])])
expected = {'audio': audio, 'tokens': tokens[0].int()}
with torch.inference_mode():
    expected['mel'] = model.encoder.feature_extractor(audio[None])
    features = model.get_audio_features(audio[None], output_hidden_states=True)
    expected['subsampled'] = features.input_hidden_state
    expected['block0'] = features.backbone_hidden_states[0]
    expected['encoder'] = features.encoder_last_hidden_state
    logits, cache = model.decode(features.encoder_last_hidden_state, tokens, use_cache=True)
    expected['prefill'] = logits[0, -1]
    next_token = int(logits[0, -1].argmax())
    expected['next_token'] = torch.tensor([next_token], dtype=torch.int32)
    extra = torch.tensor([[next_token]])
    logits, _ = model.decode(features.encoder_last_hidden_state, extra, use_cache=True, past_key_values=cache)
    expected['cached'] = logits[0, -1]
    expected['uncached'] = model.decode(features.encoder_last_hidden_state, torch.cat([tokens, extra], dim=1))[0][0, -1]
args.output.mkdir(parents=True, exist_ok=False)
save_file({name: value.contiguous() for name, value in expected.items()}, str(args.output / 'expected.safetensors'))
if args.recording:
    audio_module = importlib.import_module(package + '.audio_sheetsage2')
    generation = importlib.import_module(package + '.generation_sheetsage2')
    model.config.input_audio_length = 300
    recording = audio_module.load_audio(args.recording)
    production = generation.constrained_prompt_generate(model, recording[None], generation.FULL_TASK_PROMPTS,
        model.max_output_seq_len, autocast_dtype=None, stop_time_seconds=len(recording) / 24000,
        progress_callback=lambda count: print(f'Production reference: {count} tokens', flush=True))
    (args.output / 'production-tokens.json').write_text(json.dumps({
        'recording_sha256': sha256(args.recording), 'tokens': production.tolist(),
        'sample_count': len(recording), 'window_seconds': 300,
    }, indent=2) + '\n')
(args.output / 'provenance.json').write_text(json.dumps({
    'revision': 'b83c3d79a6d98bcffaf8f9ed112c90aa887887bc',
    'source_sha256': source_hashes, 'parent_sha256': parent_hash,
    'adapter_sha256': sha256(args.model / 'model.safetensors'),
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'dtype': 'float32', 'device': 'cpu', 'audio_seconds': 2,
}, indent=2) + '\n')
print(args.output)

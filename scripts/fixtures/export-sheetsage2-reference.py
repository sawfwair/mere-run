#!/usr/bin/env python3
"""Export small CPU oracles from the pinned SheetSage2 source, without checkpoint weights.

Run with torch/torchaudio 2.8.0, transformers 4.45.2, safetensors 0.5.3,
numpy 2.2.6, mir_eval 0.8.2, and pretty_midi 0.2.10.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import transformers
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.upstream.parent))
package = args.upstream.name
mert = importlib.import_module(package + '.modeling_mert2')
ss = importlib.import_module(package + '.modeling_sheetsage2')
configuration = importlib.import_module(package + '.configuration_sheetsage2')
vocabulary = importlib.import_module(package + '.tokenization_sheetsage2')
grammar = importlib.import_module(package + '.generation_sheetsage2')
# Only replace vocabulary construction to allow a tiny decoder embedding table.
# Model/frontend/attention/adapter arithmetic remains the pinned upstream implementation.
ss.SheetSage2Tokenizer = lambda *a, **kw: SimpleNamespace(n_tokens=32, pad_token=0)
torch.set_num_threads(1)
torch.manual_seed(831001)
ec = mert.MERT2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_mel_bins=8, subsampling_channels=[8,12,16],
    subsampling_depths=[1,1,1], conv_depthwise_kernel_size=3, variant='fs')
c = configuration.SheetSage2Config(vocab_size=32, hidden_size=16, decoder_layers=2,
    num_attention_heads=2, intermediate_size=32, max_output_seq_len=64,
    lora_rank=4, lora_alpha=8, input_audio_length=0.32, backbone_config=ec.to_dict())
model = ss.SheetSage2Model(c).eval()
model.encoder = mert.MERT2Model(ec).eval()
with torch.no_grad():
    model.layer_weight.copy_(torch.tensor([-0.8,0.1,0.6]))
    model.encoder.feature_extractor.mel_mean.copy_(torch.linspace(-2,2,8))
    model.encoder.feature_extractor.mel_std.copy_(torch.linspace(0.8,1.6,8))
    for adapter in model.adapter.layers:
        for projection in ss.PROJECTIONS:
            layer = getattr(adapter.attn,projection)
            layer.lora_B.weight.normal_(0,0.03)
parent = {k:v.clone().contiguous() for k,v in model.encoder.state_dict().items()}
adapter = {k:v.clone().contiguous() for k,v in model.state_dict().items()
    if not k.startswith('encoder.') and k not in ('decoder.embed_tokens.weight','output_projection.weight')}
model.merge_lora()
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'config.json').write_text(json.dumps(c.to_dict(),indent=2)+'\n')
save_file(parent,str(args.output/'parent.safetensors'))
save_file(adapter,str(args.output/'adapter.safetensors'))
x = torch.sin(torch.arange(7680)*0.037)*0.2 + torch.randn(7680)*0.03
ids = torch.tensor([[1,4,5,3,7]])
expected = {'audio':x,'tokens':ids[0].int()}
with torch.inference_mode():
    expected['mel'] = model.encoder.feature_extractor(x[None])
    features = model.get_audio_features(x[None],output_hidden_states=True)
    expected['subsampled'] = features.input_hidden_state
    expected['encoder'] = features.encoder_last_hidden_state
    expected['block0'] = features.backbone_hidden_states[0]
    logits,cache=model.decode(features.encoder_last_hidden_state,ids,use_cache=True)
    expected['prefill']=logits[0,-1]
    logits,_=model.decode(features.encoder_last_hidden_state,torch.tensor([[9]]),use_cache=True,past_key_values=cache)
    expected['cached']=logits[0,-1]
    expected['uncached']=model.decode(features.encoder_last_hidden_state,torch.cat([ids,torch.tensor([[9]])],dim=1))[0][0,-1]
save_file({k:v.contiguous() for k,v in expected.items()},str(args.output/'expected.safetensors'))
tok=vocabulary.SheetSage2Tokenizer()
state=grammar.PromptGrammarState(tok)
tokens=[tok.subbeat_shift_token_start,tok.time_token_start+50,tok.meter_token_start+18,
    tok.eighth_position_token_start,tok.structure_token_start+3,tok.key_token_start,
    tok.full_chord_token_start+1,tok.pitch_token_start+60,tok.duration_token_start+5,
    tok.subbeat_shift_token_start+4,tok.pitch_token_start+190,tok.eos_token]
masks=[]
for token in tokens:
    masks.append(torch.where(state.allowed('cpu'))[0].tolist())
    state.update(token)
(args.output/'grammar.json').write_text(json.dumps({'tokens':tokens,'allowed':masks}))
(args.output/'provenance.json').write_text(json.dumps({'revision':'b83c3d79a6d98bcffaf8f9ed112c90aa887887bc',
    'torch':torch.__version__,'transformers':transformers.__version__,
    'source_sha256':{name:hashlib.sha256((args.upstream/name).read_bytes()).hexdigest() for name in
        ['modeling_mert2.py','modeling_sheetsage2.py','generation_sheetsage2.py','tokenization_sheetsage2.py']}},indent=2)+'\n')

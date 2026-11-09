#!/usr/bin/env python3
"""Export bounded trained-weight YuE2 FP32 arithmetic from pinned upstream code.

Requires local checkpoints and the upstream modeling_yue2.py from revision
1116bea76032bee5668e5f097294d082fa0b84f0. No assets are downloaded. The reference
uses torch 2.8.0, transformers 4.45.2, safetensors 0.5.3 and numpy 2.2.6 on CPU.
It casts the released BF16 weights to FP32 to compare complete-layer arithmetic;
it does not claim identical random sampling between PyTorch and MLX.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import transformers
from safetensors.torch import load_file, save_file


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-file', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.source_file.read_bytes()
# A patch-based source transfer can append a blank line; verify every other byte.
source_hash = hashlib.sha256(source.rstrip(b'\n') + b'\n').hexdigest()
if source_hash != '163dc19293f7454c4c616377ea74e68f62dc5d02281175e148d59337adcb9be9':
    raise SystemExit('Unexpected upstream source')
weights_hash = file_sha256(args.model / 'model.safetensors')
if weights_hash != '1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9':
    raise SystemExit('Unexpected released checkpoint')
spec = importlib.util.spec_from_file_location('qualified_yue2', args.source_file)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
torch.set_num_threads(4)
torch.manual_seed(831001)
config = module.YuE2Config(**json.loads((args.model / 'config.json').read_text()))
model = module.YuE2ForCausalLM(config).eval()
arrays = load_file(str(args.model / 'model.safetensors'))
model.load_state_dict(arrays, strict=True)
del arrays
gc.collect()
tokens = [151643, 100, 101, 151847, 66, 67, 151848, 151851, 151853, 151854, 151855, 151852]
noise = (torch.sin(torch.arange(6 * 64, dtype=torch.float32) * 0.01) * 0.2).reshape(6, 64)
reference = {'tokens': torch.tensor(tokens, dtype=torch.int32), 'noise': noise}
with torch.inference_mode():
    cache = module.StaticKVCache(num_layers=config.num_hidden_layers, batch_size=1,
        num_kv_heads=config.num_key_value_heads, max_seq_len=64, head_dim=config.head_dim,
        dtype=torch.float32, device='cpu')
    reference['prefill'] = model(torch.tensor([tokens]), past_key_values=cache, use_cache=True).logits[0, -1]
    reference['cached'] = model(torch.tensor([[151856]]), past_key_values=cache, use_cache=True).logits[0, -1]
    reference['uncached'] = model(torch.tensor([tokens + [151856]]), use_cache=False).logits[0, -1]
    ids = torch.tensor([tokens + [0] * 8])
    ar_mask = torch.zeros_like(ids, dtype=torch.bool)
    ar_mask[:, :len(tokens)] = True
    content = torch.zeros_like(ids, dtype=torch.bool)
    content[:, len(tokens) + 1:-1] = True
    reference['velocity'] = model.nar_velocity(ids, ar_mask, ~ar_mask, content, noise, 0.4)
args.output.mkdir(parents=True, exist_ok=False)
save_file({key: value.contiguous() for key, value in reference.items()}, str(args.output / 'expected.safetensors'))
(args.output / 'provenance.json').write_text(json.dumps({
    'source_revision': '1116bea76032bee5668e5f097294d082fa0b84f0',
    'source_sha256': hashlib.sha256(source).hexdigest(), 'normalized_source_sha256': source_hash,
    'weights_sha256': weights_hash, 'torch': torch.__version__, 'transformers': transformers.__version__,
    'dtype': 'float32', 'device': 'cpu', 'prefix_tokens': len(tokens), 'latent_frames': 6,
}, indent=2) + '\n')
print(args.output)

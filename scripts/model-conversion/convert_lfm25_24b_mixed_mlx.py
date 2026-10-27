#!/usr/bin/env python3
"""Convert pinned LFM2.5 24B-A2B into measured, high-precision MLX candidates.

Developer tooling only; original BF16 tensors and immutable Hub hashes are
verified before conversion. Native inference never executes this script.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import time

REPO = 'LiquidAI/LFM2.5-24B-A2B'
REVISION = '1769516ac588c4c558089c94f27d96b0dc67b183'
PROFILES = ['reference', 'mixed8', 'mixed6-down8', 'mixed6-group32', 'mixed6-group64', 'mixed6-8-quality']

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)

def policy(profile, path):
    if profile not in PROFILES:
        raise ValueError('Unknown precision profile: ' + profile)
    if profile == 'reference' or '.feed_forward.switch_mlp.' not in path:
        return None
    if profile == 'mixed6-8-quality':
        layer = int(path.split('.')[2])
        return dict(bits=8 if layer in {2,3,6,10,14,18,22,26,30,34,38,39} else 6,group_size=32,mode='affine')
    bits = 8 if profile == 'mixed8' or (profile == 'mixed6-down8' and path.endswith('down_proj')) else 6
    return dict(bits=bits, group_size=32 if profile == 'mixed6-group32' else 64, mode='affine')

def convert(source, destination, profiles, quantization_precision="float32"):
    import mlx.core as mx
    import torch
    from safetensors import safe_open
    from huggingface_hub import HfApi
    started = time.monotonic()
    config = json.loads((source/'config.json').read_text())
    index = json.loads((source/'model.safetensors.index.json').read_text())
    assert config['model_type'] == 'lfm2_moe' and config['num_hidden_layers'] == 40
    assert config['num_experts'] == 64 and config['num_dense_layers'] == 2
    assert len(index['weight_map']) == 7610
    remote = {f.rfilename:f for f in HfApi().model_info(REPO,revision=REVISION,files_metadata=True).siblings}
    source_files = {}
    readers = {}
    for name in sorted(set(index['weight_map'].values())):
        path = source/name
        sha = digest(path)
        metadata = remote[name]
        if metadata.lfs is None or sha != metadata.lfs.sha256 or path.stat().st_size != metadata.size:
            raise ValueError('Pinned source hash/size mismatch: '+name)
        source_files[name] = dict(sha256=sha,bytes=path.stat().st_size)
        readers[name] = safe_open(path,framework='pt',device='cpu')
    roots = {p:destination/p for p in profiles}
    for root in roots.values():
        root.mkdir(parents=True,exist_ok=False)
    mapped = set()
    policies = {p:{} for p in profiles}
    maps = {p:{} for p in profiles}
    receipts = {p:{} for p in profiles}
    totals = {p:0 for p in profiles}
    def array(key):
        tensor = readers[index['weight_map'][key]].get_tensor(key)
        expected = torch.float32 if key.endswith('expert_bias') else torch.bfloat16
        if tensor.dtype != expected:
            raise ValueError('Unexpected source dtype: '+key)
        mapped.add(key)
        if tensor.dtype == torch.bfloat16:
            return mx.array(tensor.view(torch.uint16).numpy()).view(mx.bfloat16)
        return mx.array(tensor.numpy())
    def write_part(part, arrays):
        filename = f'model-{part:05d}-of-00041.safetensors'
        mx.eval(arrays)
        for profile,root in roots.items():
            output = {}
            for key,weight in arrays.items():
                path = key.removesuffix('.weight')
                q = policy(profile,path) if key.endswith('.weight') else None
                if q is None:
                    output[key] = weight
                else:
                    packed,scales,biases = mx.quantize(weight.astype(mx.float32) if quantization_precision == "float32" else weight,**q)
                    if profile == 'mixed6-8-quality':
                        scales,biases=scales.astype(mx.float16),biases.astype(mx.float16)
                    output[key]=packed
                    output[path+'.scales']=scales
                    output[path+'.biases']=biases
                    policies[profile][path]=q
            mx.eval(output)
            file = root/filename
            mx.save_safetensors(str(file),output)
            totals[profile] += sum(value.nbytes for value in output.values())
            maps[profile].update({key:filename for key in output})
            receipts[profile][filename]=dict(bytes=file.stat().st_size,sha256=digest(file))
            del output
            mx.clear_cache()
        print(json.dumps(dict(part=part,parts=41,elapsed_seconds=round(time.monotonic()-started,2),logical_bytes=totals)),flush=True)
    write_part(1,{key:array(key) for key in ['model.embed_tokens.weight','model.embedding_norm.weight']})
    for layer in range(config['num_hidden_layers']):
        prefix = f'model.layers.{layer}.'
        arrays = {}
        for key in index['weight_map']:
            if not key.startswith(prefix) or '.feed_forward.experts.' in key:
                continue
            name = key
            if layer < config['num_dense_layers']:
                for original,native in [('w1','gate_proj'),('w3','up_proj'),('w2','down_proj')]:
                    name=name.replace('.feed_forward.'+original+'.weight','.feed_forward.'+native+'.weight')
            value=array(key)
            if key.endswith('.conv.conv.weight'):
                value=value.transpose(0,2,1)
            arrays[name]=value
        if layer >= config['num_dense_layers']:
            for original,native in [('w1','gate_proj'),('w3','up_proj'),('w2','down_proj')]:
                keys=[prefix+f'feed_forward.experts.{expert}.{original}.weight' for expert in range(config['num_experts'])]
                arrays[prefix+'feed_forward.switch_mlp.'+native+'.weight']=mx.stack([array(key) for key in keys])
        write_part(layer+2,arrays)
        del arrays
        mx.clear_cache()
    if mapped != set(index['weight_map']):
        raise ValueError('Not every source tensor was accounted for exactly once')
    for profile,root in roots.items():
        converted=dict(config)
        global_quant=dict(bits=8,group_size=64,mode='affine')
        if profile != 'reference':
            converted['quantization']={**global_quant,**policies[profile]}
            converted['quantization_config']=converted['quantization']
        else:
            converted.pop('quantization',None)
            converted.pop('quantization_config',None)
        converted['mererun_quantization']=policies[profile]
        if profile == 'mixed6-8-quality': converted['mererun_fp32_activations']=True
        atomic_json(root/'config.json',converted)
        atomic_json(root/'model.safetensors.index.json',dict(metadata=dict(total_size=totals[profile]),weight_map=maps[profile]))
        for name in ['tokenizer.json','tokenizer_config.json','generation_config.json','chat_template.jinja','LICENSE']:
            shutil.copy2(source/name,root/name)
        atomic_json(root/'MERERUN_CONVERSION.json',dict(
            schema_version=1,source_repository=REPO,source_revision=REVISION,
            source_tensor_count=len(mapped),source_files=source_files,
            metadata_files={name:dict(sha256=digest(source/name),bytes=(source/name).stat().st_size) for name in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','LICENSE']},
            profile=profile,quantization=policies[profile],quantization_arithmetic=quantization_precision,affine_metadata_dtype='float16' if profile=='mixed6-8-quality' else quantization_precision,fp32_activations=profile=='mixed6-8-quality',converter_sha256=digest(__file__),output_files=receipts[profile],
            logical_weight_bytes=totals[profile],tensor_count=len(maps[profile]),
            preserved_tensors='all non-expert tensors retain original BF16/FP32 values',
            tool_versions={name:version(name) for name in ['mlx','torch','safetensors','huggingface-hub']},
            quality_qualified=False))
        (root/'README.md').write_text(f'''# LFM2.5 24B-A2B {profile}

Native MLX mixed-precision derivative of {REPO}@{REVISION}.
Only routed expert matrices are quantized. Attention, convolution, the dense
prefix, tied vocabulary embedding, routers, and norms retain original precision.
The LFM Open License v1.0 applies; see LICENSE. No quality claim follows from
conversion alone. See MERERUN_CONVERSION.json and separate qualification receipts.
''')
    print(json.dumps(dict(complete=True,logical_bytes=totals,elapsed_seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profiles',choices=PROFILES,nargs='+',default=PROFILES)
    parser.add_argument('--quantization-precision',choices=['source','float32'],default='float32',help='FP32 affine scales/biases avoid BF16 metadata rounding; non-expert tensors remain unchanged.')
    args=parser.parse_args()
    convert(args.source,args.output,args.profiles,args.quantization_precision)

#!/usr/bin/env python3
"""Validate a converted bundle's closure, hashes, precision, and BF16 preservation."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from convert_lfm25_24b_mixed_mlx import digest,atomic_json
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
receipt=json.loads((a.model/'MERERUN_CONVERSION.json').read_text())
index=json.loads((a.model/'model.safetensors.index.json').read_text())
original=json.loads((a.source/'model.safetensors.index.json').read_text())['weight_map']
assert receipt['source_revision']=='1769516ac588c4c558089c94f27d96b0dc67b183'
assert receipt['quantization_arithmetic']=='float32'
source={name:safe_open(a.source/name,framework='pt',device='cpu') for name in set(original.values())}
seen=set(); preserved=0; packed=0; metadata=0; logical=0
for name,expected in receipt['output_files'].items():
 file=a.model/name
 assert file.stat().st_size==expected['bytes'] and digest(file)==expected['sha256'],name
 with safe_open(file,framework='pt',device='cpu') as weights:
  for key in weights.keys():
   assert key not in seen and index['weight_map'][key]==name,key
   seen.add(key)
   value=weights.get_tensor(key)
   logical+=value.numel()*value.element_size()
   if '.switch_mlp.' in key:
    path=key.rsplit('.',1)[0]
    q=receipt['quantization'][path]
    down='.down_proj.' in key
    input_dim=1536 if down else 2048
    output_dim=2048 if down else 1536
    if key.endswith('.weight'):
     assert value.dtype==torch.uint32 and tuple(value.shape)==(64,output_dim,input_dim*q['bits']//32),key
     packed+=1
    else:
     assert key.endswith(('.scales','.biases')) and value.dtype==(torch.float16 if receipt.get('affine_metadata_dtype')=='float16' else torch.float32),key
     assert tuple(value.shape)==(64,output_dim,input_dim//q['group_size']),key
     assert torch.isfinite(value).all(),key
     metadata+=1
    continue
   original_key=key
   if key.startswith(('model.layers.0.','model.layers.1.')):
    for native,old in [('gate_proj','w1'),('up_proj','w3'),('down_proj','w2')]:
     original_key=original_key.replace('.feed_forward.'+native+'.weight','.feed_forward.'+old+'.weight')
   expected_value=source[original[original_key]].get_tensor(original_key)
   if key.endswith('.conv.conv.weight'): expected_value=expected_value.transpose(1,2)
   assert value.dtype==expected_value.dtype and torch.equal(value,expected_value),key
   preserved+=1
assert seen==set(index['weight_map'])
assert packed==114 and metadata==228
assert logical==index['metadata']['total_size']==receipt['logical_weight_bytes']
atomic_json(a.output,dict(schema_version=1,source_revision=receipt['source_revision'],conversion_sha256=digest(a.model/'MERERUN_CONVERSION.json'),
 profile=receipt['profile'],logical_weight_bytes=logical,tensor_count=len(seen),verified_shards=len(receipt['output_files']),
 bit_exact_preserved_non_expert_tensors=preserved,packed_expert_matrices=packed,affine_metadata_tensors=metadata,affine_metadata_dtype=receipt.get('affine_metadata_dtype','float32'),passes=True))
print(a.output.read_text())

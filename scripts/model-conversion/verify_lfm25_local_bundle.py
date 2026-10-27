#!/usr/bin/env python3
"""File-I/O-only local verification of a complete LFM2.5 mixed bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--source-config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--wait',action='store_true')
p.add_argument('--maximum-bundle-bytes',type=int)
p.add_argument('--source-generation-config',type=Path)
a=p.parse_args()
def sha(file):
 h=hashlib.sha256()
 with file.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 return h.hexdigest()
receipt=json.loads((a.model/'MERERUN_CONVERSION.json').read_text())
index=json.loads((a.model/'model.safetensors.index.json').read_text())
config=json.loads((a.model/'config.json').read_text())
original=json.loads(a.source_config.read_text())
assert config['mererun_fp32_activations'] is True
assert config['mererun_quantization']==receipt['quantization']
assert {k:v for k,v in config.items() if k not in ['quantization','quantization_config','mererun_quantization','mererun_fp32_activations']}==original
rows=[];tensors={};logical=0;started=time.monotonic()
for name,expected in sorted(receipt['output_files'].items()):
 file=a.model/name
 assert not file.is_symlink(), 'Bundle must contain self-contained shards: '+name
 while not file.exists():
  if not a.wait: raise FileNotFoundError(file)
  time.sleep(2)
 assert file.stat().st_size==expected['bytes'],name
 digest=sha(file)
 assert digest==expected['sha256'],name
 with file.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]
  assert n<1024*1024
  header=json.loads(f.read(n))
 for key,value in header.items():
  if key=='__metadata__':continue
  assert key not in tensors and index['weight_map'][key]==name,key
  tensors[key]=value
  begin,end=value['data_offsets']
  logical+=end-begin
 rows.append(dict(file=name,bytes=file.stat().st_size,sha256=digest))
 print(json.dumps(dict(verified_shards=len(rows),total_shards=len(receipt['output_files']),bytes=sum(r['bytes'] for r in rows))),flush=True)
assert len(rows)==len(receipt['output_files']) and set(tensors)==set(index['weight_map']) and len(tensors)==656
assert set(receipt['output_files'])==set(index['weight_map'].values())
assert logical==index['metadata']['total_size']==receipt['logical_weight_bytes']
assert {p.name for p in a.model.glob('*.safetensors')}==set(receipt['output_files'])
for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','LICENSE']:
 assert sha(a.model/name)==receipt['metadata_files'][name]['sha256'],name
if 'full_hessian_refinement_sha256' in receipt:
 assert sha(a.model/receipt.get('full_hessian_refinement_file','FULL_HESSIAN_REFINEMENT.json'))==receipt['full_hessian_refinement_sha256']
generation_sha=sha(a.model/'generation_config.json')
if a.source_generation_config is not None:assert generation_sha==sha(a.source_generation_config)
bundle_bytes=sum(p.stat().st_size for p in a.model.iterdir() if p.is_file())
if a.maximum_bundle_bytes is not None:assert bundle_bytes<=a.maximum_bundle_bytes
result=dict(schema_version=1,bundle_total_bytes=bundle_bytes,generation_config_sha256=generation_sha,generation_config_matches_source_snapshot=a.source_generation_config is not None,source_revision=receipt['source_revision'],conversion_sha256=sha(a.model/'MERERUN_CONVERSION.json'),
 config_sha256=sha(a.model/'config.json'),profile=receipt['profile'],logical_weight_bytes=logical,tensor_count=len(tensors),
 verified_shards=len(rows),files=rows,metadata_hashes_verified=True,source_config_preserved=True,passes=True,
 elapsed_seconds=time.monotonic()-started)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'}))

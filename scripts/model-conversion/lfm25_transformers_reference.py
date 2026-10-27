#!/usr/bin/env python3
"""Independent pinned Transformers BF16 comparison on frozen LFM2.5 tokens."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from safetensors.numpy import save_file,load_file
from qualify_lfm25_24b_mixed import sha,save,log_softmax
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--frozen',type=Path,required=True)
p.add_argument('--mlx-reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--compute-dtype',choices=['bfloat16','float32'],default='bfloat16')
a=p.parse_args()
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
a.output.mkdir(parents=True,exist_ok=True)
suite=json.loads(a.frozen.read_text())
start=time.monotonic()
model=AutoModelForCausalLM.from_pretrained(str(a.source),dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval()
rows=[]
batched_logits=None
if a.compute_dtype=='float32':
 # Promote only the active layer; preserve and restore every source tensor.
 originals={}
 def promote(module,args):
  values=[(parameter,parameter.data) for parameter in module.parameters() if parameter.dtype==torch.bfloat16]
  originals[id(module)]=values
  for parameter,value in values: parameter.data=value.float()
 def restore(module,args,output):
  for parameter,value in originals.pop(id(module)): parameter.data=value
  return output
 model.model.embed_tokens.register_forward_hook(lambda module,args,output:output.float())
 for module in [*model.model.layers,model.model.embedding_norm,model.lm_head]:
  module.register_forward_pre_hook(promote)
  module.register_forward_hook(restore)
 width=max(len(case['tokens'])-1 for case in suite['cases'])
 ids=torch.tensor([case['tokens'][:-1]+[0]*(width-len(case['tokens'])+1) for case in suite['cases']],device='cuda')
 with torch.inference_mode(): batched_logits=model(ids,use_cache=False).logits.float().cpu().numpy()
with torch.inference_mode():
 for index,case in enumerate(suite['cases']):
  ids=torch.tensor([case['tokens'][:-1]],device='cuda')
  output=model(ids,use_cache=False).logits[0,case['score_start']-1:].float().cpu().numpy() if batched_logits is None else batched_logits[index,case['score_start']-1:len(case['tokens'])-1]
  file=a.output/(case['id']+'.safetensors')
  save_file({'logits':output},str(file))
  reference=load_file(str(a.mlx_reference/file.name))['logits']
  target=np.array(case['tokens'][case['score_start']:])
  lp,lq=log_softmax(output),log_softmax(reference)
  delta=lq[np.arange(len(target)),target]-lp[np.arange(len(target)),target]
  kl=(np.exp(lp)*(lp-lq)).sum(-1)
  rows.append(dict(id=case['id'],tokens=len(target),sha256=sha(file),mean_kl=float(kl.mean()),top1_agreement=float((output.argmax(-1)==reference.argmax(-1)).mean()),delta_nll=float(-delta.mean()),max_abs_logit_delta=float(abs(output-reference).max())))
  n=sum(r['tokens'] for r in rows)
  aggregate={key:sum(r[key]*r['tokens'] for r in rows)/n for key in ['mean_kl','top1_agreement','delta_nll']}
  aggregate.update(cases=len(rows),tokens=n,perplexity_ratio=float(np.exp(aggregate['delta_nll'])))
  save(a.output/'receipt.json',dict(source_revision=suite['source_revision'],frozen_suite_sha256=sha(a.frozen),backend='Transformers PyTorch CUDA eager attention',compute_dtype=a.compute_dtype,torch_version=torch.__version__,elapsed_seconds=time.monotonic()-start,aggregate=aggregate,cases=rows))
  print(json.dumps(dict(case=case['id'],aggregate=aggregate)),flush=True)
  del output,reference,lp,lq,delta,kl
  gc.collect()

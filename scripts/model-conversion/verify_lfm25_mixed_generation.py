#!/usr/bin/env python3
"""Bounded trained-checkpoint generation and long-context diagnostic."""
import os
os.environ['MLX_ENABLE_TF32']='0'
import argparse
import json
from pathlib import Path
import time
import mlx.core as mx
from mlx_lm import load
from transformers import AutoTokenizer
from qualify_lfm25_24b_mixed import sha,save
p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--contexts',type=int,nargs='*',default=[8192,32768,131072])
p.add_argument('--max-new',type=int,default=512)
p.add_argument('--prefill-chunk-size',type=int,default=128)
p.add_argument('--kv-bits',type=int,choices=[0,8],default=8)
p.add_argument('--temperature',type=float,default=0.1)
a=p.parse_args()
original=mx.gather_qmm
def compatible(*args,global_scale=None,**kwargs):
 if global_scale is not None: raise ValueError('Unsupported scale')
 return original(*args,**kwargs)
mx.gather_qmm=compatible
from mlx_lm.models import lfm2_moe
# CUDA MLX 0.32.1 lacks some quantized-KV matmul shapes. Preserve affine
# storage and apply the mathematically equivalent dequantized SDPA fallback.
def portable_attention(queries,keys,values,cache,scale,mask,sinks=None):
 if hasattr(cache,'bits'):
  keys=mx.dequantize(*keys,group_size=cache.group_size,bits=cache.bits).astype(queries.dtype)
  values=mx.dequantize(*values,group_size=cache.group_size,bits=cache.bits).astype(queries.dtype)
 return mx.fast.scaled_dot_product_attention(queries,keys,values,scale=scale,mask=mask,sinks=sinks)
lfm2_moe.scaled_dot_product_attention=portable_attention
model,_=load(str(a.model),lazy=False)
mx.random.seed(20261013)
tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,trust_remote_code=False)
config=json.loads((a.model/'config.json').read_text())
assert config['mererun_fp32_activations']
receipt=dict(schema_version=1,conversion_sha256=sha(a.model/'MERERUN_CONVERSION.json'),mlx_enable_tf32=False,fp32_activations=True,quantized_kv_attention='dequantized SDPA fallback',temperature=a.temperature,top_k=50,top_p=1,repetition_penalty=1.05,seed=20261013,cases=[])
def run(id,tokens,max_new=None,context_limit=None):
 if max_new is None: max_new=a.max_new
 from mlx_lm.models.cache import KVCache,QuantizedKVCache
 start=time.monotonic(); cache=model.make_cache(); last=None
 if a.kv_bits: cache=[QuantizedKVCache(group_size=64,bits=a.kv_bits) if isinstance(c,KVCache) else c for c in cache]
 chunk_size=min(a.prefill_chunk_size,32) if len(tokens)>32768 else a.prefill_chunk_size
 mx.reset_peak_memory()
 for begin in range(0,len(tokens),chunk_size):
  ids=mx.array([tokens[begin:begin+chunk_size]])
  embeddings=model.model.embed_tokens(ids).astype(mx.float32)
  hidden=model.model(ids,cache=cache,input_embeddings=embeddings)
  last=hidden[:,-1:,:]
  mx.eval(last,[c.state for c in cache])
  mx.clear_cache()
  if begin%(chunk_size*64)==0: print(json.dumps(dict(case=id,prefilled=min(begin+chunk_size,len(tokens)),prompt_tokens=len(tokens),active_gpu_bytes=mx.get_active_memory())),flush=True)
 prefill=time.monotonic()-start; generated=[]
 for step in range(max_new):
  logits=model.model.embed_tokens.as_linear(last)
  scores=logits[0,-1,:]
  repeats=mx.array(sorted(set((tokens[-64:]+generated)[-64:])))
  values=scores[repeats]
  scores[repeats]=mx.where(values>=0,values/1.05,values*1.05)
  if a.temperature:
   cutoff=mx.sort(scores)[-50]
   scores=mx.where(scores>=cutoff,scores/a.temperature,-mx.inf)
   token=int(mx.random.categorical(scores).item())
  else: token=int(mx.argmax(scores).item())
  generated.append(token)
  if token==124900: break
  ids=mx.array([[token]])
  last=model.model(ids,cache=cache,input_embeddings=model.model.embed_tokens(ids).astype(mx.float32))
  mx.eval(last,[c.state for c in cache])
 row=dict(id=id,prompt_tokens=len(tokens),generated_tokens=len(generated),context_limit=context_limit,prefill_chunk_size=chunk_size,kv_bits=a.kv_bits,prefill_seconds=prefill,elapsed_seconds=time.monotonic()-start,
          peak_gpu_bytes=mx.get_peak_memory(),token_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=False),finished_eos=generated[-1]==124900)
 receipt['cases'].append(row);save(a.output,receipt);print(json.dumps(row,ensure_ascii=False),flush=True)
 del cache,last;mx.clear_cache()
for id,prompt in [('format','Reply with exactly READY in the final answer.'),('arithmetic','What is 17 plus 25? Give only the number after reasoning.'),('code','Write a Python function that returns the square of its argument.')]:
 run(id,tokenizer.encode(tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True),add_special_tokens=False))
tools=[dict(type='function',function=dict(name='calculate',description='Evaluate an arithmetic expression.',parameters=dict(type='object',properties=dict(expression=dict(type='string')),required=['expression'])))]
run('tool-call',tokenizer.encode(tokenizer.apply_chat_template([dict(role='user',content='Use the calculate tool to compute 17 plus 25.')],tools=tools,tokenize=False,add_generation_prompt=True),add_special_tokens=False))
fill=tokenizer.encode('A quiet library records the titles of books on its shelves. Each shelf has a numbered label.\n',add_special_tokens=False)
for limit in a.contexts:
 prompt=tokenizer.apply_chat_template([dict(role='user',content='CONTEXT_PLACEHOLDER\nWhat is the access code? Return only the code after reasoning.')],tokenize=False,add_generation_prompt=True)
 before,after=prompt.split('CONTEXT_PLACEHOLDER')
 prefix=tokenizer.encode(before,add_special_tokens=False);suffix=tokenizer.encode(after,add_special_tokens=False)
 needle=tokenizer.encode('\nThe access code is TANGERINE-4827.\n',add_special_tokens=False)
 n=limit-128-len(prefix)-len(suffix)-len(needle)
 padding=(fill*((n+len(fill)-1)//len(fill)))[:n]
 cut=n//5; tokens=prefix+padding[:cut]+needle+padding[cut:]+suffix
 assert len(tokens)+128==limit
 run('context-'+str(limit),tokens,128,limit)

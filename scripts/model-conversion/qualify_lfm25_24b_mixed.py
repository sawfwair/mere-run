#!/usr/bin/env python3
"""Freeze teacher-forced tokens and compare high-precision LFM2.5 candidates.

Uses full-vocabulary KL and target perplexity, with calibration-only selection.
The authored suite is a diagnostic, not a broad capability benchmark.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''): h.update(block)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def freeze(suite, source, output):
    from transformers import AutoTokenizer
    data=json.loads(suite.read_text())
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
    rows=[]
    for case in data['cases']:
        prompt=tokenizer.apply_chat_template(case['messages'],tokenize=False,add_generation_prompt=True)
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        tokens=tokenizer.encode(prompt+case['completion'],add_special_tokens=False)
        if tokens[:len(prefix)] != prefix: raise ValueError('Prompt/continuation boundary changed: '+case['id'])
        rows.append({**case,'rendered_prompt':prompt,'tokens':tokens,'score_start':len(prefix)})
    data.update(cases=rows,authored_suite_sha256=sha(suite),tokenizer_sha256=sha(source/'tokenizer.json'))
    save(output,data)


def score(model_path, frozen, output, compute_dtype="bfloat16"):
    # MLX enables TF32 by default; disable reduced-mantissa matrix arithmetic
    # before the backend initializes so quantization error is isolated.
    os.environ['MLX_ENABLE_TF32'] = '0'
    import mlx.core as mx
    from mlx_lm import load
    from safetensors.numpy import save_file
    from importlib.metadata import version
    # Pinned MLX-LM anticipates MLX's optional global_scale keyword. Released
    # MLX 0.32.1 has no such argument; affine candidates never use a global scale.
    # Reject any non-None scale rather than silently changing quantization.
    compatibility_adapter = 'none'
    if 'global_scale' not in mx.gather_qmm.__doc__:
        original_gather = mx.gather_qmm
        def gather_without_unused_scale(*args, global_scale=None, **kwargs):
            if global_scale is not None:
                raise ValueError('This MLX version cannot apply a global scale')
            return original_gather(*args, **kwargs)
        mx.gather_qmm = gather_without_unused_scale
        compatibility_adapter = 'omit unsupported global_scale=None for affine gather_qmm'
    output.mkdir(parents=True,exist_ok=True)
    suite=json.loads(frozen.read_text())
    start=time.monotonic()
    model,_=load(str(model_path),lazy=False)
    mx.eval(model.parameters())
    receipt=dict(schema_version=1,backend='MLX-CUDA',compatibility_adapter=compatibility_adapter,compute_dtype=compute_dtype,mlx_enable_tf32=False,model=str(model_path),frozen_suite_sha256=sha(frozen),
                 conversion_sha256=sha(model_path/'MERERUN_CONVERSION.json'),
                 versions={name:version(name) for name in ['mlx','mlx-lm','transformers']},cases=[])
    if compute_dtype == 'float32':
        # Right-padding cannot affect any scored prefix under causal attention.
        # Stream one promoted layer at a time; a full FP32 checkpoint is too
        # large for this GPU and need not be materialized to get FP32 arithmetic.
        import mlx.utils as utils
        from mlx_lm.models.base import create_attention_mask
        width=max(len(case['tokens'])-1 for case in suite['cases'])
        inputs=[case['tokens'][:-1]+[0]*(width-len(case['tokens'])+1) for case in suite['cases']]
        h=model.model.embed_tokens(mx.array(inputs,dtype=mx.int32)).astype(mx.float32)
        mx.eval(h)
        mask=create_attention_mask(h,None)
        def apply_promoted(module, operation):
            original=module.parameters()
            promoted=utils.tree_map(lambda v:v.astype(mx.float32) if v.dtype in [mx.bfloat16,mx.float16] else v,original)
            mx.eval(promoted)
            module.update(promoted)
            value=operation()
            mx.eval(value)
            module.update(original)
            del original,promoted
            mx.clear_cache()
            return value
        for index,layer in enumerate(model.layers):
            h=apply_promoted(layer,lambda:layer(h,mask=mask if layer.is_attention_layer else None,cache=None))
            print(json.dumps(dict(model=model_path.name,promoted_layer=index,peak_gpu_bytes=mx.get_peak_memory())),flush=True)
        h=apply_promoted(model.model.embedding_norm,lambda:model.model.embedding_norm(h))
        # Only scored hidden states need the tied 128K vocabulary projection.
        for index,case in enumerate(suite['cases']):
            end=len(case['tokens'])-1
            selected=apply_promoted(model.model.embed_tokens,lambda:model.model.embed_tokens.as_linear(h[index,case['score_start']-1:end]))
            values=np.array(selected.astype(mx.float32))
            file=output/(case['id']+'.safetensors')
            if not np.isfinite(values).all(): raise ValueError('Nonfinite FP32 logits')
            save_file({'logits':values},str(file))
            receipt['cases'].append(dict(id=case['id'],logits_file=file.name,sha256=sha(file),tokens=values.shape[0]))
            receipt['elapsed_seconds']=time.monotonic()-start
            receipt['peak_gpu_bytes']=mx.get_peak_memory()
            save(output/'receipt.json',receipt)
        return
    for case in suite['cases']:
        file=output/(case['id']+'.safetensors')
        logits=model(mx.array([case['tokens'][:-1]],dtype=mx.int32))
        selected=logits[0,case['score_start']-1:,:].astype(mx.float32)
        mx.eval(selected)
        values=np.array(selected)
        if values.shape != (len(case['tokens'])-case['score_start'],128000) or not np.isfinite(values).all():
            raise ValueError('Invalid logits: '+case['id'])
        save_file({'logits':values},str(file))
        receipt['cases'].append(dict(id=case['id'],logits_file=file.name,sha256=sha(file),tokens=values.shape[0]))
        receipt['elapsed_seconds']=time.monotonic()-start
        receipt['peak_gpu_bytes']=mx.get_peak_memory()
        save(output/'receipt.json',receipt)
        print(json.dumps(dict(model=model_path.name,case=case['id'],tokens=values.shape[0],elapsed=receipt['elapsed_seconds'])),flush=True)
        del logits,selected,values
        mx.clear_cache()
    del model
    gc.collect()
    mx.clear_cache()


def log_softmax(x):
    x=x.astype(np.float64)
    x-=x.max(-1,keepdims=True)
    return x-np.log(np.exp(x).sum(-1,keepdims=True))


def compare(frozen, candidates_root, evidence, output, profiles=None):
    from safetensors.numpy import load_file
    suite=json.loads(frozen.read_text())
    gates=suite['gates']
    def receipt(profile):
        root=evidence/profile
        r=json.loads((root/'receipt.json').read_text())
        if r['frozen_suite_sha256']!=sha(frozen): raise ValueError('Suite mismatch')
        if r.get('compute_dtype','bfloat16') != reference_dtype[0]: raise ValueError('Compute dtype mismatch')
        if len(r['cases'])!=len(suite['cases']): raise ValueError('Incomplete score')
        for c,case in zip(r['cases'],suite['cases']):
            if c['id']!=case['id'] or sha(root/c['logits_file'])!=c['sha256']: raise ValueError('Logit identity mismatch')
        return r
    reference_dtype=[json.loads((evidence/'reference'/'receipt.json').read_text()).get('compute_dtype','bfloat16')]
    reference=receipt('reference')
    reports=[]
    def aggregate(rows):
        n=sum(row['tokens'] for row in rows)
        a=dict(cases=len(rows),tokens=n)
        for key in ['mean_kl','top1_agreement','reference_nll','candidate_nll','delta_nll']:
            a[key]=sum(r[key]*r['tokens'] for r in rows)/n
        a['perplexity_ratio']=float(np.exp(a['delta_nll']))
        a['worst_case_mean_kl']=max(r['mean_kl'] for r in rows)
        a['worst_case_perplexity_ratio']=max(r['perplexity_ratio'] for r in rows)
        a['passes_diagnostic_gates']=(a['mean_kl']<=gates['mean_kl_max'] and a['worst_case_mean_kl']<=gates['case_mean_kl_max'] and a['perplexity_ratio']<=gates['perplexity_ratio_max'] and a['worst_case_perplexity_ratio']<=gates['case_perplexity_ratio_max'] and a['top1_agreement']>=gates['top1_agreement_min'])
        return a
    profiles=profiles or ['mixed8','mixed6-down8','mixed6-group32','mixed6-group64']
    receipts={p:receipt(p) for p in profiles}
    results={p:[] for p in profiles}
    for index,case in enumerate(suite['cases']):
        ref=reference['cases'][index]
        a=load_file(str(evidence/'reference'/ref['logits_file']))['logits']
        lp=log_softmax(a)
        target=np.array(case['tokens'][case['score_start']:])
        ref_target=lp[np.arange(len(target)),target]
        probability=np.exp(lp)
        for profile in profiles:
            item=receipts[profile]['cases'][index]
            b=load_file(str(evidence/profile/item['logits_file']))['logits']
            if b.shape != a.shape: raise ValueError('Logit shape mismatch')
            lq=log_softmax(b)
            delta=lq[np.arange(len(target)),target]-ref_target
            kl=(probability*(lp-lq)).sum(-1)
            results[profile].append(dict(id=case['id'],split=case['split'],language=case['language'],task=case['task'],tokens=len(target),
                mean_kl=float(kl.mean()),p95_kl=float(np.quantile(kl,.95)),top1_agreement=float((a.argmax(-1)==b.argmax(-1)).mean()),
                delta_nll=float(-delta.mean()),reference_nll=float(-ref_target.mean()),candidate_nll=float(-(ref_target+delta).mean()),
                perplexity_ratio=float(np.exp(-delta.mean())),token_logprob_rmse=float(np.sqrt(np.mean(delta**2)))))
    for profile in profiles:
        rows=results[profile]
        conversion=json.loads((candidates_root/profile/'MERERUN_CONVERSION.json').read_text())
        reports.append(dict(profile=profile,logical_weight_bytes=conversion['logical_weight_bytes'],conversion_sha256=sha(candidates_root/profile/'MERERUN_CONVERSION.json'),
            calibration=aggregate([r for r in rows if r['split']=='calibration']),heldout=aggregate([r for r in rows if r['split']=='heldout']),
            by_language={language:aggregate([r for r in rows if r['split']=='heldout' and r['language']==language]) for language in sorted({r['language'] for r in rows})},cases=rows))
    eligible=[r for r in reports if r['logical_weight_bytes']<=suite['maximum_logical_weight_bytes'] and r['calibration']['passes_diagnostic_gates']]
    selected=min(eligible,key=lambda r:r['calibration']['mean_kl']) if eligible else None
    under_budget=[r for r in reports if r['logical_weight_bytes']<=suite['maximum_logical_weight_bytes']]
    best=min(under_budget,key=lambda r:r['calibration']['mean_kl']) if under_budget else None
    # Heldout metrics do not influence candidate selection.
    result=dict(schema_version=1,compute_dtype=reference_dtype[0],source_revision=suite['source_revision'],frozen_suite_sha256=sha(frozen),gates=gates,
        maximum_logical_weight_bytes=suite['maximum_logical_weight_bytes'],selection_criterion='minimum calibration KL among candidates within weight budget passing calibration gates',
        best_under_budget_profile=best['profile'] if best else None,best_under_budget_passes_absolute_gates=best['heldout']['passes_diagnostic_gates'] if best else False,
        selected_profile=selected['profile'] if selected else None,selected_passes_heldout=selected['heldout']['passes_diagnostic_gates'] if selected else False,
        reports=reports,scope='60 original authored short teacher-forced cases, 16 languages; no broad benchmark, latency, long-context, or Apple fit claim')
    save(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))
    for report in reports: print(json.dumps({k:v for k,v in report.items() if k not in ['cases','by_language']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze'); f.add_argument('--suite',type=Path,required=True); f.add_argument('--source',type=Path,required=True); f.add_argument('--output',type=Path,required=True)
    s=commands.add_parser('score'); s.add_argument('--model',type=Path,required=True); s.add_argument('--frozen',type=Path,required=True); s.add_argument('--output',type=Path,required=True); s.add_argument('--compute-dtype',choices=['bfloat16','float32'],default='bfloat16')
    c=commands.add_parser('compare'); c.add_argument('--frozen',type=Path,required=True); c.add_argument('--candidates',type=Path,required=True); c.add_argument('--evidence',type=Path,required=True); c.add_argument('--output',type=Path,required=True); c.add_argument('--profiles',nargs='+')
    a=p.parse_args()
    if a.command=='freeze': freeze(a.suite,a.source,a.output)
    elif a.command=='score': score(a.model,a.frozen,a.output,a.compute_dtype)
    else: compare(a.frozen,a.candidates,a.evidence,a.output,a.profiles)

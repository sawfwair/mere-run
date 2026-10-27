#!/usr/bin/env python3
"""Hash-verify full-vocabulary scores and select only on calibration KL."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load
from lfm25_quality18 import sha,save

# One comparison process shares an immutable, hash-verified reference across policies.
_REFERENCE_CACHE = {}

def verified_logits(path, expected):
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected
    return load(raw)['logits']


def log_softmax(x):
    x=x.astype(np.float64);x-=x.max(-1,keepdims=True)
    return x-np.log(np.exp(x).sum(-1,keepdims=True))


def compare(suite,reference,candidate,split):
    source=json.loads((reference/'receipt.json').read_text())
    quant=json.loads((candidate/'receipt.json').read_text())
    assert source['frozen_suite_sha256']==quant['frozen_suite_sha256']==sha(suite)
    assert source['compute_dtype']==quant['compute_dtype']=='float32'
    rows=[r for r in json.loads(suite.read_text())['cases'] if r['split']==split]
    refs={r['id']:r for r in source['cases']};qs={r['id']:r for r in quant['cases']}
    assert set(qs)==set(refs)=={r['id'] for r in rows}
    scores=[]
    for row in rows:
        a,b=refs[row['id']],qs[row['id']]
        pa,pb=reference/a['logits_file'],candidate/b['logits_file']
        key=(str(pa.resolve()),a['sha256'])
        if key not in _REFERENCE_CACHE:
            x=verified_logits(pa,a['sha256']);assert np.isfinite(x).all()
            lp=log_softmax(x)
            _REFERENCE_CACHE[key]=(lp,np.exp(lp),x.argmax(-1),x.shape)
        lp,probability,reference_top1,shape=_REFERENCE_CACHE[key]
        y=verified_logits(pb,b['sha256'])
        target=np.array(row['tokens'][row['score_start']:])
        assert shape==y.shape==(len(target),128000) and np.isfinite(y).all()
        lq=log_softmax(y)
        delta=lq[np.arange(len(target)),target]-lp[np.arange(len(target)),target]
        kl=(probability*(lp-lq)).sum(-1)
        scores.append(dict(id=row['id'],split=split,language=row['language'],task=row['task'],tokens=len(target),
                           mean_kl=float(kl.mean()),top1_agreement=float((reference_top1==y.argmax(-1)).mean()),
                           delta_nll=float(-delta.mean()),perplexity_ratio=float(np.exp(-delta.mean()))))
    def aggregate(rows):
        n=sum(r['tokens'] for r in rows)
        result=dict(cases=len(rows),tokens=n)
        for metric in ['mean_kl','top1_agreement','delta_nll']:result[metric]=sum(r['tokens']*r[metric] for r in rows)/n
        result['perplexity_ratio']=float(np.exp(result['delta_nll']))
        result['worst_case_mean_kl']=max(r['mean_kl'] for r in rows)
        result['worst_case_perplexity_ratio']=max(r['perplexity_ratio'] for r in rows)
        gates=json.loads(suite.read_text())['gates']
        result['passes_absolute_gates']=(result['mean_kl']<=gates['mean_kl_max'] and result['top1_agreement']>=gates['top1_agreement_min'] and result['perplexity_ratio']<=gates['perplexity_ratio_max'] and result['worst_case_mean_kl']<=gates['case_mean_kl_max'] and result['worst_case_perplexity_ratio']<=gates['case_perplexity_ratio_max'])
        return result
    old=[r for r in scores if not r['id'].startswith(('v2-','v3-'))]
    new=[r for r in scores if r['id'].startswith('v2-')]
    prose=[r for r in scores if r['id'].startswith('v3-')]
    result=dict(aggregate=aggregate(scores),original_diagnostic=aggregate(old),authored_expansion=aggregate(new),cases=scores)
    if prose:result['prose_continuation']=aggregate(prose)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['calibration','heldout'],required=True);p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--select',action='store_true');a=p.parse_args()
    if a.select and a.split!='calibration':raise ValueError('Heldout cannot select a policy')
    reports=[]
    for profile in a.profiles:
        model=a.models/profile;conversion=json.loads((model/'MERERUN_CONVERSION.json').read_text())
        score_receipt=json.loads((a.scores/profile/'receipt.json').read_text())
        assert score_receipt['conversion_sha256']==sha(model/'MERERUN_CONVERSION.json')
        report=dict(profile=profile,logical_weight_bytes=conversion['logical_weight_bytes'],conversion_sha256=sha(model/'MERERUN_CONVERSION.json'),**compare(a.suite,a.reference,a.scores/profile,a.split))
        reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
    eligible=[r for r in reports if 18_000_000_000<=r['logical_weight_bytes']<=19_000_000_000]
    selected=min(eligible,key=lambda r:r['aggregate']['mean_kl']) if a.select else None
    receipt=dict(schema_version=1,reference_receipt_sha256=sha(a.reference/'receipt.json'),comparison_tool_sha256=sha(Path(__file__)),source_revision='1769516ac588c4c558089c94f27d96b0dc67b183',frozen_suite_sha256=sha(a.suite),split=a.split,selection_criterion='minimum full-vocabulary calibration KL among tested 18-19 GB candidates',selected_profile=selected['profile'] if selected else None,reports=reports)
    save(a.output,receipt)
    if selected:print('SELECTED:',selected['profile'],flush=True)

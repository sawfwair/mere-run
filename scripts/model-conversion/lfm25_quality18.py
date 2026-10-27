#!/usr/bin/env python3
"""Calibrate group-local affine quantization and allocate an 18-19 GB budget.

Owned experiment tooling. Uses routed activation block covariances, group-local
Hessian error feedback, and affine least-squares refinement. This is not full
cross-group GPTQ. Allocation proxies never replace end-to-end logit scoring.
"""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

REVISION = '1769516ac588c4c558089c94f27d96b0dc67b183'
OPTIONS = [(4,32),(5,32),(6,32),(4,64),(5,64),(6,64),(8,64)]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n'); temp.replace(path)


def pack_codes(codes, bits):
    """Pack contiguous little-endian affine codes into MLX uint32 words."""
    if codes.shape[-1] % 32 or bits not in [4,5,6,8]:
        raise ValueError('Invalid affine packing shape or precision')
    x = codes.reshape(*codes.shape[:-1], -1, 32).astype(np.uint32)
    out = np.zeros((*x.shape[:-1], bits), dtype=np.uint32)
    for i in range(32):
        word, shift = divmod(i*bits, 32)
        out[..., word] |= x[..., i] << np.uint32(shift)
        if shift+bits > 32:
            out[..., word+1] |= x[..., i] >> np.uint32(32-shift)
    return out.reshape(*codes.shape[:-1], -1)


def mlx_setup():
    # Load Torch's pinned NCCL first; MLX's bundled NCCL lacks a Torch symbol.
    import torch
    import mlx.core as mx
    # The pinned MLX-LM accepts an optional scale newer than MLX 0.32.1.
    if 'global_scale' not in mx.gather_qmm.__doc__:
        original = mx.gather_qmm
        def compatible(*args, global_scale=None, **kwargs):
            if global_scale is not None:
                raise ValueError('Non-affine global scales are unsupported')
            return original(*args, **kwargs)
        mx.gather_qmm = compatible
    return mx


def promoted(module, operation):
    import mlx.core as mx
    from mlx.utils import tree_map
    original = module.parameters()
    values = tree_map(lambda x: x.astype(mx.float32) if x.dtype in [mx.bfloat16,mx.float16] else x, original)
    mx.eval(values); module.update(values)
    output = operation(); mx.eval(output)
    module.update(original)
    del original, values
    mx.clear_cache()
    return output


def stats(source, frozen, output):
    mx = mlx_setup()
    from mlx_lm import load
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.base import create_attention_mask
    model, _ = load(str(source), lazy=False)
    suite = json.loads(frozen.read_text())
    rows = [r for r in suite['cases'] if r['split'] == 'calibration']
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); moments = {}
    receipt = dict(schema_version=1, source_revision=REVISION, frozen_suite_sha256=sha(frozen),
                   calibration_cases=len(rows), calibration_input_tokens=sum(len(r['tokens'])-1 for r in rows),
                   method='routing-score-squared weighted input and down-activation block covariances',
                   block_size=64, maximum_batch_size=16, layers=[])
    for begin in range(0,len(rows),16):
        batch=rows[begin:begin+16];lengths=np.array([len(r['tokens'])-1 for r in batch]);width=int(lengths.max())
        ids=mx.array([r['tokens'][:-1]+[0]*(width-len(r['tokens'])+1) for r in batch],dtype=mx.int32)
        h=model.model.embed_tokens(ids).astype(mx.float32);mx.eval(h)
        mask=create_attention_mask(h,None)
        valid=(np.arange(width)[None,:]<lengths[:,None]).reshape(-1)
        for i,layer in enumerate(model.model.layers):
            def operation():
                if i<2:return layer(h,mask=mask if layer.is_attention_layer else None,cache=None)
                norm=layer.operator_norm(h)
                r=layer.self_attn(norm,mask=mask,cache=None) if layer.is_attention_layer else layer.conv(norm,cache=None)
                x=layer.ffn_norm(h+r);moe=layer.feed_forward
                routing=mx.sigmoid(moe.gate(x));inds=mx.argpartition(routing+moe.expert_bias,kth=-moe.top_k,axis=-1)[...,-moe.top_k:]
                scores=mx.take_along_axis(routing,inds,axis=-1)
                if moe.norm_topk_prob:scores=scores/(scores.sum(-1,keepdims=True)+1e-6)
                scores=scores*moe.routed_scaling_factor
                mx.eval(x,scores,inds)
                flat_x=x.reshape(-1,x.shape[-1]);flat_scores=scores.reshape(-1,moe.top_k)
                indices=np.array(inds).reshape(-1,moe.top_k);down_dim=moe.switch_mlp.down_proj.weight.shape[-1]
                hs_x,hs_down,counts=[],[],[]
                for expert in range(moe.num_experts):
                    tokens,slots=np.nonzero((indices==expert)&valid[:,None]);counts.append(len(tokens))
                    if not len(tokens):
                        hs_x.append(mx.zeros((x.shape[-1]//64,64,64)));hs_down.append(mx.zeros((down_dim//64,64,64)));continue
                    ti,si=mx.array(tokens.astype(np.int32)),mx.array(slots.astype(np.int32));score=flat_scores[ti,si,None]
                    expert_x=flat_x[ti]
                    gate=expert_x@moe.switch_mlp.gate_proj.weight[expert].T;up=expert_x@moe.switch_mlp.up_proj.weight[expert].T
                    a=(expert_x*score).reshape(len(tokens),-1,64);b=(swiglu(gate,up)*score).reshape(len(tokens),-1,64)
                    hx=mx.einsum('ngd,nge->gde',a,a);hd=mx.einsum('ngd,nge->gde',b,b)
                    mx.eval(hx,hd);hs_x.append(hx);hs_down.append(hd)
                hx,hd=mx.stack(hs_x),mx.stack(hs_down)
                if i in moments:
                    old=moments[i];hx=hx+old['h_input'];hd=hd+old['h_down'];counts=np.array(counts)+old['counts']
                mx.eval(hx,hd);moments[i]=dict(h_input=hx,h_down=hd,counts=np.array(counts))
                return layer(h,mask=mask if layer.is_attention_layer else None,cache=None)
            h=promoted(layer,operation)
            print(json.dumps(dict(phase='stats',batch=begin//16,layer=i,elapsed=time.monotonic()-start,peak_gpu_bytes=mx.get_peak_memory())),flush=True)
            mx.clear_cache()
        del h,ids
        mx.clear_cache()
    for i,m in sorted(moments.items()):
        file=output/f'layer-{i:02d}.safetensors';counts=m['counts'].tolist()
        mx.save_safetensors(str(file),dict(h_input=m['h_input'],h_down=m['h_down'],counts=mx.array(counts)))
        receipt['layers'].append(dict(layer=i,sha256=sha(file),routed_counts=counts,unvisited_experts=sum(n==0 for n in counts)))
    receipt['elapsed_seconds']=time.monotonic()-start;save(output/'receipt.json',receipt)



def calibrated_affine(weight, blocks, bits, group):
    """Return codes and FP16 metadata using only calibration block statistics."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    E,O,D = weight.shape
    H = blocks
    if group == 32:
        H = torch.stack((H[...,:32,:32],H[...,32:,32:]),dim=2).reshape(E,-1,32,32)
    H = H.float()
    observed = H.diagonal(dim1=-2,dim2=-1).sum(-1) > 0
    H = torch.where(observed[...,None,None], H, torch.eye(group,device=H.device)[None,None])
    w = weight.reshape(E,O,-1,group).float()
    diag = H.diagonal(dim1=-2,dim2=-1)[:,None]
    lo, hi = w.amin(-1), w.amax(-1)
    levels = 2**bits-1
    best = None
    for alpha in [1.,.995,.99,.98,.96,.92]:
        minimum = torch.where(lo<0,lo*alpha,lo)
        maximum = torch.where(hi>0,hi*alpha,hi)
        scale = ((maximum-minimum)/levels).to(torch.float16).float()
        scale = torch.where(scale>0,scale,torch.ones_like(scale))
        bias = minimum.to(torch.float16).float()
        codes = torch.round((w-bias[...,None])/scale[...,None]).clamp(0,levels)
        error = ((codes*scale[...,None]+bias[...,None]-w).square()*diag).sum(-1)
        if best is None:
            best = [error,scale,bias,codes.to(torch.uint8)]
        else:
            take = error < best[0]
            for j,value in enumerate([error,scale,bias,codes.to(torch.uint8)]):
                best[j] = torch.where(take if j<3 else take[...,None],value,best[j])
    _,scale,bias,codes = best
    del best
    # Group-local Cholesky error feedback, retaining a true group covariance.
    damp = .01*H.diagonal(dim1=-2,dim2=-1).mean(-1)
    L = torch.linalg.cholesky(H+torch.diag_embed(damp[...,None].expand(-1,-1,group)))
    R = torch.linalg.cholesky(torch.cholesky_inverse(L),upper=True)
    work = w.clone(); feedback = torch.empty_like(codes)
    for j in range(group):
        values = work[...,j]
        q = torch.round((values-bias)/scale).clamp(0,levels)
        feedback[...,j] = q.to(torch.uint8)
        error = (values-(q*scale+bias))/R[:,None,:,j,j]
        work[...,j:] -= error[...,None]*R[:,None,:,j,j:]
    del work,L,R
    def loss(c,s,b):
        err = c.float()*s[...,None]+b[...,None]-w
        return torch.einsum('eogd,egdf,eogf->eog',err,H,err).clamp_min(0)
    error = loss(codes,scale,bias)
    fb_error = loss(feedback,scale,bias)
    take = fb_error < error
    feedback_fraction = float(take.float().mean().item())
    codes = torch.where(take[...,None],feedback,codes)
    error = torch.minimum(error,fb_error)
    # Exact two-parameter affine least squares for fixed integer codes.
    cf = codes.float()
    ones_h = H.sum(-1)
    a00 = torch.einsum('eogd,egdf,eogf->eog',cf,H,cf)
    a01 = (cf*ones_h[:,None]).sum(-1)
    a11 = H.sum((-1,-2))[:,None]
    rhs0 = torch.einsum('eogd,egdf,eogf->eog',cf,H,w)
    rhs1 = (w*ones_h[:,None]).sum(-1)
    det = a00*a11-a01.square()
    fitted_s = ((rhs0*a11-rhs1*a01)/det.clamp_min(1e-20)).to(torch.float16).float()
    fitted_b = ((rhs1*a00-rhs0*a01)/det.clamp_min(1e-20)).to(torch.float16).float()
    valid = (det>1e-12) & (fitted_s>0) & torch.isfinite(fitted_s) & torch.isfinite(fitted_b)
    fitted_s = torch.where(valid,fitted_s,scale); fitted_b = torch.where(valid,fitted_b,bias)
    fitted_error = loss(codes,fitted_s,fitted_b)
    take = fitted_error < error
    scale = torch.where(take,fitted_s,scale); bias = torch.where(take,fitted_b,bias)
    error = torch.minimum(error,fitted_error)
    recoded = torch.round((w-bias[...,None])/scale[...,None]).clamp(0,levels).to(torch.uint8)
    recoded_error = loss(recoded,scale,bias)
    take = recoded_error < error
    codes = torch.where(take[...,None],recoded,codes)
    error = torch.minimum(error,recoded_error)
    energy = torch.einsum('eogd,egdf,eogf->eog',w,H,w).sum()
    metrics = dict(relative_block_output_mse=float((error.sum()/energy).item()),
                   feedback_selected_fraction=feedback_fraction)
    return (codes.reshape(E,O,D).cpu().numpy(),scale.to(torch.float16).cpu().numpy(),
            bias.to(torch.float16).cpu().numpy(),metrics)


def quantize(source, stat_root, output):
    import torch
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    torch.set_num_threads(8)
    index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    readers = {name:safe_open(source/name,framework='pt',device='cpu') for name in set(index.values())}
    output.mkdir(parents=True,exist_ok=True)
    receipt = dict(schema_version=1,source_revision=REVISION,calibration_receipt_sha256=sha(stat_root/'receipt.json'),banks={})
    if (output/'receipt.json').exists():
        receipt=json.loads((output/'receipt.json').read_text())
    start=time.monotonic()
    for layer in range(2,40):
        h = load_file(str(stat_root/f'layer-{layer:02d}.safetensors'))
        for original,projection in [('w1','gate_proj'),('w3','up_proj'),('w2','down_proj')]:
            path = f'model.layers.{layer}.feed_forward.switch_mlp.{projection}'
            keys = [f'model.layers.{layer}.feed_forward.experts.{e}.{original}.weight' for e in range(64)]
            W = torch.stack([readers[index[k]].get_tensor(k).float() for k in keys]).to('cuda')
            H = torch.from_numpy(h['h_down' if projection=='down_proj' else 'h_input']).to('cuda')
            trials = receipt['banks'].setdefault(path,{})
            for bits,group in OPTIONS:
                key = f'b{bits}-g{group}'
                file = output/f'layer-{layer:02d}-{projection}-{key}.safetensors'
                if key in trials and file.exists() and sha(file)==trials[key]['sha256']:
                    continue
                codes,scales,biases,metrics = calibrated_affine(W,H,bits,group)
                arrays={path+'.weight':pack_codes(codes,bits),path+'.scales':scales,path+'.biases':biases}
                save_file(arrays,str(file))
                trials[key]=dict(bits=bits,group_size=group,mode='affine',file=file.name,sha256=sha(file),
                                 logical_bytes=sum(x.nbytes for x in arrays.values()),**metrics)
                save(output/'receipt.json',receipt)
                print(json.dumps(dict(phase='quantize',layer=layer,projection=projection,option=key,elapsed=time.monotonic()-start,**metrics)),flush=True)
                del codes,scales,biases,arrays
                torch.cuda.empty_cache()
            del W,H
            torch.cuda.empty_cache()


def allocation(banks, allowed, budget, trunk):
    choices={p:[(k,v) for k,v in values.items() if k in allowed] for p,values in banks.items()}
    if any(not x for x in choices.values()):raise ValueError('No eligible precision for a bank')
    def select(penalty):
        return {p:min(options,key=lambda item:item[1]['relative_block_output_mse']+penalty*item[1]['logical_bytes']/1e9)[0] for p,options in choices.items()}
    def size(policy):return trunk+sum(banks[p][key]['logical_bytes'] for p,key in policy.items())
    low,high=0.,1.
    while size(select(high))>budget:high*=10
    best=select(high)
    for _ in range(100):
        middle=(low+high)/2;policy=select(middle)
        if size(policy)>budget:low=middle
        else:high=middle;best=policy
    # Spend leftover capacity on the largest marginal calibration-error benefit.
    while True:
        remaining=budget-size(best);moves=[]
        for p,key in best.items():
            old=banks[p][key]
            for k,v in choices[p]:
                cost=v['logical_bytes']-old['logical_bytes'];gain=old['relative_block_output_mse']-v['relative_block_output_mse']
                if 0<cost<=remaining and gain>0:moves.append((gain/cost,p,k))
        if not moves:break
        _,p,key=max(moves);best[p]=key
    return best,size(best)


def assemble(source,trials,output,budget):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    output.mkdir(parents=True,exist_ok=True)
    index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    readers={n:safe_open(source/n,framework='pt',device='cpu') for n in set(index.values())}
    trunk={}
    for key in index:
        if '.feed_forward.experts.' in key:continue
        name=key
        if '.layers.0.' in name or '.layers.1.' in name:
            for a,b in [('w1','gate_proj'),('w3','up_proj'),('w2','down_proj')]:name=name.replace('.feed_forward.'+a+'.weight','.feed_forward.'+b+'.weight')
        tensor=readers[index[key]].get_tensor(key)
        if key.endswith('.conv.conv.weight'):tensor=tensor.transpose(1,2).contiguous()
        trunk[name]=tensor
    trunk_file=output/'model-trunk.safetensors';save_file(trunk,str(trunk_file))
    trunk_bytes=sum(x.numel()*x.element_size() for x in trunk.values());del trunk
    banks=json.loads((trials/'receipt.json').read_text())['banks']
    families={'hybrid':{f'b{b}-g{g}' for b,g in OPTIONS},'group32':{'b4-g32','b5-g32','b6-g32'},
              'group64':{'b4-g64','b5-g64','b6-g64','b8-g64'},'four-six':{'b4-g32','b6-g32'},'five-six':{'b5-g64','b6-g64'}}
    receipts=[]
    for family,allowed in families.items():
        policy,total=allocation(banks,allowed,budget,trunk_bytes)
        if not 18e9<=total<=19e9:raise ValueError('Allocated model is outside the required size range')
        root=output/(family+'-'+str(int(budget/1e6))+'mb');root.mkdir(exist_ok=False)
        os.link(trunk_file,root/'model-trunk.safetensors')
        from safetensors import safe_open as open_np
        with open_np(trunk_file,framework='np') as f:weight_map={k:'model-trunk.safetensors' for k in f.keys()}
        quant={};files={};selected={}
        files['model-trunk.safetensors']=dict(bytes=trunk_file.stat().st_size,sha256=sha(trunk_file))
        for i,(path,key) in enumerate(sorted(policy.items())):
            item=banks[path][key];name=f'model-experts-{i:03d}.safetensors';os.link(trials/item['file'],root/name)
            for suffix in ['weight','scales','biases']:weight_map[path+'.'+suffix]=name
            quant[path]={k:item[k] for k in ['bits','group_size','mode']};files[name]=dict(bytes=(root/name).stat().st_size,sha256=item['sha256']);selected[path]=key
        config=json.loads((source/'config.json').read_text());config['quantization']={'bits':8,'group_size':64,'mode':'affine',**quant};config['quantization_config']=config['quantization'];config['mererun_quantization']=quant;config['mererun_fp32_activations']=True
        save(root/'config.json',config);save(root/'model.safetensors.index.json',dict(metadata=dict(total_size=total),weight_map=weight_map))
        for name in ['tokenizer.json','tokenizer_config.json','generation_config.json','chat_template.jinja','LICENSE']:shutil.copy2(source/name,root/name)
        conversion=dict(schema_version=1,source_repository='LiquidAI/LFM2.5-24B-A2B',source_revision=REVISION,
                        profile=root.name,logical_weight_bytes=total,quality_qualified=False,quantization=quant,
                        affine_metadata_dtype='float16',quantization_arithmetic='float32',fp32_activations=True,output_files=files,
                        calibration_receipt_sha256=sha(trials/'receipt.json'),allocation=selected,
                        method='routed activation block covariance; group-local error feedback and affine refinement; calibration proxy budget allocation',
                        converter_sha256=sha(Path(__file__)),source_tensor_count=len(index),tensor_count=len(weight_map),
                        source_files=json.loads((source/'SOURCE_AUDIT.json').read_text())['source_files'],
                        metadata_files=json.loads((source/'SOURCE_AUDIT.json').read_text())['metadata_files'])
        save(root/'MERERUN_CONVERSION.json',conversion)
        (root/'README.md').write_text('# Experimental LFM2.5 24B-A2B '+root.name+'\n\nSource: LiquidAI/LFM2.5-24B-A2B@'+REVISION+'\n\nOriginal LFM Open License applies; see LICENSE. Calibration does not by itself establish quality. See MERERUN_CONVERSION.json and independent comparison receipts.\n')
        receipts.append(dict(profile=root.name,logical_weight_bytes=total,conversion_sha256=sha(root/'MERERUN_CONVERSION.json')))
    save(output/('candidates-'+str(int(budget/1e6))+'mb.json'),receipts)
    print(json.dumps(receipts,indent=2))


def expand_expert_reference(model):
    """Expand stored affine codes exactly to FP32 for dense CUDA reference GEMMs."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    count=0
    for layer in model.model.layers:
        mlp=getattr(layer.feed_forward,'switch_mlp',None)
        if mlp is None:continue
        for name in ['gate_proj','up_proj','down_proj']:
            quant=getattr(mlp,name)
            if not isinstance(quant,QuantizedSwitchLinear):continue
            assert quant.mode=='affine' and quant.global_scale is None
            dense=SwitchLinear.__new__(SwitchLinear);nn.Module.__init__(dense)
            dense.weight=mx.dequantize(quant.weight,quant.scales.astype(mx.float32),quant.biases.astype(mx.float32),group_size=quant.group_size,bits=quant.bits,mode='affine')
            if 'bias' in quant:dense.bias=quant.bias.astype(mx.float32)
            mx.eval(dense.weight);setattr(mlp,name,dense);count+=1
        mx.clear_cache()
    return count


def score(model_path,frozen,output,split,dequantized_reference=False,limit=None):
    mx=mlx_setup()
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from safetensors.numpy import save_file
    suite=json.loads(frozen.read_text());rows=[r for r in suite['cases'] if split=='all' or r['split']==split]
    if limit is not None:rows=rows[:limit]
    output.mkdir(parents=True,exist_ok=True);load_start=time.monotonic();model,_=load(str(model_path),lazy=False)
    load_seconds=time.monotonic()-load_start;expansion_start=time.monotonic()
    expanded=expand_expert_reference(model) if dequantized_reference else 0
    expansion_seconds=time.monotonic()-expansion_start;start=time.monotonic()
    receipt=dict(schema_version=1,backend='MLX-CUDA strict FP32',compute_dtype='float32',mlx_enable_tf32=False,
                 source_revision=REVISION,frozen_suite_sha256=sha(frozen),config_sha256=sha(model_path/'config.json'),split=split,maximum_batch_size=16,
                 expert_matrix_path='dequantized FP32 dense reference' if expanded else 'original module kernels',dequantized_expert_banks=expanded,
                 load_seconds=load_seconds,reference_expansion_seconds=expansion_seconds,scorer_sha256=sha(Path(__file__)),cases=[])
    conversion=model_path/'MERERUN_CONVERSION.json'
    if conversion.exists():receipt['conversion_sha256']=sha(conversion)
    else:receipt['source_audit_sha256']=sha(model_path/'SOURCE_AUDIT.json')
    for begin in range(0,len(rows),16):
        batch=rows[begin:begin+16];width=max(len(r['tokens'])-1 for r in batch)
        ids=mx.array([r['tokens'][:-1]+[0]*(width-len(r['tokens'])+1) for r in batch],dtype=mx.int32)
        h=model.model.embed_tokens(ids).astype(mx.float32);mx.eval(h);mask=create_attention_mask(h,None)
        for i,layer in enumerate(model.model.layers):
            h=promoted(layer,lambda:layer(h,mask=mask if layer.is_attention_layer else None,cache=None))
        h=promoted(model.model.embedding_norm,lambda:model.model.embedding_norm(h));embed=model.model.embed_tokens
        for i,row in enumerate(batch):
            selected=h[i,row['score_start']-1:len(row['tokens'])-1];logits=promoted(embed,lambda:embed.as_linear(selected))
            value=np.array(logits);assert value.shape==(len(row['tokens'])-row['score_start'],128000)
            file=output/(row['id']+'.safetensors');save_file({'logits':value},str(file))
            receipt['cases'].append(dict(id=row['id'],logits_file=file.name,sha256=sha(file),tokens=value.shape[0]))
            receipt['elapsed_seconds']=time.monotonic()-start;receipt['peak_gpu_bytes']=mx.get_peak_memory();save(output/'receipt.json',receipt)
            del logits,value;mx.clear_cache()
        print(json.dumps(dict(phase='score',model=model_path.name,completed_cases=len(receipt['cases']),elapsed=time.monotonic()-start)),flush=True)
        del h,ids;mx.clear_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    for name in ['stats','quantize','assemble','score']:
        s=commands.add_parser(name)
        if name!='score':s.add_argument('--source',type=Path,required=True)
        s.add_argument('--output',type=Path,required=True)
        if name in ['stats','score']:s.add_argument('--frozen',type=Path,required=True)
        if name=='quantize':s.add_argument('--stats',type=Path,required=True)
        if name=='assemble':s.add_argument('--trials',type=Path,required=True);s.add_argument('--budget',type=int,default=18900000000)
        if name=='score':
            s.add_argument('--model',type=Path,required=True);s.add_argument('--split',choices=['all','calibration','heldout'],default='all')
            s.add_argument('--dequantized-reference',action='store_true');s.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.command=='stats':stats(a.source,a.frozen,a.output)
    elif a.command=='quantize':quantize(a.source,a.stats,a.output)
    elif a.command=='assemble':assemble(a.source,a.trials,a.output,a.budget)
    else:score(a.model,a.frozen,a.output,a.split,a.dequantized_reference,a.limit)

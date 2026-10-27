#!/usr/bin/env python3
"""Refine a frozen mixed policy with full-column Hessian error feedback.

Keeps each selected bank's bits, group size, FP16 affine coefficients, and byte
budget. Only integer codes change. Rows retain their original codes unless the
new codes lower full-covariance calibration error. End-to-end calibration KL
must still decide whether to retain this refinement.
"""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from lfm25_quality18 import mlx_setup, pack_codes, promoted, save, sha


def unpack_codes(packed, bits, columns):
    result = np.empty((*packed.shape[:-1], columns), dtype=np.uint8)
    for column in range(columns):
        word, shift = divmod(column * bits, 32)
        value = packed[..., word].astype(np.uint64) >> shift
        if shift + bits > 32:
            value |= packed[..., word + 1].astype(np.uint64) << (32 - shift)
        result[..., column] = (value & (2**bits - 1)).astype(np.uint8)
    return result


def refine_codes(weight, covariance, original, scales, biases, bits, group, block=128):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    E, O, D = weight.shape
    H = covariance.float()
    observed = H.diagonal(dim1=-2, dim2=-1).sum(-1) > 0
    H = torch.where(observed[:, None, None], H, torch.eye(D, device=H.device)[None])
    damping = .01 * H.diagonal(dim1=-2, dim2=-1).mean(-1)
    permutation = H.diagonal(dim1=-2, dim2=-1).argsort(-1, descending=True)
    ordered_h = H.gather(1, permutation[:, :, None].expand(-1, -1, D)).gather(2, permutation[:, None, :].expand(-1, D, -1))
    L = torch.linalg.cholesky(ordered_h + torch.diag_embed(damping[:, None].expand(-1, D)))
    R = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    del L, ordered_h
    work = weight.gather(2, permutation[:, None, :].expand(-1, O, -1)).clone()
    groups = (permutation // group)[:, None, :].expand(-1, O, -1)
    s = scales.gather(2, groups).float()
    b = biases.gather(2, groups).float()
    ordered_codes = torch.empty_like(original)
    for begin in range(0, D, block):
        end = min(begin + block, D)
        values = work[:, :, begin:end].clone()
        errors = torch.empty_like(values)
        for column in range(begin, end):
            j = column - begin
            v = values[:, :, j]
            q = torch.round((v - b[:, :, column]) / s[:, :, column]).clamp(0, 2**bits - 1)
            ordered_codes[:, :, column] = q.to(torch.uint8)
            error = (v - (q * s[:, :, column] + b[:, :, column])) / R[:, column, column, None]
            errors[:, :, j] = error
            values[:, :, j:] -= error[:, :, None] * R[:, None, column, column:end]
        if end < D:
            work[:, :, end:] -= torch.bmm(errors, R[:, begin:end, end:])
    inverse = permutation.argsort(-1)
    codes = ordered_codes.gather(2, inverse[:, None, :].expand(-1, O, -1))
    del work, ordered_codes, R, s, b, groups
    def loss(c):
        values = c.reshape(E, O, -1, group).float() * scales[:, :, :, None] + biases[:, :, :, None]
        err = values.reshape(E, O, D) - weight
        return (torch.bmm(err, H) * err).sum(-1).clamp_min(0)
    old_loss, new_loss = loss(original), loss(codes)
    take = new_loss < old_loss
    codes = torch.where(take[:, :, None], codes, original)
    return codes, dict(original_full_hessian_loss=float(old_loss.sum().item()),
                       refined_full_hessian_loss=float(torch.minimum(old_loss, new_loss).sum().item()),
                       improved_row_fraction=float(take.float().mean().item()),
                       unvisited_experts=int((~observed).sum().item()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'model', 'frozen', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    mx = mlx_setup()
    import torch
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    from mlx_lm import load
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.base import create_attention_mask
    from mlx.utils import tree_map
    torch.set_num_threads(8)
    model, _ = load(str(a.source), lazy=False)
    suite = json.loads(a.frozen.read_text())
    rows = [r for r in suite['cases'] if r['split'] == 'calibration']
    index = json.loads((a.source / 'model.safetensors.index.json').read_text())['weight_map']
    readers = {name: safe_open(a.source / name, framework='pt', device='cpu') for name in set(index.values())}
    conversion = json.loads((a.model / 'MERERUN_CONVERSION.json').read_text())
    converted_index = json.loads((a.model / 'model.safetensors.index.json').read_text())
    a.output.mkdir(exist_ok=False)
    for file in a.model.iterdir():
        if file.is_file():
            if file.suffix == '.safetensors': os.link(file, a.output / file.name)
            else: shutil.copy2(file, a.output / file.name)
    batches = []
    for begin in range(0, len(rows), 16):
        batch = rows[begin:begin+16]
        lengths = np.array([len(r['tokens'])-1 for r in batch]); width = int(lengths.max())
        ids = mx.array([r['tokens'][:-1] + [0]*(width-len(r['tokens'])+1) for r in batch], dtype=mx.int32)
        h = model.model.embed_tokens(ids).astype(mx.float32); mx.eval(h)
        batches.append((h, (np.arange(width)[None, :] < lengths[:, None]).reshape(-1)))
    receipt = dict(schema_version=1, frozen_suite_sha256=sha(a.frozen),
                   parent_conversion_sha256=sha(a.model/'MERERUN_CONVERSION.json'),
                   calibration_cases=len(rows), method='full-column covariance, descending activation order, 1 percent damping, static original affine coefficients, cross-block error feedback; retain baseline per output row if lower measured full-covariance loss',
                   tool_sha256=sha(Path(__file__)), layers=[])
    start = time.monotonic()
    for i, layer in enumerate(model.model.layers):
        original = layer.parameters()
        layer.update(tree_map(lambda x: x.astype(mx.float32) if x.dtype in [mx.bfloat16, mx.float16] else x, original))
        mx.eval(layer.parameters())
        hx = hd = None; counts = np.zeros(64, dtype=np.int64)
        next_batches = []
        for h, valid in batches:
            mask = create_attention_mask(h, None)
            if i >= 2:
                norm = layer.operator_norm(h)
                r = layer.self_attn(norm, mask=mask, cache=None) if layer.is_attention_layer else layer.conv(norm, cache=None)
                x = layer.ffn_norm(h+r); moe = layer.feed_forward
                routing = mx.sigmoid(moe.gate(x))
                inds = mx.argpartition(routing+moe.expert_bias, kth=-moe.top_k, axis=-1)[..., -moe.top_k:]
                scores = mx.take_along_axis(routing, inds, axis=-1)
                if moe.norm_topk_prob: scores = scores/(scores.sum(-1, keepdims=True)+1e-6)
                scores *= moe.routed_scaling_factor
                mx.eval(x, inds, scores)
                flat = x.reshape(-1, x.shape[-1]); scores = scores.reshape(-1, moe.top_k)
                indices = np.array(inds).reshape(-1, moe.top_k)
                xs, ds = [], []
                down_dim = moe.switch_mlp.down_proj.weight.shape[-1]
                for expert in range(moe.num_experts):
                    tokens, slots = np.nonzero((indices == expert) & valid[:, None]); counts[expert] += len(tokens)
                    if not len(tokens):
                        xs.append(mx.zeros((x.shape[-1], x.shape[-1]))); ds.append(mx.zeros((down_dim, down_dim))); continue
                    ti, si = mx.array(tokens.astype(np.int32)), mx.array(slots.astype(np.int32))
                    expert_x = flat[ti]; score = scores[ti, si, None]
                    gate = expert_x @ moe.switch_mlp.gate_proj.weight[expert].T
                    up = expert_x @ moe.switch_mlp.up_proj.weight[expert].T
                    xx = expert_x*score; dd = swiglu(gate, up)*score
                    xx, dd = xx.T @ xx, dd.T @ dd
                    mx.eval(xx, dd); xs.append(xx); ds.append(dd)
                xx, dd = mx.stack(xs), mx.stack(ds)
                hx = xx if hx is None else hx+xx; hd = dd if hd is None else hd+dd
                mx.eval(hx, hd)
            out = layer(h, mask=mask if layer.is_attention_layer else None, cache=None)
            mx.eval(out); next_batches.append((out, valid))
        layer.update(original); del original
        batches = next_batches; mx.clear_cache()
        if i < 2: continue
        layer_result = dict(layer=i, routed_counts=counts.tolist(), projections={})
        for source_name, projection, moments in [('w1','gate_proj',hx), ('w3','up_proj',hx), ('w2','down_proj',hd)]:
            path = f'model.layers.{i}.feed_forward.switch_mlp.{projection}'
            policy = conversion['quantization'][path]; bits, group = policy['bits'], policy['group_size']
            name = converted_index['weight_map'][path+'.weight']
            arrays = load_file(str(a.model/name)); packed = arrays[path+'.weight']
            E, O, D = 64, packed.shape[1], packed.shape[2]*32//bits
            codes = unpack_codes(packed, bits, D)
            keys = [f'model.layers.{i}.feed_forward.experts.{e}.{source_name}.weight' for e in range(64)]
            weight = torch.stack([readers[index[k]].get_tensor(k).float() for k in keys]).to('cuda')
            covariance = torch.from_numpy(np.array(moments)).to('cuda')
            codes = torch.from_numpy(codes).to('cuda')
            scales = torch.from_numpy(arrays[path+'.scales'].astype(np.float32)).to('cuda')
            biases = torch.from_numpy(arrays[path+'.biases'].astype(np.float32)).to('cuda')
            refined, metrics = refine_codes(weight, covariance, codes, scales, biases, bits, group)
            arrays[path+'.weight'] = pack_codes(refined.cpu().numpy(), bits)
            # Replace the destination hardlink without modifying its parent.
            dest = a.output/name; temp = dest.with_suffix('.refined')
            save_file(arrays, str(temp)); temp.replace(dest)
            conversion['output_files'][name] = dict(bytes=dest.stat().st_size, sha256=sha(dest))
            layer_result['projections'][projection] = metrics
            print(json.dumps(dict(phase='full-hessian', layer=i, projection=projection, elapsed=time.monotonic()-start, **metrics)), flush=True)
            del weight, covariance, codes, scales, biases, refined, arrays
            torch.cuda.empty_cache()
        del hx, hd, moments
        mx.clear_cache()
        receipt['layers'].append(layer_result); receipt['elapsed_seconds']=time.monotonic()-start
        save(a.output/'FULL_HESSIAN_REFINEMENT.json',receipt)
    conversion['profile']=a.output.name
    conversion['parent_conversion_sha256']=receipt['parent_conversion_sha256']
    conversion['full_hessian_refinement_sha256']=sha(a.output/'FULL_HESSIAN_REFINEMENT.json')
    conversion['method'] += '; full-column Hessian integer-code refinement with static affine metadata'
    conversion['converter_sha256']=sha(Path(__file__))
    save(a.output/'MERERUN_CONVERSION.json',conversion)
    assert sha(a.model/'MERERUN_CONVERSION.json') == receipt['parent_conversion_sha256']
    for name, expected in json.loads((a.model/'MERERUN_CONVERSION.json').read_text())['output_files'].items():
        assert sha(a.model/name) == expected['sha256'], 'Parent changed: '+name
    (a.output/'README.md').write_text('# Experimental '+a.output.name+'\n\nOriginal LFM Open License applies; see LICENSE. Full-column Hessian refinement preserves the parent bit/group policy and FP16 affine metadata. Quality depends on separate calibration and heldout receipts. See MERERUN_CONVERSION.json and FULL_HESSIAN_REFINEMENT.json.\n')
    print('Full Hessian refinement complete; calibration KL must decide retention.',flush=True)


if __name__ == '__main__':
    main()

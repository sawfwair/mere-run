#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx==0.32.4", "numpy==2.3.5"]
# ///
"""Profile SOV expert arithmetic using released weights and native captured routes."""
import argparse
import json
from pathlib import Path
import statistics
import time
import mlx.core as mx
import numpy as np
GELU = mx.compile(lambda x: 0.5 * x * (1 + mx.tanh((2 / np.pi) ** 0.5 * (x + 0.044715 * x ** 3))))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--activations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=15)
    args = p.parse_args()
    index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.layers.0.experts.'
    weights = mx.load(str(args.checkpoint / index[prefix + 'gate_up_proj']))
    combined = weights[prefix + 'gate_up_proj']
    down = weights[prefix + 'down_proj']
    del weights
    mid = combined.shape[1] // 2
    gate, up = (combined[:, :mid], combined[:, mid:])
    q8 = {g: [mx.quantize(w, group_size=g, bits=8) for w in [combined, down]] for g in [32, 64]}
    q8f = {g: [mx.quantize(w.astype(mx.float32), group_size=g, bits=8) for w in [combined, down]] for g in [32, 64]}
    mx.eval(combined, down, q8, q8f)
    source = mx.load(str(args.activations))
    rows = []
    for kind in ['verification', 'prefill']:
        x = source[kind + '_input']
        indices = source[kind + '_indices']
        mix = source[kind + '_weights']
        k = x.shape[-1]
        count = indices.size
        topk = indices.shape[-1]
        order = mx.argsort(indices.reshape(-1))
        inverse = mx.argsort(order)
        sorted_indices = indices.reshape(-1)[order]
        routed = x.reshape(-1, k)[order // topk].reshape(count, 1, k)
        mx.eval(routed, sorted_indices, inverse, mix)

        def finish(value):
            z = value[inverse].reshape(*indices.shape, k)
            return (z * mix[..., None]).astype(x.dtype).sum(axis=-2)

        def bf16(a, sorted_flag=True, fused=False):
            if fused:
                pair = mx.gather_mm(a, combined.swapaxes(-1, -2), rhs_indices=sorted_indices, sorted_indices=sorted_flag)
                g, u = (pair[..., :mid], pair[..., mid:])
            else:
                g = mx.gather_mm(a, gate.swapaxes(-1, -2), rhs_indices=sorted_indices, sorted_indices=sorted_flag)
                u = mx.gather_mm(a, up.swapaxes(-1, -2), rhs_indices=sorted_indices, sorted_indices=sorted_flag)
            z = GELU(g) * u
            return finish(mx.gather_mm(z, down.swapaxes(-1, -2), rhs_indices=sorted_indices, sorted_indices=sorted_flag))

        def quant(a, group, precise=False):

            def mm(a, w):
                return mx.gather_qmm(a, w[0], w[1], w[2], rhs_indices=sorted_indices, transpose=True, group_size=group, bits=8, sorted_indices=True).astype(x.dtype)
            packed = q8f[group] if precise else q8[group]
            pair = mm(a, packed[0])
            g, u = (pair[..., :mid], pair[..., mid:])
            return finish(mm(GELU(g) * u, packed[1]))
        functions = {'reference': lambda a: bf16(a), 'bf16-gemv': lambda a: bf16(a, False), 'bf16-fused': lambda a: bf16(a, True, True), 'bf16-fused-gemv': lambda a: bf16(a, False, True)}
        functions.update({f'q8-g{g}': lambda a, g=g: quant(a, g) for g in q8})
        functions.update({f'q8f-g{g}': lambda a, g=g: quant(a, g, True) for g in q8f})
        fns = {name: mx.compile(fn) for name, fn in functions.items()}
        end = time.perf_counter() + 1
        while time.perf_counter() < end:
            for fn in fns.values():
                mx.eval(fn(routed))
        timings = {name: [] for name in fns}
        outputs = {}
        for i in range(args.repetitions):
            for name in list(fns) if i % 2 == 0 else list(reversed(fns)):
                start = time.perf_counter()
                y = fns[name](routed)
                mx.eval(y)
                timings[name].append(time.perf_counter() - start)
                outputs[name] = np.array(y.astype(mx.float32))
        ref = outputs['reference'].astype(np.float64)
        row = dict(kind=kind, shape=list(x.shape), uniqueExperts=int(len(np.unique(np.array(indices)))), variants={})
        for name, t in timings.items():
            out = outputs[name].astype(np.float64)
            row['variants'][name] = dict(medianSeconds=statistics.median(t), seconds=t, relativeRMSE=float(np.linalg.norm(out - ref) / np.linalg.norm(ref)), bitIdentical=bool(np.array_equal(out, ref)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
if __name__ == '__main__':
    main()

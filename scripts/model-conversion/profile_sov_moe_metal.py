#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx==0.32.4", "numpy==2.3.5"]
# ///
"""Screen the shared native Q8 expert shader against MLX on captured SOV routes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--activations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=25)
    parser.add_argument('--reuse-rows', nargs='+', type=int, choices=[1, 2, 4], default=[1, 2, 4])
    parser.add_argument('--simdgroups', nargs='+', type=int, choices=[1, 2, 4, 8], default=[2])
    parser.add_argument('--legacy-bit-unpack', action='store_true', help='Use the original shifts instead of byte-vector unpacking.')
    parser.add_argument('--output-columns', nargs='+', type=int, choices=[1, 2, 4, 8, 16, 32], default=[8])
    parser.add_argument('--reference-layout', action='store_true', help='Dispatch output tiles before routes instead of the optimized route-first order.')
    args = parser.parse_args()
    shader = Path(__file__).resolve().parents[2] / 'Sources/MereRunGemmaModel/Kernels/Gemma4Q8Experts.metal'
    kernel = mx.fast.metal_kernel(name='gemma4_q8_expert_reuse',
        input_names=['input', 'weights', 'scales', 'biases', 'indices'],
        output_names=['output'], source=shader.read_text())
    index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.layers.0.experts.'
    weights = {}
    for key in ['gate_up_proj', 'down_proj']:
        arrays = mx.load(str(args.checkpoint / index[prefix + key]))
        weights[key] = mx.quantize(arrays[prefix + key], group_size=64, bits=8)
    mx.eval(weights)
    mx.clear_cache()
    activations = mx.load(str(args.activations))
    gelu = mx.compile(lambda x: 0.5 * x * (1 + mx.tanh((2 / np.pi) ** 0.5 * (x + 0.044715 * x**3))))
    reports = []
    for kind in ['verification', 'prefill']:
        x, indices, mix = (activations[kind + suffix] for suffix in ['_input', '_indices', '_weights'])
        count, topk, width = indices.size, indices.shape[-1], x.shape[-1]
        order = mx.argsort(indices.reshape(-1))
        inverse = mx.argsort(order)
        sorted_indices = indices.reshape(-1)[order]
        routed = x.reshape(-1, width)[order // topk].reshape(count, 1, width)
        mx.eval(routed, sorted_indices, inverse, mix)

        def block(a, reuse=None, simdgroups=2, columns=8):
            def project(value, packed):
                outputs, inner = packed[0].shape[1], packed[0].shape[2] * 4
                if reuse is None:
                    return mx.gather_qmm(value, *packed, rhs_indices=sorted_indices,
                        transpose=True, group_size=64, bits=8, sorted_indices=True)
                tiles = outputs // (columns * simdgroups)
                grid = (tiles * simdgroups * 32, count, 1) if args.reference_layout else (count * simdgroups * 32, tiles, 1)
                return kernel(inputs=[value, *packed, sorted_indices],
                    template=[('R', count), ('N', outputs), ('K', inner), ('Rows', reuse), ('O', columns), ('SG', simdgroups),
                              ('ByteLoads', not args.legacy_bit_unpack), ('RouteFirst', not args.reference_layout),
                              ('VPT', 8 if inner % 256 == 0 else 4)],
                    grid=grid, threadgroup=(32 * simdgroups, 1, 1),
                    output_shapes=[(count, 1, outputs)], output_dtypes=[mx.bfloat16])[0]
            pair = project(a, weights['gate_up_proj'])
            middle = pair.shape[-1] // 2
            output = project(gelu(pair[..., :middle]) * pair[..., middle:], weights['down_proj'])
            output = output[inverse].reshape(*indices.shape, width)
            return (output * mix[..., None]).astype(x.dtype).sum(axis=-2)

        # Matrix prefill is an unchanged fallback in the native policy. Custom
        # vectors are screened only on the small verification workload.
        functions = {'mlx': mx.compile(lambda a: block(a))}
        if kind == 'verification':
            for reuse in args.reuse_rows:
                for groups in args.simdgroups:
                    for columns in args.output_columns:
                        if any(w[0].shape[1] % (columns * groups) for w in weights.values()):
                            continue
                        name = f'metal-reuse-{reuse}-sg-{groups}-columns-{columns}'
                        functions[name] = mx.compile(lambda a, reuse=reuse, groups=groups, columns=columns: block(a, reuse, groups, columns))
        for function in functions.values():
            mx.eval(function(routed))
        until = time.perf_counter() + 2
        while time.perf_counter() < until:
            for function in functions.values():
                mx.eval(function(routed))
        timings = {name: [] for name in functions}
        outputs = {}
        for repetition in range(args.repetitions):
            order_ = list(functions) if repetition % 2 == 0 else list(reversed(functions))
            for name in order_:
                started = time.perf_counter()
                result = functions[name](routed)
                mx.eval(result)
                timings[name].append(time.perf_counter() - started)
                outputs[name] = np.array(result.astype(mx.float32))
        reference = outputs['mlx'].astype(np.float64)
        row = dict(kind=kind, shape=list(x.shape), uniqueExperts=len(np.unique(np.array(indices))), variants={})
        for name, samples in timings.items():
            result = outputs[name].astype(np.float64)
            row['variants'][name] = dict(medianSeconds=statistics.median(samples), seconds=samples,
                bitIdentical=bool(np.array_equal(reference, result)),
                relativeRMSE=float(np.linalg.norm(result - reference) / np.linalg.norm(reference)))
        reports.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(shaderSHA256=hashlib.sha256(shader.read_bytes()).hexdigest(), byteLoads=not args.legacy_bit_unpack,
        routeFirst=not args.reference_layout,
        scope='Prepared sorted routes; expert projections, activation, scatter and mixing included. Native end-to-end qualification separate.',
        reports=reports), indent=2) + '\n')


if __name__ == '__main__':
    main()

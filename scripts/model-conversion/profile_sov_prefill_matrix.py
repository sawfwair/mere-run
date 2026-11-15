#!/usr/bin/env python3
"""Screen M5 grouped prefill TensorOps on native packed weights and activations.

Job planning, sorting and route gathering are excluded in every variant. A win
here is an upper bound; it still needs native dispatch and end-to-end validation.
"""
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
    parser.add_argument('--capture', type=Path, required=True, help='Native research output prefix.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=15)
    args = parser.parse_args()
    a = mx.load(str(args.capture) + '-routes.safetensors')
    weights = mx.load(str(args.capture) + '-experts.safetensors')
    def projection(name):
        return tuple(weights['switch_glu.' + name + '.' + key] for key in ['weight', 'scales', 'biases'])
    up, down = projection('gate_up_proj'), projection('down_proj')
    shader = Path(__file__).with_name('SOVPrefillQ8.metal')
    header = '#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n#pragma clang fp reassociate(off)\n'
    kernel = mx.fast.metal_kernel('sov_prefill_q8_matrix',
        ['input', 'weights', 'scales', 'biases', 'indices', 'jobs'], ['output'], shader.read_text(), header=header)
    gelu = mx.compile(lambda x: 0.5 * x * (1 + mx.tanh(float(np.sqrt(np.float32(2) / np.float32(np.pi))) * (x + 0.044715 * x**3))))
    reports = []
    for kind in ['prefill', 'verification']:
        x, ids, mix = (a[kind + suffix] for suffix in ['_input', '_indices', '_weights'])
        count, topk, width = ids.size, ids.shape[-1], x.shape[-1]
        order = mx.argsort(ids.reshape(-1)); inverse = mx.argsort(order)
        indices = ids.reshape(-1)[order]
        routed = x.reshape(-1, width)[order // topk].reshape(count, 1, width)
        mx.eval(indices, routed)
        expert_ids = np.array(indices)
        _, starts, counts = np.unique(expert_ids, return_index=True, return_counts=True)
        jobs = {bm: mx.array([row for start, length in zip(starts, counts) for row in range(int(start), int(start + length), bm)], dtype=mx.uint32)
                for bm in [8, 16, 32]}
        mx.eval(jobs)
        def block(value, geometry=None, affine=False):
            def mm(value, packed):
                if geometry is None:
                    return mx.gather_qmm(value, *packed, rhs_indices=indices, transpose=True, group_size=64, bits=8, sorted_indices=True)
                bm, bn = geometry
                n, k = packed[0].shape[1], packed[0].shape[2] * 4
                return kernel(inputs=[value, *packed, indices, jobs[bm]],
                    template=[('R', count), ('N', n), ('K', k), ('BM', bm), ('BN', bn), ('BK', 64), ('GroupAffine', affine)],
                    grid=(jobs[bm].size * 32, n // bn, 1), threadgroup=(32, 1, 1),
                    output_shapes=[(count, 1, n)], output_dtypes=[mx.bfloat16])[0]
            pair = mm(value, up); mid = pair.shape[-1] // 2
            result = mm(gelu(pair[..., :mid]) * pair[..., mid:], down)
            return (result[inverse].reshape(*ids.shape, width) * mix[..., None]).astype(x.dtype).sum(axis=-2)
        functions = {'mlx': mx.compile(lambda value: block(value))}
        for bm, bn in [(8, 32), (16, 32), (16, 64), (32, 64)]:
            for affine in [False, True]:
                name = f'{bm}x{bn}-' + ('group-affine' if affine else 'bf16-dequant')
                functions[name] = mx.compile(lambda value, shape=(bm, bn), affine=affine: block(value, shape, affine))
        outputs = {}
        samples = {name: [] for name in functions}
        for name, function in functions.items():
            for _ in range(3): mx.eval(function(routed))
        for iteration in range(args.repetitions):
            names = list(functions) if iteration % 2 == 0 else list(reversed(functions))
            for name in names:
                start = time.perf_counter(); result = functions[name](routed); mx.eval(result)
                samples[name].append(time.perf_counter() - start); outputs[name] = result
        reference = np.array(outputs['mlx'].astype(mx.float32)).astype(np.float64)
        variants = {}
        for name, result in outputs.items():
            values = np.array(result.astype(mx.float32)).astype(np.float64)
            variants[name] = dict(seconds=samples[name], medianSeconds=statistics.median(samples[name]),
                bitIdentical=bool(np.array_equal(values, reference)), maxAbsoluteError=float(np.max(np.abs(values-reference))),
                relativeRMSE=float(np.linalg.norm(values-reference)/np.linalg.norm(reference)))
        native = np.array(a[kind + '_output'].astype(mx.float32))
        report = dict(kind=kind, tokens=x.shape[1], nativeReferenceBitIdentical=bool(np.array_equal(native, reference)), variants=variants)
        reports.append(report); print(json.dumps(report), flush=True)
    args.output.write_text(json.dumps(dict(mlx=mx.__version__, shaderSHA256=hashlib.sha256(shader.read_bytes()).hexdigest(),
        scope=__doc__, reports=reports), indent=2) + '\n')


if __name__ == '__main__':
    main()

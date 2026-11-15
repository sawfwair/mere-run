#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx==0.32.4", "numpy==2.3.5"]
# ///
"""Compare the shared fused BF16 head shader with native SOV norm/RoPE graphs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--activations', type=Path, required=True, help='Native decoder profile prefix.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=60)
    args = parser.parse_args()
    shader = Path(__file__).resolve().parents[2] / 'Sources/MereRunGemmaModel/Kernels/Gemma4HeadNormRoPE.metal'
    kernel = mx.fast.metal_kernel(name='gemma4_head_norm_rope',
        input_names=['q_raw', 'k_raw', 'v_raw', 'q_weight', 'k_weight', 'eps', 'offset', 'freqs', 'log_base'],
        output_names=['q_out', 'k_out', 'v_out'], source=shader.read_text())
    config = json.loads((args.checkpoint / 'config.json').read_text())['text_config']
    reports = []
    for layer in [0, 5]:
        arrays = mx.load(str(args.activations) + f'-attention-{layer}.safetensors')
        q, k, v, qw, kw = (arrays[name] for name in ['q_raw', 'k_raw', 'v_raw', 'q_norm', 'k_norm'])
        full = config['layer_types'][layer] == 'full_attention'
        params = config['rope_parameters'][config['layer_types'][layer]]
        dim = config['global_head_dim'] if full else config['head_dim']
        count, heads, kvheads = q.shape[1], q.shape[-1] // dim, k.shape[-1] // dim
        rotated = int(params.get('partial_rotary_factor', 1) * (dim // 2))
        base = params['rope_theta']
        offset_value = arrays['keys'].shape[2] - count
        frequencies = mx.power(mx.array(base, dtype=mx.float32), mx.arange(0, rotated * 2, 2, dtype=mx.float32) / dim)
        full_frequencies = mx.concatenate([frequencies, mx.full((dim // 2 - rotated,), float('inf'))])
        eps = mx.array([config['rms_norm_eps']])
        offset = mx.array([offset_value], dtype=mx.int32)
        logbase = mx.array([math.log2(base)])
        mx.eval(q, k, v, qw, kw, frequencies, full_frequencies, eps, offset, logbase)

        def reference(direct):
            def rope(value):
                if not full:
                    return mx.fast.rope(value, dims=dim, traditional=False, base=base, scale=1, offset=offset_value)
                if direct:
                    return mx.fast.rope(value, dims=dim, traditional=False, base=None, scale=1,
                        offset=offset_value, freqs=full_frequencies)
                half = dim // 2
                left, right = value[..., :half], value[..., half:]
                packed = mx.concatenate([left[..., :rotated], right[..., :rotated]], axis=-1)
                packed = mx.fast.rope(packed, dims=rotated * 2, traditional=False, base=None,
                    scale=1, offset=offset_value, freqs=frequencies)
                return mx.concatenate([packed[..., :rotated], left[..., rotated:], packed[..., rotated:], right[..., rotated:]], axis=-1)
            return [rope(mx.fast.rms_norm(q.reshape(1, count, heads, dim), qw, config['rms_norm_eps']).transpose(0, 2, 1, 3)),
                rope(mx.fast.rms_norm(k.reshape(1, count, kvheads, dim), kw, config['rms_norm_eps']).transpose(0, 2, 1, 3)),
                mx.fast.rms_norm(v.reshape(1, count, kvheads, dim), None, config['rms_norm_eps']).transpose(0, 2, 1, 3)]

        def fused():
            return kernel(inputs=[q, k, v, qw, kw, eps, offset, frequencies, logbase],
                template=[('QH', heads), ('KH', kvheads), ('T', count), ('D', dim), ('RotHalf', rotated), ('Proportional', full)],
                grid=(dim // 4, (heads + 2 * kvheads) * count, 1), threadgroup=(dim // 4, 1, 1),
                output_shapes=[(1, heads, count, dim), (1, kvheads, count, dim), (1, kvheads, count, dim)],
                output_dtypes=[mx.bfloat16] * 3)

        functions = {'original': lambda: reference(False), 'direct-rope': lambda: reference(True), 'fused': fused}
        until = time.perf_counter() + 2
        while time.perf_counter() < until:
            for function in functions.values():
                mx.eval(function())
        times = {name: [] for name in functions}
        outputs = {}
        for iteration in range(args.repetitions):
            order = list(functions)
            shift = iteration % len(order)
            order = order[shift:] + order[:shift]
            if iteration % 2:
                order.reverse()
            for name in order:
                start = time.perf_counter()
                values = functions[name]()
                mx.eval(values)
                times[name].append(time.perf_counter() - start)
                outputs[name] = [np.array(value.astype(mx.float32)).astype(np.float64) for value in values]
        variants = {}
        for name, samples in times.items():
            parity = {}
            for kind, expected, value in zip(['queries', 'keys', 'values'], outputs['original'], outputs[name]):
                parity[kind] = dict(bitIdentical=bool(np.array_equal(expected, value)),
                    maxAbsoluteError=float(np.max(np.abs(expected - value))),
                    relativeRMSE=float(np.linalg.norm(expected - value) / np.linalg.norm(expected)))
            variants[name] = dict(medianSeconds=statistics.median(samples), seconds=samples, parity=parity)
        report = dict(layer=layer,dim=dim,offset=offset_value,variants=variants,
            medianPairedReduction=statistics.median(1-b/a for a,b in zip(times['direct-rope'], times['fused'])))
        reports.append(report)
        print(json.dumps(report), flush=True)
    args.output.write_text(json.dumps(dict(shaderSHA256=hashlib.sha256(shader.read_bytes()).hexdigest(),
        scope='Captured raw projections; norms, RoPE and layout included. Isolated operators, not resident inference.',
        reports=reports), indent=2) + '\n')


if __name__ == '__main__':
    main()

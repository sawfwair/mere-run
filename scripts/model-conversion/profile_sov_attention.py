#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx==0.32.4", "numpy==2.3.5"]
# ///
"""Screen attention dispatch and partial RoPE on native SOV activation captures."""
import argparse
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx
import numpy as np


def benchmark(functions, repetitions):
    times = {name: [] for name in functions}
    outputs = {}
    until = time.perf_counter() + 2
    while time.perf_counter() < until:
        for function in functions.values():
            mx.eval(function())
    for iteration in range(repetitions):
        order = list(functions) if iteration % 2 == 0 else list(reversed(functions))
        for name in order:
            started = time.perf_counter()
            value = functions[name]()
            mx.eval(value)
            times[name].append(time.perf_counter() - started)
            outputs[name] = np.array(value.astype(mx.float32)).astype(np.float64)
    reference = outputs[next(iter(functions))]
    return {name: dict(medianSeconds=statistics.median(samples), seconds=samples,
        bitIdentical=bool(np.array_equal(outputs[name], reference)),
        maxAbsoluteError=float(np.max(np.abs(outputs[name] - reference))),
        relativeRMSE=float(np.linalg.norm(outputs[name] - reference) / np.linalg.norm(reference)))
        for name, samples in times.items()}


def screen_sdpa(arrays, repetitions):
    q, k, v, mask = (arrays[key] for key in ['queries', 'keys', 'values', 'mask'])
    mx.eval(q, k, v, mask)
    functions = {
        'reference': lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1, mask=mask),
        'bool-mask': lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1, mask=mask == 0),
        'causal': lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1, mask='causal')}
    # Native captures are the first verification block, before window eviction.
    # Chunking screens vector dispatch and intentionally changes accumulation.
    for count in [1, 2, 4, 8]:
        if count * q.shape[1] // k.shape[1] > 32:
            continue
        functions[f'chunk-{count}'] = lambda count=count: mx.concatenate([
            mx.fast.scaled_dot_product_attention(q[:, :, i:i+count], k, v, scale=1,
                mask=mask[:, :, i:i+count]) for i in range(0, q.shape[2], count)], axis=2)
    return dict(queryShape=list(q.shape), keyShape=list(k.shape), variants=benchmark(functions, repetitions))


def screen_rope(arrays, base, partial, repetitions):
    dim = arrays['q_pre_rope'].shape[-1]
    half, rotated = dim // 2, int(partial * (dim // 2))
    frequencies = mx.power(mx.array(base, dtype=mx.float32), mx.arange(0, rotated * 2, 2, dtype=mx.float32) / dim)
    full_frequencies = mx.concatenate([frequencies, mx.full((half - rotated,), float('inf'))])
    mx.eval(frequencies, full_frequencies)

    def reference(value, offset):
        left, right = value[..., :half], value[..., half:]
        packed = mx.concatenate([left[..., :rotated], right[..., :rotated]], axis=-1)
        packed = mx.fast.rope(packed, dims=rotated * 2, traditional=False, base=None,
            scale=1, offset=offset, freqs=frequencies)
        return mx.concatenate([packed[..., :rotated], left[..., rotated:], packed[..., rotated:], right[..., rotated:]], axis=-1)

    rows = []
    for offset in [0, 779, 1024, 65537]:
        for kind in ['q_pre_rope', 'k_pre_rope']:
            value = arrays[kind]
            functions = {'reference': lambda: reference(value, offset),
                'direct': lambda: mx.fast.rope(value, dims=dim, traditional=False, base=None,
                    scale=1, offset=offset, freqs=full_frequencies)}
            rows.append(dict(kind=kind, offset=offset, variants=benchmark(functions, repetitions)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--activations', type=Path, required=True, help='Native profile prefix, before -attention-N.safetensors.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=35)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / 'config.json').read_text())['text_config']
    rows = []
    for index in [0, 5]:
        arrays = mx.load(str(args.activations) + f'-attention-{index}.safetensors')
        row = dict(layer=index, sdpa=screen_sdpa(arrays, args.repetitions))
        if config['layer_types'][index] == 'full_attention':
            params = config['rope_parameters']['full_attention']
            row['rope'] = screen_rope(arrays, params['rope_theta'], params['partial_rotary_factor'], args.repetitions)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(scope='Isolated captured operators; not resident transcription latency.', reports=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()

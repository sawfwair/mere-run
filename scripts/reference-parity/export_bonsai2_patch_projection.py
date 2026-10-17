#!/usr/bin/env python3
"""Export bounded upstream Conv3d results from real vision projection weights."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if importlib.metadata.version('mlx') != '0.32.0':
        raise ValueError('Use the pinned upstream MLX environment')
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    from safetensors import safe_open
    destination = root / 'cpu-patch-projection'
    destination.mkdir(exist_ok=True)
    reports = []
    for size in ['2', '4', '9']:
        output = destination / (size + '.safetensors')
        if output.exists():
            raise FileExistsError('Preserve completed patch-projection evidence')
        model = root.parent / 'models' / ('text-chat-bonsai-2-' + size + 'b-2bit')
        reference = root / ('reference-' + size) / 'vision-chart.safetensors'
        with safe_open(str(model / 'model.safetensors'), framework='numpy') as source:
            weight = mx.array(source.get_tensor('vision_tower.patch_embed.proj.weight'))
            bias = mx.array(source.get_tensor('vision_tower.patch_embed.proj.bias'))
        patches = mx.load(str(reference))['patches'][:128].astype(mx.float16)
        conv = nn.Conv3d(3, weight.shape[0], kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)
        conv.weight, conv.bias = weight, bias
        expected = conv(patches.reshape(-1, 3, 2, 16, 16).moveaxis(1, 4)).reshape(-1, weight.shape[0])
        mx.eval(expected)
        mx.save_safetensors(str(output), {'weight': weight, 'bias': bias, 'patches': patches, 'expected': expected})
        reports.append({'size': size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                        'reference_sha256': hashlib.file_digest(reference.open('rb'), 'sha256').hexdigest()})
    report = {'reference': 'MLX 0.32.0 nn.Conv3d on CPU with original checkpoint weights',
              'input': 'First 128 frozen chart patches', 'rows': 128, 'patch_shape': [3, 2, 16, 16], 'models': reports}
    (destination / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

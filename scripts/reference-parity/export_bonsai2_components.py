#!/usr/bin/env python3
"""Export bounded real-checkpoint packed rows through upstream Packed on CPU.

This checks embedding and tied/untied head arithmetic without loading a model.
The selected rows and all outputs stay in the external parity evidence root.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if importlib.metadata.version('mlx') != '0.32.0':
        raise ValueError('Use the pinned upstream MLX environment')
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import numpy as np
    from safetensors import safe_open
    reports = []
    destination = root / 'cpu-components'
    destination.mkdir(exist_ok=True)
    for size in ['2', '4', '9']:
        runtime = root.parent / 'metadata' / size / 'runtime'
        metadata = json.loads((runtime.parent / 'metadata.json').read_text())
        for path in runtime.glob('*.py'):
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            expected = next(row['blobId'] for row in metadata['siblings'] if row['rfilename'] == 'runtime/' + path.name)
            if blob != expected:
                raise ValueError('Upstream source changed: ' + str(path))
        # Each size loads its own verified module, rather than a cached sibling.
        sys.path.insert(0, str(runtime))
        for name in ['runtime', 'codec']:
            sys.modules.pop(name, None)
        from runtime import Packed
        model = root.parent / 'models' / ('text-chat-bonsai-2-' + size + 'b-2bit')
        config = json.loads((model / 'config.json').read_text())
        rows = [198, 785, 1000, 10000]
        arrays = {}
        with safe_open(str(model / 'model.safetensors'), framework='numpy') as source:
            for name, module_path in [('embedding', 'model.embed_tokens'), ('head', 'model.embed_tokens' if config['tie_word_embeddings'] else 'lm_head')]:
                record = next(row for row in config['modules'] if row['path'] == module_path)
                prefix = 'language_model.' + module_path
                packed = []
                for field in ['weight', 'scales', 'biases']:
                    slices = source.get_slice(prefix + '.' + field)
                    value = mx.array(np.concatenate([slices[row:row+1] for row in rows]))
                    arrays[name + '_' + field] = value
                    packed.append(value)
                signs = mx.array(source.get_tensor(prefix + '.signs'))
                arrays[name + '_signs'] = signs
                module = Packed(packed, record['block'], signs, record['embedding'], mx.float16)
                if name == 'embedding':
                    ids = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
                    arrays['expected_embedding'] = module(ids)
                else:
                    width = signs.shape[0]
                    hidden = mx.sin(mx.arange(3 * width).astype(mx.float32) * .013).reshape(1, 3, width).astype(mx.float16)
                    arrays['hidden'] = hidden
                    arrays['expected_head'] = module.as_linear(hidden) if record['embedding'] else module(hidden)
        mx.eval(*arrays.values())
        output = destination / (size + '.safetensors')
        if output.exists():
            raise FileExistsError('Preserve completed component evidence')
        mx.save_safetensors(str(output), arrays)
        reports.append({'size': size, 'revision': metadata['sha'], 'rows': rows,
                        'blocks': {name: next(row['block'] for row in config['modules'] if row['path'] == path)
                                   for name, path in [('embedding', 'model.embed_tokens'), ('head', 'model.embed_tokens' if config['tie_word_embeddings'] else 'lm_head')]},
                        'sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
        print(size, 'CPU packed component reference exported', flush=True)
        sys.path.pop(0)
    (destination / 'reference.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()

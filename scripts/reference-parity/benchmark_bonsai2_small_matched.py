#!/usr/bin/env python3
"""Time fixed-continuation checkpoint forwards under the native test's admission lease."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--size', choices=['2', '4', '9'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mlx-version', choices=['0.32.0', '0.32.1'],
                        default=os.environ.get('MERERUN_BONSAI_MATCHED_REFERENCE_MLX_VERSION', '0.32.1'))
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve the previous cohort before collecting another.'
    root = args.root.resolve(strict=True)
    runtime = root / 'metadata' / args.size / 'runtime'
    metadata = json.loads((runtime.parent / 'metadata.json').read_text())
    blobs = {entry['rfilename']: entry['blobId'] for entry in metadata['siblings']}
    for path in runtime.glob('*.py'):
        data = path.read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blobs['runtime/' + path.name]
    assert importlib.metadata.version('mlx') == args.mlx_version
    sys.path.insert(0, str(runtime))
    from vision_artifact import load_vl_model
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    mx.set_default_device(mx.gpu)
    model, processor, _ = load_vl_model(root / 'models' / f'text-chat-bonsai-2-{args.size}b-2bit')
    lm = model.language_model
    inputs = json.loads((root / f'matched-inputs-{args.size}.json').read_text())
    records = []
    for probe in inputs['probes']:
        images = [Image.open(root / 'fixtures' / probe['image']).convert('RGB')] if probe.get('image') else None
        content = [{'type': 'image'}, {'type': 'text', 'text': probe['prompt']}] if images else probe['prompt']
        prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=True)
        encoded = processor(text=[prompt], images=images, return_tensors='np', padding=False)
        ids = np.asarray(encoded['input_ids'])
        assert ids.reshape(-1).tolist() == probe['prompt_ids']
        for repetition in range(4):
            tokens = probe['continuation'][:128] if repetition == 0 else probe['continuation']
            lm._position_ids = None
            lm._rope_deltas = None
            cache = lm.make_cache()
            input_ids = mx.array(ids, dtype=mx.int32)
            kwargs = {}
            pixels = None
            if images:
                pixels = mx.array(np.asarray(encoded['pixel_values']))
                kwargs['image_grid_thw'] = mx.array(np.asarray(encoded['image_grid_thw']), dtype=mx.int32)
                frozen = mx.load(str(root / probe['image_features']))
                assert frozen['grid'].tolist() == kwargs['image_grid_thw'].tolist()
                kwargs['cached_image_features'] = frozen['image_embeddings']
            features = model.get_input_embeddings(input_ids, pixel_values=pixels, **kwargs)
            for start in range(0, ids.size, 512):
                end = min(start + 512, ids.size)
                positions = {'position_ids': lm._position_ids[:, :, start:end]} if images else {}
                result = lm(input_ids[:, start:end], inputs_embeds=features.inputs_embeds[:, start:end],
                            cache=cache, **positions)
                mx.eval(result.logits)
            mx.synchronize()
            greedy = []
            started = time.perf_counter_ns()
            for token in tokens:
                result = lm(mx.array([[token]], dtype=mx.int32), cache=cache)
                greedy.append(int(mx.argmax(result.logits[0, -1].astype(mx.float32)).item()))
            mx.synchronize()
            seconds = (time.perf_counter_ns() - started) / 1e9
            records.append({'modality': probe['id'], 'repetition': repetition, 'tokenCount': len(tokens),
                            'promptIDs': ids.reshape(-1).tolist(), 'continuation': tokens,
                            'argmax': greedy, 'seconds': seconds})
            print('BONSAI_MATCHED', probe['id'], 'repetition=' + str(repetition),
                  'tokens=' + str(len(tokens)), 'seconds=' + str(seconds), flush=True)
    report = {'stack': 'reference', 'size': args.size, 'records': records,
              'versions': {name: importlib.metadata.version(name) for name in
                           ['mlx', 'mlx-lm', 'mlx-vlm', 'transformers', 'numpy', 'pillow']}}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    assert len(records) == 8 and all(r['seconds'] > 0 for r in records)


if __name__ == '__main__':
    main()

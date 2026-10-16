#!/usr/bin/env python3
"""Independent pinned Bonsai reference export; never used for product inference.

Run under the opt-in Swift test's normal machine-admission reservation. The
reference pack loader and neural model are unchanged. Greedy reference tokens
become a common teacher-forced continuation for the native comparison.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--size', choices=['2', '4', '9'], required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    runtime = root.parent / 'metadata' / args.size / 'runtime'
    receipt = json.loads((model_path / 'SOURCE-RECEIPT.json').read_text())
    if not receipt['verified']:
        raise ValueError('A source-verified checkpoint is required')
    for name, version in [('mlx', '0.32.0'), ('mlx-vlm', '0.6.3'), ('transformers', '5.5.0')]:
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Use the pinned upstream {name}=={version}')
    metadata = json.loads((runtime.parent / 'metadata.json').read_text())
    if metadata['sha'] != receipt['revision'] or metadata['id'] != receipt['repo']:
        raise ValueError('Reference source revision does not match the checkpoint')
    identities = {entry['rfilename']: entry['blobId'] for entry in metadata['siblings']}
    runtime_blobs = {}
    for path in runtime.glob('*.py'):
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != identities['runtime/' + path.name]:
            raise ValueError('Upstream runtime source changed: ' + path.name)
        runtime_blobs[path.name] = blob
    for entry in receipt['files']:
        if digest(model_path / entry['file']) != entry['sha256']:
            raise ValueError('Checkpoint source changed: ' + entry['file'])
    exporter_sha256 = digest(Path(__file__))
    sys.path.insert(0, str(runtime))
    from vision_artifact import load_vl_model
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    destination = root / ('reference-' + args.size)
    destination.mkdir(exist_ok=True)
    if (destination / 'results.json').exists():
        raise FileExistsError('Preserve the completed reference export')
    started = time.monotonic()
    model, processor, config = load_vl_model(model_path)
    plan = json.loads((root / 'cases.json').read_text())
    results = []
    for case in plan['cases']:
        print('START', case['id'], flush=True)
        content = case['prompt']
        images = None
        if 'image' in case:
            content = [{'type': 'image'}, {'type': 'text', 'text': content}]
            images = [Image.open(root.parent / 'fixtures' / case['image']).convert('RGB')]
        messages = [{'role': 'system', 'content': plan['system']}, {'role': 'user', 'content': content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=case['thinking'])
        encoded = processor(text=[prompt], images=images, return_tensors='np', padding=False)
        ids = mx.array(np.asarray(encoded['input_ids']), dtype=mx.int32)
        pixels = mx.array(np.asarray(encoded['pixel_values'])) if images else None
        grid = mx.array(np.asarray(encoded['image_grid_thw']), dtype=mx.int32) if images else None
        lm = model.language_model
        lm._position_ids = None
        lm._rope_deltas = None
        kwargs = {'image_grid_thw': grid} if images else {}
        features = model.get_input_embeddings(ids, pixel_values=pixels, **kwargs)
        embeds = features.inputs_embeds
        arrays = {'input_embeddings': embeds, 'prompt_ids': ids}
        if images:
            arrays['patches'] = pixels
            arrays['grid'] = grid
            arrays['positions'] = lm._position_ids
            positions = np.flatnonzero(np.asarray(ids).reshape(-1) == model.config.image_token_index)
            arrays['image_embeddings'] = mx.take(embeds[0], mx.array(positions, dtype=mx.int32), axis=0)
        rope_delta = int(np.asarray(lm._rope_deltas).reshape(-1)[0]) if lm._rope_deltas is not None else 0
        cache = lm.make_cache()
        out = None
        for start in range(0, ids.shape[1], 512):
            end = min(start + 512, ids.shape[1])
            # Supply the model's own full-prompt mRoPE slice. Automatic cached
            # decode positions are linear and cannot represent a later image
            # chunk that still contains spatial tokens.
            positions = {'position_ids': arrays['positions'][:, :, start:end]} if images else {}
            out = lm(ids[:, start:end], inputs_embeds=embeds[:, start:end], cache=cache, **positions)
            mx.eval(out.logits)
        logits = []
        tokens = []
        for step in range(case['steps']):
            row = out.logits[0, -1].astype(mx.float32)
            mx.eval(row)
            logits.append(row)
            token = int(mx.argmax(row).item())
            tokens.append(token)
            if token in model.config.eos_token_id:
                break
            if step + 1 < case['steps']:
                out = lm(mx.array([[token]], dtype=mx.int32), cache=cache)
                mx.eval(out.logits)
        arrays['logits'] = mx.stack(logits)
        mx.eval(*arrays.values())
        output = destination / (case['id'] + '.safetensors')
        mx.save_safetensors(str(output), arrays)
        results.append({'id': case['id'], 'prompt_ids': np.asarray(ids).reshape(-1).tolist(),
                        'continuation': tokens, 'text': processor.tokenizer.decode(tokens),
                        'grid': np.asarray(grid).reshape(-1).tolist() if images else [],
                        'rope_delta': rope_delta, 'arrays': output.name, 'arrays_sha256': digest(output),
                        'rendered_prompt': prompt})
        print('END', case['id'], len(tokens), flush=True)
        mx.clear_cache()
    report = {'cases': results, 'source': receipt, 'runtime_sha256': {p.name: digest(p) for p in runtime.glob('*.py')},
              'runtime_upstream_git_blobs': runtime_blobs, 'exporter_sha256': exporter_sha256,
              'versions': {n: importlib.metadata.version(n) for n in ['mlx', 'mlx-vlm', 'mlx-lm', 'transformers', 'pillow', 'numpy']},
              'seconds': time.monotonic() - started, 'peak_memory_bytes': mx.get_peak_memory()}
    (destination / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

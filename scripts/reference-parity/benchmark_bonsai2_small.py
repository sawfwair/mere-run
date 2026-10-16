#!/usr/bin/env python3
"""Measure the unmodified pinned Bonsai pack under the opt-in test's admission lease."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def machine():
    rows = subprocess.check_output(['ps', '-axo', 'pid,pcpu,rss,ucomm'], text=True).splitlines()[1:]
    processes = []
    for line in rows:
        fields = line.strip().split(None, 3)
        if len(fields) == 4 and fields[3] in {'mere.run', 'xctest', 'swift-frontend', 'clang'}:
            processes.append({'pid': int(fields[0]), 'cpu_percent': float(fields[1]),
                              'rss_kib': int(fields[2]), 'executable': fields[3]})
    return {'time': time.time(), 'load_average': os.getloadavg(), 'processes': processes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--size', choices=['2', '4', '9'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--admission-state', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    model_path = root / 'models' / f'text-chat-bonsai-2-{args.size}b-2bit'
    runtime = root / 'metadata' / args.size / 'runtime'
    receipt = json.loads((model_path / 'SOURCE-RECEIPT.json').read_text())
    metadata = json.loads((runtime.parent / 'metadata.json').read_text())
    assert receipt['verified'] and metadata['sha'] == receipt['revision']
    assert metadata['id'] == receipt['repo']
    blobs = {entry['rfilename']: entry['blobId'] for entry in metadata['siblings']}
    for path in runtime.glob('*.py'):
        data = path.read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blobs['runtime/' + path.name]
    for entry in receipt['files']:
        assert digest(model_path / entry['file']) == entry['sha256'], entry['file']
    versions = {name: importlib.metadata.version(name)
                for name in ['mlx', 'mlx-metal', 'mlx-lm', 'mlx-vlm', 'transformers', 'pillow', 'numpy']}
    assert versions['mlx'] == '0.32.0' and versions['mlx-vlm'] == '0.6.3'
    assert versions['transformers'] == '5.5.0'
    sys.path.insert(0, str(runtime))
    from vision_artifact import load_vl_model
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_vlm import stream_generate

    args.output.mkdir()
    quiet = 0
    for _ in range(120):
        observed = machine()
        admission = json.loads(args.admission_state.read_text())
        queued = {ticket['processID'] for ticket in admission['tickets'] if ticket['state'] == 'queued'}
        others = [process for process in observed['processes']
                  if process['pid'] != os.getppid() and process['pid'] not in queued]
        with (args.output / 'quiet-start.jsonl').open('a') as output:
            output.write(json.dumps({'time': observed['time'], 'others': others, 'tickets': admission['tickets']}) + '\n')
        quiet = quiet + 1 if not others else 0
        if quiet == 3:
            break
        time.sleep(5)
    else:
        raise RuntimeError('No quiet start for upstream throughput')
    started = time.monotonic()
    model, processor, config = load_vl_model(model_path)
    load_seconds = time.monotonic() - started
    records = []
    baseline = json.loads((root / 'baseline-requests.json').read_text())
    inputs = []
    for modality in ['text', 'vision']:
        body = {**baseline[modality], 'model': f'text-chat-bonsai-2-{args.size}b-2bit'}
        content = body['messages'][0]['content']
        images = None
        if modality == 'vision':
            prompt_text = next(item['text'] for item in content if item['type'] == 'text')
            content = [{'type': 'image'}, {'type': 'text', 'text': prompt_text}]
            images = [Image.open(root / 'fixtures/shapes.png').convert('RGB')]
        else:
            prompt_text = content
        prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True)
        encoded = processor(text=[prompt], images=images, return_tensors='np', padding=False)
        ids = np.asarray(encoded['input_ids'])
        grid = np.asarray(encoded['image_grid_thw']) if images else None
        image_tokens = int(np.prod(grid[0]) // (config['vision_config']['spatial_merge_size'] ** 2)) if images else 0
        inputs.append({'modality': modality, 'prompt': prompt_text, 'rendered_prompt': prompt,
                       'prompt_ids': ids.reshape(-1).tolist(), 'image_tokens': image_tokens,
                       'grid': grid.reshape(-1).tolist() if images else [],
                       'patch_sha256': hashlib.sha256(np.asarray(encoded['pixel_values']).tobytes()).hexdigest() if images else None})
        for repetition in range(4):
            budget = min(128, body['max_tokens']) if repetition == 0 else body['max_tokens']
            label = f"{args.size}b-{modality}-" + ('warmup' if repetition == 0 else f'measured-{repetition}')
            model.language_model._position_ids = None
            model.language_model._rope_deltas = None
            kwargs = {'input_ids': mx.array(ids, dtype=mx.int32),
                      'pixel_values': mx.array(np.asarray(encoded['pixel_values'])) if images else None,
                      'mask': mx.array(np.asarray(encoded['attention_mask'])) if 'attention_mask' in encoded else None}
            if images:
                kwargs['image_grid_thw'] = mx.array(grid, dtype=mx.int32)
            before = machine()
            mx.synchronize()
            start = time.monotonic()
            tokens, segments, last_count = [], [], 0
            print('START', label, flush=True)
            final = None
            for result in stream_generate(model, processor, prompt, max_tokens=budget,
                                          temperature=body['temperature'], top_p=body['top_p'],
                                          top_k=body['top_k'], min_p=body['min_p'],
                                          prefill_step_size=512, max_kv_size=8192,
                                          enable_thinking=True, **kwargs):
                segments.append(result.text)
                if result.generation_tokens > last_count:
                    tokens.append(int(np.asarray(result.token).reshape(-1)[0]))
                    last_count = result.generation_tokens
                final = result
            mx.synchronize()
            wall = time.monotonic() - start
            assert final is not None and len(tokens) == final.generation_tokens
            assert final.generation_tps > 0 and final.prompt_tokens == ids.size
            record = {'label': label, 'measurement': 'discarded warm-up' if repetition == 0 else 'resident request after warm-up',
                      'generated_tokens': final.generation_tokens, 'prompt_tokens': final.prompt_tokens,
                      'decode_tokens_per_second': final.generation_tps,
                      'decode_seconds': final.generation_tokens / final.generation_tps,
                      'prefill_seconds': final.prompt_tokens / final.prompt_tps,
                      'wall_seconds': wall, 'finish_reason': final.finish_reason,
                      'peak_memory_bytes': int(final.peak_memory * 1e9), 'tokens': tokens,
                      'response': ''.join(segments), 'before': before, 'after': machine(),
                      'request_body_sha256': hashlib.sha256(json.dumps({**body, 'max_tokens': budget}, sort_keys=True).encode()).hexdigest()}
            records.append(record)
            report = {'size': args.size, 'source': receipt, 'versions': versions,
                      'runtime_sha256': {p.name: digest(p) for p in runtime.glob('*.py')},
                      'script_sha256': digest(Path(__file__)), 'load_seconds': load_seconds,
                      'process_id': os.getpid(), 'admission_parent_pid': os.getppid(), 'inputs': inputs,
                      'records': records, 'metric': 'Unmodified mlx-vlm stream_generate final generation_tps; excludes prefill and includes token sampling/detokenization. Native timings use product decode counters. Timers are not instruction-for-instruction equivalents.'}
            (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            print('END', label, round(final.generation_tps, 2), flush=True)
    assert len(records) == 8


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Refresh pinned Turbo metadata and tensor schemas without downloading weights."""
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

REPO = 'Qwen/Qwen-Image-2.1-Turbo'
REVISION = 'd65dbc9a7e8f6b5479e33dee6030eaab2a906509'
BASE = f'https://huggingface.co/{REPO}/resolve/{REVISION}/'
ROOT = Path(__file__).resolve().parents[2] / 'Tests/MereRunCoreTests/Fixtures/QwenImage21Turbo'
CONFIGS = {
    'LICENSE': 'LICENSE',
    'model_index.json': 'model_index.json',
    'scheduler/scheduler_config.json': 'scheduler.json',
    'processor/processor_config.json': 'processor.json',
    'text_encoder/config.json': 'text-encoder.json',
    'transformer/config.json': 'transformer.json',
    'vae/config.json': 'vae.json',
}
WEIGHTS = [
    'text_encoder/model.safetensors',
    'transformer/diffusion_pytorch_model-00001-of-00002.safetensors',
    'transformer/diffusion_pytorch_model-00002-of-00002.safetensors',
    'vae/diffusion_pytorch_model.safetensors',
]


def fetch(path, end=None):
    url = BASE + path
    headers = {}
    if end is not None:
        url += f'?header_end={end}'
        headers['Range'] = f'bytes=0-{end}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        if end is not None and response.status != 206:
            raise RuntimeError('Server ignored Range; refusing a checkpoint download')
        return response.read(end + 1 if end is not None else 1_000_000)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source, name in CONFIGS.items():
        data = fetch(source)
        hashes[source] = hashlib.sha256(data).hexdigest()
        (ROOT / name).write_bytes(data)
    (ROOT / 'Notice').write_text(
        'Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 '
        'Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.\n')
    shapes = {}
    for path in WEIGHTS:
        size = struct.unpack('<Q', fetch(path, 7))[0]
        if size > 2_000_000:
            raise RuntimeError('Unexpected safetensors header size')
        data = fetch(path, size + 7)[8:]
        hashes[path + ':header'] = hashlib.sha256(data).hexdigest()
        header = json.loads(data)
        component = path.split('/')[0]
        tensors = {key: value for key, value in header.items() if key != '__metadata__'}
        if any(value['dtype'] != 'BF16' for value in tensors.values()):
            raise RuntimeError(f'Unexpected non-BF16 tensors in {path}')
        shapes.setdefault(component, {}).update({key: value['shape'] for key, value in tensors.items()})
    (ROOT / 'weight-shapes.json').write_text(json.dumps(shapes, indent=2, sort_keys=True) + '\n')
    evidence = {'repo': REPO, 'revision': REVISION, 'sha256': hashes,
                'scope': 'Configurations and safetensors headers only; no trained-weight inference.'}
    (ROOT / 'source.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    print('Saved pinned configs and BF16 tensor schemas; no weight payloads downloaded.')


if __name__ == '__main__':
    main()

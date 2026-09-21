#!/usr/bin/env python3
"""Refresh pinned configs and tensor-shape fixtures using HTTP range reads only."""
import json
from pathlib import Path
import struct
import urllib.request

REVISION = 'b3179ad355be050328e483a9dfdd9e60cd62adfa'
BASE = f'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/{REVISION}/'
ROOT = Path(__file__).resolve().parents[2] / 'Tests/ImageRuntimeTests/Fixtures/QwenImage21'
FILES = [
    'transformer/diffusion_pytorch_model-00001-of-00002.safetensors',
    'transformer/diffusion_pytorch_model-00002-of-00002.safetensors',
    'vae/diffusion_pytorch_model.safetensors',
]


def read_prefix(path, end):
    request = urllib.request.Request(BASE + path + f'?header_end={end}', headers={'Range': f'bytes=0-{end}'})
    with urllib.request.urlopen(request) as response:
        if response.status != 206:
            raise RuntimeError('Server did not honor the range request; refusing a checkpoint download')
        return response.read(end + 1)


shapes = {'transformer': {}, 'vae': {}}
for path in FILES:
    size = struct.unpack('<Q', read_prefix(path, 7))[0]
    if size > 2_000_000:
        raise RuntimeError('Unexpected safetensors header size')
    header = json.loads(read_prefix(path, 7 + size)[8:])
    component = path.split('/')[0]
    shapes[component].update({key: value['shape'] for key, value in header.items() if key != '__metadata__'})
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'official-weight-shapes.json').write_text(json.dumps(shapes, indent=2) + '\n')
for component in shapes:
    with urllib.request.urlopen(BASE + component + '/config.json') as response:
        (ROOT / f'official-{component}-config.json').write_bytes(response.read())
print('Updated pinned configs and tensor schemas; no weight data downloaded.')

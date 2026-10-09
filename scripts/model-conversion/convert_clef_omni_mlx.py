#!/usr/bin/env python3
"""Convert pinned Clef Omni into native MLX affine-Q4 experts, streaming by shard.

Run on a CUDA MLX host. Attention, routers, embeddings, media towers and joint
head retain source precision. Talker/code2wav weights are omitted. No remote
checkpoint Python code is executed. Qualification is separate from conversion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import time

REPOSITORY = 'Cloudflare/clef-omni'
REVISION = '0db1cd2607d76a7bdb2a382f659e7b313079f84b'
QUANTIZATION = dict(bits=4, group_size=64, mode='affine', scope='thinker_moe_experts')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def quantized(key):
    return key.startswith('thinker.model.layers.') and '.mlp.experts.' in key and key.endswith('.weight')


def convert(source: Path, output: Path):
    import mlx.core as mx
    import torch
    from safetensors import safe_open
    from huggingface_hub import HfApi, snapshot_download

    if output.exists(): raise FileExistsError('Use a fresh output directory.')
    started = time.monotonic()
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=source,
        allow_patterns=['config.json', 'processor_config.json', 'tokenizer*.json', 'joint_head*',
                        'model*.safetensors*', 'LICENSE', 'README.md'])
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    config = json.loads((source / 'config.json').read_text())
    assert config['model_type'] == 'qwen3_omni_moe'
    assert config['thinker_config']['text_config']['num_hidden_layers'] == 48
    assert config['thinker_config']['text_config']['num_experts'] == 128
    selected = {k:v for k,v in index['weight_map'].items() if k.startswith('thinker.')}
    output.mkdir(parents=True)
    weights, files, source_files, metrics = {}, {}, {}, []
    logical_bytes = 0
    for filename in sorted(set(selected.values())):
        path = source / filename
        sha = digest(path)
        metadata = remote[filename]
        if metadata.lfs is None or sha != metadata.lfs.sha256 or path.stat().st_size != metadata.size:
            raise ValueError('Pinned source hash/size mismatch: '+filename)
        source_files[filename] = dict(sha256=sha, bytes=path.stat().st_size)
        arrays = {}
        with safe_open(path, framework='pt', device='cpu') as reader:
            for key in sorted(k for k,v in selected.items() if v == filename):
                tensor = reader.get_tensor(key)
                if tensor.dtype != torch.bfloat16: raise ValueError('Unexpected source dtype: '+key)
                value = mx.array(tensor.view(torch.uint16).numpy()).view(mx.bfloat16)
                if quantized(key):
                    if value.shape[-1] % 64: raise ValueError('Unsupported expert input width: '+key)
                    packed, scales, biases = mx.quantize(value, group_size=64, bits=4)
                    mx.eval(packed, scales, biases)
                    arrays[key] = packed
                    arrays[key.removesuffix('.weight')+'.scales'] = scales
                    arrays[key.removesuffix('.weight')+'.biases'] = biases
                    # Bounded weight reconstruction diagnostics; these are not model quality scores.
                    if len(metrics) < 24:
                        reconstructed = mx.dequantize(packed, scales, biases, group_size=64, bits=4)
                        diff = value.astype(mx.float32) - reconstructed.astype(mx.float32)
                        metrics.append(dict(tensor=key, relative_mse=float((mx.square(diff).mean() / mx.maximum(mx.square(value.astype(mx.float32)).mean(), 1e-20)).item())))
                else: arrays[key] = value
        mx.eval(arrays)
        destination = output / filename
        mx.save_safetensors(str(destination), arrays)
        logical_bytes += sum(v.nbytes for v in arrays.values())
        weights.update({k:filename for k in arrays})
        files[filename] = dict(bytes=destination.stat().st_size, sha256=digest(destination))
        print(json.dumps(dict(shard=filename, logical_bytes=logical_bytes, tensors=len(weights))), flush=True)
        del arrays
        mx.clear_cache()
    for name in ['processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'joint_head_config.json', 'joint_head.safetensors', 'LICENSE']:
        if not (source / name).exists(): raise FileNotFoundError(name)
        shutil.copyfile(source / name, output / name)
        files[name] = dict(bytes=(output/name).stat().st_size, sha256=digest(output/name))
    config['quantization'] = QUANTIZATION
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'model.safetensors.index.json').write_text(json.dumps(dict(metadata=dict(total_size=logical_bytes), weight_map=weights),indent=2)+'\n')
    shutil.copyfile(source/'README.md', output/'UPSTREAM_README.md')
    for name in ['config.json', 'model.safetensors.index.json', 'UPSTREAM_README.md']:
        files[name] = dict(bytes=(output/name).stat().st_size, sha256=digest(output/name))
    receipt = dict(converter_sha256=digest(Path(__file__)), source_repository=REPOSITORY, source_revision=REVISION, quantization=QUANTIZATION,
                   logical_thinker_bytes=logical_bytes, bundle_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
                   quality_qualified=False, local_36gb_qualified=False, source_files=source_files, files=files,
                   reconstruction_diagnostics=metrics, mlx_version=version('mlx'),
                   elapsed_seconds=time.monotonic()-started)
    (output/'conversion-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['logical_thinker_bytes','bundle_bytes','elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.source,args.output)

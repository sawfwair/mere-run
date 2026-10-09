#!/usr/bin/env python3
"""Reproducible mixed Q4/Q8 PPLX 9B packing; requires mlx[cuda12]==0.32.2.

Conversion only. Published artifacts execute through native Swift/MLX.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

MODELS = {
    'late': ('perplexity-ai/pplx-embed-v2-late-9b', '77e936a1b18ed2ac00b7c76fccd70dc6a1bb1c18'),
    'context': ('perplexity-ai/pplx-embed-v2-context-9b-preview', 'b667039ee8b438a6350fbc91bbcecd86f9d363ba'),
}
ARTIFACTS = {
    'late': 'Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit',
    'context': 'Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit',
}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def bits_for(name, shape, profile="mixed4"):
    if len(shape) != 2 or shape[-1] % 64:
        return None
    if name == 'language_model.embed_tokens.weight':
        return 8
    if re.fullmatch(r'language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight', name):
        return 8 if profile == "q8" else 4
    return None


def checksums(root):
    files = {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(root.rglob('*')) if p.is_file() and '.cache' not in p.parts and p.name != 'SHA256SUMS'}
    (root / 'SHA256SUMS').write_text(''.join(f"{v['sha256']}  {k}\n" for k, v in files.items()))
    return files


def convert(kind, source, output, profile):
    import mlx.core as mx
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    repository, revision = MODELS[kind]
    artifact = "Sawfwair/" + repository.split("/")[1] + "-MLX-" + ("8bit" if profile == "q8" else "Mixed-4bit")
    projection_bits = 8 if profile == "q8" else 4
    info = HfApi(token=False).model_info(repository, revision=revision, files_metadata=True)
    assert info.sha == revision
    snapshot_download(repository, revision=revision, local_dir=source, token=False,
                      ignore_patterns=['assets/*', '.gitattributes'])
    output.mkdir(parents=True, exist_ok=False)
    remote = {p.rfilename: p for p in info.siblings}
    provenance, errors, modules, mapping = {}, {}, {}, {}
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '.cache' in path.parts:
            continue
        name = path.relative_to(source).as_posix()
        sha, entry = digest(path), remote[name]
        if path.stat().st_size != entry.size or (entry.lfs and entry.lfs.sha256 != sha):
            raise ValueError('Source integrity mismatch: ' + name)
        provenance[name] = {'bytes': entry.size, 'sha256': sha}
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.safetensors' and path.parent == source:
            arrays, converted = mx.load(str(path)), {}
            for key, weight in arrays.items():
                if weight.dtype != mx.float32:
                    raise ValueError('Expected source FP32: ' + key)
                bits = bits_for(key, weight.shape, profile)
                if bits is None:
                    converted[key] = weight
                    continue
                prefix = key.removesuffix('.weight')
                packed, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
                mx.eval(packed, scales, biases)
                reconstructed = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
                rel = float(mx.sqrt(mx.sum((weight - reconstructed)**2) / mx.sum(weight**2)).item())
                errors[key] = {'bits': bits, 'group_size': 64, 'relative_l2': rel,
                               'dequantized_sample': reconstructed[0, :128].tolist()}
                modules[prefix.removeprefix('language_model.')] = {'bits': bits, 'group_size': 64, 'mode': 'affine'}
                converted[key], converted[prefix + '.scales'], converted[prefix + '.biases'] = packed, scales, biases
                del reconstructed
                mx.clear_cache()
            mx.save_safetensors(str(target), converted, metadata={'format': 'mlx', 'source_revision': revision})
            with safe_open(target, framework='numpy') as reader:
                for key in reader.keys():
                    if key in mapping: raise ValueError('Duplicate tensor: ' + key)
                    mapping[key] = name
            print(json.dumps({'converted': name, 'bytes': target.stat().st_size}), flush=True)
            del arrays, converted
            mx.clear_cache()
        elif name == 'README.md':
            shutil.copyfile(path, output / 'UPSTREAM_MODEL_CARD.md')
        elif name != 'model.safetensors.index.json':
            shutil.copyfile(path, target)
    if len(modules) != 201:  # 24*6 linear-attention/MLP + 8*7 full-attention/MLP + embedding.
        raise ValueError(f'Unexpected packed module coverage: {len(modules)}')
    config = json.loads((output / 'config.json').read_text())
    config['quantization'] = {'bits': projection_bits, 'group_size': 64, 'mode': 'affine', 'modules': modules}
    config['_conversion_notice'] = f'Sawfwair MLX Q{projection_bits}/Q8; FP32 gates, norms, vision and output heads. See MODIFICATIONS.md.'
    write_json(output / 'config.json', config)
    write_json(output / 'model.safetensors.index.json', {'metadata': {'total_size': sum((output / p).stat().st_size for p in set(mapping.values()))}, 'weight_map': mapping})
    write_json(output / 'PPLX_CONVERSION.json', {'schema_version': 1, 'kind': kind,
        'source_repository': repository, 'source_revision': revision, 'artifact_repository': artifact, 'profile': profile,
        'converter_sha256': digest(Path(__file__)), 'mlx_version': '0.32.2', 'source_files': provenance,
        'weight_reconstruction': errors, 'quality_qualified': False,
        'notes': ['No training or fine-tuning. Source FP32 vision, gates, convolution, norms and final projection preserved.',
                  'Weight reconstruction alone does not qualify retrieval quality or Apple memory fit.']})
    shutil.copyfile(__file__, output / Path(__file__).name)
    (output / 'MODIFICATIONS.md').write_text(f'# Modifications by Sawfwair\n\nSource: {repository}@{revision}.\n\n'
        f'Token embeddings use MLX affine Q8/group-64; selected transformer projections use Q{projection_bits}/group-64. '
        'Recurrent gates, convolution, norms, vision and final embedding heads retain FP32. '
        'Explicit per-module packing metadata and a complete tensor index were added. No fine-tuning or training. '
        'Original model card and reference code are retained. The upstream model card declares MIT.\n')
    (output / 'README.md').write_text(f'---\nlicense: mit\nbase_model: {repository}\npipeline_tag: feature-extraction\ntags: [mlx, quantized, embeddings]\n---\n\n'
        f'# {artifact.split("/")[1]}\n\n'
        f'Unofficial Q{projection_bits}/Q8 MLX quantization for native Swift/MLX execution in '
        '[mere.run](https://github.com/sawfwair/mere-run). Runtime support is under development; released binaries may not include it.\n\n'
        f'Q{projection_bits}/group-64 transformer projections, Q8/group-64 token embeddings, FP32 recurrent gates, norms, convolution, vision and output projection. '
        'This is weight quantization; the contextual model still returns its original int8-valued vectors.\n\n'
        'See PPLX_QUALIFICATION.json for the bounded diagnostic scope, UPSTREAM_MODEL_CARD.md, '
        'PPLX_CONVERSION.json, MODIFICATIONS.md and SHA256SUMS. Broad retrieval benchmarks and maximum-context memory fit remain unqualified.\n\n'
        f'```sh\nhf download {artifact} --local-dir ./pplx-{kind}-mixed\n'
        f'mere.run text embed --model ./pplx-{kind}-mixed --task query "What is photosynthesis?"\n```\n')
    checksums(output)
    print(json.dumps({'status': 'converted', 'kind': kind, 'packed_modules': len(modules)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=MODELS, required=True)
    parser.add_argument('--profile', choices=['mixed4','q8'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.kind, args.source, args.output, args.profile or ("mixed4" if args.kind == "late" else "q8"))

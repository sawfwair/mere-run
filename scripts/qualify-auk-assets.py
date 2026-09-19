#!/usr/bin/env python3
"""Verify original AuK qualification assets against pinned Hugging Face metadata."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', type=Path, required=True)
p.add_argument('--flash', type=Path, required=True)
p.add_argument('--thinker', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sources = [
    ('tencent/AuK', '790742b71a4430120daf2b2099192abae449eb9f', a.base),
    ('tencent/AuK-Flash', '575b92f0895f75180bf2cbd35f2e176c5732b8ed', a.flash),
    ('Qwen/Qwen2.5-Omni-3B', 'f75b40e3da2003cdd6e1829b1f420ca70797c34e', a.thinker),
]
receipts = []
for repo, revision, root in sources:
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError('Source revision mismatch')
    for entry in info.siblings:
        # The pinned index places all Thinker text/audio tensors in shards 1 and 2.
        if repo == 'Qwen/Qwen2.5-Omni-3B' and entry.rfilename == 'model-00003-of-00003.safetensors':
            continue
        path = root / entry.rfilename
        required = path.suffix in ['.safetensors', '.json', '.yaml'] or path.name == 'LICENSE'
        if not required:
            continue
        size = path.stat().st_size
        digest = hashlib.sha256() if entry.lfs else hashlib.sha1()
        if not entry.lfs:
            digest.update(('blob ' + str(size) + '\0').encode())
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(block)
        expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
        actual = digest.hexdigest()
        if size != entry.size or actual != expected:
            raise RuntimeError('Asset mismatch: ' + str(path))
        receipts.append({'repo': repo, 'revision': revision, 'file': entry.rfilename, 'bytes': size,
                         'algorithm': 'sha256' if entry.lfs else 'git-blob-sha1', 'digest': actual})
        print('Verified', repo, entry.rfilename, flush=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(receipts, indent=2))

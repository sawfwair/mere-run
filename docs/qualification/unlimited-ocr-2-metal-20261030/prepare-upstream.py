"""Fetch pinned reference code and link the already verified checkpoint into a temporary package."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = 'c14071b549f4fcab10909f3c4acc6a2edb366947'
SOURCES = ['_sparse_base_impl.py', 'configuration_deepseek_v2.py', 'conversation.py',
           'dash_attention.py', 'deepencoder.py', 'modeling_deepseekv2.py',
           'modeling_unlimitedocr.py', 'packing_patch.py', 'sparse_gather_e2e_triton.py',
           'sparse_gather_triton.py']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = []
    for name in SOURCES:
        url = f'https://huggingface.co/baidu/Unlimited-OCR-2/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url) as response:
            data = response.read()
        (args.output / name).write_bytes(data)
        receipt.append(dict(file=name, sha256=hashlib.sha256(data).hexdigest()))
    (args.output / '__init__.py').touch()
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                 'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
                 'model-00002-of-00002.safetensors']:
        target = args.output / name
        if not target.exists():
            target.symlink_to((args.checkpoint / name).resolve(strict=True))
    (args.output / 'source-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()

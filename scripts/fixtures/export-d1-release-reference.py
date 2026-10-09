#!/usr/bin/env python3
"""Export trained-checkpoint answers with LiquidAI's original code, for tests only.

Requires torch, transformers, torchvision, pillow, and soundfile. The checkpoint
folders must contain the pinned original Python source, configs, tokenizer, and
weights. Inputs and media live in --output; no files or code are downloaded.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import transformers
from PIL import Image
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, required=True, help='Contains causal/ and omni/ original checkpoints.')
    parser.add_argument('--output', type=Path, required=True, help='Contains text.json, image.json, audio.json and their relative media.')
    parser.add_argument('--family', choices=['causal', 'omni'], required=True)
    args = parser.parse_args()
    root = (args.checkpoints / args.family).resolve()
    namespace = 'd1_released_' + args.family
    package = types.ModuleType(namespace)
    package.__path__ = [str(root)]
    sys.modules[namespace] = package
    module = importlib.import_module(namespace + '.modeling_d1')
    cls = module.D1Model if args.family == 'causal' else module.D1OmniModel
    dtype = torch.bfloat16 if args.family == 'causal' else torch.float32
    kwargs = {'attn_implementation': 'sdpa'} if args.family == 'causal' else {}
    torch.set_num_threads(6)
    model = cls.from_pretrained(root, dtype=dtype, local_files_only=True, **kwargs).eval()
    for media in ['text', 'image'] if args.family == 'causal' else ['text', 'image', 'audio']:
        request = json.loads((args.output / (media + '.json')).read_text())
        images = [Image.open(args.output / path).convert('RGB') for path in request.get('images', [])] or None
        audio = sf.read(args.output / request['audio'], dtype='float32')[0] if 'audio' in request else None
        answers, tokens = {}, 0
        for name, question in request['questions'].items():
            inputs = {'images': images}
            if audio is not None:
                inputs['audio'] = audio
            # The native runtime executes each question independently; match that shape.
            with torch.inference_mode():
                result = model.system_one(request['state'], {name: question}, **inputs)
            answers.update(result['answers'])
            tokens += result['usage']['input_tokens']
        reference = {
            'family': args.family, 'media': media, 'dtype': str(dtype), 'device': 'cpu',
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'answers': answers, 'usage': {'input_tokens': tokens, 'output_tokens': 0},
        }
        destination = args.output / ('reference-' + args.family + '-' + media + '.json')
        destination.write_text(json.dumps(reference, indent=2) + '\n')
        print(destination)


if __name__ == '__main__':
    main()

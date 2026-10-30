"""Run the pinned upstream PyTorch code against the same weights and page images."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', default='receipt-base,w9-form-base')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--dtype', choices=['bfloat16'], default='bfloat16')
    parser.add_argument('--ngram-size', type=int, default=0)
    parser.add_argument('--max-tokens', type=int)
    args = parser.parse_args()
    root = args.source_root.resolve()
    entry = root / 'modeling_unlimitedocr.py'
    original = entry.with_suffix('.py.original')
    if not original.exists():
        original.write_bytes(entry.read_bytes())
    source = original.read_text()
    assert source.count('.unsqueeze(-1).cuda()') == 2
    adapted = source.replace('.unsqueeze(-1).cuda()', '.unsqueeze(-1).to(inputs_embeds.device)')
    entry.write_text(adapted)
    sys.path.insert(0, str(root.parent))
    module = importlib.import_module(root.name + '.modeling_unlimitedocr')
    import torch
    import transformers
    import PIL
    from transformers import AutoTokenizer

    torch.set_float32_matmul_precision('highest')
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    start = time.monotonic()
    model = module.UnlimitedOCRForCausalLM.from_pretrained(
        root, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
        attn_implementation='eager'
    ).eval().to(args.device)
    print(f'Upstream loaded on {args.device} ({args.dtype}) in {time.monotonic()-start:.1f}s', flush=True)
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    selected = set(args.cases.split(','))
    args.output.mkdir(parents=True, exist_ok=True)
    for case in manifest['cases']:
        if case['id'] not in selected:
            continue
        start = time.monotonic()
        result = model.infer(
            tokenizer, image_file=[str((args.inputs / page['image']).resolve()) for page in case['pages']],
            crop_mode=None if case['crop'] == 'auto' else case['crop'] == 'tiles',
            max_new_tokens=args.max_tokens or case['maxTokens'], temperature=0,
            no_repeat_ngram_size=args.ngram_size, ngram_window=1024 if len(case['pages']) > 1 else 128,
            eval_mode=True, return_dict=True, tps_interval=256
        )
        record = dict(caseID=case['id'], revision='c14071b549f4fcab10909f3c4acc6a2edb366947',
                      device=args.device, dtype=args.dtype, torch=torch.__version__,
                      transformers=transformers.__version__, pillow=PIL.__version__,
                      noRepeatNgramSize=args.ngram_size, seconds=time.monotonic()-start,
                      maxTokens=args.max_tokens or case['maxTokens'],
                      adaptations=['Two CUDA-only image-mask transfers use the input embedding device.'],
                      upstreamEntrySHA256=hashlib.sha256(source.encode()).hexdigest(), **result)
        name = f"{case['id']}-{args.dtype}-ngram{args.ngram_size}"
        (args.output / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        (args.output / (name + '.txt')).write_text(result['text'] + '\n')
        print(name, len(result['token_ids']), result['finish_reason'], f"{record['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()

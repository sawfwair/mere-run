"""Capture the pinned upstream price decision under an identical token prefix."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ['source-root', 'image', 'reference', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--step', type=int, default=341)
    parser.add_argument('--capture-all-steps', action='store_true')
    parser.add_argument('--capture-leaves', action='store_true')
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root.parent))
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    module = importlib.import_module(root.name + '.modeling_unlimitedocr')
    base = importlib.import_module(root.name + '.modeling_deepseekv2')
    torch.set_float32_matmul_precision('highest')
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    reference = json.loads(args.reference.read_text())
    tokens = reference['token_ids']
    model = module.UnlimitedOCRForCausalLM.from_pretrained(
        root, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation='eager').eval().to('mps')
    state = dict(step=0, attention=0, router=0)
    tensors = {}

    def store(name, value):
        tensors[name] = value.detach().cpu().contiguous().clone()

    def inputs(layer):
        def capture(mod, values):
            if state['step'] == args.step:
                store(f'layer.{layer}.input', values[0].reshape(-1, 1280))
            if state['step'] == 0 and layer == 0:
                store('embeddings', values[0].reshape(-1, 1280))
        return capture

    def outputs(layer):
        def capture(mod, values, value):
            if args.capture_all_steps:
                store(f'step.{state["step"]}.decoder.block.{layer}', value[0].reshape(-1, 1280))
            if state['step'] == args.step:
                store(f'layer.{layer}.output', value[0].reshape(-1, 1280))
        return capture

    original_attention = base._decode_attend
    def attention(q, k, v, topi, bias, valid, positions, block_size, window, scale):
        output = original_attention(q, k, v, topi, bias, valid, positions, block_size, window, scale)
        if state['step'] == args.step:
            layer = state['attention']
            for name, value in [('q', q[0]), ('k', k[0]), ('v', v[0]), ('topi', topi[0]),
                                ('bias', bias[0]), ('valid', valid[0]), ('output', output[0])]:
                store(f'attention.{layer}.{name}', value)
            state['attention'] += 1
        return output
    base._decode_attend = attention
    original_scores = base._block_scores_infer
    def scores(router, iq, ik, positions, nblk):
        result = original_scores(router, iq, ik, positions, nblk)
        if state['step'] == args.step:
            layer = state['router']
            for name, value in [('iq', iq[0]), ('ik', ik[0]), ('head', router.head_cls), ('scores', result[0])]:
                store(f'attention.{layer}.{name}', value)
            state['router'] += 1
        return result
    base._block_scores_infer = scores
    for layer, block in enumerate(model.model.layers):
        block.register_forward_pre_hook(inputs(layer))
        block.register_forward_hook(outputs(layer))
        if args.capture_leaves:
            for name, child in block.named_modules():
                if not name:
                    continue
                def leaf(mod, values, value, key=f'layer.{layer}.{name}'):
                    if state['step'] != args.step:
                        return
                    if isinstance(value, torch.Tensor):
                        store(key, value)
                    elif isinstance(value, tuple):
                        for part, item in enumerate(value):
                            if isinstance(item, torch.Tensor):
                                store(f'{key}.{part}', item)
                child.register_forward_hook(leaf)
    def norm(mod, values, value):
        if args.capture_all_steps:
            store(f'step.{state["step"]}.decoder.norm', value[:, -1:])
    model.model.norm.register_forward_hook(norm)
    def logits(mod, values, output):
        step = state['step']
        state['step'] += 1
        if step == args.step:
            store('logits', output[0, -1])
            return output
        # Teacher forcing changes token selection only, preserving model math.
        forced = torch.full_like(output, -float('inf'))
        forced[..., tokens[step]] = 0
        return forced
    model.lm_head.register_forward_hook(logits)
    result = model.infer(tokenizer, image_file=str(args.image.resolve()), crop_mode=False,
                         max_new_tokens=args.step + 1, temperature=0, no_repeat_ngram_size=0,
                         eval_mode=True, return_dict=True)
    values, indices = torch.topk(tensors['logits'].float(), 8)
    record = dict(step=args.step, tokens=tokens, promptTokens=result['prompt_token_ids'],
                  candidates=[dict(id=int(i), text=tokenizer.decode([int(i)]), score=float(v))
                              for i, v in zip(indices, values)],
                  zeroTokens={text: tokenizer.encode(text, add_special_tokens=False) for text in ['00', '000']},
                  selected=result['token_ids'][-1], torch=torch.__version__, prefixTeacherForced=True)
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output / 'upstream.safetensors'))
    (args.output / 'upstream.json').write_text(json.dumps(record, indent=2) + '\n')
    print(record['candidates'], flush=True)


if __name__ == '__main__':
    main()

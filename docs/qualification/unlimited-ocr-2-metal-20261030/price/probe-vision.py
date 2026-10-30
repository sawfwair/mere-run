"""Isolate MPS BF16 vision attention using the real first SAM window inputs."""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ['source-root', 'capture', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve().parent))
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    deep = importlib.import_module(args.source_root.name + '.deepencoder')
    torch.set_float32_matmul_precision('highest')
    captured = load_file(str(args.capture))
    projected = captured['sam.first.qkv'].to('mps')
    batch, height, width, _ = projected.shape
    q, k, v = projected.reshape(batch, height * width, 3, 12, 64).permute(2, 0, 3, 1, 4).unbind(0)
    index = json.loads((args.source_root / 'model.safetensors.index.json').read_text())['weight_map']
    relative = []
    for axis in ['h', 'w']:
        name = 'model.sam_model.blocks.0.attn.rel_pos_' + axis
        with safe_open(str(args.source_root / index[name]), framework='pt') as weights:
            relative.append(weights.get_tensor(name).to('mps'))
    rh, rw = deep.add_decomposed_rel_pos(q.reshape(-1, height * width, 64), *relative,
                                        (height, width), (height, width))
    bias = (rh + rw).reshape(batch, 12, height * width, height * width)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    scores = (q @ k.transpose(-1, -2)) * 0.125 + bias
    variants = dict(
        bf16=(torch.softmax(scores, dim=-1) @ v),
        bf16ScoresFloatSoftmax=(torch.softmax(scores.float(), dim=-1) @ v.float()).to(torch.bfloat16),
        float=((torch.softmax((q.float() @ k.float().transpose(-1, -2)) * 0.125 + bias.float(), dim=-1)) @ v.float()).to(torch.bfloat16),
        splitScale=(torch.softmax(((q.float() * math.sqrt(0.125)) @ (k.float() * math.sqrt(0.125)).transpose(-1, -2)) + bias.float(), dim=-1) @ v.float()).to(torch.bfloat16))
    metrics = {}
    for name, value in variants.items():
        delta = (value.float() - reference.float()).abs()
        metrics[name] = dict(maximum=delta.max().item(), unequal=(value != reference).sum().item(),
                             rmse=delta.square().mean().sqrt().item())
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep one real window for a compact native regression, including all heads.
    save_file({name: value[:1].detach().cpu().contiguous() for name, value in
               [('q', q), ('k', k), ('v', v), ('bias', bias), ('output', reference)]}, str(args.output / 'vision.safetensors'))
    (args.output / 'vision.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(metrics, flush=True)


if __name__ == '__main__':
    main()

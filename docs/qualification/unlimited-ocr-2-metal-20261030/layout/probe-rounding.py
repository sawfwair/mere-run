"""Check actual differing projection cells against CPU FP64 accumulation.

The FP64 values diagnose reduction error; they are not a replacement model dtype.
Use --prepare before the native diagnostic to write the BF16 activation sweep.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--prefill', type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    if args.prepare:
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        x = bits.view(torch.bfloat16)
        x = x[torch.isfinite(x) & (x.abs() <= 32)].to('mps')
        save_file(dict(x=x.cpu(), silu=torch.nn.functional.silu(x).cpu(),
                       gelu=torch.nn.functional.gelu(x).cpu(), sigmoid=x.sigmoid().cpu()),
                  str(args.capture / 'activations.safetensors'))
        return
    index = json.loads((args.source_root / 'model.safetensors.index.json').read_text())['weight_map']

    def weight(name):
        with safe_open(str(args.source_root / index[name]), framework='pt') as source:
            return source.get_tensor(name)

    native = load_file(str(args.capture / 'native-states.safetensors'))
    reference = load_file(str(args.capture / 'upstream.safetensors'))
    record = dict(torch=torch.__version__, oracle='CPU FP64 dot products, then BF16 round-to-nearest-even',
                  note='Sampled differing cells; backend agreement is not a correctness oracle.', projections={})

    def row(exact, actual, expected, position):
        nearest = exact.to(torch.bfloat16).item()
        return dict(position=position, fp64=exact.item(), nearestBF16=nearest,
                    native=actual.item(), upstream=expected.item(),
                    nativeNearest=actual.item() == nearest, upstreamNearest=expected.item() == nearest)

    x = reference['layer.0.post_attention_layernorm'].reshape(-1).double()
    for name in ['gate_proj', 'up_proj']:
        key = 'layer.0.mlp.' + name
        actual, expected = native[key].flatten(), reference[key].flatten()
        differing = (actual != expected).nonzero().flatten()
        w = weight('model.layers.0.mlp.' + name + '.weight')
        record['projections'][key] = [row((x * w[i].double()).sum(), actual[i], expected[i], int(i))
                                       for i in differing[:8]]
    if args.prefill:
        prefill = load_file(str(args.prefill))
        w = weight('model.sam_model.patch_embed.proj.weight').double()
        bias = weight('model.sam_model.patch_embed.proj.bias').double()
        for name in ['prefill.sam.patch', 'prefill.sam.patch.chw', 'prefill.sam.patch.hwc']:
            actual, expected = native[name], prefill['sam.patch']
            differing = (actual != expected).nonzero()
            samples = []
            for batch, y, x, channel in differing[:8].tolist():
                patch = prefill['pixels'][batch, y*16:(y+1)*16, x*16:(x+1)*16].permute(2, 0, 1).double()
                exact = (patch * w[channel]).sum() + bias[channel]
                at = (batch, y, x, channel)
                samples.append(row(exact, actual[at], expected[at], list(at)))
            record['projections'][name] = dict(unequal=len(differing), total=actual.numel(), samples=samples)
    (args.capture / 'rounding.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Compare projection accumulation on the exact upstream decision state."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ['source-root', 'capture']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--native-state', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve().parent))
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    base = importlib.import_module(args.source_root.name + '._sparse_base_impl')
    tensors = load_file(str(args.capture / 'upstream.safetensors'))
    index = json.loads((args.source_root / 'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        with safe_open(str(args.source_root / index[name]), framework='pt') as source:
            return source.get_tensor(name).to('mps')
    norm = base.DeepseekV2RMSNorm(1280, eps=1e-6).to(device='mps', dtype=torch.bfloat16)
    norm.weight.data = weight('model.norm.weight')
    head = weight('lm_head.weight')
    hidden = norm(tensors['layer.11.output'].to('mps'))
    if args.native_state:
        hidden = load_file(str(args.native_state))['head.input'].to('mps')
    original = torch.nn.functional.linear(hidden, head)
    promoted = torch.nn.functional.linear(hidden.float(), head.float()).to(torch.bfloat16)
    if not args.native_state:
        assert torch.equal(original.cpu().flatten(), tensors['logits'].flatten())
    save_file(dict(hidden=hidden.detach().cpu().contiguous(), original=original.detach().cpu().contiguous(),
                   promoted=promoted.detach().cpu().contiguous()), str(args.capture / ('head-native.safetensors' if args.native_state else 'head.safetensors')))
    print(dict(original=original[0, [504, 1320]].tolist(), promoted=promoted[0, [504, 1320]].tolist(),
               unequal=(original != promoted).sum().item()), flush=True)


if __name__ == '__main__':
    main()

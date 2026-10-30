"""Regenerate BF16 attention goldens using the pinned upstream CPU implementation.

Run in a temporary PyTorch environment with --source-root pointing to the
package prepared by the qualification report's prepare-upstream.py script.
Production and the Swift tests don't require Python or PyTorch.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root.parent))
    import torch
    base = importlib.import_module(root.name + '.modeling_deepseekv2')
    dash = importlib.import_module(root.name + '.dash_attention')
    path = Path(__file__).parent
    data = json.loads((path / 'attention.json').read_text())
    count = len(data['tokens'])
    q, k, v = [torch.tensor(data[name]).reshape(1, 1, count, 4).to(torch.bfloat16) for name in ['q', 'k', 'v']]
    iq, ik = [torch.tensor(data[name]).reshape(1, count, 4) for name in ['indexQ', 'indexK']]
    head = torch.tensor(data['head'])
    router = SimpleNamespace(block_size=2, local_blocks=2, local_window=4,
                             stage0='head_cls', head_cls=head, q_chunk=64, _scale=0.5)
    params = dict(idx_scale=4 / math.sqrt(4), floor=0.0001, sigma=1, K=2, topm=3)
    positions = torch.arange(count)[None]
    scores = base._block_scores_infer(router, iq, ik, positions, (count + 1) // 2)
    topi, bias, valid = base._infer_route_select(scores, None, params)
    raw_prefill = dash.DashAttention._forward_dense(router, q, k, v, topi, bias, valid, count, (count + 1) // 2, None)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    def fp32_sdpa(q, k, v, attn_mask=None, **kwargs):
        return sdpa(q.float(), k.float(), v.float(), attn_mask=attn_mask.float(), **kwargs).to(q.dtype)
    torch.nn.functional.scaled_dot_product_attention = fp32_sdpa
    try:
        prefill = dash.DashAttention._forward_dense(router, q, k, v, topi, bias, valid, count, (count + 1) // 2, None)
    finally:
        torch.nn.functional.scaled_dot_product_attention = sdpa
    float_mask = dash.DashAttention._forward_dense(router, q.float(), k.float(), v.float(), topi, bias, valid, count, (count + 1) // 2, None).to(torch.bfloat16)
    assert (prefill != float_mask).any(), 'Fixture must expose the prefill BF16 mask/output boundary.'
    decode_positions = [7, 17, count - 1]
    decode = []
    for position in decode_positions:
        end = position + 1
        qp = torch.tensor([[position]])
        scores = base._block_scores_infer(router, iq[:, position:end], ik[:, :end], qp, (end + 1) // 2)
        topi, bias, valid = base._infer_route_select(scores, None, params)
        out = base._decode_attend(q[:, :, position:end], k[:, :, :end], v[:, :, :end], topi, bias, valid, qp, 2, 4, 0.5)
        decode.append(out.to(torch.bfloat16).float().flatten().tolist())
    record = dict(revision='c14071b549f4fcab10909f3c4acc6a2edb366947', torch=torch.__version__,
                  count=count, q=q.float().flatten().tolist(), k=k.float().flatten().tolist(), v=v.float().flatten().tolist(),
                  indexQ=iq.flatten().tolist(), indexK=ik.flatten().tolist(), head=head.tolist(),
                  sdpaAccumulation='FP32; upstream rounds its prefill mask to BF16 before the controlled promotion',
                  rawPrefill=raw_prefill.float().flatten().tolist(), prefill=prefill.float().flatten().tolist(),
                  decodePositions=decode_positions, decode=decode)
    grid = (torch.sin(torch.arange(105).float()) / 10).reshape(1, 7, 5, 3).to(torch.bfloat16)
    resized = torch.nn.functional.interpolate(grid.permute(0, 3, 1, 2).float(), size=(4, 3),
                                             mode='bicubic', align_corners=False, antialias=True)
    record.update(positionInput=grid.float().flatten().tolist(),
                  positionOutput=resized.permute(0, 2, 3, 1).to(torch.bfloat16).float().flatten().tolist())
    (path / 'precision.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()

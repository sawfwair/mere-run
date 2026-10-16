#!/usr/bin/env python3
"""Export unmodified upstream vision stage outputs for the frozen chart input."""
import argparse
from pathlib import Path
import sys
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--size', choices=['2', '4', '9'], required=True)
    args = parser.parse_args()
    root = args.root
    sys.path.insert(0, str(root.parent / 'metadata' / args.size / 'runtime'))
    from vision_artifact import load_vl_model
    import mlx.core as mx
    fixture = os.environ.get('MERERUN_TEST_BONSAI_ROTARY_FIXTURE')
    if fixture:
        import hashlib
        import json
        from mlx_vlm.models.qwen3_vl.vision import VisionModel
        from mlx_vlm.models.qwen3_vl.config import VisionConfig
        arrays = {}
        for label, device in [('cpu', mx.cpu), ('gpu', mx.gpu)]:
            mx.set_default_device(device)
            for head_dim in [64, 72]:
                tower = VisionModel(VisionConfig(depth=0, hidden_size=head_dim, intermediate_size=128,
                    out_hidden_size=head_dim, num_heads=1, patch_size=16, temporal_patch_size=2))
                arrays[label + '_' + str(head_dim)] = tower.rot_pos_emb(mx.array([[1, 4, 6]], dtype=mx.int32))
                mx.eval(arrays[label + '_' + str(head_dim)])
        destination = Path(fixture)
        if destination.exists():
            raise FileExistsError('Preserve completed rotary fixtures')
        mx.save_safetensors(str(destination), arrays)
        destination.with_suffix('.json').write_text(json.dumps({'reference': 'mlx-vlm 0.6.3 VisionModel.rot_pos_emb; MLX 0.32.0 CPU and Metal',
            'grid': [1, 4, 6], 'head_dimensions': [64, 72], 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2) + '\n')
        return
    model, _, _ = load_vl_model(root.parent / 'models' / ('text-chat-bonsai-2-' + args.size + 'b-2bit'), load_processor=False)
    vision = model.vision_tower
    source = mx.load(str(root / ('reference-' + args.size) / 'vision-chart.safetensors'))
    grid = source['grid']
    patches = source['patches'].astype(vision.patch_embed.proj.weight.dtype)
    hidden = vision.patch_embed(patches)
    pos = vision.fast_pos_embed_interpolate(grid)
    rotary = vision.rot_pos_emb(grid)
    external_rope = os.environ.get('MERERUN_TEST_BONSAI_STAGE_ROTARY_FROM')
    if external_rope:
        rotary = mx.load(external_rope)['rotary_freqs']
    arrays = {'patch_embed': hidden, 'pos_embed': pos, 'rotary_freqs': rotary}
    hidden = hidden + pos
    arrays['input'] = hidden
    cu = mx.array([0, hidden.shape[0]], dtype=mx.int32)
    for index, block in enumerate(vision.blocks):
        hidden = block(hidden, cu_seqlens=cu, rotary_pos_emb=rotary)
        mx.eval(hidden)
        arrays['block_' + str(index)] = hidden
    arrays['merged'] = vision.merger(hidden)
    mx.eval(*arrays.values())
    suffix = '-external-rope' if external_rope else ''
    if os.environ.get('MERERUN_TEST_BONSAI_STAGE_BACKEND_CONTROL') == '0.32.1':
        import importlib.metadata
        if importlib.metadata.version('mlx') != '0.32.1':
            raise ValueError('Backend control must use MLX 0.32.1')
        suffix += '-backend0321'
    destination = root / ('reference-vision-stages-' + args.size + suffix + '.safetensors')
    if destination.exists():
        raise FileExistsError('Preserve completed stage evidence')
    mx.save_safetensors(str(destination), arrays)
    print('Exported upstream vision stages', args.size, flush=True)


if __name__ == '__main__':
    main()

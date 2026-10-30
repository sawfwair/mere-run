"""Capture pinned upstream intermediates for numerical differential diagnosis."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    entry = root / 'modeling_unlimitedocr.py'
    original = entry.with_suffix('.py.original')
    source = original.read_text() if original.exists() else entry.read_text()
    assert source.count('.unsqueeze(-1).cuda()') == 2
    entry.write_text(source.replace('.unsqueeze(-1).cuda()', '.unsqueeze(-1).to(inputs_embeds.device)'))
    sys.path.insert(0, str(root.parent))
    module = importlib.import_module(root.name + '.modeling_unlimitedocr')
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    base = importlib.import_module(root.name + '.modeling_deepseekv2')
    apply_rope = base._llama_apply_rotary_pos_emb
    def rope(q, k, cos, sin, *extra, **kwargs):
        result = apply_rope(q, k, cos, sin, *extra, **kwargs)
        if 'decoder.first.rope_q' not in tensors:
            tensors['decoder.first.rope_q'] = result[0][0].detach().cpu().contiguous().clone()
            tensors['decoder.first.rope_k'] = result[1][0].detach().cpu().contiguous().clone()
        return result
    base._llama_apply_rotary_pos_emb = rope

    torch.set_float32_matmul_precision('highest')
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = module.UnlimitedOCRForCausalLM.from_pretrained(
        root, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation='eager').eval().to('mps')
    tensors = {}

    def hook(name, channels_first=False, flatten=False):
        def capture(mod, inputs, output):
            if name in tensors:
                return
            value = output[0] if isinstance(output, tuple) else output
            if channels_first:
                value = value.permute(0, 2, 3, 1)
            if flatten:
                value = value.reshape(-1, value.shape[-1])
            tensors[name] = value.detach().cpu().contiguous().clone()
        return capture

    sam = model.model.sam_model
    deep = importlib.import_module(root.name + '.deepencoder')
    tensors['probe.sam_pos_40'] = deep.get_abs_pos_sam(sam.pos_embed.detach().cpu(), 40).contiguous()
    table = model.model.vision_model.embeddings.position_embedding.weight.detach().cpu()
    tensors['probe.clip_pos_10'] = deep.get_abs_pos(table.unsqueeze(0), 100)[:, 1:].reshape(1, 10, 10, 1024).contiguous()
    def pixels(mod, inputs):
        tensors['pixels'] = inputs[0].detach().cpu().permute(0, 2, 3, 1).contiguous().clone()
    sam.register_forward_pre_hook(pixels)
    sam.patch_embed.register_forward_hook(hook('sam.patch'))
    for i, block in enumerate(sam.blocks):
        block.register_forward_hook(hook(f'sam.block.{i}'))
    for i, layer in enumerate(sam.neck):
        layer.register_forward_hook(hook(f'sam.neck.{i}', channels_first=True))
    sam.net_2.register_forward_hook(hook('sam.net2', channels_first=True))
    sam.net_3.register_forward_hook(hook('sam.net3', channels_first=True))
    for name, layer in [('norm1', sam.blocks[0].norm1), ('qkv', sam.blocks[0].attn.qkv),
                        ('proj', sam.blocks[0].attn.proj), ('norm2', sam.blocks[0].norm2),
                        ('up', sam.blocks[0].mlp.lin1), ('gelu', sam.blocks[0].mlp.act),
                        ('down', sam.blocks[0].mlp.lin2)]:
        layer.register_forward_hook(hook('sam.first.' + name))
    clip = model.model.vision_model
    clip.pre_layrnorm.register_forward_hook(hook('clip.pre'))
    for i, layer in enumerate(clip.transformer.layers):
        layer.register_forward_hook(hook(f'clip.block.{i}'))
    first_clip = clip.transformer.layers[0]
    for name, layer in [('norm1', first_clip.layer_norm1), ('qkv', first_clip.self_attn.qkv_proj),
                        ('attention', first_clip.self_attn.out_proj), ('norm2', first_clip.layer_norm2),
                        ('up', first_clip.mlp.fc1), ('down', first_clip.mlp.fc2)]:
        layer.register_forward_hook(hook('clip.first.' + name))
    def quick_input(mod, inputs):
        tensors['clip.first.quick_gelu'] = inputs[0].detach().cpu().contiguous().clone()
    first_clip.mlp.fc2.register_forward_pre_hook(quick_input)
    model.model.projector.layers.register_forward_hook(hook('projector', flatten=True))
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(hook(f'decoder.block.{i}', flatten=True))
    first_decoder = model.model.layers[0]
    for name, layer in [('norm1', first_decoder.input_layernorm), ('attention', first_decoder.self_attn),
                        ('norm2', first_decoder.post_attention_layernorm), ('up', first_decoder.mlp.up_proj),
                        ('gate', first_decoder.mlp.gate_proj), ('down', first_decoder.mlp.down_proj),
                        ('q_proj', first_decoder.self_attn.q_proj), ('k_proj', first_decoder.self_attn.k_proj),
                        ('v_proj', first_decoder.self_attn.v_proj)]:
        layer.register_forward_hook(hook('decoder.first.' + name, flatten=True))
    def gated_input(mod, inputs):
        if 'decoder.first.gated' not in tensors:
            tensors['decoder.first.gated'] = inputs[0].detach().cpu().reshape(-1, inputs[0].shape[-1]).contiguous().clone()
    first_decoder.mlp.down_proj.register_forward_pre_hook(gated_input)
    model.model.norm.register_forward_hook(hook('decoder.norm', flatten=True))
    def embedding(mod, inputs):
        if 'embeddings' not in tensors:
            tensors['embeddings'] = inputs[0].detach().cpu().reshape(-1, inputs[0].shape[-1]).contiguous().clone()
    model.model.layers[0].register_forward_pre_hook(embedding)
    model.lm_head.register_forward_hook(hook('logits', flatten=True))
    result = model.infer(tokenizer, image_file=str(args.image.resolve()), crop_mode=False,
                         max_new_tokens=1, temperature=0, eval_mode=True, return_dict=True)
    tensors['prompt_ids'] = torch.tensor(result['prompt_token_ids'])
    q = tensors['decoder.first.rope_q'][:, :64].unsqueeze(0)
    k = tensors['decoder.first.rope_k'][:, :64].unsqueeze(0)
    v = tensors['decoder.first.v_proj'][:64].reshape(64, 10, 128).transpose(0, 1).unsqueeze(0)
    for name, value in [('q', q[0]), ('k', k[0]), ('v', v[0])]:
        tensors['probe.attention.' + name] = value.contiguous()
    for device in ['cpu', 'mps']:
        for dtype in [torch.bfloat16, torch.float32]:
            value = original_sdpa(
                q.to(device=device, dtype=dtype), k.to(device=device, dtype=dtype),
                v.to(device=device, dtype=dtype), is_causal=True)
            tensors[f'probe.attention.{device}_{dtype}'] = value[0].to(torch.bfloat16).cpu().contiguous()
    image = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    tensors['decoded_rgb'] = torch.tensor(list(image.getdata()), dtype=torch.uint8).reshape(image.height, image.width, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    print(json.dumps({name: list(value.shape) for name, value in tensors.items()}, indent=2))


if __name__ == '__main__':
    main()

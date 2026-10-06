#!/usr/bin/env python3
"""Independent, layer-streamed BF16 reference from the original checkpoint.

Implements published Kolibri arithmetic with PyTorch operators, not vLLM's
fused kernels. This cross-check cannot certify bitwise official-vLLM parity.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def run(source, suite, output, case_id):
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    readers = {name: safe_open(source / name, framework="pt", device="cpu") for name in set(index.values())}
    seq = next(r for r in json.loads(suite.read_text()) if r["id"] == case_id)
    tokens = seq["tokens"][:-1]
    length = len(tokens)
    def weight(key): return readers[index[key]].get_tensor(key).cuda()
    def linear(x, key): return torch.nn.functional.linear(x, weight(key + ".weight"))
    def norm(x, key):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + config["rms_norm_eps"]) * weight(key).float()).to(x.dtype)
    x = weight("model.embed_tokens.weight")[tokens].unsqueeze(0)
    dim, nq, nkv = config["head_dim"], config["num_attention_heads"], config["num_key_value_heads"]
    positions = torch.arange(length, device="cuda")
    phase = positions[:, None].float() * (config["rope_theta"] ** (-torch.arange(0, dim, 2, device="cuda").float() / dim))[None, :]
    def rope(y):
        a, b = y[..., :dim // 2].float(), y[..., dim // 2:].float()
        return torch.cat([a * phase.cos() - b * phase.sin(), b * phase.cos() + a * phase.sin()], -1).to(y.dtype)
    def mlp(y, prefix):
        gate, up = linear(y, prefix + ".gate_proj"), linear(y, prefix + ".up_proj")
        return linear(torch.nn.functional.silu(gate) * up, prefix + ".down_proj")
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        y = norm(x, p + "input_layernorm.weight")
        q = norm(linear(y, p + "self_attn.q_proj").reshape(1, length, nq, dim), p + "self_attn.q_norm.weight").transpose(1, 2)
        k = norm(linear(y, p + "self_attn.k_proj").reshape(1, length, nkv, dim), p + "self_attn.k_norm.weight").transpose(1, 2)
        v = linear(y, p + "self_attn.v_proj").reshape(1, length, nkv, dim).transpose(1, 2)
        allowed = positions[None, :] <= positions[:, None]
        if config["layer_types"][layer] == "sliding_attention":
            q, k = rope(q), rope(k)
            allowed &= positions[None, :] > positions[:, None] - config["sliding_window"]
        attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True)
        attention = attention.transpose(1, 2).reshape(1, length, nq * dim)
        x = x + norm(linear(attention, p + "self_attn.o_proj"), p + "post_attn_norm.weight")
        y = norm(x, p + "post_attention_layernorm.weight")
        logits = torch.nn.functional.linear(y.float(), weight(p + "mlp.gate.weight").float())
        ids = (logits + weight(p + "moe.router.expert_bias").float()).topk(config["num_experts_per_tok"], -1).indices
        scores = logits.gather(-1, ids).sigmoid()
        routed = torch.zeros_like(y, dtype=torch.float32)
        for expert in ids.unique().tolist():
            row, slot = (ids[0] == expert).nonzero(as_tuple=True)
            expert_output = mlp(y[0, row], p + f"mlp.experts.{expert}")
            routed[0, row] += expert_output.float() * scores[0, row, slot, None]
        moe = routed.to(y.dtype) + mlp(y, p + "mlp.shared_experts")
        x = x + norm(moe, p + "post_ffn_norm.weight")
        torch.cuda.synchronize()
        print(f"layer={layer}", flush=True)
        del y, q, k, v, attention, logits, routed, moe
        gc.collect()
        torch.cuda.empty_cache()
    logits = torch.nn.functional.linear(norm(x, "model.norm.weight").float(), weight("lm_head.weight").float())[0, seq["scoreStart"] - 1:]
    output.mkdir(parents=True, exist_ok=False)
    save_file({"logits": logits.cpu().contiguous()}, str(output / (case_id + ".safetensors")))
    (output / "reference.json").write_text(json.dumps(dict(sequence=seq, suiteSHA256=hashlib.sha256(suite.read_bytes()).hexdigest(),
          implementation="independent PyTorch operators; original BF16 tensors; no TF32", torch_version=torch.__version__), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", default="cal-en")
    args = parser.parse_args()
    run(args.source, args.suite, args.output, args.case_id)

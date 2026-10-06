#!/usr/bin/env python3
"""Independent PyTorch arithmetic fixture for Kolibri's published architecture.

Tiny deterministic tensors exercise both attention types, sigmoid routing,
correction bias, shared experts, sandwich norms, and a sliding-window crossing.
This fixture is not a full-checkpoint or official vLLM parity claim.
"""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file


def export(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    c = dict(model_type="kolibri1", hidden_size=64, num_hidden_layers=2,
             num_attention_heads=6, num_key_value_heads=2, head_dim=16,
             max_position_embeddings=128, rms_norm_eps=1e-6, vocab_size=97,
             rope_theta=10000, num_experts=7, num_experts_per_tok=3,
             moe_intermediate_size=64, shared_expert_intermediate_size=64,
             norm_topk_prob=False, sliding_window=5,
             layer_types=["sliding_attention", "full_attention"], eos_token_id=96,
             mererun_quantization={})
    torch.manual_seed(713)
    weights = {}
    def weight(key, shape, norm=False):
        w = torch.ones(shape) + torch.randn(shape) * .03 if norm else torch.randn(shape) * .08
        weights[key] = w
        return w
    weight("model.embed_tokens.weight", (97, 64))
    weight("lm_head.weight", (97, 64))
    weight("model.norm.weight", (64,), True)
    for i in range(2):
        p = f"model.layers.{i}."
        for name in ["input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"]:
            weight(p + name + ".weight", (64,), True)
        for name, shape in [("q_proj", (96, 64)), ("k_proj", (32, 64)), ("v_proj", (32, 64)), ("o_proj", (64, 96))]:
            weight(p + "self_attn." + name + ".weight", shape)
        for name in ["q_norm", "k_norm"]: weight(p + "self_attn." + name + ".weight", (16,), True)
        weight(p + "mlp.gate.weight", (7, 64))
        weight(p + "mlp.gate.e_score_correction_bias", (7,))
        for name in ["gate_proj", "up_proj", "down_proj"]:
            weight(p + "mlp.experts." + name + ".weight", (7, 64, 64))
            weight(p + "mlp.shared_experts." + name + ".weight", (64, 64))
    tokens = [3, 11, 7, 23, 41, 19, 61, 37, 2, 17, 53]
    x = weights["model.embed_tokens.weight"][tokens].unsqueeze(0)
    def norm(x, key):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weights[key]
    def linear(x, key): return torch.nn.functional.linear(x, weights[key + ".weight"])
    def rope(x):
        frequency = 10000 ** (-torch.arange(0, 16, 2).float() / 16)
        phase = torch.arange(len(tokens)).float()[:, None] * frequency[None, :]
        first, second = x[..., :8], x[..., 8:]
        return torch.cat([first * phase.cos() - second * phase.sin(), second * phase.cos() + first * phase.sin()], -1)
    def mlp(x, p):
        return linear(torch.nn.functional.silu(linear(x, p + ".gate_proj")) * linear(x, p + ".up_proj"), p + ".down_proj")
    layers = []
    for i in range(2):
        p = f"model.layers.{i}."
        a = norm(x, p + "input_layernorm.weight")
        q = norm(linear(a, p + "self_attn.q_proj").reshape(1, -1, 6, 16), p + "self_attn.q_norm.weight").transpose(1, 2)
        k = norm(linear(a, p + "self_attn.k_proj").reshape(1, -1, 2, 16), p + "self_attn.k_norm.weight").transpose(1, 2)
        v = linear(a, p + "self_attn.v_proj").reshape(1, -1, 2, 16).transpose(1, 2)
        if i == 0: q, k = rope(q), rope(k)
        positions = torch.arange(len(tokens))
        allowed = positions[None, :] <= positions[:, None]
        if i == 0: allowed &= positions[None, :] > positions[:, None] - 5
        scores = q @ k.repeat_interleave(3, dim=1).transpose(-1, -2) / 4
        scores = scores.masked_fill(~allowed, -float("inf"))
        attn = (scores.softmax(-1) @ v.repeat_interleave(3, dim=1)).transpose(1, 2).reshape(1, -1, 96)
        x = x + norm(linear(attn, p + "self_attn.o_proj"), p + "post_attn_norm.weight")
        a = norm(x, p + "post_attention_layernorm.weight")
        logits = linear(a, p + "mlp.gate")
        ids = (logits + weights[p + "mlp.gate.e_score_correction_bias"]).topk(3, -1).indices
        probabilities = logits.gather(-1, ids).sigmoid()
        expert_outputs = []
        for row in range(len(tokens)):
            out = 0
            for slot in range(3):
                expert = int(ids[0, row, slot])
                g = torch.nn.functional.linear(a[0, row], weights[p + "mlp.experts.gate_proj.weight"][expert])
                u = torch.nn.functional.linear(a[0, row], weights[p + "mlp.experts.up_proj.weight"][expert])
                d = torch.nn.functional.linear(torch.nn.functional.silu(g) * u, weights[p + "mlp.experts.down_proj.weight"][expert])
                out = out + d * probabilities[0, row, slot]
            expert_outputs.append(out)
        moe = torch.stack(expert_outputs).unsqueeze(0) + mlp(a, p + "mlp.shared_experts")
        x = x + norm(moe, p + "post_ffn_norm.weight")
        layers.append(x.flatten().tolist())
    logits = linear(norm(x, "model.norm.weight"), "lm_head")
    save_file(weights, str(output / "weights.safetensors"))
    (output / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    (output / "reference.json").write_text(json.dumps({"tokens": tokens, "logits": logits.flatten().tolist(), "layers": layers, "shape": list(logits.shape)}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    export(p.parse_args().output)

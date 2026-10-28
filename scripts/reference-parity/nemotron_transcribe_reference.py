#!/usr/bin/env python3
"""Dependency-free scalar reference for the NeMo Transcribe inference equations.

Regenerate: python3 scripts/reference-parity/nemotron_transcribe_reference.py
Source equations: NVIDIA-NeMo/Speech 1c1b25339ca2b31794ba4b299a5ea6a2c83c64b5,
transformer_encoder.py and parallel_expert_encoder.py; Nemotron-H Mamba-2/MoE.
This deliberately uses scalar Python math, independent of the Swift/MLX graph.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tensors = {}


def tensor(key, shape, base=0.0, scale=0.12):
    count = math.prod(shape)
    phase = sum(map(ord, key)) * 0.017
    values = [base + scale * math.sin(phase + i * 0.37) for i in range(count)]
    tensors[key] = {"shape": shape, "values": values}
    return values


def matrix(key):
    data = tensors[key]
    cols = data["shape"][-1]
    return [data["values"][i:i + cols] for i in range(0, len(data["values"]), cols)]


def add(x, y):
    return [a + b for a, b in zip(x, y)]


def linear(x, prefix, bias=False):
    result = [sum(a * b for a, b in zip(x, row)) for row in matrix(prefix + ".weight")]
    return add(result, tensors[prefix + ".bias"]["values"]) if bias else result


def norm(x, prefix, rms=False):
    mean = 0.0 if rms else sum(x) / len(x)
    centered = [v - mean for v in x]
    variance = sum(v * v for v in centered) / len(x)
    result = [v / math.sqrt(variance + 1e-5) * w for v, w in zip(centered, tensors[prefix + ".weight"]["values"])]
    return result if rms else add(result, tensors[prefix + ".bias"]["values"])


def softmax(x):
    exp = [math.exp(v - max(x)) for v in x]
    return [v / sum(exp) for v in exp]


def gelu(x):
    return [v * 0.5 * (1 + math.erf(v / math.sqrt(2))) for v in x]


def norm_weights(prefix, width):
    tensor(prefix + ".weight", [width], 1.0)
    tensor(prefix + ".bias", [width], scale=0.02)


def dense_weights(prefix, output, input_size, bias=False):
    tensor(prefix + ".weight", [output, input_size])
    if bias:
        tensor(prefix + ".bias", [output], scale=0.02)


# One tiny audio Transformer with frame stacking, half-head RoPE, Q/K LayerNorm.
dense_weights("encoder.pre_encode.proj", 8, 16)
for name in ["embed_norm", "final_norm", "layers.0.norm1", "layers.0.norm2"]:
    norm_weights("encoder." + name, 8)
for name in ["q_norm", "k_norm"]:
    norm_weights("encoder.layers.0.attn." + name, 4)
dense_weights("encoder.layers.0.attn.w_qkv", 24, 8)
dense_weights("encoder.layers.0.attn.out_proj", 8, 8, True)
dense_weights("encoder.layers.0.ffn.net.0", 16, 8, True)
dense_weights("encoder.layers.0.ffn.net.3", 8, 16, True)
features = [[math.sin(i * 0.19), math.cos(i * 0.31)] for i in range(17)]
padded = features + [[0, 0] for _ in range(7)]
hidden = [norm(linear(sum(padded[i:i + 8], []), "encoder.pre_encode.proj"), "encoder.embed_norm") for i in range(0, 24, 8)]
projected = [linear(norm(row, "encoder.layers.0.norm1"), "encoder.layers.0.attn.w_qkv") for row in hidden]
q, k, v = [], [], []
for pos, row in enumerate(projected):
    heads = []
    for part in range(3):
        values = []
        for head in range(2):
            piece = row[part * 8 + head * 4:part * 8 + (head + 1) * 4]
            if part < 2:
                piece = norm(piece, "encoder.layers.0.attn." + ("q_norm" if part == 0 else "k_norm"))
                a, b = piece[:2]
                piece = [a * math.cos(pos) - b * math.sin(pos), b * math.cos(pos) + a * math.sin(pos)] + piece[2:]
            values.append(piece)
        heads.append(values)
    q.append(heads[0]); k.append(heads[1]); v.append(heads[2])
output = []
for pos, row in enumerate(hidden):
    attended = []
    for head in range(2):
        scores = [sum(a * b for a, b in zip(q[pos][head], key[head])) / 2 for key in k]
        probabilities = softmax(scores)
        attended += [sum(probabilities[t] * v[t][head][d] for t in range(3)) for d in range(4)]
    residual = add(row, linear(attended, "encoder.layers.0.attn.out_proj", True))
    ff = gelu(linear(norm(residual, "encoder.layers.0.norm2"), "encoder.layers.0.ffn.net.0", True))
    output.append(norm(add(residual, linear(ff, "encoder.layers.0.ffn.net.3", True)), "encoder.final_norm"))

# A full Mamba -> GQA -> MoE language stack with original NeMo matrix layouts.
config = {
    "model_type": "nemotron_h", "vocab_size": 12, "hidden_size": 8,
    "layers_block_type": ["mamba", "attention", "moe"], "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 4, "norm_eps": 1e-5,
    "mamba_head_dim": 2, "mamba_num_heads": 2, "ssm_state_size": 2,
    "n_groups": 1, "conv_kernel": 3, "time_step_min": 0.001,
    "n_routed_experts": 4, "num_experts_per_tok": 2,
    "moe_intermediate_size": 6, "moe_shared_expert_intermediate_size": 8,
    "routed_scaling_factor": 2.5, "norm_topk_prob": True, "n_group": 1,
    "topk_group": 1, "mlp_hidden_act": "relu2",
    "time_step_limit": [0.0, {"__float__": "Infinity"}], "rescale_prenorm_residual": True,
}
tensor("llm.model.embed_tokens.weight", [12, 8])
for layer in range(3):
    tensor(f"llm.model.layers.{layer}.norm.weight", [8], 1.0)
tensor("llm.model.norm.weight", [8], 1.0)
dense_weights("llm.lm_head", 12, 8)
p = "llm.model.layers.0.mixer"
dense_weights(p + ".in_proj", 14, 8)
dense_weights(p + ".out_proj", 8, 4)
tensor(p + ".conv1d.weight", [8, 1, 3])
tensor(p + ".conv1d.bias", [8], scale=0.02)
tensor(p + ".norm.weight", [4], 1.0)
for name in ["A_log", "D", "dt_bias"]:
    tensor(p + "._fp32_params." + name, [2], scale=0.1)
p = "llm.model.layers.1.mixer"
for name, width in [("q_proj", 8), ("k_proj", 4), ("v_proj", 4), ("o_proj", 8)]:
    dense_weights(p + "." + name, width, 8)
p = "llm.model.layers.2.mixer"
dense_weights(p + ".gate", 4, 8)
tensor(p + ".gate.e_score_correction_bias", [4], scale=0.2)
tensor(p + ".experts.gate_and_up_projs", [4, 8, 6])
tensor(p + ".experts.down_projs", [4, 6, 8])
dense_weights(p + ".shared_experts.up_proj", 8, 8)
dense_weights(p + ".shared_experts.down_proj", 8, 8)
h = [matrix("llm.model.embed_tokens.weight")[i] for i in [1, 2, 3]]
p = "llm.model.layers.0.mixer"
projected = [linear(norm(row, "llm.model.layers.0.norm", True), p + ".in_proj") for row in h]
convs = []
for pos, row in enumerate(projected):
    current = []
    for channel in range(8):
        value = tensors[p + ".conv1d.bias"]["values"][channel]
        for kernel in range(3):
            frame = pos - 2 + kernel
            if frame >= 0:
                value += projected[frame][4 + channel] * tensors[p + ".conv1d.weight"]["values"][channel * 3 + kernel]
        current.append(value / (1 + math.exp(-value)))
    convs.append(current)
state = [[[0.0] * 2 for _ in range(2)] for _ in range(2)]
for pos, row in enumerate(projected):
    y = []
    for head in range(2):
        delta = max(0.001, math.log1p(math.exp(row[12 + head] + tensors[p + "._fp32_params.dt_bias"]["values"][head])))
        decay = math.exp(-math.exp(tensors[p + "._fp32_params.A_log"]["values"][head]) * delta)
        for d in range(2):
            xv = convs[pos][head * 2 + d]
            for n in range(2):
                state[head][d][n] = state[head][d][n] * decay + delta * convs[pos][4 + n] * xv
            y.append(sum(state[head][d][n] * convs[pos][6 + n] for n in range(2)) + xv * tensors[p + "._fp32_params.D"]["values"][head])
    gated = [value * gate / (1 + math.exp(-gate)) for value, gate in zip(y, row[:4])]
    # Grouped gated RMS normalization: one group in this fixture.
    mixed = linear(norm(gated, p + ".norm", True), p + ".out_proj")
    h[pos] = add(h[pos], mixed)
p = "llm.model.layers.1.mixer"
normalized = [norm(row, "llm.model.layers.1.norm", True) for row in h]
queries = [linear(row, p + ".q_proj") for row in normalized]
keys = [linear(row, p + ".k_proj") for row in normalized]
values = [linear(row, p + ".v_proj") for row in normalized]
for pos in range(3):
    attended = []
    for head in range(2):
        query = queries[pos][head * 4:(head + 1) * 4]
        probs = softmax([sum(a * b for a, b in zip(query, key)) / 2 for key in keys[:pos + 1]])
        attended += [sum(prob * value[d] for prob, value in zip(probs, values)) for d in range(4)]
    h[pos] = add(h[pos], linear(attended, p + ".o_proj"))
p = "llm.model.layers.2.mixer"
for pos in range(3):
    row = norm(h[pos], "llm.model.layers.2.norm", True)
    scores = [1 / (1 + math.exp(-v)) for v in linear(row, p + ".gate")]
    bias = tensors[p + ".gate.e_score_correction_bias"]["values"]
    chosen = sorted(range(4), key=lambda i: scores[i] + bias[i], reverse=True)[:2]
    routed = [0.0] * 8
    for expert in chosen:
        up_data = tensors[p + ".experts.gate_and_up_projs"]["values"][expert * 48:(expert + 1) * 48]
        down_data = tensors[p + ".experts.down_projs"]["values"][expert * 48:(expert + 1) * 48]
        up = [max(0, sum(row[i] * up_data[i * 6 + j] for i in range(8))) ** 2 for j in range(6)]
        scale = scores[expert] / sum(scores[i] for i in chosen) * 2.5
        routed = add(routed, [scale * sum(up[i] * down_data[i * 8 + j] for i in range(6)) for j in range(8)])
    shared = [max(0, v) ** 2 for v in linear(row, p + ".shared_experts.up_proj")]
    h[pos] = add(h[pos], add(routed, linear(shared, p + ".shared_experts.down_proj")))
logits = linear(norm(h[-1], "llm.model.norm", True), "llm.lm_head")
fixture = {
    "tensors": tensors, "input": {"shape": [1, 17, 2], "values": sum(features, [])},
    "output": {"shape": [1, 3, 8], "values": sum(output, [])},
    "logits": {"shape": [1, 1, 12], "values": logits}, "config": config,
}
path = ROOT / "Tests/NemotronTranscribeRuntimeTests/Fixtures/reference.json"
path.write_text(json.dumps(fixture, indent=2) + "\n")
print(path)

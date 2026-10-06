#!/usr/bin/env python3
"""Independent scalar reference for EmbeddingGemma 2's synthetic Swift fixture.

Uses only Python's standard library; no checkpoint assets or inference runtime.
Equations follow Transformers' modeling_embedding_gemma2.py as read 2026-10-06.
Run from the repo root to regenerate the JSON fixture, or pass --check to verify it.
"""

import argparse
import json
import math
from pathlib import Path


CONFIG = {
    "model_type": "embedding_gemma2",
    "text_config": {
        "model_type": "embedding_gemma2_text", "hidden_size": 8,
        "intermediate_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "hidden_size_per_layer_input": 4, "embedding_dim": 8, "vocab_size": 16,
        "max_position_embeddings": 8192, "rms_norm_eps": 1e-6,
        "sliding_window": 1, "layer_types": ["sliding_attention", "full_attention"],
        "per_layer_config": {"01": {"head_dim": 8, "num_key_value_heads": 1}},
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
        "hidden_activation": "gelu_pytorch_tanh", "attention_bias": False,
        "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
    },
}


def parameter(name, size):
    if name.endswith("layer_scalar"):
        return [0.875] * size
    if "norm" in name:
        return [1 + (i % 5 - 2) * 0.03125 for i in range(size)]
    phase = sum(name.encode("utf-8")) % 7
    return [(i % 13 - 6) * 0.03125 + phase * 0.0078125 for i in range(size)]


def linear(x, name, output_size):
    weights = parameter(name + ".weight", len(x) * output_size)
    return [sum(a * b for a, b in zip(x, weights[i * len(x):(i + 1) * len(x)]))
            for i in range(output_size)]


def norm(x, name=None):
    scale = (sum(a * a for a in x) / len(x) + 1e-6) ** -0.5
    weights = parameter(name + ".weight", len(x)) if name else [1] * len(x)
    return [a * scale * b for a, b in zip(x, weights)]


def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def add(x, y):
    return [a + b for a, b in zip(x, y)]


def rope(x, position, base):
    half = len(x) // 2
    result = list(x)
    for i in range(half):
        angle = position * base ** (-2 * i / len(x))
        c, s = math.cos(angle), math.sin(angle)
        result[i] = x[i] * c - x[i + half] * s
        result[i + half] = x[i + half] * c + x[i] * s
    return result


def encode(ids, valid):
    embeddings = parameter("embed_tokens.weight", 16 * 8)
    hidden = [[a * math.sqrt(8) for a in embeddings[token * 8:(token + 1) * 8]] for token in ids]
    ple = []
    for row in hidden:
        projected = [a / math.sqrt(8) for a in linear(row, "ple.per_layer_model_projection", 8)]
        ple.append([norm(projected[i * 4:(i + 1) * 4], "ple.per_layer_projection_norm") for i in range(2)])
    for layer in range(2):
        prefix = f"layers.{layer}"
        attn = prefix + ".self_attn"
        head_dim = 4 if layer == 0 else 8
        base = 10000 if layer == 0 else 1000000
        queries, keys, values = [], [], []
        for position, row in enumerate(hidden):
            x = norm(row, prefix + ".input_layernorm")
            q = linear(x, attn + ".q_proj", 2 * head_dim)
            queries.append([rope(norm(q[i * head_dim:(i + 1) * head_dim], attn + ".q_norm"), position, base) for i in range(2)])
            keys.append(rope(norm(linear(x, attn + ".k_proj", head_dim), attn + ".k_norm"), position, base))
            values.append(norm(linear(x, attn + ".v_proj", head_dim)))
        next_hidden = []
        for position, row in enumerate(hidden):
            attended = []
            for head in range(2):
                indices = [j for j in range(len(ids)) if valid[j] and (layer == 1 or abs(position - j) <= 1)]
                scores = [sum(a * b for a, b in zip(queries[position][head], keys[j])) for j in indices]
                largest = max(scores)
                probs = [math.exp(a - largest) for a in scores]
                total = sum(probs)
                attended += [sum(probs[k] / total * values[j][d] for k, j in enumerate(indices)) for d in range(head_dim)]
            x = add(row, norm(linear(attended, attn + ".o_proj", 8), prefix + ".post_attention_layernorm"))
            mlp_input = norm(x, prefix + ".pre_feedforward_layernorm")
            gate = linear(mlp_input, prefix + ".mlp.gate_proj", 16)
            up = linear(mlp_input, prefix + ".mlp.up_proj", 16)
            mlp = linear([gelu(a) * b for a, b in zip(gate, up)], prefix + ".mlp.down_proj", 8)
            x = add(x, norm(mlp, prefix + ".post_feedforward_layernorm"))
            gate = linear(x, prefix + ".ple_block.per_layer_input_gate", 4)
            projected = linear([gelu(a) * b for a, b in zip(gate, ple[position][layer])], prefix + ".ple_block.per_layer_projection", 8)
            x = add(x, norm(projected, prefix + ".ple_block.post_per_layer_input_norm"))
            next_hidden.append([a * 0.875 for a in x])
        hidden = next_hidden
    return [linear(norm(row, "norm"), "embedding_projection", 8) for row in hidden]


def fixture():
    ids = [[2, 4, 6, 1, 0], [2, 5, 7, 8, 1]]
    mask = [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    outputs = [encode(row, valid) for row, valid in zip(ids, mask)]
    return {"config": CONFIG, "input_ids": ids, "attention_mask": mask, "token_embeddings": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "Tests/MereRunCoreTests/Fixtures/EmbeddingGemma2/synthetic.json"
    text = json.dumps(fixture(), indent=2) + "\n"
    if args.check:
        if path.read_text() != text:
            raise SystemExit("EmbeddingGemma 2 reference fixture differs; regenerate and review it.")
        print("EmbeddingGemma 2 scalar reference fixture matches.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(path)

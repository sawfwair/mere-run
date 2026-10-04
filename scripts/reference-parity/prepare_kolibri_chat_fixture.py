#!/usr/bin/env python3
"""Prepare small synthetic mixed-quantized weights with the real chat tokenizer.

This checks runtime integration, not model quality. Run on the conversion worker;
only the resulting approximately 40 MB fixture is needed on the test machine.
"""
import argparse
import json
from pathlib import Path
import shutil
import mlx.core as mx


def prepare(fixture, tokenizer_source, output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((fixture / "config.json").read_text())
    source_config = json.loads((tokenizer_source / "config.json").read_text())
    config.update(vocab_size=source_config["vocab_size"], eos_token_id=source_config["eos_token_id"],
                  max_position_embeddings=4096)
    weights = mx.load(str(fixture / "weights.safetensors"))
    weights = {key: value if key.endswith("e_score_correction_bias") else value.astype(mx.bfloat16)
               for key, value in weights.items()}
    mx.random.seed(29)
    for key in ["model.embed_tokens.weight", "lm_head.weight"]:
        weights[key] = (mx.random.normal((config["vocab_size"], config["hidden_size"])) * .02).astype(mx.bfloat16)
    policies = {}
    for key in list(weights):
        if not key.endswith("_proj.weight"): continue
        path = key.removesuffix(".weight")
        bits = 2 if ".experts." in path else 8
        policies[path] = dict(bits=bits, group_size=32)
        q, s, b = mx.quantize(weights[key], group_size=32, bits=bits)
        weights[key], weights[path + ".scales"], weights[path + ".biases"] = q, s, b
    config["mererun_quantization"] = policies
    mx.save_safetensors(str(output / "weights.safetensors"), weights)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "weights.safetensors" for key in weights}}))
    for name in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        shutil.copy2(tokenizer_source / name, output / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["fixture", "tokenizer-source", "output"]: parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.fixture, args.tokenizer_source, args.output)

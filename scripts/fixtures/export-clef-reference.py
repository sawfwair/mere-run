#!/usr/bin/env python3
"""Export a tiny Clef head and schema fixture from the pinned upstream MLX loader.

Reference tooling only; inference is entirely Swift. Requires mlx and numpy.
Download clef_mlx.py from mlx-community/clef-4bit revision
e0a23bd4406c15075b7473616429c46f3fd130a9 and pass its path as --reference.
No learned checkpoint weights are bundled.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

REFERENCE_SHA256 = "f1abbe542ee3e98764d71fd5fc8489e794db5619abe50c32160ec863e1e0e7a2"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, help="Optional pinned checkpoint tokenizer for a real-vocabulary schema fixture")
    args = parser.parse_args()
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("The reference loader does not match the pinned revision")
    spec = importlib.util.spec_from_file_location("clef_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(231)
    config = dict(hidden_size=8, width=8, routing_layers=2, layers=2, heads=2, feedforward=12)
    head = reference.JointSchemaHead(**config)
    weights = {}
    for key, initial in tree_flatten(head.parameters()):
        values = rng.normal(0, 0.15, initial.shape).astype(np.float32)
        if "norm" in key and key.endswith(".weight"):
            values += 1
        weights[key] = mx.array(values)
    head.load_weights(list(weights.items()), strict=True)
    tokenizer = lambda text, **kwargs: SimpleNamespace(input_ids=list(text.encode("utf-8")))
    request = {
        "state": {"z": "Checkout outage café", "a": [True, 2, None]},
        "questions": {
            "z_route": {"type": "choice", "instructions": "Select the team", "criteria": {
                "technical": "An outage", "billing": "An invoice", "other": None}},
            "urgency": {"type": "score", "criteria": ["Low", "Medium", "High", "Critical"]},
            "a_outage": {"type": "noul", "instructions": "Is the system down?"},
        },
        "max_tokens": 2048,
    }
    encoded = reference.encode_record(tokenizer, request, max_length=request["max_tokens"])
    ids = mx.array(encoded.input_ids)
    hidden = mx.array(rng.normal(0, 1, (len(encoded.input_ids), 8)).astype(np.float32))
    lexical = mx.array(rng.normal(0, 1, (256, 8)).astype(np.float32))
    logits = head(hidden, ids, encoded, lambda token_ids: lexical[token_ids])
    original = {
        key.replace(".feedforward.fc1.", ".feedforward.0.")
        .replace(".feedforward.fc2.", ".feedforward.3.")
        .replace("scorer1.", "residual_scorer.0.")
        .replace("scorer2.", "residual_scorer.3."): value
        for key, value in weights.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output / "head.safetensors"), original)
    expected = {"hidden": hidden, "lexical": lexical}
    expected.update({f"logits_{index}": value for index, value in enumerate(logits)})
    mx.save_safetensors(str(args.output / "reference.safetensors"), expected)
    (args.output / "joint_head_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    spans = [{"id": q.question_id, "type": q.question_type, "question_span": list(q.question_span),
              "option_spans": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
             for q in encoded.questions]
    (args.output / "sequence.json").write_text(json.dumps({"ids": list(encoded.input_ids), "fields": spans}) + "\n")
    if args.tokenizer_root:
        from tokenizers import Tokenizer
        tokenizer_file = args.tokenizer_root / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        wrapper = lambda text, **kwargs: SimpleNamespace(input_ids=tokenizer.encode(text, add_special_tokens=False).ids)
        real = reference.encode_record(wrapper, request, max_length=request["max_tokens"])
        real_spans = [{"id": q.question_id, "type": q.question_type, "question_span": list(q.question_span),
                       "option_spans": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
                      for q in real.questions]
        (args.output / "sequence_qwen.json").write_text(json.dumps({
            "ids": list(real.input_ids), "fields": real_spans,
            "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        }) + "\n")
    (args.output / "provenance.json").write_text(json.dumps({
        "repository": "mlx-community/clef-4bit", "revision": "e0a23bd4406c15075b7473616429c46f3fd130a9",
        "reference_sha256": REFERENCE_SHA256, "seed": 231,
        "mlx_version": mx.__version__, "dtype": "float32", "learned_weights": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

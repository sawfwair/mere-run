#!/usr/bin/env python3
"""Audit packed checkpoint headers against the pinned original logical schemas."""
import argparse
import json
from pathlib import Path
import struct

from convert_qwen_image21_turbo_mlx import bits_for, REVISION


def audit(root, schemas):
    counts = {}
    for component, shapes in schemas.items():
        actual = {}
        owners = {}
        for path in sorted((root / component).glob("*.safetensors")):
            with path.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(size))
            for key, tensor in header.items():
                if key == "__metadata__":
                    continue
                if key in actual:
                    raise ValueError("Duplicate tensor: " + key)
                actual[key] = tensor
                owners[key] = path.name
        expected = {}
        packed_count = 0
        for key, shape in shapes.items():
            if component == "text_encoder" and key == "lm_head.weight":
                continue
            bits = bits_for(component, key, shape)
            if bits:
                prefix = key.removesuffix(".weight")
                expected[key] = ("U32", [shape[0], shape[1] * bits // 32])
                for suffix in (".scales", ".biases"):
                    expected[prefix + suffix] = ("BF16", [shape[0], shape[1] // 64])
                packed_count += 1
            else:
                expected[key] = ("BF16", shape)
        if actual.keys() != expected.keys():
            raise ValueError("Tensor set mismatch: " + component)
        for key, (dtype, shape) in expected.items():
            if actual[key]["dtype"] != dtype or actual[key]["shape"] != shape:
                raise ValueError("Tensor dtype/shape mismatch: " + component + "/" + key)
        for path in (root / component).glob("*.safetensors.index.json"):
            if json.loads(path.read_text())["weight_map"] != owners:
                raise ValueError("Index ownership mismatch: " + component)
        counts[component] = {"tensors": len(actual), "quantized_layers": packed_count}
    return {"source_revision": REVISION, "header_audit_passed": True, "components": counts,
            "scope": "Tensor closure, packed dtype/shapes, unchanged dense precision and shard ownership; no image-quality claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--schemas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.artifact, json.loads(args.schemas.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

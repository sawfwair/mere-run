"""Export Falcon normalization values from pinned PyTorch reference operations.

Requires torch==2.13.0 and einops==0.8.1. Reads the exact upstream source supplied
with --reference; it does not download or load model weights.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import einops as E
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[2]
    / "Tests/MereRunCoreTests/Fixtures/falcon-functional-normalization.json",
)
args = parser.parse_args()
source = args.reference
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert torch.__version__.split("+")[0] == "2.13.0", "Use the pinned PyTorch version."
assert E.__version__ == "0.8.1", "Use the pinned einops version."
assert source_hash == (
    "987b371068e6120e5f08458b37dec638d88e1ecb330a6661eeaeba6dd4a27552"
), "Reference source identity mismatch."

nodes = []
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "repeat_kv":
        nodes.append(node)
    if isinstance(node, ast.ClassDef) and node.name == "Attention":
        node.body = [
            method for method in node.body
            if isinstance(method, ast.FunctionDef)
            and method.name in ["__init__", "_pre_attention_qkv"]
        ]
        nodes.append(node)
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *nodes,
    ],
    type_ignores=[],
)
namespace = {"torch": torch, "nn": torch.nn, "F": F, "E": E, "T": torch.Tensor}
exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)

config = SimpleNamespace(
    n_kv_heads=1, n_heads=2, head_dim=4, dim=4, norm_eps=0.1
)
attention = namespace["Attention"](config, 0)
inputs = torch.tensor(
    [[[-0.001, 0.002, -0.003, 0.004], [0.1, -0.2, 0.3, -0.4]]],
    dtype=torch.float32,
)
wqkv = (torch.arange(64, dtype=torch.float32).reshape(16, 4) - 31) * 0.003
attention.wqkv.weight.data.copy_(wqkv)
queries, keys, _ = attention._pre_attention_qkv(inputs)

# Native loading reorders interleaved upstream rows into gate rows then up rows.
w13 = (torch.arange(24, dtype=torch.float32).reshape(6, 4) - 11) * 0.03
w2 = (torch.arange(12, dtype=torch.float32).reshape(4, 3) - 5) * 0.02
normalized = F.rms_norm(inputs, (4,))
packed = F.linear(normalized, w13)
mlp = F.linear(packed[..., 0::2].relu().square() * packed[..., 1::2], w2)
fixture = {
    "provenance": {
        "torchVersion": torch.__version__,
        "upstreamRevision": "54916b3dec58565fafc6d82eb3051fe7246ab666",
        "sourceSha256": source_hash,
        "scope": (
            "Attention constructor and pre-attention method extracted unchanged by AST. "
            "Feed-forward interleaved-gate equation evaluated with PyTorch, replacing "
            "the Triton implementation. Final normalization uses explicit config "
            "epsilon. No full-reference inference claim."
        ),
    },
    "input": inputs.flatten().tolist(),
    "wqkv": wqkv.flatten().tolist(),
    "w13": torch.cat([w13[0::2], w13[1::2]]).flatten().tolist(),
    "w2": w2.flatten().tolist(),
    "normalized": normalized.flatten().tolist(),
    "queries": queries.permute(0, 2, 1, 3).flatten().tolist(),
    "keys": keys.permute(0, 2, 1, 3).flatten().tolist(),
    "mlp": mlp.flatten().tolist(),
    "finalNorm": F.rms_norm(inputs, (4,), eps=0.1).flatten().tolist(),
}
args.output.write_text(json.dumps(fixture, indent=2) + "\n")
print(json.dumps(fixture["provenance"]))

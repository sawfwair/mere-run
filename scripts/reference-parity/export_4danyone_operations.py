#!/usr/bin/env python3
"""Freeze small 4DAnyone operation cases and replay them on PyTorch CPU or CUDA.

Use --checkpoint once to extract trained weights. The resulting cases file
can be replayed with --cases on a CUDA host without transferring the model.
Dependencies: torch==2.8.0 numpy==1.26.4 safetensors==0.6.2 packaging==25.0.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

CHECKPOINT_SHA256 = "aff60b0db2d333bd9e960a9cf3333cc8dd40fe76614a22f75a0da72be4e8289f"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint", type=Path)
    inputs.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--activation-fixture", type=Path)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(8)
    torch.manual_seed(4196)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; no CPU fallback is allowed.")
    if args.checkpoint:
        if digest(args.checkpoint) != CHECKPOINT_SHA256:
            raise SystemExit("Released checkpoint checksum mismatch")
        with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
            cases = {
                "linear.weight": checkpoint.get_tensor("blocks.0.self_attn.q.weight").clone(),
                "linear.bias": checkpoint.get_tensor("blocks.0.self_attn.q.bias").clone(),
                "rms.weight": checkpoint.get_tensor("blocks.0.self_attn.norm_q.weight").clone(),
                "layer.weight": checkpoint.get_tensor("blocks.0.norm3.weight").clone(),
                "layer.bias": checkpoint.get_tensor("blocks.0.norm3.bias").clone(),
            }
        cases["activation.input"] = torch.linspace(-12, 12, 16384).bfloat16()
        cases["linear.input"] = torch.randn(5, 12, 3072).bfloat16()
        cases["norm.input"] = (torch.randn(5, 12, 3072) * 3).bfloat16()
        for name in ["query", "key", "value"]:
            cases["attention." + name] = torch.randn(5, 24, 12, 128).bfloat16()
        args.cases = args.output / "operations-cases.safetensors"
        save_file(cases, args.cases, metadata={"checkpoint_sha256": CHECKPOINT_SHA256, "seed": "4196"})
    else:
        cases = load_file(args.cases)

    frozen_hash = digest(args.cases)
    cases = {key: value.to(args.device) for key, value in cases.items()}
    result = {}
    started = time.monotonic()
    with torch.inference_mode():
        x = cases["activation.input"]
        result["gelu"] = functional.gelu(x, approximate="tanh")
        result["silu"] = functional.silu(x)
        result["linear"] = functional.linear(cases["linear.input"], cases["linear.weight"], cases["linear.bias"])
        x = cases["norm.input"]
        result["layer.bf16"] = functional.layer_norm(
            x, (3072,), cases["layer.weight"], cases["layer.bias"], 1e-6)
        result["layer.fp32"] = functional.layer_norm(
            x.float(), (3072,), cases["layer.weight"].float(), cases["layer.bias"].float(), 1e-6)
        # Exact arithmetic order of the pinned upstream RMSNorm.forward.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        result["rms"] = normalized.bfloat16() * cases["rms.weight"]
        result["attention"] = functional.scaled_dot_product_attention(
            cases["attention.query"], cases["attention.key"], cases["attention.value"])
        if args.device == "cuda":
            torch.cuda.synchronize()
    result = {key: value.detach().cpu().contiguous() for key, value in result.items()}
    reference = args.output / ("operations-reference-" + args.device + ".safetensors")
    save_file(result, reference)
    receipt = dict(
        torchVersion=torch.__version__, device=args.device, dtype="bfloat16",
        casesSHA256=frozen_hash, referenceSHA256=digest(reference), elapsedSeconds=time.monotonic() - started,
        scope="Primitive replay on identical frozen inputs; not full-model or CUDA-autocast graph parity.",
    )
    if args.device == "cuda":
        receipt.update(gpu=torch.cuda.get_device_name(), cudaVersion=torch.version.cuda)
    reference.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.activation_fixture:
        args.activation_fixture.parent.mkdir(parents=True, exist_ok=True)
        save_file({"input": cases["activation.input"].cpu(), "gelu": result["gelu"], "silu": result["silu"]},
                  args.activation_fixture)
        fixture_receipt = dict(receipt, fixtureSHA256=digest(args.activation_fixture))
        args.activation_fixture.with_suffix(".json").write_text(json.dumps(fixture_receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a real DreamX camera-attention block fixture from the released weights."""

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import types


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dreamx_inference_utils(dreamx_root):
    package_name = "dreamx_block_fixture_utils"
    package = types.ModuleType(package_name)
    package.__path__ = [str(dreamx_root / "utils")]
    sys.modules[package_name] = package
    load_source_module(
        f"{package_name}.pose_utils",
        dreamx_root / "utils" / "pose_utils.py",
    )
    return load_source_module(
        f"{package_name}.inference_utils",
        dreamx_root / "utils" / "inference_utils.py",
    )


def rms_norm(value, weight, epsilon=1e-6):
    import torch

    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return normalized.to(value.dtype) * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamx-root", type=Path, required=True)
    parser.add_argument("--camera-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open
    from safetensors.torch import save_file

    inference = load_dreamx_inference_utils(args.dreamx_root)
    prope = load_source_module(
        "dreamx_camera_prope",
        args.dreamx_root / "wan" / "modules" / "camera_prope.py",
    )
    pixel_frame_count = 17
    poses = inference.ActionToPoseFromID(
        ["j"],
        [3],
        duration=pixel_frame_count,
    )
    camera, _ = inference.GetPoseEmbedsFromPosesPrope(
        poses[:pixel_frame_count],
        h=704,
        w=1280,
        target_length=pixel_frame_count,
        dtype=torch.float32,
    )
    views = camera["viewmats"].unsqueeze(0)
    intrinsics = camera["K"].unsqueeze(0)

    prefix = f"blocks.{args.block}.cam_self_attn."
    with safe_open(args.camera_weights, framework="pt", device="cpu") as source:
        weights = {key.removeprefix(prefix): source.get_tensor(key).float() for key in source.keys() if key.startswith(prefix)}
    if len(weights) != 10:
        raise RuntimeError(f"Expected 10 camera tensors for {prefix}, found {len(weights)}")

    sequence = views.shape[1] * 2
    dimensions = weights["q_proj.weight"].shape[1]
    heads = 24
    head_dimension = dimensions // heads
    values = torch.arange(sequence * dimensions, dtype=torch.float32).reshape(1, sequence, dimensions)
    model_input = torch.sin(values / 113) * 0.25 + torch.cos(values / 47) * 0.125

    query = rms_norm(functional.linear(model_input, weights["q_proj.weight"], weights["q_proj.bias"]), weights["norm_q.weight"])
    key = rms_norm(functional.linear(model_input, weights["k_proj.weight"], weights["k_proj.bias"]), weights["norm_k.weight"])
    value = functional.linear(model_input, weights["v_proj.weight"], weights["v_proj.bias"])
    query = query.reshape(1, sequence, heads, head_dimension).transpose(1, 2)
    key = key.reshape(1, sequence, heads, head_dimension).transpose(1, 2)
    value = value.reshape(1, sequence, heads, head_dimension).transpose(1, 2)
    query, key, value, apply_output = prope.prope_qkv(
        query,
        key,
        value,
        viewmats=views,
        Ks=intrinsics,
    )
    attended = functional.scaled_dot_product_attention(query, key, value)
    attended = apply_output(attended).transpose(1, 2).reshape(1, sequence, dimensions)
    output = functional.linear(attended, weights["out_proj.weight"], weights["out_proj.bias"])
    tensors = {
        "input": model_input.contiguous(),
        "view_matrices": views.contiguous(),
        "intrinsics": intrinsics.contiguous(),
        "query": query.contiguous(),
        "key": key.contiguous(),
        "value": value.contiguous(),
        "output": output.contiguous(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={
        "source": "GD-ML/DreamX-World-5B-Cam",
        "revision": "a4379c7723f6ebd02139e2e8fd62d6ef523e86e3",
        "block": str(args.block),
        "attention_scale": str(1 / math.sqrt(head_dimension)),
    })
    print(args.output)


if __name__ == "__main__":
    main()

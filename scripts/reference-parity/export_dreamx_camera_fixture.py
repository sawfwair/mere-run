#!/usr/bin/env python3
"""Export DreamX trajectory and PRoPE tensors for the Swift/MLX parity tests."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types


def tensor_payload(tensor):
    value = tensor.detach().cpu().float().contiguous()
    return {"shape": list(value.shape), "values": value.flatten().tolist()}


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dreamx_inference_utils(dreamx_root):
    package_name = "dreamx_fixture_utils"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    inference_module = load_dreamx_inference_utils(args.dreamx_root)
    prope_module = load_source_module(
        "dreamx_camera_prope",
        args.dreamx_root / "wan" / "modules" / "camera_prope.py",
    )

    torch.manual_seed(0)
    segments = [("w", 3), ("j", 5)]
    pixel_frame_count = 17
    duration = -(-pixel_frame_count // len(segments))
    poses = inference_module.ActionToPoseFromID(
        [action for action, _ in segments],
        [speed for _, speed in segments],
        duration=duration,
    )
    camera, _ = inference_module.GetPoseEmbedsFromPosesPrope(
        poses[:pixel_frame_count],
        h=704,
        w=1280,
        target_length=pixel_frame_count,
        dtype=torch.float32,
    )
    view_matrices = camera["viewmats"].unsqueeze(0)
    intrinsics = camera["K"].unsqueeze(0)
    frame_count = view_matrices.shape[1]
    sequence = frame_count * 2
    heads = 2
    dimension = 8
    total = heads * sequence * dimension
    query = torch.arange(total, dtype=torch.float32).reshape(1, heads, sequence, dimension) / 101
    key = (torch.arange(total, dtype=torch.float32) + 7).reshape(1, heads, sequence, dimension) / 97
    value = (torch.arange(total, dtype=torch.float32) - 11).reshape(1, heads, sequence, dimension) / 89
    output_input = (torch.arange(total, dtype=torch.float32) + 3).reshape(1, heads, sequence, dimension) / 83
    encoded_query, encoded_key, encoded_value, apply_output = prope_module.prope_qkv(
        query,
        key,
        value,
        viewmats=view_matrices,
        Ks=intrinsics,
    )
    output = {
        "source_revision": "AMAP-ML/DreamX-World@a1f4c6e",
        "trajectory": {
            "segments": segments,
            "pixel_frame_count": pixel_frame_count,
        },
        "view_matrices": tensor_payload(view_matrices),
        "intrinsics": tensor_payload(intrinsics),
        "query_input": tensor_payload(query),
        "query_output": tensor_payload(encoded_query),
        "key_input": tensor_payload(key),
        "key_output": tensor_payload(encoded_key),
        "value_input": tensor_payload(value),
        "value_output": tensor_payload(encoded_value),
        "output_input": tensor_payload(output_input),
        "output_output": tensor_payload(apply_output(output_input)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

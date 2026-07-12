#!/usr/bin/env python3
"""Export a tiny full-transformer DreamX-5B-Cam BF16 reference fixture."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dreamx_transformer(dreamx_root):
    package_name = "dreamx_fixture_models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(dreamx_root / "models")]
    sys.modules[package_name] = package
    load_source_module(
        f"{package_name}.attention_utils",
        dreamx_root / "models" / "attention_utils.py",
    )
    load_source_module(
        f"{package_name}.prope_utils",
        dreamx_root / "models" / "prope_utils.py",
    )
    distributed = types.ModuleType("dist")
    distributed.get_sequence_parallel_rank = lambda: 0
    distributed.get_sequence_parallel_world_size = lambda: 1
    distributed.get_sp_group = lambda: None
    distributed.usp_attn_forward = None
    distributed.sp_prope_forward = None
    sys.modules["dist"] = distributed
    return load_source_module(
        f"{package_name}.wan_transformer3d",
        dreamx_root / "models" / "wan_transformer3d.py",
    )


def load_dreamx_inference_utils(dreamx_root):
    package_name = "dreamx_transformer_fixture_utils"
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


def upstream_base_tensor(key, value):
    if key == "patch_embedding_proj.weight":
        return "patch_embedding.weight", value.reshape(3072, 48, 1, 2, 2)
    exact = {
        "patch_embedding_proj.bias": "patch_embedding.bias",
        "text_embedding_0.weight": "text_embedding.0.weight",
        "text_embedding_0.bias": "text_embedding.0.bias",
        "text_embedding_1.weight": "text_embedding.2.weight",
        "text_embedding_1.bias": "text_embedding.2.bias",
        "time_embedding_0.weight": "time_embedding.0.weight",
        "time_embedding_0.bias": "time_embedding.0.bias",
        "time_embedding_1.weight": "time_embedding.2.weight",
        "time_embedding_1.bias": "time_embedding.2.bias",
        "time_projection.weight": "time_projection.1.weight",
        "time_projection.bias": "time_projection.1.bias",
    }
    if key in exact:
        return exact[key], value
    if ".ffn.fc1." in key:
        return key.replace(".ffn.fc1.", ".ffn.0."), value
    if ".ffn.fc2." in key:
        return key.replace(".ffn.fc2.", ".ffn.2."), value
    return key, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamx-root", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--camera-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    transformer_module = load_dreamx_transformer(args.dreamx_root)
    inference = load_dreamx_inference_utils(args.dreamx_root)
    with torch.device("meta"):
        model = transformer_module.Wan2_2Transformer3DModel(
            model_type="ti2v",
            in_dim=48,
            dim=3072,
            ffn_dim=14336,
            freq_dim=256,
            text_dim=4096,
            out_dim=48,
            num_heads=24,
            num_layers=30,
            cross_attn_norm=True,
            add_control_adapter=True,
            cam_method="prope",
        )

    state = {}
    with safe_open(args.base_weights, framework="pt", device="cpu") as source:
        for key in source.keys():
            mapped_key, value = upstream_base_tensor(key, source.get_tensor(key))
            state[mapped_key] = value
    with safe_open(args.camera_weights, framework="pt", device="cpu") as source:
        for key in source.keys():
            state[key] = source.get_tensor(key).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    del state
    head_dimension = model.dim // model.num_heads
    model.freqs = torch.cat([
        transformer_module.rope_params(1024, head_dimension - 4 * (head_dimension // 6)),
        transformer_module.rope_params(1024, 2 * (head_dimension // 6)),
        transformer_module.rope_params(1024, 2 * (head_dimension // 6)),
    ], dim=1)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm) and module.elementwise_affine:
            module.weight.data = module.weight.data.float()
            module.bias.data = module.bias.data.float()
    model.eval()

    pixel_frame_count = 17
    poses = inference.ActionToPoseFromID(["j"], [3], duration=pixel_frame_count)
    camera, _ = inference.GetPoseEmbedsFromPosesPrope(
        poses[:pixel_frame_count],
        h=704,
        w=1280,
        target_length=pixel_frame_count,
        dtype=torch.bfloat16,
    )
    sequence = 10
    latent_values = torch.arange(48 * 5 * 2 * 4, dtype=torch.float32).reshape(48, 5, 2, 4)
    latent = (torch.sin(latent_values / 79) * 0.25).to(torch.bfloat16)
    context_values = torch.arange(4 * 4096, dtype=torch.float32).reshape(4, 4096)
    context = (torch.cos(context_values / 113) * 0.125).to(torch.bfloat16)
    timesteps = torch.full((1, sequence), 1000, dtype=torch.float32)
    timesteps[:, :2] = 0
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(
            x=[latent],
            t=timesteps,
            context=[context],
            seq_len=sequence,
            y_camera={
                "viewmats": camera["viewmats"].unsqueeze(0),
                "K": camera["K"].unsqueeze(0),
            },
        )[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({
        "input": latent.contiguous(),
        "timesteps": timesteps.contiguous(),
        "context": context.contiguous(),
        "view_matrices": camera["viewmats"].unsqueeze(0).contiguous(),
        "intrinsics": camera["K"].unsqueeze(0).contiguous(),
        "output": output.float().contiguous(),
    }, args.output, metadata={
        "source": "AMAP-ML/DreamX-World",
        "source_revision": "f2bf6bf9ab716e9e9b5a27288abc5b9b97420adb",
        "camera_revision": "a4379c7723f6ebd02139e2e8fd62d6ef523e86e3",
        "compute_dtype": "bfloat16",
    })
    print(args.output)


if __name__ == "__main__":
    main()

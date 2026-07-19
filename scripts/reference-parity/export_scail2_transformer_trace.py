#!/usr/bin/env python3
"""Export a learned SCAIL-2 PyTorch transformer oracle.

The native bundle contains the exact pinned upstream weights in an MLX-friendly
key/layout. This tool can either reverse those mechanical changes or load the
pinned upstream checkpoint directly, then runs deterministic animation and
replacement forwards through 1 to 40 blocks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as functional
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file


CODE_REVISION = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
MODEL_REVISION = "150cc0ca4e98e50e60b9295dacde39442fdccab2"


class DeviceAMP:
    def __init__(self, device_type: str) -> None:
        self.device_type = device_type

    def autocast(self, dtype=torch.float16, enabled=True):
        if not enabled:
            return torch.autocast(device_type=self.device_type, enabled=False)
        if dtype == torch.float32:
            return torch.autocast(device_type=self.device_type, enabled=False)
        return torch.autocast(device_type=self.device_type, dtype=dtype)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream_model_module(root: Path):
    wan = types.ModuleType("wan")
    wan.__path__ = [str(root / "wan")]
    modules = types.ModuleType("wan.modules")
    modules.__path__ = [str(root / "wan/modules")]
    sys.modules["wan"] = wan
    sys.modules["wan.modules"] = modules
    return load_module("wan.modules.model_scail2", root / "wan/modules/model_scail2.py")


def verify_inputs(upstream: Path, model_root: Path) -> None:
    import subprocess

    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if code_revision != CODE_REVISION:
        raise RuntimeError(f"Expected code {CODE_REVISION}, found {code_revision}")
    provenance = json.loads((model_root / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("code_revision") != CODE_REVISION:
        raise RuntimeError("Native bundle code provenance does not match the pinned reference")
    if provenance.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("Native bundle model provenance does not match the pinned reference")


def reference_attention(q, k, v, **_):
    # The released CUDA path uses FlashAttention with no effective padding for
    # this single-sample fixture. PyTorch SDPA computes the same attention on
    # Apple Silicon and keeps the reference model executable without CUDA.
    output = functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous().to(q.dtype)


def reference_timestep_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
    # Upstream computes this small table in float64. MPS has no float64, so do
    # that exact operation on CPU and transfer the resulting float32 values.
    half = dim // 2
    cpu_position = position.detach().to(device="cpu").to(dtype=torch.float64)
    frequencies = torch.pow(
        10000,
        -torch.arange(half, dtype=torch.float64).div(half),
    )
    sinusoid = torch.outer(cpu_position, frequencies)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(
        device=position.device,
        dtype=torch.float32,
    )


def cpu_float64_bridge(function):
    """Run upstream's exact float64 RoPE body on CPU for an MPS oracle."""

    def bridged(input_tensor, *args, **kwargs):
        device = input_tensor.device
        cpu_args = [value.cpu() if isinstance(value, torch.Tensor) else value for value in args]
        cpu_kwargs = {
            key: value.cpu() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        return function(input_tensor.cpu(), *cpu_args, **cpu_kwargs).to(device)

    return bridged


def upstream_key(native_key: str) -> tuple[str, tuple[int, ...] | None]:
    mapped = native_key
    replacements = {
        "patch_embedding_proj.": "patch_embedding.",
        "patch_embedding_pose_proj.": "patch_embedding_pose.",
        "patch_embedding_mask_proj.": "patch_embedding_mask.",
        "text_embedding_0.": "text_embedding.0.",
        "text_embedding_1.": "text_embedding.2.",
        "time_embedding_0.": "time_embedding.0.",
        "time_embedding_1.": "time_embedding.2.",
        "time_projection.": "time_projection.1.",
        "img_emb.layer_0.": "img_emb.proj.0.",
        "img_emb.layer_1.": "img_emb.proj.1.",
        "img_emb.layer_3.": "img_emb.proj.3.",
        "img_emb.layer_4.": "img_emb.proj.4.",
    }
    for source, target in replacements.items():
        if source in mapped:
            mapped = mapped.replace(source, target)
            break
    mapped = re.sub(r"^(blocks\.0\.ffn)\.fc1\.", r"\1.0.", mapped)
    mapped = re.sub(r"^(blocks\.0\.ffn)\.fc2\.", r"\1.2.", mapped)
    shapes = {
        "patch_embedding.weight": (5120, 20, 1, 2, 2),
        "patch_embedding_pose.weight": (5120, 20, 1, 2, 2),
        "patch_embedding_mask.weight": (5120, 28, 1, 2, 2),
    }
    return mapped, shapes.get(mapped)


def selected_block(key: str, layer_count: int) -> bool:
    if not key.startswith("blocks."):
        return True
    return int(key.split(".", 2)[1]) < layer_count


def load_selected_state(model_root: Path, layer_count: int) -> dict[str, torch.Tensor]:
    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    selected = {
        key: filename
        for key, filename in index.items()
        if selected_block(key, layer_count)
    }
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in selected.items():
        by_file[filename].append(key)
    state: dict[str, torch.Tensor] = {}
    for filename, keys in by_file.items():
        with safe_open(model_root / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                target, shape = upstream_key(key)
                value = handle.get_tensor(key)
                state[target] = value.reshape(shape) if shape is not None else value
    return state


def load_source_selected_state(
    checkpoint: Path,
    upstream: Path,
    layer_count: int,
) -> dict[str, torch.Tensor]:
    converter = load_module("scail2_upstream_convert", upstream / "convert.py")
    safe_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        loaded = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    source = loaded["module"]
    state: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        for target, mapped in converter.get_new_mappings(key, value).items():
            if selected_block(target, layer_count):
                # Official inference wraps the whole forward in BF16 autocast,
                # then explicitly disables autocast for timestep projection,
                # adaptive modulation, and the output head. MPS does not cast
                # Conv3D/Linear operands through CUDA autocast, so materialize
                # the same operational dtypes before moving the oracle model.
                float32_island = (
                    target.startswith("time_embedding.")
                    or target.startswith("time_projection.")
                    or target.startswith("head.")
                    or target.endswith("modulation")
                )
                state[target] = mapped.to(
                    torch.float32 if float32_island else torch.bfloat16
                )
    return state


def deterministic(shape: tuple[int, ...], scale: float, phase: float) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    values = torch.arange(count, dtype=torch.float32)
    return torch.sin(values * scale + phase).reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--layers", type=int, default=1, choices=range(1, 41))
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    model_root = args.model_root.resolve()
    verify_inputs(upstream, model_root)
    module = load_upstream_model_module(upstream)
    module.flash_attention = reference_attention
    module.sinusoidal_embedding_1d = reference_timestep_embedding
    module.rope_apply_ref = cpu_float64_bridge(module.rope_apply_ref)
    module.rope_apply_video = cpu_float64_bridge(module.rope_apply_video)
    module.rope_apply_pose = cpu_float64_bridge(module.rope_apply_pose)

    with torch.device("meta"):
        model = module.SCAIL2Model(
            model_type="i2v",
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=20,
            mask_dim=28,
            dim=5120,
            ffn_dim=13824,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=40,
            num_layers=args.layers,
            cross_attn_norm=True,
            pose_rope_shift=[0, 0, 120],
            eps=1e-6,
        )
    if args.source_checkpoint is None:
        state = load_selected_state(model_root, args.layers)
        weight_source = "converted pinned weights"
    else:
        state = load_source_selected_state(
            args.source_checkpoint.resolve(),
            upstream,
            args.layers,
        )
        weight_source = "pinned upstream checkpoint"
    incompatibility = model.load_state_dict(state, strict=True, assign=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"State mismatch: {incompatibility}")
    device = torch.device(args.device)
    if args.source_checkpoint is not None:
        # The official model marks timestep/modulation/head islands with
        # `amp.autocast(dtype=float32)`. Route those CUDA-scoped contexts to
        # the selected reference device while preserving their FP32 behavior.
        module.amp = DeviceAMP(device.type)
    model.freqs = torch.cat(
        [
            module.rope_params(8192, 44),
            module.rope_params(8192, 42),
            module.rope_params(8192, 42),
        ],
        dim=1,
    ).to(device)
    model = model.eval().to(device)

    traces: dict[str, torch.Tensor] = {}
    inputs = {
        "video": deterministic((16, 1, 4, 4), 0.031, 0.1),
        "reference": deterministic((16, 1, 4, 4), 0.029, 0.2),
        "reference_mask": deterministic((28, 1, 4, 4), 0.023, 0.3).mul(0.5).add(0.5),
        "driving": deterministic((16, 1, 2, 2), 0.037, 0.4),
        "driving_mask": deterministic((28, 1, 2, 2), 0.019, 0.5).mul(0.5).add(0.5),
        "text": deterministic((1, 8, 4096), 0.0013, 0.6),
        "image": deterministic((1, 257, 1280), 0.0011, 0.7),
        "timestep": torch.tensor([999], dtype=torch.float32),
    }
    for key, value in inputs.items():
        traces[f"transformer_{key}"] = value.contiguous()
    typed = {key: value.to(device=device, dtype=torch.bfloat16) for key, value in inputs.items()}
    typed["timestep"] = inputs["timestep"].to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
    ):
        for name, replacement in (("animation", False), ("replacement", True)):
            captures: dict[str, list[torch.Tensor]] = defaultdict(list)
            handles = []
            for module_name in (
                "time_embedding",
                "time_projection",
                "text_embedding",
                "img_emb",
                "blocks.0.self_attn",
                "blocks.0.norm3",
                "blocks.0.cross_attn",
                "blocks.0.norm2",
                "blocks.0.ffn",
                "head",
            ):
                target = model.get_submodule(module_name)
                handles.append(target.register_forward_hook(
                    lambda _module, _inputs, output, key=module_name:
                        captures[key].append(output.detach())
                ))
            for block_index, block in enumerate(model.blocks):
                block_name = f"blocks.{block_index}"
                handles.append(block.register_forward_hook(
                    lambda _module, _inputs, output, key=block_name:
                        captures[key].append(output.detach())
                ))
                handles.append(block.register_forward_pre_hook(
                    lambda _module, inputs, key=block_name:
                        captures[f"{key}.input"].append(inputs[0].detach())
                ))
            for module_name in (
                "blocks.0.self_attn",
                "blocks.0.norm3",
                "blocks.0.norm2",
                "blocks.0.ffn",
            ):
                target = model.get_submodule(module_name)
                handles.append(target.register_forward_pre_hook(
                    lambda _module, inputs, key=module_name:
                        captures[f"{key}.input"].append(inputs[0].detach())
                ))
            output = model(
                x=[typed["video"]],
                pose_latents=[typed["driving"]],
                driving_masks=[typed["driving_mask"]],
                ref_latents=[typed["reference"]],
                ref_masks=[typed["reference_mask"]],
                t=typed["timestep"],
                context=[typed["text"].squeeze(0)],
                seq_len=9,
                replace_flag=replacement,
                clip_fea=typed["image"],
            )[0]
            for handle in handles:
                handle.remove()
            trace_names = {
                "time_embedding": "timestep_embedding",
                "time_projection": "modulation",
                "text_embedding": "text_conditioning",
                "img_emb": "image_conditioning",
                "blocks.0.self_attn.input": "self_input",
                "blocks.0.self_attn": "self_output",
                "blocks.0.norm3.input": "post_self",
                "blocks.0.norm3": "cross_input",
                "blocks.0.cross_attn": "cross_output",
                "blocks.0.norm2.input": "post_cross",
                "blocks.0.ffn.input": "ffn_input",
                "blocks.0.ffn": "ffn_output",
                "head": "head_patches",
            }
            for block_index in range(args.layers):
                trace_names[f"blocks.{block_index}.input"] = f"block_{block_index}_input"
                trace_names[f"blocks.{block_index}"] = f"block_{block_index}_output"
            trace_names["blocks.0.input"] = "assembled_tokens"
            for capture_name, trace_name in trace_names.items():
                traces[f"transformer_{name}_{trace_name}"] = (
                    captures[capture_name][0].float().cpu().contiguous()
                )
            traces[f"transformer_{name}_output"] = output.float().cpu().contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        traces,
        str(args.output),
        metadata={
            "code_revision": CODE_REVISION,
            "model_revision": MODEL_REVISION,
            "scope": f"official PyTorch {args.layers}-block forward from {weight_source}",
            "layers": str(args.layers),
        },
    )
    print(f"Wrote {len(traces)} tensors to {args.output}")


if __name__ == "__main__":
    main()

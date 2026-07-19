#!/usr/bin/env python3
"""Export deterministic pinned-upstream SCAIL-2 parity traces.

This script intentionally runs only the reference operations that do not need
the 14B checkpoint or CUDA: official-example preprocessing, mask packing,
CLIP preprocessing, mixed RoPE, and the exact Flow-UniPC trajectory.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import save_file


UPSTREAM_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_upstream(root: Path) -> None:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected upstream {UPSTREAM_COMMIT}, found {actual}")
    required = [
        root / "wan/utils/fm_solvers_unipc.py",
        root / "wan/utils/scail_utils.py",
        root / "wan/modules/model_scail2.py",
        root / "examples/animation_001/ref.jpg",
        root / "examples/animation_001/ref_mask.jpg",
        root / "examples/replace_001/ref.png",
        root / "examples/replace_001/ref_mask.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing pinned upstream files: {missing}")


def load_scail_utils(root: Path):
    # The utility module imports decord at module load even though the still
    # image and mask functions exercised here do not use it. Keep parity
    # tooling lightweight by supplying only that unused import surface.
    decord = types.ModuleType("decord")
    decord.VideoReader = object
    decord.bridge = types.SimpleNamespace(set_bridge=lambda _: None)
    sys.modules.setdefault("decord", decord)
    return load_module("scail2_upstream_utils", root / "wan/utils/scail_utils.py")


def load_rope_module(root: Path):
    # Load the pinned module without executing wan/__init__.py and its full
    # generation dependency graph. Relative imports still resolve normally.
    wan = types.ModuleType("wan")
    wan.__path__ = [str(root / "wan")]
    modules = types.ModuleType("wan.modules")
    modules.__path__ = [str(root / "wan/modules")]
    sys.modules["wan"] = wan
    sys.modules["wan.modules"] = modules
    return load_module("wan.modules.model_scail2", root / "wan/modules/model_scail2.py")


def pil_tensor(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    rgb = torch.from_numpy(np.asarray(image).copy())
    normalized = transforms.ToTensor()(image).mul(2).sub(1).unsqueeze(0)
    return rgb, normalized


def clip_preprocess(cropped: torch.Tensor) -> torch.Tensor:
    resized = functional.interpolate(
        cropped,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    ).mul(0.5).add(0.5)
    mean = resized.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = resized.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (resized - mean) / std


def export_example_preprocessing(
    traces: dict[str, torch.Tensor],
    root: Path,
    scail_utils,
    example: str,
    reference_name: str,
    mask_name: str,
) -> None:
    directory = root / "examples" / example
    source_rgb, source = pil_tensor(directory / reference_name)
    mask_rgb, source_mask = pil_tensor(directory / mask_name)
    target_height, target_width = 512, 896
    if (source.shape[2] < source.shape[3] and target_height > target_width) or (
        source.shape[2] > source.shape[3] and target_height < target_width
    ):
        target_height, target_width = target_width, target_height

    cropped = scail_utils.resize_for_rectangle_crop(
        source,
        (target_height, target_width),
        reshape_mode="center",
    )
    cropped_mask = scail_utils.resize_for_rectangle_crop(
        source_mask,
        (target_height, target_width),
        reshape_mode="center",
    )
    half = functional.interpolate(
        cropped,
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    )
    encoded_mask = scail_utils.extract_and_compress_mask_to_latent(
        cropped_mask.permute(1, 0, 2, 3),
        additional_spatial_downsample=1,
        temporal_compression_stride=4,
    )
    prefix = example.removesuffix("_001")
    traces[f"{prefix}_source_rgb_hwc_u8"] = source_rgb.contiguous()
    traces[f"{prefix}_mask_source_rgb_hwc_u8"] = mask_rgb.contiguous()
    traces[f"{prefix}_cropped_nhwc"] = cropped.permute(0, 2, 3, 1).contiguous()
    traces[f"{prefix}_mask_cropped_nhwc"] = cropped_mask.permute(0, 2, 3, 1).contiguous()
    traces[f"{prefix}_half_nhwc"] = half.permute(0, 2, 3, 1).contiguous()
    traces[f"{prefix}_mask_encoded_cthw"] = encoded_mask.contiguous()
    traces[f"{prefix}_clip_nhwc"] = clip_preprocess(cropped).permute(0, 2, 3, 1).contiguous()


def export_scheduler(traces: dict[str, torch.Tensor], root: Path, steps: int, shift: float) -> None:
    scheduler_module = load_module(
        "scail2_upstream_unipc",
        root / "wan/utils/fm_solvers_unipc.py",
    )
    scheduler = scheduler_module.FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device="cpu", shift=shift)
    traces["scheduler_timesteps"] = scheduler.timesteps.to(torch.float32).contiguous()
    traces["scheduler_sigmas"] = scheduler.sigmas.to(torch.float32).contiguous()

    values = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    sample = torch.cos(values * 0.17 + 0.31) * 0.4
    traces["scheduler_sample_00"] = sample.contiguous()
    for index, timestep in enumerate(scheduler.timesteps):
        model_output = (
            torch.sin(values * 0.11 + (index + 1) * 0.07) * 0.23
            + torch.cos(values * 0.03 - index * 0.13) * 0.05
        )
        traces[f"scheduler_model_output_{index:02d}"] = model_output.contiguous()
        sample = scheduler.step(
            model_output.unsqueeze(0) if model_output.ndim == 3 else model_output,
            timestep,
            sample,
            return_dict=False,
        )[0]
        traces[f"scheduler_sample_{index + 1:02d}"] = sample.contiguous()


def make_rope_frequencies(rope_module, maximum: int, head_dimension: int) -> torch.Tensor:
    split = head_dimension // 6
    return torch.cat(
        [
            rope_module.rope_params(maximum, head_dimension - 4 * split),
            rope_module.rope_params(maximum, 2 * split),
            rope_module.rope_params(maximum, 2 * split),
        ],
        dim=1,
    )


def export_rope(traces: dict[str, torch.Tensor], root: Path) -> None:
    rope_module = load_rope_module(root)
    frames, height, width = 3, 4, 4
    additional_count = 1
    head_dimension = 12
    heads = 2
    frequencies = make_rope_frequencies(rope_module, 256, head_dimension)
    additional_length = additional_count * height * width
    reference_length = height * width
    video_length = frames * height * width
    pose_length = frames * (height // 2) * (width // 2)
    total_length = additional_length + reference_length + video_length + pose_length
    values = torch.arange(total_length * heads * head_dimension, dtype=torch.float32)
    input_tensor = torch.sin(values * 0.007 + 0.19).reshape(
        1, total_length, heads, head_dimension
    )
    traces["rope_frequencies_real"] = frequencies.real.to(torch.float32).contiguous()
    traces["rope_frequencies_imag"] = frequencies.imag.to(torch.float32).contiguous()
    traces["rope_input"] = input_tensor.contiguous()
    common = {
        "freqs": frequencies,
        "ref_length": reference_length,
        "seq_length": video_length,
        "pose_length": pose_length,
        "additional_ref_length": additional_length,
        "rope_T": frames,
        "rope_H": height,
        "rope_W": width,
        "rope_ref_T": {"ref": 1, "additional_ref": additional_count},
        "hidden_size_head": head_dimension,
    }
    for mode, replacement in (("animation", False), ("replacement", True)):
        base_video_shift = 0 if replacement else 1
        kwargs = dict(common)
        kwargs["rope_T_shift"] = {
            "additional_ref": 0,
            "ref": additional_count,
            "pose": base_video_shift + additional_count,
            "video": base_video_shift + additional_count,
        }
        kwargs["rope_H_shift"] = {
            "ref": 120 if replacement else 0,
            "additional_ref": 120 if replacement else 0,
            "pose": 0,
            "video": 0,
        }
        kwargs["rope_W_shift"] = {
            "ref": 0,
            "additional_ref": 0,
            "pose": 120,
            "video": 0,
        }
        traces[f"rope_{mode}_output"] = rope_module.rope_apply_scail(
            input_tensor,
            **kwargs,
        ).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--shift", type=float, default=3.0)
    args = parser.parse_args()
    root = args.upstream.resolve()
    verify_upstream(root)
    traces: dict[str, torch.Tensor] = {}
    scail_utils = load_scail_utils(root)
    export_example_preprocessing(
        traces, root, scail_utils, "animation_001", "ref.jpg", "ref_mask.jpg"
    )
    export_example_preprocessing(
        traces, root, scail_utils, "replace_001", "ref.png", "ref_mask.png"
    )
    export_scheduler(traces, root, args.steps, args.shift)
    export_rope(traces, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        traces,
        str(args.output),
        metadata={
            "upstream_commit": UPSTREAM_COMMIT,
            "steps": str(args.steps),
            "shift": str(args.shift),
        },
    )
    print(f"Wrote {len(traces)} tensors to {args.output}")


if __name__ == "__main__":
    main()

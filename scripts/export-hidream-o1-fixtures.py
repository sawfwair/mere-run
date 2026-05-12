#!/usr/bin/env python3
"""Export HiDream O1 parity fixtures for the native Swift runtime.

This script mirrors the upstream HiDream-O1-Image sample builders and scheduler
setup. It writes small JSON/NPZ artifacts for Swift parity tests without
requiring a full image generation run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="Path to HiDream-O1-Image checkout")
    parser.add_argument("--model-path", required=True, help="Path to HiDream HF model root")
    parser.add_argument("--output-dir", required=True, help="Fixture output directory")
    parser.add_argument("--prompt", default="a red cube on a white background")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--model-type", choices=["dev", "full"], default="dev")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--mode", choices=["t2i", "reference"], default="t2i")
    parser.add_argument("--ref-image", action="append", default=[])
    parser.add_argument("--keep-original-aspect", action="store_true")
    parser.add_argument(
        "--export-pixel-head",
        action="store_true",
        help="Load the model and export timestep/x_embedder/final_layer2 outputs for parity tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream = Path(args.upstream).resolve()
    sys.path.insert(0, str(upstream))

    from transformers import AutoConfig, AutoProcessor  # pylint: disable=import-error
    from inference import add_special_tokens, get_tokenizer  # pylint: disable=import-error
    from models.pipeline import (  # pylint: disable=import-error
        CONDITION_IMAGE_SIZE,
        DEFAULT_TIMESTEPS,
        PATCH_SIZE,
        TIMESTEP_TOKEN_NUM,
        TENSOR_TRANSFORM,
        build_scheduler,
        build_t2i_text_sample,
    )
    from models.utils import (  # pylint: disable=import-error
        calculate_dimensions,
        find_closest_resolution,
        get_rope_index_fix_point,
        resize_pilimage,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)
    model_config = AutoConfig.from_pretrained(args.model_path)

    guidance_scale = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = 0.0 if args.model_type == "dev" else 5.0
    scheduler = build_fixture_scheduler(args, DEFAULT_TIMESTEPS, build_scheduler)

    if args.mode == "reference":
        if not args.ref_image:
            raise SystemExit("--mode reference requires at least one --ref-image")
        exported = build_reference_samples(
            prompt=args.prompt,
            ref_image_paths=[Path(path) for path in args.ref_image],
            height=args.height,
            width=args.width,
            tokenizer=tokenizer,
            processor=processor,
            model_config=model_config,
            guidance_scale=guidance_scale,
            keep_original_aspect=args.keep_original_aspect,
            resize_pilimage=resize_pilimage,
            calculate_dimensions=calculate_dimensions,
            find_closest_resolution=find_closest_resolution,
            get_rope_index_fix_point=get_rope_index_fix_point,
            tensor_transform=TENSOR_TRANSFORM,
            patch_size=PATCH_SIZE,
            condition_image_size=CONDITION_IMAGE_SIZE,
            timestep_token_num=TIMESTEP_TOKEN_NUM,
        )
    else:
        exported = build_text_samples(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            tokenizer=tokenizer,
            processor=processor,
            model_config=model_config,
            guidance_scale=guidance_scale,
            build_t2i_text_sample=build_t2i_text_sample,
        )

    metadata = {
        "prompt": args.prompt,
        "height": exported["target_height"],
        "width": exported["target_width"],
        "requested_height": args.height,
        "requested_width": args.width,
        "seed": args.seed,
        "steps": len(scheduler.timesteps),
        "mode": args.mode,
        "model_type": args.model_type,
        "guidance_scale": guidance_scale,
        "keep_original_aspect": args.keep_original_aspect,
        "ref_images": args.ref_image,
        "default_timesteps": DEFAULT_TIMESTEPS,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez(out_dir / "sample_and_scheduler.npz", **sample_npz_arrays(exported, scheduler))

    if args.export_pixel_head:
        export_pixel_head(args.model_path, out_dir)


def build_fixture_scheduler(args: argparse.Namespace, default_timesteps: list[int], build_scheduler):
    if args.model_type == "dev":
        timesteps = default_timesteps if args.steps is None else default_timesteps[: args.steps]
        if not timesteps:
            raise SystemExit("--steps must be >= 1")
        return build_scheduler(
            num_inference_steps=len(timesteps),
            timesteps_list=timesteps,
            shift=1.0,
            device=torch.device("cpu"),
            scheduler_name="flash",
        )
    return build_scheduler(
        num_inference_steps=args.steps or 50,
        timesteps_list=None,
        shift=3.0,
        device=torch.device("cpu"),
        scheduler_name="default",
    )


def build_text_samples(
    *,
    prompt: str,
    height: int,
    width: int,
    tokenizer,
    processor,
    model_config,
    guidance_scale: float,
    build_t2i_text_sample,
) -> dict:
    cond = build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config)
    uncond = None
    if guidance_scale > 1.0:
        uncond = build_t2i_text_sample(" ", height, width, tokenizer, processor, model_config)
    return {
        "target_width": width,
        "target_height": height,
        "cond": cond,
        "uncond": uncond,
        "reference_sizes": [],
        "condition_grid_thw": np.zeros((0, 3), dtype=np.int64),
        "reference_grid_thw": np.zeros((0, 3), dtype=np.int64),
        "reference_patches": np.zeros((0,), dtype=np.float32),
        "condition_pixel_values": np.zeros((0,), dtype=np.float32),
        "condition_image_grid_thw": np.zeros((0, 3), dtype=np.int64),
    }


def build_reference_samples(
    *,
    prompt: str,
    ref_image_paths: list[Path],
    height: int,
    width: int,
    tokenizer,
    processor,
    model_config,
    guidance_scale: float,
    keep_original_aspect: bool,
    resize_pilimage,
    calculate_dimensions,
    find_closest_resolution,
    get_rope_index_fix_point,
    tensor_transform,
    patch_size: int,
    condition_image_size: int,
    timestep_token_num: int,
) -> dict:
    from einops import rearrange  # pylint: disable=import-error
    from PIL import Image  # pylint: disable=import-error

    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    spatial_merge_size = model_config.vision_config.spatial_merge_size

    preresized_ref_pil = None
    if keep_original_aspect and len(ref_image_paths) == 1:
        pil_orig = Image.open(ref_image_paths[0]).convert("RGB")
        preresized_ref_pil = resize_pilimage(pil_orig, 2048, patch_size)
        width, height = preresized_ref_pil.size
    else:
        width, height = find_closest_resolution(width, height)

    if preresized_ref_pil is not None:
        ref_pils = [preresized_ref_pil]
    else:
        ref_pils = [Image.open(path).convert("RGB") for path in ref_image_paths]

    ref_pils_resized = []
    ref_images = []
    for pil in ref_pils:
        pil_r = pil if preresized_ref_pil is not None and pil is preresized_ref_pil else resize_pilimage(
            pil,
            reference_max_size(height=height, width=width, count=len(ref_pils)),
            patch_size,
        )
        ref_pils_resized.append(pil_r)
        ref_images.append(
            rearrange(
                tensor_transform(pil_r),
                "C (H p1) (W p2) -> (H W) (C p1 p2)",
                p1=patch_size,
                p2=patch_size,
            )
        )

    reference_lengths = [image.shape[0] for image in ref_images]
    total_reference_length = sum(reference_lengths)
    reference_patches = torch.cat(ref_images, dim=0).unsqueeze(0).cpu().numpy()

    condition_size = condition_max_size(condition_image_size, len(ref_pils))
    ref_pils_vlm = []
    for pil_r in ref_pils_resized:
        cond_w, cond_h = calculate_dimensions(condition_size, pil_r.width / pil_r.height)
        ref_pils_vlm.append(pil_r.resize((cond_w, cond_h), resample=Image.LANCZOS))

    target_grid_thw = torch.tensor([1, height // patch_size, width // patch_size], dtype=torch.int64).unsqueeze(0)
    reference_grid_thw = torch.zeros((len(ref_pils), 3), dtype=torch.int64)
    for index, pil_r in enumerate(ref_pils_resized):
        ref_w, ref_h = pil_r.size
        reference_grid_thw[index] = torch.tensor([1, ref_h // patch_size, ref_w // patch_size], dtype=torch.int64)

    samples = []
    condition_grid_thw = None
    condition_pixel_values = None
    condition_image_grid_thw = None
    for caption in [prompt] + ([" "] if guidance_scale > 1.0 else []):
        sample, cond_grid, proc = build_one_reference_sample(
            caption=caption,
            ref_pils_vlm=ref_pils_vlm,
            tokenizer=tokenizer,
            processor=processor,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            spatial_merge_size=spatial_merge_size,
            target_grid_thw=target_grid_thw,
            reference_grid_thw=reference_grid_thw,
            reference_lengths=reference_lengths,
            total_reference_length=total_reference_length,
            get_rope_index_fix_point=get_rope_index_fix_point,
            timestep_token_num=timestep_token_num,
        )
        samples.append(sample)
        if condition_grid_thw is None:
            condition_grid_thw = cond_grid.cpu().numpy()
            condition_pixel_values = proc.pixel_values.cpu().float().numpy()
            condition_image_grid_thw = proc.image_grid_thw.cpu().numpy()

    return {
        "target_width": width,
        "target_height": height,
        "cond": samples[0],
        "uncond": samples[1] if len(samples) > 1 else None,
        "reference_sizes": [[pil.width, pil.height] for pil in ref_pils_resized],
        "condition_grid_thw": condition_grid_thw,
        "reference_grid_thw": reference_grid_thw.cpu().numpy(),
        "reference_patches": reference_patches,
        "condition_pixel_values": condition_pixel_values,
        "condition_image_grid_thw": condition_image_grid_thw,
    }


def build_one_reference_sample(
    *,
    caption: str,
    ref_pils_vlm: list,
    tokenizer,
    processor,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
    target_grid_thw: torch.Tensor,
    reference_grid_thw: torch.Tensor,
    reference_lengths: list[int],
    total_reference_length: int,
    get_rope_index_fix_point,
    timestep_token_num: int,
) -> tuple[dict, torch.Tensor, object]:
    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")
    content = [{"type": "image"} for _ in ref_pils_vlm]
    content.append({"type": "text", "text": caption})
    messages = [{"role": "user", "content": content}]
    template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    proc = processor(text=[template_caption], images=ref_pils_vlm, padding="longest", return_tensors="pt")
    input_ids_2 = tokenizer.encode(
        boi_token + tms_token * timestep_token_num,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

    condition_grid_thw = proc.image_grid_thw.clone()
    for index in range(len(ref_pils_vlm)):
        condition_grid_thw[index, 1] //= spatial_merge_size
        condition_grid_thw[index, 2] //= spatial_merge_size
    all_grid_thw = torch.cat([condition_grid_thw, target_grid_thw, reference_grid_thw], dim=0)

    target_length = int(target_grid_thw[0, 1].item() * target_grid_thw[0, 2].item())
    vision_tokens = [vision_tokens_for_length(target_length, image_token_id, vision_start_token_id, input_ids.dtype)]
    for length in reference_lengths:
        vision_tokens.append(vision_tokens_for_length(length, image_token_id, vision_start_token_id, input_ids.dtype))
    input_ids_pad = torch.cat([input_ids, torch.cat(vision_tokens, dim=1)], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=all_grid_thw,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[0] * len(ref_pils_vlm) + [1] + [1] * len(ref_pils_vlm),
    )
    text_length = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]
    token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    generation_start = text_length - timestep_token_num
    generation_end = generation_start + target_length + timestep_token_num
    token_types_raw[0, generation_start:generation_end] = 1
    token_types_raw[0, generation_end : generation_end + total_reference_length] = 2
    token_types_raw[0, text_length - timestep_token_num : text_length] = 3

    return (
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_types": (token_types_raw > 0).to(token_types_raw.dtype),
            "vinput_mask": torch.logical_or(token_types_raw == 1, token_types_raw == 2),
        },
        condition_grid_thw,
        proc,
    )


def reference_max_size(*, height: int, width: int, count: int) -> int:
    max_size = max(height, width)
    if count == 1:
        return max_size
    if count == 2:
        return max_size * 48 // 64
    if count <= 4:
        return max_size // 2
    if count <= 8:
        return max_size * 24 // 64
    return max_size // 4


def condition_max_size(condition_image_size: int, count: int) -> int:
    if count <= 4:
        return condition_image_size
    if count <= 8:
        return condition_image_size * 48 // 64
    return condition_image_size // 2


def vision_tokens_for_length(length: int, image_token_id: int, vision_start_token_id: int, dtype) -> torch.Tensor:
    tokens = torch.full((1, length), image_token_id, dtype=dtype)
    tokens[0, 0] = vision_start_token_id
    return tokens


def sample_npz_arrays(exported: dict, scheduler) -> dict:
    cond = exported["cond"]
    arrays = {
        "input_ids": cond["input_ids"].cpu().numpy(),
        "position_ids": cond["position_ids"].cpu().numpy(),
        "token_types": cond["token_types"].cpu().numpy(),
        "vinput_mask": cond["vinput_mask"].cpu().numpy(),
        "timesteps": scheduler.timesteps.cpu().numpy(),
        "sigmas": scheduler.sigmas.cpu().numpy(),
        "reference_sizes": np.asarray(exported["reference_sizes"], dtype=np.int64),
        "condition_grid_thw": exported["condition_grid_thw"],
        "reference_grid_thw": exported["reference_grid_thw"],
        "reference_patches": exported["reference_patches"],
        "condition_pixel_values": exported["condition_pixel_values"],
        "condition_image_grid_thw": exported["condition_image_grid_thw"],
    }
    if exported["uncond"] is not None:
        arrays.update(
            {
                "uncond_input_ids": exported["uncond"]["input_ids"].cpu().numpy(),
                "uncond_position_ids": exported["uncond"]["position_ids"].cpu().numpy(),
                "uncond_token_types": exported["uncond"]["token_types"].cpu().numpy(),
                "uncond_vinput_mask": exported["uncond"]["vinput_mask"].cpu().numpy(),
            }
        )
    return arrays


def export_pixel_head(model_path: str, out_dir: Path) -> None:
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # pylint: disable=import-error

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    pixel_model = model.model
    hidden_size = pixel_model.config.text_config.hidden_size
    patch_dim = 3 * 32 * 32

    timestep = torch.tensor([0.5], dtype=torch.float32)
    patches = torch.linspace(-1.0, 1.0, steps=2 * patch_dim, dtype=torch.float32).reshape(2, patch_dim)
    hidden = torch.linspace(-0.5, 0.5, steps=2 * hidden_size, dtype=torch.float32).reshape(2, hidden_size)

    with torch.no_grad():
        timestep_freq = pixel_model.t_embedder1.timestep_embedding(timestep * 1000, 256)
        timestep_out = pixel_model.t_embedder1(timestep)
        patch_out = pixel_model.x_embedder(patches)
        final_out = pixel_model.final_layer2(hidden)

    np.savez(
        out_dir / "pixel_head.npz",
        timestep=timestep.cpu().numpy(),
        timestep_freq=timestep_freq.cpu().numpy(),
        timestep_out=timestep_out.cpu().numpy(),
        patch_input=patches.cpu().numpy(),
        patch_out=patch_out.cpu().numpy(),
        final_input=hidden.cpu().numpy(),
        final_out=final_out.cpu().numpy(),
    )


if __name__ == "__main__":
    main()

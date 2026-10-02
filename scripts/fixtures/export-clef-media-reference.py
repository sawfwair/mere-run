#!/usr/bin/env python3
"""Export independent Clef RGB resize fixtures with Pillow and mlx-vlm.

Reference tooling only. Requires Pillow 12.3.0, mlx-vlm 0.7.4, and numpy.
Pass processor_config.json from the pinned Clef checkpoint. No checkpoint
weights are loaded or bundled. Native inference does not use Python.
"""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import PIL
import numpy as np
import mlx.core as mx
from mlx_vlm.models.qwen3_vl.vision import VisionConfig, VisionModel
from PIL import Image, ImageDraw
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLImageProcessor,
    _smart_resize_video,
)


def pattern(width, height, shift=0):
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 7):
        draw.rectangle((x, 0, x + 3, height - 1), fill=(20, 60, 220))
    draw.ellipse((width // 4 + shift, height // 6,
                  width * 3 // 4, height * 5 // 6), fill=(235, 35, 30))
    return image


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--processor-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if PIL.__version__ != "12.3.0" or version("mlx-vlm") != "0.7.4":
        raise ValueError("Use the pinned Pillow 12.3.0 and mlx-vlm 0.7.4 references")
    config = json.loads(args.processor_config.read_text())
    output = args.output / "media"
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for kind, images in [
        ("image", [pattern(301, 197)]),
        ("video", [pattern(83, 49, shift=i * 3) for i in range(4)]),
    ]:
        settings = config[f"{kind}_processor"]
        source_width, source_height = images[0].size
        if kind == "image":
            processor = Qwen3VLImageProcessor(
                min_pixels=settings["size"]["shortest_edge"],
                max_pixels=settings["size"]["longest_edge"],
            )
            height, width = processor._resolved_size(source_height, source_width)
        else:
            height, width = _smart_resize_video(
                len(images), source_height, source_width,
                min_pixels=settings["size"]["shortest_edge"],
                max_pixels=settings["size"]["longest_edge"],
            )
        inputs, expected = [], []
        for index, image in enumerate(images):
            name = f"{kind}-{index}"
            image.save(output / f"{name}-source.png")
            image.resize((width, height), Image.Resampling.BICUBIC).save(
                output / f"{name}-reference.png")
            inputs.append(f"media/{name}-source.png")
            expected.append(f"media/{name}-reference.png")
        cases.append({"kind": kind, "inputs": inputs, "expected": expected,
                      "grid": [1 if kind == "image" else len(images) // 2,
                               height // 16, width // 16]})
    (args.output / "processor_config.json").write_bytes(args.processor_config.read_bytes())
    # Isolated learned-position arithmetic from the actual reference vision model.
    # Use a tiny feature width but the checkpoint's 48x48 learned position grid.
    rng = np.random.default_rng(2481)
    tower = VisionModel(VisionConfig(depth=0, hidden_size=8, intermediate_size=16,
                                    num_heads=2, out_hidden_size=8, num_position_embeddings=2304))
    weights = mx.array(rng.normal(0, 0.2, (2304, 8)).astype(np.float32)).astype(mx.bfloat16)
    arrays = {"weight": weights}
    grids = [[1, 14, 22], [2, 4, 6], [1, 16, 16]]
    for index, grid in enumerate(grids):
        tower.pos_embed.weight = weights
        arrays[f"bf16_{index}"] = tower.fast_pos_embed_interpolate(mx.array([grid]))
        tower.pos_embed.weight = weights.astype(mx.float32)
        arrays[f"fp32_{index}"] = tower.fast_pos_embed_interpolate(mx.array([grid]))
    mx.save_safetensors(str(args.output / "vision-interpolation.safetensors"), arrays)
    (args.output / "media-reference.json").write_text(json.dumps({
        "pillow_version": PIL.__version__, "mlx_vlm_version": version("mlx-vlm"),
        "processor_config_sha256": hashlib.sha256(args.processor_config.read_bytes()).hexdigest(),
        "normalized_8bit_lut": ((np.arange(256, dtype=np.float32)
                                * config["image_processor"]["rescale_factor"] - np.float32(0.5))
                               / np.float32(0.5)).tolist(),
        "cases": cases,
        "position_grids": grids,
        "video_geometry": [
            {"frames": frames, "width": width, "height": height,
             "minPixels": low, "maxPixels": high,
             "expectedHeight": result[0], "expectedWidth": result[1]}
            for frames, width, height, low, high in [
                (5, 512, 512, 4096, 1_200_000),
                (5, 32, 32, 6000, 1_000_000),
            ]
            for result in [_smart_resize_video(frames, height, width,
                                              min_pixels=low, max_pixels=high)]
        ],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

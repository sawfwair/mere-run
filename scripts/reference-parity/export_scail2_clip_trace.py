#!/usr/bin/env python3
"""Export the official 31-block SCAIL-2 CLIP visual trace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file


CLIP_SHA256 = "020daacf4c0ce94284df584d243cefb6ddfadefff8772226a8e85431df5de2da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_clip_module(root: Path):
    wan = types.ModuleType("wan")
    wan.__path__ = [str(root / "wan")]
    modules = types.ModuleType("wan.modules")
    modules.__path__ = [str(root / "wan/modules")]
    attention = types.ModuleType("wan.modules.attention")
    attention.flash_attention = reference_attention
    tokenizers = types.ModuleType("wan.modules.tokenizers")
    tokenizers.HuggingfaceTokenizer = object
    xlm = types.ModuleType("wan.modules.xlm_roberta")
    xlm.XLMRoberta = torch.nn.Module
    sys.modules.update({
        "wan": wan,
        "wan.modules": modules,
        "wan.modules.attention": attention,
        "wan.modules.tokenizers": tokenizers,
        "wan.modules.xlm_roberta": xlm,
    })
    path = root / "wan/modules/clip.py"
    spec = importlib.util.spec_from_file_location("wan.modules.clip", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.flash_attention = reference_attention
    return module


def reference_attention(q, k, v, **_):
    output = functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous().to(q.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocess-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if checkpoint.stat().st_size != 2_528_485_611 or sha256(checkpoint) != CLIP_SHA256:
        raise RuntimeError("CLIP checkpoint does not match the pinned SCAIL-2 source")
    module = load_clip_module(args.upstream.resolve())
    with torch.device("meta"):
        model = module.VisionTransformer(
            image_size=224,
            patch_size=14,
            dim=1280,
            mlp_ratio=4,
            out_dim=1024,
            num_heads=16,
            num_layers=32,
            pool_type="token",
            pre_norm=True,
            post_norm=False,
            activation="gelu",
        )
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("visual."): value
        for key, value in loaded.items()
        if key.startswith("visual.")
    }
    incompatibility = model.load_state_dict(state, strict=True, assign=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"State mismatch: {incompatibility}")
    device = torch.device(args.device)
    model = model.eval().to(device=device, dtype=torch.float16)
    preprocessing = load_file(str(args.preprocess_trace.resolve()))["animation_clip_nhwc"]
    input_nhwc = preprocessing.to(torch.float32).contiguous()
    input_nchw = input_nhwc.permute(0, 3, 1, 2).to(device=device, dtype=torch.float16)
    traces = {
        "clip_input_nhwc": input_nhwc,
    }
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        hidden = model.patch_embedding(input_nchw).flatten(2).permute(0, 2, 1)
        hidden = torch.cat([model.cls_embedding.expand(hidden.shape[0], -1, -1), hidden], dim=1)
        hidden = model.pre_norm(hidden + model.pos_embedding)
        traces["clip_pre_norm_ntc"] = hidden.float().cpu().contiguous()
        for index, block in enumerate(model.transformer[:-1]):
            hidden = block(hidden)
            traces[f"clip_block_{index:02d}_ntc"] = hidden.float().cpu().contiguous()
    traces["clip_output_ntc"] = traces["clip_block_30_ntc"].clone()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        traces,
        str(args.output),
        metadata={"checkpoint_sha256": CLIP_SHA256, "blocks": "31"},
    )
    print(f"Wrote {len(traces)} tensors to {args.output}")


if __name__ == "__main__":
    main()

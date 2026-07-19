#!/usr/bin/env python3
"""Export official-prompt UMT5-XXL traces from the pinned SCAIL-2 source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


T5_SHA256 = "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
TOKENIZER_SHA256 = "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
PROMPTS = [
    "The girl is dancing",
    "A blond white male wearing a black suit, trousers, and leather shoes is playing the violin on the street while pedestrians walk past him.",
    "",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_t5_module(root: Path):
    wan = types.ModuleType("wan")
    wan.__path__ = [str(root / "wan")]
    modules = types.ModuleType("wan.modules")
    modules.__path__ = [str(root / "wan/modules")]
    tokenizers_module = types.ModuleType("wan.modules.tokenizers")
    tokenizers_module.HuggingfaceTokenizer = object
    sys.modules.update({
        "wan": wan,
        "wan.modules": modules,
        "wan.modules.tokenizers": tokenizers_module,
    })
    path = root / "wan/modules/t5.py"
    spec = importlib.util.spec_from_file_location("wan.modules.t5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    current_device = torch.cuda.current_device
    torch.cuda.current_device = lambda: 0
    try:
        spec.loader.exec_module(module)
    finally:
        torch.cuda.current_device = current_device
    return module


def tokenize(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # SCAIL-2 constructs HuggingfaceTokenizer with the local umt5-xxl
    # directory. AutoTokenizer's T5 post-processor is significant here: an
    # empty negative prompt must contain EOS only, without a metaspace token.
    tokenizer = AutoTokenizer.from_pretrained(path.parent)
    ids: list[list[int]] = []
    masks: list[list[int]] = []
    lengths: list[int] = []
    for prompt in PROMPTS:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)[:512]
        lengths.append(len(encoded))
        ids.append(encoded + [0] * (512 - len(encoded)))
        masks.append([1] * len(encoded) + [0] * (512 - len(encoded)))
    return (
        torch.tensor(ids, dtype=torch.int64),
        torch.tensor(masks, dtype=torch.int64),
        torch.tensor(lengths, dtype=torch.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    if checkpoint.stat().st_size != 11_361_920_418 or sha256(checkpoint) != T5_SHA256:
        raise RuntimeError("T5 checkpoint does not match the pinned SCAIL-2 source")
    if tokenizer_path.stat().st_size != 16_837_417 or sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise RuntimeError("Tokenizer does not match the pinned SCAIL-2 source")
    module = load_t5_module(args.upstream.resolve())
    with torch.device("meta"):
        model = module.umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.bfloat16,
            device="meta",
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatibility = model.load_state_dict(state, strict=True, assign=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"State mismatch: {incompatibility}")
    device = torch.device(args.device)
    model = model.eval().to(device)
    token_ids, mask, lengths = tokenize(tokenizer_path)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model(token_ids.to(device), mask.to(device))
    traces = {
        "t5_token_ids": token_ids.to(torch.int32).contiguous(),
        "t5_mask": mask.to(torch.int32).contiguous(),
        "t5_lengths": lengths.contiguous(),
        "t5_output_ntc": output.float().cpu().contiguous(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        traces,
        str(args.output),
        metadata={
            "checkpoint_sha256": T5_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "prompts": "official animation, official replacement, empty negative",
        },
    )
    print(f"Wrote {len(traces)} tensors to {args.output}")


if __name__ == "__main__":
    main()

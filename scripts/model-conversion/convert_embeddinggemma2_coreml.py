#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["coremltools==9.0", "numpy==2.3.5", "torch==2.7.0", "safetensors==0.8.0"]
# ///
"""Extend the pinned EmbeddingGemma 2 Core ML bundle to 8K text and 30 s audio.

Offline conversion tooling only. Inference remains native Swift/Core ML. The
source checkpoint and base bundle must already be installed; originals are never
modified. Config publication happens after every requested graph is exported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

SOURCE_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
SOURCE_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
BASE_REVISION = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4"
TEXT_LENGTHS = (1024, 2048, 4096, 8192)
TEXT_PACKAGE = "EmbeddingGemma2TextExtended.mlpackage"
AUDIO_PACKAGE = "EmbeddingGemma2AudioExtended.mlpackage"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_wrapper(base):
    import torch
    from torch import nn

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base

        def forward(self, inputs_embeds, attention_mask):
            x = inputs_embeds.float()
            attention_mask = attention_mask.float()
            s = x.shape[1]
            m = self.base
            ple = m.per_layer_projection_norm(m.per_layer_model_projection(x).view(1, s, m.cfg.n_layers, m.cfg.ple_dim))
            full, sliding = m._biases(attention_mask, s)
            for i, layer in enumerate(m.layers):
                kind = m.cfg.layer_types[i]
                cos = getattr(m, "cos_" + kind)[:s].to(x.dtype)[None, None]
                sin = getattr(m, "sin_" + kind)[:s].to(x.dtype)[None, None]
                x = layer(x, ple[:, :, i, :], cos, sin, sliding if kind == "sliding_attention" else full)
            x = m.embedding_projection(m.norm(x)).float()
            weights = attention_mask.float()
            pooled = (x * weights[:, :, None]).sum(1) / weights.sum(-1, keepdim=True)
            return pooled / pooled.norm(dim=-1, keepdim=True)

    return Text().eval()


def convert_text(source: Path, output: Path, scratch: Path) -> None:
    import coremltools as ct
    import numpy as np
    import torch
    import embeddinggemma2_export_text as graph

    # Keep RMSNorm/softmax in float32, preserve the original RoPE positions and
    # bidirectional distance <= 512 local mask beyond the old 512-token export.
    graph.PRECISION_MODE = "fp32"
    graph.GQA_MODE = "expand"
    base = graph.load_model(str(source), torch.float32, max_seq=max(TEXT_LENGTHS))
    del base.embed_tokens  # host BF16 lookup is shared with the short graphs
    wrapper = text_wrapper(base)
    descriptor = ct.utils.MultiFunctionDescriptor()
    with torch.inference_mode():
        for length in TEXT_LENGTHS:
            package = scratch / f"text_{length}.mlpackage"
            if not package.exists():
                print(f"Tracing text {length}", flush=True)
                examples = (torch.zeros(1, length, 512, dtype=torch.float16), torch.ones(1, length, dtype=torch.float16))
                traced = torch.jit.trace(wrapper, examples, check_trace=False)
                model = ct.convert(
                    traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                    compute_precision=ct.precision.FLOAT32,
                    inputs=[ct.TensorType(name="inputs_embeds", shape=(1, length, 512), dtype=np.float16),
                            ct.TensorType(name="attention_mask", shape=(1, length), dtype=np.float16)],
                    outputs=[ct.TensorType(name="embedding", dtype=np.float32)], skip_model_load=True,
                )
                model.save(str(package))
                del model, traced, examples
            descriptor.add_function(str(package), "main", f"embed_{length}")
    descriptor.default_function_name = "embed_1024"
    ct.utils.save_multifunction(descriptor, str(output / TEXT_PACKAGE))


def convert_audio(source: Path, output: Path) -> None:
    import coremltools as ct
    import numpy as np
    import torch
    import embeddinggemma2_export_audio as graph

    # The native Accelerate frontend supplies log-mel frames and all three masks.
    graph.PRECISION_MODE = "fp32"
    model = graph.load_model(str(source), torch.float32)
    examples = (torch.zeros(1, 3000, 128), *graph.masks_for(2999, 3000, dtype=torch.float32))
    print("Tracing 30-second audio", flush=True)
    with torch.inference_mode():
        traced = torch.jit.trace(model, examples, check_trace=False)
    converted = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.float32)
                for name, value in zip(("features", "frame_mask", "half_mask", "token_mask"), examples)],
        outputs=[ct.TensorType(name="audio_tokens", dtype=np.float32)], skip_model_load=True,
    )
    converted.save(str(output / AUDIO_PACKAGE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original Google checkpoint directory")
    parser.add_argument("--base-bundle", type=Path, required=True, help="Pinned FluidInference Core ML bundle")
    parser.add_argument("--output", type=Path, required=True, help="New expanded bundle directory")
    parser.add_argument("--resume", action="store_true", help="Resume this tool's existing output after verifying its base files")
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in (args.source, args.base_bundle)):
        parser.error("Output must be outside both source directories.")
    if sha256(args.source / "model.safetensors") != SOURCE_SHA256:
        parser.error("Source checkpoint does not match the pinned Google revision.")
    if sha256(args.source / "config.json") != "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3":
        parser.error("Source configuration differs from the pinned checkpoint.")
    pins = json.loads(Path(__file__).with_name("embeddinggemma2_coreml_base_pins.json").read_text())
    for name, digest in pins.items():
        if sha256(args.base_bundle / name) != digest:
            parser.error(f"Base bundle differs from the pinned export: {name}")
    if args.output.exists():
        if not args.resume:
            parser.error("Output already exists. Use a new directory or --resume for a previous conversion.")
        for name, digest in pins.items():
            retained = "BASE_MODEL.md" if name == "README.md" and (args.output / "BASE_MODEL.md").exists() else name
            if name != "config.json" and sha256(args.output / retained) != digest:
                parser.error(f"Existing output does not retain the pinned base file: {name}")
    expected = {"coremltools": "9.0", "torch": "2.7.0", "numpy": "2.3.5", "safetensors": "0.8.0"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        parser.error(f"Toolchain differs from pinned versions: {versions}")
    config = json.loads((args.base_bundle / "config.json").read_text())
    if config["format"] != "coreml" or config["base_model"] != "google/embeddinggemma-2":
        parser.error("Expected the pinned FluidInference Core ML base bundle.")
    if not args.output.exists():
        args.output.mkdir(parents=True)
        for name in pins:
            destination = args.output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.base_bundle / name, destination)
    import torch
    torch.set_num_threads(4)
    graph_names = ("embeddinggemma2_export_text.py", "embeddinggemma2_export_audio.py", Path(__file__).name)
    graph_sources = {name: sha256(Path(__file__).with_name(name)) for name in graph_names}
    generation = hashlib.sha256(json.dumps(graph_sources, sort_keys=True).encode()).hexdigest()[:12]
    scratch = args.output.parent / (args.output.name + "-conversion-" + generation)
    scratch.mkdir(exist_ok=True)
    convert_text(args.source, args.output, scratch)
    convert_audio(args.source, args.output)
    config["precision"] = "mixed"
    config["extended"] = {
        "version": 1,
        "text_package": TEXT_PACKAGE,
        "lengths": list(TEXT_LENGTHS),
        "audio_package": AUDIO_PACKAGE,
        "audio_frames": 3000,
        "audio_seconds": 30,
        "text_precision": "float32",
        "audio_precision": "float32",
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
    }
    notices = Path(__file__).resolve().parents[2] / "Sources/MereRunCoreMLModel"
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(notices / name, args.output / name)
    shutil.copy2(args.base_bundle / "README.md", args.output / "BASE_MODEL.md")
    (args.output / "README.md").write_text(
        "# EmbeddingGemma 2 expanded Core ML bundle\n\n"
        "Native mere.run inference on Apple Silicon macOS 15 or later, including macOS 26.\n\n"
        "- Context: 8192 total tokens, including text, task prefixes, BOS/EOS, and media.\n"
        "- Audio: more than 10 ms and at most 30 seconds per ordered segment.\n"
        "- Long text: fixed 1024/2048/4096/8192-position functions, float32 CPU/GPU, fused attention.\n"
        "- Long audio: native Accelerate log-mel features and a float32 Core ML encoder.\n"
        "- Short inputs and images retain the original FluidInference functions.\n"
        "- Outputs: normalized 128/256/512/768-dimensional vectors.\n\n"
        "Use this directory with `mere.run text embed --model /path/to/this/bundle`.\n"
        "Python is conversion tooling only. Text truncation preserves BOS/EOS; mixed\n"
        "records and audio over the limits are rejected without truncating media.\n\n"
        f"Google source: {SOURCE_REVISION}. FluidInference base: {BASE_REVISION}.\n"
        "See conversion.json for graph source, toolchain, and artifact hashes.\n"
        "See BASE_MODEL.md for the original short-input model card and LICENSE/NOTICE for attribution.\n"
    )
    # Hash every graph and table so the expanded artifact has a reproducible closure.
    hashes = {str(p.relative_to(args.output)): sha256(p) for p in sorted(args.output.rglob("*"))
              if p.is_file() and p.name not in ("config.json", "conversion.json")}
    receipt = {"schema_version": 1, "source_repository": "google/embeddinggemma-2", "source_revision": SOURCE_REVISION,
               "source_sha256": SOURCE_SHA256, "base_repository": "FluidInference/embeddinggemma-2-coreml",
               "base_revision": BASE_REVISION, "toolchain": versions, "files": hashes,
               "graph_sources": graph_sources}
    (args.output / "conversion.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Expanded Core ML bundle: {args.output}", flush=True)


if __name__ == "__main__":
    main()

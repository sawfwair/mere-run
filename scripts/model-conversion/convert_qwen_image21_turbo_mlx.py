#!/usr/bin/env python3
"""Build the pinned Turbo mixed Q4/Q8 MLX artifact on a remote CUDA host.

Requires mlx[cuda12]==0.32.2 and huggingface-hub==1.28.0. Conversion tooling
only: the published checkpoint executes through native Swift/MLX.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPOSITORY = "Qwen/Qwen-Image-2.1-Turbo"
REVISION = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
ARTIFACT = "Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def bits_for(component, name, shape):
    if not name.endswith(".weight") or len(shape) != 2 or shape[-1] % 64:
        return None
    if component == "transformer":
        return 4 if name.startswith("transformer_blocks.") else None
    if component == "text_encoder" and not any(
        item in name for item in ("embed_tokens", "pos_embed", "position_embedding", "lm_head")
    ):
        return 8
    return None


def convert(source, output):
    import mlx.core as mx
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=False)
    info = api.model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=source, token=False,
                      allow_patterns=["LICENSE", "README.md", "model_index.json", "processor/*",
                                      "scheduler/*", "text_encoder/*", "transformer/*", "vae/*"])
    output.mkdir(parents=True, exist_ok=False)
    files = {entry.rfilename: entry for entry in info.siblings}
    provenance = {}
    errors = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
        sha = digest(path)
        entry = files[relative]
        if path.stat().st_size != entry.size or (entry.lfs and entry.lfs.sha256 != sha):
            raise ValueError("Pinned source integrity mismatch: " + relative)
        provenance[relative] = {"bytes": path.stat().st_size, "sha256": sha}
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".safetensors" and relative.split("/")[0] in ("transformer", "text_encoder"):
            component = relative.split("/")[0]
            arrays = mx.load(str(path))
            converted = {}
            for name, weight in arrays.items():
                if name == "lm_head.weight":
                    continue  # Native conditioning returns hidden states and does not use this head.
                bits = bits_for(component, name, weight.shape)
                if bits is None:
                    converted[name] = weight
                    continue
                prefix = name.removesuffix(".weight")
                packed, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
                mx.eval(packed, scales, biases)
                reconstructed = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
                numerator = mx.sum((weight.astype(mx.float32) - reconstructed.astype(mx.float32)) ** 2)
                denominator = mx.sum(weight.astype(mx.float32) ** 2)
                relative_l2 = float(mx.sqrt(numerator / denominator).item())
                errors[component + "/" + name] = {"bits": bits, "group_size": 64, "relative_l2": relative_l2}
                converted[name] = packed
                converted[prefix + ".scales"] = scales
                converted[prefix + ".biases"] = biases
                del reconstructed, numerator, denominator
                mx.clear_cache()
            mx.save_safetensors(str(target), converted, metadata={"format": "mlx", "source_revision": REVISION,
                                "modification": "MLX affine mixed Q4 transformer blocks / Q8 encoder linear weights"})
            print(json.dumps({"converted": relative, "bytes": target.stat().st_size}), flush=True)
            del arrays, converted
            mx.clear_cache()
        elif relative == "README.md":
            shutil.copyfile(path, output / "UPSTREAM_MODEL_CARD.md")
        else:
            shutil.copyfile(path, target)

    for component, stem, bits in [("transformer", "diffusion_pytorch_model", 4), ("text_encoder", "model", 8)]:
        config_path = output / component / "config.json"
        config = json.loads(config_path.read_text())
        config["quantization"] = {"bits": bits, "group_size": 64, "mode": "affine"}
        config["_conversion_notice"] = "Modified by Sawfwair: MLX affine mixed Q4/Q8 packing; see MODIFICATIONS.md."
        write_json(config_path, config)
        index_path = output / component / (stem + ".safetensors.index.json")
        if index_path.exists():
            from safetensors import safe_open
            mapping = {}
            total = 0
            for shard in sorted((output / component).glob("*.safetensors")):
                with safe_open(shard, framework="numpy") as reader:
                    for key in reader.keys():
                        mapping[key] = shard.name
                total += shard.stat().st_size
            write_json(index_path, {"metadata": {"total_size": total, "modification": "Sawfwair mixed Q4/Q8 packing; see MODIFICATIONS.md."}, "weight_map": mapping})

    write_json(output / "mererun_model.json", {
        "schemaVersion": 3, "id": "image-qwen-21-turbo-mixed-4bit", "engine": "qwen-image-21",
        "family": "qwen", "tier": "turbo", "variant": "distilled", "precision": "int4",
        "defaults": {"steps": 8, "cfg": 1}, "supports": ["txt2img", "reference_edit"],
        "components": {key: {"type": "local", "path": value} for key, value in
                       {"tokenizer": "processor", "text_encoder": "text_encoder", "transformer": "transformer",
                        "vae": "vae", "scheduler": "scheduler"}.items()},
        "upstreamRepoId": REPOSITORY + "@" + REVISION, "createdAt": "2026-10-09T00:00:00Z"
    })
    write_json(output / "QWEN21_CONVERSION.json", {
        "schema_version": 1, "source_repository": REPOSITORY, "source_revision": REVISION,
        "artifact_repository": ARTIFACT, "converter_sha256": digest(Path(__file__)),
        "mlx_version": "0.32.2", "quantization": {"mode": "affine", "group_size": 64,
        "transformer_blocks_bits": 4, "encoder_linear_bits": 8}, "source_files": provenance,
        "weight_reconstruction": errors, "quality_qualified": False,
        "notes": ["VAE, embeddings, norms, transformer input/output and modulation remain BF16.",
                  "Unused lm_head.weight omitted. Original Turbo sampling grid preserved.",
                  "Weight reconstruction error does not establish image quality or Apple memory fit."]
    })
    shutil.copyfile(__file__, output / Path(__file__).name)
    (output / "Notice").write_text(
        "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 "
        "Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.\n\nBuilt with Qwen.\n")
    (output / "MODIFICATIONS.md").write_text(
        "# Modifications\n\nDerived from " + REPOSITORY + "@" + REVISION + ".\n\n"
        "Transformer block linear weights use MLX affine Q4/group-64. Encoder linear weights use Q8/group-64. "
        "Unused language-model output head omitted. Other tensors retain source precision. "
        "Component configs and tensor indexes updated for packed weights. Original Turbo schedule preserved. "
        "No fine-tuning or retraining. Governed by the original Qwen Research License; see LICENSE.\n")
    (output / "README.md").write_text(
        "---\nlicense: other\nlicense_name: qwen-research\nlicense_link: LICENSE\n"
        "base_model: Qwen/Qwen-Image-2.1-Turbo\n"
        "pipeline_tag: text-to-image\ntags: [mlx, qwen-image, quantized, image-to-image]\n---\n\n"
        "# Qwen Image 2.1 Turbo — MLX Mixed 4-bit\n\nBuilt with Qwen.\n\n"
        "Unofficial mixed-precision MLX quantization of Qwen/Qwen-Image-2.1-Turbo. No fine-tuning or retraining. "
        "Q4/group-64 transformer blocks, Q8/group-64 encoder linear weights, BF16 VAE and remaining layers.\n\n"
        "Requires the native Qwen Image 2.1 mixed-weight loader in [mere.run](https://github.com/sawfwair/mere-run). "
        "Runtime support is under development; released binaries may not yet include it.\n\n"
        "```sh\nhf download " + ARTIFACT + " --local-dir ./qwen21-turbo-mixed\n"
        "mere.run image generate --model ./qwen21-turbo-mixed --prompt 'A blue ceramic teapot' "
        "--width 1024 --height 1024 --steps 8 --output ./teapot.png\n```\n\n"
        "The original Turbo sigma grid and CFG 1 defaults are retained. "
        "This artifact is not the base Qwen 2.1 model or a third-party Turbo LoRA.\n\n"
        "Experimental: weight reconstruction metrics are recorded in QWEN21_CONVERSION.json. "
        "Image quality, editing fidelity and hardware fit require separate qualification. "
        "Quantization reduces weight memory, not all activation memory.\n\n"
        "See LICENSE, UPSTREAM_MODEL_CARD.md, MODIFICATIONS.md and SHA256SUMS. "
        "The original Qwen Research License applies; this conversion does not grant commercial rights.\n")
    sums = "".join(digest(path) + "  " + path.relative_to(output).as_posix() + "\n"
                   for path in sorted(output.rglob("*")) if path.is_file())
    (output / "SHA256SUMS").write_text(sums)
    print(json.dumps({"status": "converted", "repository": ARTIFACT,
                      "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
                      "quantized_layers": len(errors)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output)

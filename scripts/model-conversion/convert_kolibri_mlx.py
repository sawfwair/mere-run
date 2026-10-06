#!/usr/bin/env python3
"""Stream pinned Kolibri BF16 into native reference/Q8/mixed-Q2 layouts.

Remote release tooling only. Requires mlx==0.32.2, safetensors==0.8.0,
huggingface-hub==1.28.0, numpy==2.3.5, and torch for BF16 mmap access.
No complete model is instantiated. Calibration/holdout evaluation is separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

REPOSITORY = "Aleph-Alpha/Kolibri-1-BF16"
REVISION = "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"
SOURCE_IMPLEMENTATION = "049a6a7bd2405b27d6d280d256bd3d585191c7ae"
PROFILES = ["reference", "q8", "mixed2", "mixed2-refit", "mixed2-down3"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: object):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def policy(profile: str, path: str) -> dict | None:
    if profile == "reference" or not path.endswith(("_proj", "lm_head")):
        return None
    if path == "lm_head":
        return None  # Preserve the FP32-compute vocabulary head's BF16 weights.
    if ".mlp.experts." in path and profile != "q8":
        if profile == "mixed2-down3" and path.endswith("down_proj"):
            return {"bits": 3, "group_size": 64}
        return {"bits": 2, "group_size": 128}
    return {"bits": 8, "group_size": 64}


def refit_q2(weight, packed, scales, biases, mx):
    """Refine affine levels; accept only groups whose stored-BF16 L2 error improves.

    This is an unweighted control, not an activation-calibrated quality claim.
    Native held-out logprobs decide whether to retain it.
    """
    shape = weight.shape
    x = weight.astype(mx.float32).reshape(*shape[:-1], shape[-1] // 128, 128)
    initial_s, initial_b = scales.astype(mx.float32)[..., None], biases.astype(mx.float32)[..., None]
    shifts = mx.arange(16, dtype=mx.uint32) * 2
    original_codes = ((packed[..., None] >> shifts) & 3).reshape(x.shape).astype(mx.float32)
    s, b = initial_s, initial_b
    for _ in range(6):
        codes = mx.clip(mx.round((x - b) / mx.where(mx.abs(s) > 1e-12, s, 1e-12)), 0, 3)
        mean_q = codes.mean(-1, keepdims=True)
        mean_x = x.mean(-1, keepdims=True)
        variance = ((codes - mean_q) ** 2).mean(-1, keepdims=True)
        slope = ((codes - mean_q) * (x - mean_x)).mean(-1, keepdims=True) / mx.maximum(variance, 1e-12)
        valid = (variance > 0) & (mx.abs(slope) > 1e-12)
        s = mx.where(valid, slope, s).astype(mx.bfloat16).astype(mx.float32)
        b = mx.where(valid, mean_x - s * mean_q, b).astype(mx.bfloat16).astype(mx.float32)
    codes = mx.clip(mx.round((x - b) / mx.where(mx.abs(s) > 1e-12, s, 1e-12)), 0, 3)
    initial_codes = original_codes
    improved = ((x - (s * codes + b)) ** 2).sum(-1, keepdims=True) < ((x - (initial_s * initial_codes + initial_b)) ** 2).sum(-1, keepdims=True)
    s = mx.where(improved, s, initial_s)
    b = mx.where(improved, b, initial_b)
    codes = mx.where(improved, codes, initial_codes).astype(mx.uint32).reshape(*shape[:-1], shape[-1] // 16, 16)
    shifts = mx.arange(16, dtype=mx.uint32) * 2
    packed = (codes << shifts).sum(-1).astype(mx.uint32)
    return packed, s.squeeze(-1).astype(mx.bfloat16), b.squeeze(-1).astype(mx.bfloat16)


def convert(source: Path, destination: Path, profiles: list[str], resume: bool = False):
    import mlx.core as mx
    import numpy as np
    import torch
    from safetensors import safe_open
    from huggingface_hub import HfApi

    remote = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    pinned_files = {f.rfilename: f for f in remote.siblings}
    index = json.loads((source / "model.safetensors.index.json").read_text())
    config = json.loads((source / "config.json").read_text())
    assert config["model_type"] == "kolibri1" and config["num_hidden_layers"] == 50 and config["num_experts"] == 384
    assert index["metadata"]["total_size"] == 156_206_149_120
    receipts = {}
    readers = {}
    print("Verifying pinned source shard hashes", flush=True)
    for filename in sorted(set(index["weight_map"].values())):
        file = source / filename
        expected = pinned_files[filename]
        sha = digest(file)
        if expected.lfs is None or sha != expected.lfs.sha256 or file.stat().st_size != expected.size:
            raise ValueError(f"Source SHA-256/size mismatch: {filename}")
        receipts[filename] = {"sha256": sha, "bytes": file.stat().st_size}
        readers[filename] = safe_open(file, framework="pt", device="cpu")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "verified-source.json", {"revision": REVISION, "files": receipts})
    print("Source shard hashes verified", flush=True)
    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        # Verify small files against the exact pinned Hub revision too.
        from huggingface_hub import hf_hub_download
        pinned = Path(hf_hub_download(REPOSITORY, filename, revision=REVISION))
        if digest(source / filename) != digest(pinned):
            raise ValueError(f"Source metadata differs from pinned revision: {filename}")

    def array(key):
        tensor = readers[index["weight_map"][key]].get_tensor(key)
        if tensor.dtype == torch.bfloat16:
            return mx.array(tensor.view(torch.uint16).numpy()).view(mx.bfloat16)
        return mx.array(tensor.numpy())

    roots = {profile: destination / ("Kolibri-1-MLX-" + profile) for profile in profiles}
    policies = {profile: {} for profile in profiles}
    maps = {profile: {} for profile in profiles}
    hashes = {profile: {} for profile in profiles}
    logical_bytes = {profile: 0 for profile in profiles}
    for root in roots.values():
        if root.exists() and not resume: raise FileExistsError(f"Use fresh output roots: {root}")
        root.mkdir(parents=True, exist_ok=resume)

    def write_part(filename, arrays):
        started = time.monotonic()
        for profile, root in roots.items():
            file = root / filename
            expected_keys = set()
            for key in arrays:
                path = key.removesuffix(".weight")
                q = policy(profile, path) if key.endswith(".weight") else None
                expected_keys.add(key)
                if q is not None:
                    policies[profile][path] = q
                    expected_keys.update([path + ".scales", path + ".biases"])
            if resume and file.exists():
                existing = mx.load(str(file))
                if set(existing) != expected_keys: raise ValueError(f"Incomplete resumed shard: {file}")
                for key, value in arrays.items():
                    path = key.removesuffix(".weight")
                    q = policy(profile, path) if key.endswith(".weight") else None
                    shape = list(value.shape)
                    if q is None:
                        if existing[key].shape != value.shape or existing[key].dtype != value.dtype:
                            raise ValueError(f"Resumed tensor mismatch: {key}")
                    else:
                        packed_shape = shape[:-1] + [shape[-1] * q["bits"] // 32]
                        group_shape = shape[:-1] + [shape[-1] // q["group_size"]]
                        if existing[key].shape != tuple(packed_shape) or existing[key].dtype != mx.uint32:
                            raise ValueError(f"Resumed packed tensor mismatch: {key}")
                        for suffix in [".scales", ".biases"]:
                            if existing[path + suffix].shape != tuple(group_shape) or existing[path + suffix].dtype != value.dtype:
                                raise ValueError(f"Resumed affine tensor mismatch: {path + suffix}")
                maps[profile].update({key: filename for key in existing})
                hashes[profile][filename] = {"sha256": digest(file), "bytes": file.stat().st_size}
                logical_bytes[profile] += sum(a.nbytes for a in existing.values())
                del existing
                continue
            output = {}
            for key, weight in arrays.items():
                path = key.removesuffix(".weight")
                quantization = policy(profile, path) if key.endswith(".weight") else None
                if quantization is None:
                    output[key] = weight
                    continue
                policies[profile][path] = quantization
                q, s, b = mx.quantize(weight, group_size=quantization["group_size"], bits=quantization["bits"])
                if profile == "mixed2-refit" and quantization["bits"] == 2:
                    q, s, b = refit_q2(weight, q, s, b, mx)
                output[key] = q
                output[path + ".scales"] = s
                output[path + ".biases"] = b
            mx.eval(output)
            # Network filesystems make mmap serialization expensive. Serialize on the
            # container disk, then stream the finished file into the persistent volume.
            with tempfile.TemporaryDirectory(prefix="kolibri-shard-") as temporary:
                local = Path(temporary) / filename
                mx.save_safetensors(str(local), output)
                local_digest = digest(local)
                staging = file.with_suffix(file.suffix + ".partial")
                with local.open("rb") as src, staging.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                staging.replace(file)
            maps[profile].update({key: filename for key in output})
            hashes[profile][filename] = {"sha256": local_digest, "bytes": file.stat().st_size}
            logical_bytes[profile] += sum(a.nbytes for a in output.values())
            del output
            mx.clear_cache()
        print(json.dumps({"shard": filename, "seconds": time.monotonic() - started}), flush=True)

    write_part("global.safetensors", {key: array(key) for key in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]})
    mapped_source_keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        weights = {}
        keys = [key for key in index["weight_map"] if key.startswith(prefix) and ".mlp.experts." not in key]
        for key in keys:
            canonical = key.replace(".moe.router.expert_bias", ".mlp.gate.e_score_correction_bias")
            weights[canonical] = array(key)
            if canonical.endswith(".e_score_correction_bias"):
                weights[canonical] = weights[canonical].astype(mx.float32)
            mapped_source_keys.add(key)
        for projection in ["gate_proj", "up_proj", "down_proj"]:
            source_keys = [prefix + f"mlp.experts.{expert}.{projection}.weight" for expert in range(config["num_experts"])]
            bank = mx.stack([array(key) for key in source_keys], axis=0)
            mx.eval(bank)
            weights[prefix + f"mlp.experts.{projection}.weight"] = bank
            mapped_source_keys.update(source_keys)
        mx.eval(weights)
        write_part(f"layer-{layer:03d}.safetensors", weights)
        del weights
        mx.clear_cache()
    if mapped_source_keys != set(index["weight_map"]):
        raise ValueError("Conversion did not account for every source tensor exactly once")
    for profile, root in roots.items():
        native_config = dict(config)
        native_config.pop("quantization_config", None)
        native_config["mererun_quantization"] = policies[profile]
        atomic_json(root / "config.json", native_config)
        atomic_json(root / "model.safetensors.index.json", {"metadata": {"total_size": logical_bytes[profile]}, "weight_map": maps[profile]})
        for filename in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "LICENSE", "README.md"]:
            if (source / filename).exists(): shutil.copy2(source / filename, root / filename)
        atomic_json(root / "KOLIBRI_CONVERSION.json", {
            "schema_version": 1, "source_repository": REPOSITORY, "source_revision": REVISION,
            "architecture_revision": SOURCE_IMPLEMENTATION, "profile": profile,
            "source_files": receipts, "output_files": hashes[profile],
            "logical_weight_bytes": logical_bytes[profile], "tensor_count": len(maps[profile]),
            "source_tensor_count": len(mapped_source_keys), "native_expert_order": "numeric",
            "quality_qualified": False,
        })
        print(json.dumps({"profile": profile, "weight_bytes": logical_bytes[profile]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--profiles", nargs="+", choices=PROFILES, default=["reference", "mixed2", "mixed2-refit", "mixed2-down3", "q8"])
    p.add_argument("--resume", action="store_true", help="Resume complete native shards in these output roots.")
    p.add_argument("--plan", action="store_true")
    a = p.parse_args()
    if a.plan:
        print(json.dumps({"repository": REPOSITORY, "revision": REVISION, "profiles": a.profiles, "source_weight_bytes": 156_206_149_120}, indent=2))
        return
    if a.source is None or a.output is None: p.error("--source and --output are required")
    convert(a.source.resolve(), a.output.resolve(), a.profiles, resume=a.resume)


if __name__ == "__main__": main()

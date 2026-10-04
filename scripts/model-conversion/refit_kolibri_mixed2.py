#!/usr/bin/env python3
"""Fit Q2 affine levels using native calibration-only input second moments.

This diagonal activation-weighted fit is a candidate, not a quality guarantee.
Heldout paired logprobs determine whether it should replace the baseline.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
from convert_kolibri_mlx import digest, atomic_json


def refit(weight, packed, scales, biases, moment, mx):
    shape = weight.shape
    x = weight.astype(mx.float32).reshape(*shape[:-1], shape[-1] // 128, 128)
    h = moment.astype(mx.float32)
    h = mx.maximum(h, h.mean() * .001).reshape(shape[-1] // 128, 128)
    hsum = mx.maximum(h.sum(-1, keepdims=True), 1e-20)
    shifts = mx.arange(16, dtype=mx.uint32) * 2
    original_codes = ((packed[..., None] >> shifts) & 3).reshape(x.shape).astype(mx.float32)
    initial_s, initial_b = scales.astype(mx.float32)[..., None], biases.astype(mx.float32)[..., None]
    s, b, codes = initial_s, initial_b, original_codes
    for _ in range(8):
        mean_q = (h * codes).sum(-1, keepdims=True) / hsum
        mean_x = (h * x).sum(-1, keepdims=True) / hsum
        variance = (h * (codes - mean_q) ** 2).sum(-1, keepdims=True) / hsum
        slope = (h * (codes - mean_q) * (x - mean_x)).sum(-1, keepdims=True) / hsum / mx.maximum(variance, 1e-20)
        valid = (variance > 1e-12) & (mx.abs(slope) > 1e-12)
        s = mx.where(valid, slope, s).astype(mx.bfloat16).astype(mx.float32)
        b = mx.where(valid, mean_x - s * mean_q, b).astype(mx.bfloat16).astype(mx.float32)
        denominator = mx.where(mx.abs(s) > 1e-12, s, 1e-12)
        codes = mx.clip(mx.round((x - b) / denominator), 0, 3)
    before = (h * (x - (initial_s * original_codes + initial_b)) ** 2).sum(-1, keepdims=True)
    after = (h * (x - (s * codes + b)) ** 2).sum(-1, keepdims=True)
    improved = after < before
    s, b = mx.where(improved, s, initial_s), mx.where(improved, b, initial_b)
    codes = mx.where(improved, codes, original_codes).astype(mx.uint32).reshape(*shape[:-1], shape[-1] // 16, 16)
    packed = (codes << shifts).sum(-1).astype(mx.uint32)
    fraction = improved.astype(mx.float32).mean().item()
    ratio = (mx.where(improved, after, before).sum() / mx.maximum(before.sum(), 1e-20)).item()
    return packed, s.squeeze(-1).astype(mx.bfloat16), b.squeeze(-1).astype(mx.bfloat16), fraction, ratio


def run(reference, baseline, moments, reference_results, output):
    import mlx.core as mx
    if output.exists(): raise FileExistsError("Use a fresh output directory")
    reference_receipt = json.loads((reference / "KOLIBRI_CONVERSION.json").read_text())
    receipt = json.loads((baseline / "KOLIBRI_CONVERSION.json").read_text())
    if reference_receipt["source_revision"] != receipt["source_revision"] or receipt["profile"] != "mixed2":
        raise ValueError("Source/reference/baseline identity differs")
    for root, manifest in [(reference, reference_receipt), (baseline, receipt)]:
        for filename, expected in manifest["output_files"].items():
            if digest(root / filename) != expected["sha256"]: raise ValueError(f"Checksum mismatch: {filename}")
    scoring = json.loads((reference_results / "receipt.json").read_text())
    calibration_ids = [r["sequence"]["id"] for r in scoring["cases"] if r["sequence"]["split"] == "calibration"]
    if scoring.get("calibrationSHA256") != digest(moments) or scoring.get("calibrationCaseIDs") != calibration_ids or not calibration_ids:
        raise ValueError("Calibration provenance does not match the native reference results")
    if scoring.get("conversionSHA256") != digest(reference / "KOLIBRI_CONVERSION.json"):
        raise ValueError("Calibration reference checkpoint differs")
    stats = mx.load(str(moments))
    config = json.loads((baseline / "config.json").read_text())
    output.mkdir(parents=True)
    hashes, fitting = {}, {}
    for filename in sorted(receipt["output_files"]):
        started = time.monotonic()
        arrays = mx.load(str(baseline / filename))
        full = mx.load(str(reference / filename))
        for key in list(arrays):
            path = key.removesuffix(".weight")
            policy = config["mererun_quantization"].get(path)
            if not key.endswith(".weight") or not policy or policy["bits"] != 2: continue
            if path not in stats: raise ValueError(f"Missing calibration statistic: {path}")
            packed, scales, biases, fraction, ratio = refit(full[key], arrays[key], arrays[path + ".scales"],
                                                          arrays[path + ".biases"], stats[path], mx)
            arrays[key], arrays[path + ".scales"], arrays[path + ".biases"] = packed, scales, biases
            fitting[path] = {"improved_group_fraction": fraction, "weighted_error_ratio": ratio}
        mx.eval(arrays)
        with tempfile.TemporaryDirectory(prefix="kolibri-fit-") as temporary:
            local = Path(temporary) / filename
            mx.save_safetensors(str(local), arrays)
            sha = digest(local)
            with local.open("rb") as src, (output / filename).open("wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        hashes[filename] = {"sha256": sha, "bytes": (output / filename).stat().st_size}
        del arrays, full
        mx.clear_cache()
        print(json.dumps({"shard": filename, "seconds": time.monotonic() - started}), flush=True)
    for filename in ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "LICENSE"]:
        if (baseline / filename).exists(): shutil.copy2(baseline / filename, output / filename)
    receipt.update(profile="mixed2-calibrated", output_files=hashes, quality_qualified=False,
                   calibration_sha256=digest(moments), calibration_suite_sha256=scoring["suiteSHA256"],
                   calibration_case_ids=calibration_ids, fitting=fitting,
                   fit="diagonal input second moments; BF16 affine levels; eight Lloyd iterations; groupwise weighted-error acceptance")
    atomic_json(output / "KOLIBRI_CONVERSION.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["reference", "baseline", "moments", "reference-results", "output"]: parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.baseline, args.moments, args.reference_results, args.output)

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "safetensors==0.8.0"]
# ///
"""Compare resident SOV receipts and projected tensors from the same recording.

Qualification requires identical greedy tokens, finite tensors, relative RMSE
at most 5%, and cosine similarity at least 0.999. Receipts preserve all measured
runs; compare medians after unmeasured warmups, outside Instruments recording.
"""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mlx", type=Path)
    parser.add_argument("coreml", type=Path)
    parser.add_argument("--mlx-embeddings", type=Path, required=True)
    parser.add_argument("--coreml-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = [json.loads(p.read_text()) for p in (args.mlx, args.coreml)]
    for key in ("audioSHA256", "audioSeconds", "maxTokens", "dflash"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Benchmark inputs differ: {key}")
    if baseline.get("prefillChunkSize", 256) != candidate.get("prefillChunkSize", 256):
        raise ValueError("Benchmark inputs differ: prefillChunkSize")
    if not baseline["dflash"]:
        raise ValueError("This comparison requires DFlash enabled for both providers")
    if any(r["warmups"] < 1 or len(r["measurements"]) < 3 for r in (baseline, candidate)):
        raise ValueError("Use at least one warmup and three measured repetitions")
    a, b = [load_file(p)["audio_embeddings"].astype(np.float64) for p in (args.mlx_embeddings, args.coreml_embeddings)]
    if a.shape != b.shape:
        raise ValueError("Projected tensor shapes differ")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    rmse = float(np.sqrt(np.mean((a-b)**2)))
    relative = rmse / float(np.sqrt(np.mean(a*a)))
    cosine = float(np.sum(a*b) / (np.linalg.norm(a)*np.linalg.norm(b)))
    tokens = baseline["measurements"][0]["tokens"]
    parity = all(m["tokens"] == tokens for r in (baseline,candidate) for m in r["measurements"])
    stages = ["featureSeconds","encoderProjectorSeconds","prefillSeconds","generationSeconds","cleanupSeconds","totalSeconds"]
    medians = {p: {s: statistics.median(m[s] for m in r["measurements"]) for s in stages} for p,r in [("mlx",baseline),("coreml",candidate)]}
    receipt = dict(schemaVersion=1, tokenParity=parity, finite=finite, rmse=rmse, relativeRMSE=relative, cosine=cosine,
                   maxAbsoluteError=float(np.max(np.abs(a-b))), relativeRMSELimit=0.05, cosineMinimum=0.999,
                   medianSeconds=medians, encoderSpeedup=medians["mlx"]["encoderProjectorSeconds"]/medians["coreml"]["encoderProjectorSeconds"],
                   endToEndSpeedup=medians["mlx"]["totalSeconds"]/medians["coreml"]["totalSeconds"])
    receipt["passed"] = parity and finite and relative <= 0.05 and cosine >= 0.999
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not receipt["passed"]:
        raise SystemExit("SOV provider qualification failed")


if __name__ == "__main__":
    main()

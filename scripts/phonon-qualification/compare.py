#!/usr/bin/env python3
"""Compare released-weight float32 PCM and step traces (requires NumPy)."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("native", type=Path)
parser.add_argument("upstream", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
rows = []
for ref in json.loads((args.upstream / "reference.json").read_text()):
    name = ref["id"]
    actual = np.fromfile(args.native / (name + ".f32"), np.float32).astype(np.float64)
    expected = np.fromfile(args.upstream / (name + ".f32"), np.float32).astype(np.float64)
    native_steps = json.loads((args.native / (name + ".steps.json")).read_text())
    reference_steps = json.loads((args.upstream / (name + ".steps.json")).read_text())
    if actual.size != expected.size or len(native_steps) != len(reference_steps):
        raise SystemExit(f"Frame/sample count mismatch: {name}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise SystemExit(f"Non-finite audio: {name}")
    delta = actual - expected
    latent_delta = np.array([s["latent"] for s in native_steps]) - np.array([s["latent"] for s in reference_steps])
    eos_delta = np.array([s["eos"] for s in native_steps]) - np.array([s["eos"] for s in reference_steps])
    rows.append(dict(id=name, samples=actual.size, frames=len(native_steps), chunks=len(ref["chunks"]),
                     pcm_rmse=float(np.sqrt(np.mean(delta**2))), pcm_max_abs=float(np.max(np.abs(delta))),
                     pcm_snr_db=float(10 * np.log10(np.sum(expected**2) / np.sum(delta**2))),
                     correlation=float(np.corrcoef(actual, expected)[0, 1]),
                     latent_max_abs=float(np.max(np.abs(latent_delta))), eos_max_abs=float(np.max(np.abs(eos_delta)))))
args.output.write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps({"cases": len(rows), "minimum_correlation": min(r["correlation"] for r in rows),
                  "minimum_snr_db": min(r["pcm_snr_db"] for r in rows)}, indent=2))

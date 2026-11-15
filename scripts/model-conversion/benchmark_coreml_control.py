#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["coremltools==9.0", "numpy==2.3.5", "safetensors==0.8.0"]
# ///
"""Benchmark a published Core ML encoder separately from SOV qualification.

The encoder's own metadata supplies feature names. Results describe that model's
work, including any decoder cross-KV projection it contains; they establish no
SOV output parity. Download and pin the artifact before invoking this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, choices=[1000, 3000], default=3000)
    parser.add_argument("--compute-units", choices=["CPU_AND_NE", "CPU_AND_GPU"], default="CPU_AND_NE")
    parser.add_argument("--repetitions", type=int, default=9)
    args = parser.parse_args()
    import coremltools as ct
    import numpy as np
    from safetensors.numpy import load_file
    metadata = json.loads((args.artifact / "metadata.json").read_text())[0]
    name = metadata["inputSchema"][0]["name"]
    inputs = {name: np.ascontiguousarray(load_file(args.features)["mel"][:1, :, None, :args.frames], dtype=np.float16)}
    started = time.perf_counter()
    model = ct.models.CompiledMLModel(str(args.artifact), compute_units=getattr(ct.ComputeUnit, args.compute_units))
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model.predict(inputs)
    first_seconds = time.perf_counter() - started
    model.predict(inputs)
    seconds = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        result = model.predict(inputs)
        seconds.append(time.perf_counter() - started)
    receipt = dict(artifact=str(args.artifact), computeUnits=args.compute_units,
        os=platform.platform(), inputShape=list(inputs[name].shape),
        inputSHA256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
        outputShapes={k: list(v.shape) for k, v in result.items()},
        finite=all(bool(np.isfinite(v).all()) for v in result.values()),
        loadSeconds=load_seconds, firstPredictionSeconds=first_seconds,
        warmups=2, predictionSeconds=seconds, medianSeconds=statistics.median(seconds),
        files=[dict(path=str(p.relative_to(args.artifact)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted(args.artifact.rglob("*")) if p.is_file()])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()

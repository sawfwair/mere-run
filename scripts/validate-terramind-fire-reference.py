#!/usr/bin/env python3
"""Compare native TerraMind Fire logits with the official TerraTorch task."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.numpy import load_file
from terratorch.tasks import SemanticSegmentationTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--native-output", type=pathlib.Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    hyperparameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError("checkpoint does not contain Lightning hyper_parameters")
    model_args = dict(hyperparameters["model_args"])
    model_args["backbone_pretrained"] = False
    task = SemanticSegmentationTask.load_from_checkpoint(
        args.checkpoint.resolve(),
        map_location="cpu",
        model_args=model_args,
        strict=True,
        weights_only=True,
    )
    task.model.eval()

    arrays = load_file(args.input.resolve())
    reference_input = {
        name: torch.from_numpy(np.ascontiguousarray(arrays[name].astype(np.float32)))
        for name in ("S2L2A", "S1RTC", "DEM")
    }
    with torch.inference_mode():
        reference = task.model(reference_input).output.cpu().numpy()
    native = load_file(args.native_output.resolve())["logits"]
    if reference.shape != native.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape}, native {native.shape}")

    difference = np.abs(reference.astype(np.float64) - native.astype(np.float64))
    passed = bool(np.allclose(reference, native, atol=args.atol, rtol=args.rtol))
    reference_mask = reference.argmax(axis=1)
    native_mask = native.argmax(axis=1)
    intersection = np.logical_and(reference_mask == 1, native_mask == 1).sum()
    union = np.logical_or(reference_mask == 1, native_mask == 1).sum()
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "checkpoint": str(args.checkpoint.resolve()),
        "input": str(args.input.resolve()),
        "native_output": str(args.native_output.resolve()),
        "shape": list(reference.shape),
        "atol": args.atol,
        "rtol": args.rtol,
        "maximum_absolute_error": float(difference.max(initial=0)),
        "mean_absolute_error": float(difference.mean()),
        "mask_agreement": float((reference_mask == native_mask).mean()),
        "positive_mask_jaccard": float(intersection / union) if union else 1.0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"reference validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

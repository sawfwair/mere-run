#!/usr/bin/env python3
"""Compare native TESSERA v2 embeddings with the pinned PyTorch reference."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.numpy import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--native-output", type=pathlib.Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def load_reference_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("tessera_v2_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_inputs(module, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    s2 = (arrays["S2"].astype(np.float32) - module.S2_BAND_MEAN) / module.S2_BAND_STD
    s2 = np.concatenate([s2, arrays["S2_DOY"].astype(np.float32)[..., None]], axis=-1)

    s1_parts: list[np.ndarray] = []
    teacher_bands: list[np.ndarray] = []
    teacher_days: list[np.ndarray] = []
    is_teacher = hasattr(module, "S1_BAND_MEAN")
    sources = (
        ("S1_ASC", None if is_teacher else module.S1A_BAND_MEAN,
         None if is_teacher else module.S1A_BAND_STD),
        ("S1_DESC", None if is_teacher else module.S1D_BAND_MEAN,
         None if is_teacher else module.S1D_BAND_STD),
    )
    for prefix, mean, standard_deviation in sources:
        bands = arrays.get(prefix)
        day_of_year = arrays.get(f"{prefix}_DOY")
        if bands is None and day_of_year is None:
            continue
        if bands is None or day_of_year is None:
            raise ValueError(f"{prefix} and {prefix}_DOY must be present together")
        if is_teacher:
            teacher_bands.append(bands.astype(np.float32))
            teacher_days.append(day_of_year.astype(np.float32))
        else:
            normalized = (bands.astype(np.float32) - mean) / standard_deviation
            s1_parts.append(
                np.concatenate([normalized, day_of_year.astype(np.float32)[..., None]], axis=-1)
            )
    if is_teacher and teacher_bands:
        merged_bands = np.concatenate(teacher_bands, axis=1)
        merged_days = np.concatenate(teacher_days, axis=1)
        normalized = (merged_bands - module.S1_BAND_MEAN) / module.S1_BAND_STD
        s1_parts.append(np.concatenate([normalized, merged_days[..., None]], axis=-1))
    if not s1_parts:
        raise ValueError("input must contain a Sentinel-1 ascending or descending sequence")
    return s2, np.concatenate(s1_parts, axis=1)


def main() -> int:
    args = parse_args()
    module = load_reference_module(args.reference_model.resolve())
    arrays = load_file(args.input.resolve())
    s2, s1 = prepare_inputs(module, arrays)
    model = module.load_model(args.checkpoint.resolve(), torch.device("cpu"))
    with torch.inference_mode():
        reference = model(
            torch.from_numpy(np.ascontiguousarray(s2)),
            torch.from_numpy(np.ascontiguousarray(s1)),
        ).cpu().numpy()
    native = load_file(args.native_output.resolve())["embeddings"]
    if reference.shape != native.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape}, native {native.shape}")

    difference = np.abs(reference.astype(np.float64) - native.astype(np.float64))
    passed = bool(np.allclose(reference, native, atol=args.atol, rtol=args.rtol))
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "reference": str(args.reference_model.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "input": str(args.input.resolve()),
        "native_output": str(args.native_output.resolve()),
        "shape": list(reference.shape),
        "atol": args.atol,
        "rtol": args.rtol,
        "maximum_absolute_error": float(difference.max(initial=0)),
        "mean_absolute_error": float(difference.mean()),
        "reference_l2_norm": float(np.linalg.norm(reference)),
        "native_l2_norm": float(np.linalg.norm(native)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"reference validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

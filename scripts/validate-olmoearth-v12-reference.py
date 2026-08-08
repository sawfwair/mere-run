#!/usr/bin/env python3
"""Compare native OlmoEarth v1.2 embeddings with the pinned PyTorch reference."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import _load_model_from_config, _load_state_dict
from safetensors.numpy import load_file


MODALITIES = {
    "S2L2A": ("sentinel2_l2a", Modality.SENTINEL2_L2A, "S2L2A_EMBEDDINGS"),
    "S1RTC": ("sentinel1", Modality.SENTINEL1, "S1RTC_EMBEDDINGS"),
    "LANDSAT": ("landsat", Modality.LANDSAT, "LANDSAT_EMBEDDINGS"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=pathlib.Path, required=True)
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--native-output", type=pathlib.Path, required=True)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--input-resolution", type=int, default=10)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = _load_model_from_config(args.configuration.resolve())
    model.load_state_dict(_load_state_dict(args.weights.resolve()))
    model.eval()

    raw = load_file(args.input.resolve())
    timestamps = torch.from_numpy(np.ascontiguousarray(raw["TIMESTAMPS"].astype(np.int64)))
    normalizer = Normalizer(Strategy.COMPUTED)
    sample_values: dict[str, torch.Tensor] = {"timestamps": timestamps}
    present: list[tuple[str, str]] = []
    for input_name, (field, modality, output_name) in MODALITIES.items():
        if input_name not in raw:
            continue
        normalized = normalizer.normalize(modality, raw[input_name].astype(np.float32))
        value = torch.from_numpy(np.ascontiguousarray(normalized.astype(np.float32)))
        mask_shape = (*value.shape[:-1], 1)
        mask = torch.full(mask_shape, float(MaskValue.ONLINE_ENCODER.value))
        sample_values[field] = value
        sample_values[f"{field}_mask"] = mask
        present.append((field, output_name))
    if not present:
        raise ValueError("input must contain S2L2A, S1RTC, or LANDSAT")

    sample = MaskedOlmoEarthSample(**sample_values)
    with torch.inference_mode():
        token_result = model.encoder(
            sample,
            fast_pass=True,
            patch_size=args.patch_size,
            input_res=args.input_resolution,
        )["tokens_and_masks"]
    native = load_file(args.native_output.resolve())

    comparisons: dict[str, dict[str, float | list[int] | str]] = {}
    passed = True
    for field, output_name in present:
        reference_tokens = getattr(token_result, field)
        reference = reference_tokens.mean(dim=(3, 4)).cpu().numpy()
        actual = native[output_name]
        if reference.shape != actual.shape:
            raise ValueError(
                f"{output_name} shape mismatch: reference {reference.shape}, native {actual.shape}"
            )
        difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
        modality_passed = bool(
            np.allclose(reference, actual, atol=args.atol, rtol=args.rtol)
        )
        passed = passed and modality_passed
        comparisons[output_name] = {
            "status": "passed" if modality_passed else "failed",
            "shape": list(reference.shape),
            "maximum_absolute_error": float(difference.max(initial=0)),
            "mean_absolute_error": float(difference.mean()),
            "reference_l2_norm": float(np.linalg.norm(reference)),
            "native_l2_norm": float(np.linalg.norm(actual)),
        }

    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "configuration": str(args.configuration.resolve()),
        "weights": str(args.weights.resolve()),
        "input": str(args.input.resolve()),
        "native_output": str(args.native_output.resolve()),
        "patch_size": args.patch_size,
        "input_resolution": args.input_resolution,
        "atol": args.atol,
        "rtol": args.rtol,
        "comparisons": comparisons,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"reference validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

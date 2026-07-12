#!/usr/bin/env python3
"""Export frozen TripoSR reference tensors from the pinned official code.

Run this only in an isolated reference environment. mere.run never invokes
Python for inference. The script verifies the official checkpoint before using
PyTorch's weights-only loader and requires an explicit checkout of
VAST-AI-Research/TripoSR at the pinned commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch


CHECKPOINT_SIZE = 1_677_246_742
CHECKPOINT_SHA256 = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
UPSTREAM_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_pins(checkpoint: Path, upstream: Path) -> None:
    if checkpoint.stat().st_size != CHECKPOINT_SIZE or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint does not match the pinned TripoSR artifact")
    git_head = upstream / ".git" / "HEAD"
    if not git_head.is_file():
        raise ValueError("--upstream must be a Git checkout of VAST-AI-Research/TripoSR")
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"upstream commit mismatch: expected {UPSTREAM_COMMIT}, found {commit}")


def deterministic_input() -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(512, dtype=np.float32),
        np.arange(512, dtype=np.float32),
        indexing="ij",
    )
    return np.stack(
        [x / 511.0, y / 511.0, ((x * 17 + y * 29) % 257) / 256.0],
        axis=-1,
    )[None]


def export(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    require_pins(checkpoint, upstream)
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(upstream))
    # Mesh extraction/background removal are intentionally outside this tensor
    # fixture. Stub only those optional imports so the reference environment
    # does not need a compiled torchmcubes wheel or an ONNX rembg runtime.
    if importlib.util.find_spec("torchmcubes") is None:
        torchmcubes = types.ModuleType("torchmcubes")
        torchmcubes.marching_cubes = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("marching_cubes is outside this parity fixture")
        )
        sys.modules["torchmcubes"] = torchmcubes
    if importlib.util.find_spec("rembg") is None:
        rembg = types.ModuleType("rembg")
        rembg.remove = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("rembg is outside this parity fixture")
        )
        sys.modules["rembg"] = rembg
    from einops import rearrange
    from omegaconf import OmegaConf
    from tsr.system import TSR

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)

    image = deterministic_input()
    input_tensor = torch.from_numpy(image).to(args.device)
    with torch.no_grad():
        rgb_cond = input_tensor[:, None]
        normalized = (
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1)
            - model.image_tokenizer.image_mean
        ) / model.image_tokenizer.image_std
        vit = model.image_tokenizer.model(
            rearrange(normalized, "B N C H W -> (B N) C H W"),
            interpolate_pos_encoding=True,
        )
        image_tokens = vit.last_hidden_state.permute(0, 2, 1)
        encoder_hidden = rearrange(image_tokens, "B C N -> B N C")
        tokens_initial = model.tokenizer(1)
        tokens_final = model.backbone(tokens_initial, encoder_hidden_states=encoder_hidden)
        scene_code = model.post_processor(model.tokenizer.detokenize(tokens_final))
        positions = torch.tensor(
            [
                [-0.87, -0.87, -0.87],
                [-0.25, 0.125, 0.5],
                [0.0, 0.0, 0.0],
                [0.7, -0.4, 0.2],
                [0.87, 0.87, 0.87],
            ],
            dtype=torch.float32,
            device=args.device,
        )
        field = model.renderer.query_triplane(model.decoder, positions, scene_code[0])

    tensors = {
        "input": image,
        "image_tokens_bcn": image_tokens.float().cpu().numpy(),
        "tokens_initial_bcn": tokens_initial.float().cpu().numpy(),
        "tokens_final_bcn": tokens_final.float().cpu().numpy(),
        "scene_code_bpc_hw": scene_code.float().cpu().numpy(),
        "query_positions": positions.float().cpu().numpy(),
        "query_density": field["density"].float().cpu().numpy(),
        "query_density_activated": field["density_act"].float().cpu().numpy(),
        "query_color": field["color"].float().cpu().numpy(),
    }
    records = {}
    for name, tensor in tensors.items():
        path = output / f"{name}.npy"
        np.save(path, tensor, allow_pickle=False)
        records[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": sha256(path),
        }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "checkpointRevision": "5b521936b01fbe1890f6f9baed0254ab6351c04a",
                "checkpointSHA256": CHECKPOINT_SHA256,
                "device": args.device,
                "tensors": records,
                "upstreamCommit": UPSTREAM_COMMIT,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--upstream", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return result


if __name__ == "__main__":
    export(parser().parse_args())

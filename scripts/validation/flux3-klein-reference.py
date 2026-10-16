#!/usr/bin/env python3
"""Generate tiny CPU parity fixtures from pinned BFL code, never product inference.

Python 3.12; torch==2.8.0 numpy==2.3.3 safetensors==0.6.2 packaging==25.0.
The reference source is Apache-2.0 (see THIRD_PARTY_NOTICES.md).
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import urllib.request

import torch
from safetensors.torch import save_file

REVISION = "45e57c9fb06f9751cdced29a65c5d2171003be05"
DIGEST = "d4c0d2aea5282ae63ac32a1f3029a738ddab951bb1033d6f3a2e9bd192c89934"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Tests/ImageRuntimeTests/Fixtures/Flux3"


def reference(name, digest):
    source = urllib.request.urlopen(
        f"https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/{REVISION}/src/flux3/{name}.py"
    ).read()
    assert hashlib.sha256(source).hexdigest() == digest
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / f"flux3_reference_{name}.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(f"flux3_reference_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(1)
    module = reference("dit", DIGEST)
    module.HIDDEN, module.HEADS = 16, 2
    module.DEPTH, module.DEPTH_SINGLE = 1, 2
    module.AXES_DIM = (2, 2, 2, 2)
    module.IN_CHANNELS = {"image": 8, "image_cond": 8, "text": 12}
    torch.manual_seed(42)
    model = module.DiT(flow_map=True).eval()
    for name, parameter in model.named_parameters():
        with torch.no_grad():
            parameter.copy_(torch.randn_like(parameter) * 0.08 + (1 if name.endswith(".scale") else 0))
    inputs = {
        "image": torch.randn(3, 8), "text": torch.randn(5, 12), "references": torch.randn(4, 8),
        "image_ids": torch.tensor([[0, 0, 0, 0], [0, 0, .5, 0], [0, .5, 0, 0]]),
        "text_ids": torch.tensor([[0, 0, 0, i] for i in range(5)], dtype=torch.float32),
        "reference_ids": torch.tensor([[0, 0, 0, -1], [0, 0, .5, -1], [0, 0, 0, -2], [0, .5, 0, -2]]),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(OUTPUT / "weights.safetensors"))
    save_file(inputs, str(OUTPUT / "inputs.safetensors"))
    for dtype, suffix in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        model.to(dtype)

        def predict(image, time, target, references):
            data = {"image": image, "text": inputs["text"].to(dtype),
                    "image_cond": inputs["references"].to(dtype) if references else torch.empty(0, 8, dtype=dtype)}
            ids = {"image": inputs["image_ids"], "text": inputs["text_ids"],
                   "image_cond": inputs["reference_ids"] if references else torch.empty(0, 4)}
            times = {m: torch.full((v.shape[0],), time if m == "image" else 0, dtype=dtype) for m, v in data.items()}
            pack = module.Packed(data, ids, times, {m: [(0, v.shape[0])] for m, v in data.items()})
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                return model(pack, torch.tensor([target], dtype=dtype)).float()

        results = {
            "text_only": predict(inputs["image"], .75, .5, False),
            "references": predict(inputs["image"], .413725, .125, True),
            "changed_target": predict(inputs["image"], .413725, .25, True),
        }
        schedule = torch.linspace(1, 0, 4)
        schedule = 6.93 * schedule / (1 + (6.93 - 1) * schedule)
        state = inputs["image"].clone()
        for step in range(3):
            state = state + (schedule[step + 1] - schedule[step]) * predict(
                state, schedule[step].item(), schedule[step + 1].item(), True)
        results["trajectory"] = state
        save_file({k: v.contiguous() for k, v in results.items()}, str(OUTPUT / f"results-{suffix}.safetensors"))

    vae_module = reference("autoencoder", "960e2a98046bcb78f2d36fbbd4848db726f14810c7887485f7e2682346f95189")
    vae_output = ROOT / "Tests/MereRunCoreTests/Fixtures/Flux3"
    vae_output.mkdir(exist_ok=True)
    with torch.device("meta"):
        original = vae_module.AutoEncoder()
    import json
    (vae_output / "vae-source-shapes.json").write_text(
        json.dumps({k: list(v.shape) for k, v in original.state_dict().items()}, indent=2) + "\n")
    vae_module.CH, vae_module.CH_MULT = 32, (1, 1, 1, 1)
    torch.manual_seed(43)
    vae = vae_module.AutoEncoder().eval()
    with torch.no_grad():
        for name, parameter in vae.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.04 + (1 if "norm" in name and name.endswith("weight") else 0))
        vae.bn.running_mean.copy_(torch.randn(128) * 0.1)
        vae.bn.running_var.copy_(torch.rand(128) + 0.8)
        pixels = torch.randn(3, 32, 32).clamp(-1, 1)
        encoded, ids = vae.encode(pixels)
        decoded = vae.decode(encoded, ids)
        results = {"pixels": pixels, "encoded": encoded.to(torch.bfloat16).float(), "decode_tokens": encoded,
                   "decoded": ((decoded.clamp(-1, 1) + 1) / 2).unsqueeze(0)}
    save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, str(vae_output / "vae-weights.safetensors"))
    save_file({k: v.contiguous() for k, v in results.items()}, str(vae_output / "vae-results.safetensors"))


if __name__ == "__main__":
    main()

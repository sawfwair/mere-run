#!/usr/bin/env python3
"""Generate small CPU fixtures from pinned Apache-2.0 Diffusers model code.

Test-only dependencies: torch 2.15.0, diffusers 0.42.1, safetensors, numpy.
Pass --reference-dir to use previously downloaded, hash-checked source files.
No checkpoint weights are downloaded or executed by this script.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import urllib.request

import torch
from safetensors.torch import save_file

REVISION = "f598e6e6a99528858b962c1399c45b59a6c25484"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Tests/H3RuntimeTests/Fixtures/H3Image"


def reference(name, folder, digest, directory):
    source = (directory / f"{name}.py").read_bytes() if directory else urllib.request.urlopen(
        f"https://raw.githubusercontent.com/huggingface/diffusers/{REVISION}/src/diffusers/{folder}/{name}.py"
    ).read()
    assert hashlib.sha256(source).hexdigest() == digest
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / f"{name}.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(f"diffusers.{folder.replace('/', '.')}.h3_image_reference", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def mapped_weights(model, decoder=False):
    result = {}
    state = model.state_dict()
    for key, value in state.items():
        if ".to_k." in key or ".to_v." in key or key.startswith("pdd_proj_out."):
            continue
        if ".to_q." in key:
            value = torch.cat([state[key.replace(".to_q.", f".to_{part}.")] for part in ["q", "k", "v"]])
            key = key.replace(".to_q.", ".to_qkv." if decoder else ".qkv_proj.")
        # Original H3 stores [gate; value]; Diffusers SwiGLU stores [value; gate].
        if ".ff.net.0.proj." in key:
            up, gate = value.chunk(2, dim=0)
            value = torch.cat([gate, up])
        if decoder:
            key = key.replace(".to_out.0.", ".to_out.").replace(".ff.net.0.proj.", ".ff.linear_in.").replace(".ff.net.2.", ".ff.linear_out.")
        else:
            key = key.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
            key = key.replace("transformer_blocks.", "blocks.").replace("context_embedder.", "condition_proj.")
            key = key.replace("proj_in.", "video_patch_proj.") if key.startswith("proj_in.") else key
            key = key.replace("time_embedder.linear_1.", "time_embedder.proj_in.").replace("time_embedder.linear_2.", "time_embedder.proj_out.")
            key = key.replace(".attn.norm_q.", ".attn.q_norm.").replace(".attn.norm_k.", ".attn.k_norm.")
            key = key.replace(".attn.to_out.0.", ".attn.out_proj.")
            key = key.replace(".ff.net.0.proj.", ".mlp.fc1.").replace(".ff.net.2.", ".mlp.fc2.")
            key = key.replace("norm_out.norm.", "final_layer.norm.").replace("norm_out.linear.", "final_layer.adaln_proj.linear.")
        result[key] = value.contiguous()
    if not decoder:
        result["final_layer.pdd_weight"] = torch.stack([head.weight for head in model.pdd_proj_out]).contiguous()
        result["final_layer.pdd_bias"] = torch.stack([head.bias for head in model.pdd_proj_out]).contiguous()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    transformer_source = reference("transformer_minimax_h3_image", "models/transformers", "9a83cb4ca0f28ca74bcb56da2ae75a309fc91c7e73f17248fb4c2ac8b8fff400", args.reference_dir)
    vae_source = reference("autoencoder_kl_minimax_h3_image", "models/autoencoders", "eec19a79e4b5a57ac365632a6a11fbddf2a325d357953b8119d61dc75621fc25", args.reference_dir)
    scheduler_source = reference("scheduling_minimax_h3", "schedulers", "ee59acb005d7b6e38c520b9c8846f41c9e4fc7fbdbc0128ee5880002a2b2faf4", args.reference_dir)
    model = transformer_source.MiniMaxH3ImageTransformer2DModel(num_attention_heads=2, attention_head_dim=8,
        hidden_size=12, num_layers=2, num_refiner_layers=1, ffn_dim=16, text_dim=10,
        freq_dim=8, time_embed_hidden_dim=12, time_embed_dim=6, rope_freq_dim=1).eval()
    decoder = vae_source.MiniMaxH3ImageViTDecoder2d(num_layers=2, num_attention_heads=2, attention_head_dim=8).eval()
    for module in [model, decoder]:
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                offset = 1 if "norm" in name and name.endswith("weight") else 0
                parameter.copy_(torch.randn_like(parameter) * .08 + offset)
    inputs = {"text": torch.randn(1, 3, 10), "image": torch.randn(1, 4, 96),
              "reference": torch.randn(1, 1, 96), "decoder_latents": torch.randn(1, 24, 2, 3)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_file(inputs, str(OUTPUT / "inputs.safetensors"))
    sigmas = [1.0, 0.9675236940383911, 0.8875485062599182, .75, 0.5327739119529724, 0.23200887441635132, 0]
    times = torch.tensor([1 - s for s in sigmas[:-1]], dtype=torch.float32)
    schedule = torch.cat([1 - times, torch.zeros(1)])
    for mixed in [False, True]:
        suffix = "bf16" if mixed else "fp32"
        if mixed:
            model.to(torch.bfloat16)
            for module in [model.time_embedder, model.proj_in, model.pdd_proj_out, model.rope]:
                module.float()
        save_file(mapped_weights(model), str(OUTPUT / f"weights-{suffix}.safetensors"))

        def predict(image, step, references):
            # Independent explicit grid: three text rows, an optional 1-token reference, 2x2 target.
            positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
            if references:
                positions += [[3, 0, 0]]
            positions += [[3 + int(references), y, x] for y in [0, 16] for x in [0, 16]]
            rows = torch.cat([inputs["reference"], image], dim=1) if references else image
            tags = torch.tensor([1, 0, 1] + [0] * rows.shape[1])
            time_ids = torch.tensor([0, 0, 0] + ([1] if references else []) + [0] * 4)
            with torch.no_grad():
                return model(rows, inputs["text"], torch.tensor([times[step], .999]), time_ids, tags,
                    torch.tensor(positions), torch.arange(3, len(positions)), torch.arange(3), step,
                    return_dict=False)[0][:, -4:].float()

        outputs = {}
        for references in [False, True]:
            label = "edit" if references else "text"
            state = inputs["image"].clone()
            scheduler = scheduler_source.MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
            scheduler.set_timesteps(sigmas=schedule)
            for step in range(6):
                outputs[f"{label}_step_{step}"] = predict(inputs["image"], step, references)
                state = scheduler.step(predict(state, step, references), scheduler.timesteps[step], state, return_dict=False)[0]
            outputs[f"{label}_trajectory"] = state
        save_file(outputs, str(OUTPUT / f"outputs-{suffix}.safetensors"))
    with torch.no_grad():
        save_file(mapped_weights(decoder, decoder=True), str(OUTPUT / "decoder-weights.safetensors"))
        save_file({"rgb": decoder(inputs["decoder_latents"])}, str(OUTPUT / "decoder-output.safetensors"))


if __name__ == "__main__":
    main()

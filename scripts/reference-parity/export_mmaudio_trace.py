#!/usr/bin/env python3
"""Export deterministic PyTorch MMAudio component traces for Swift parity tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import open_clip
import torch
from safetensors.torch import load_file, save_file
from torchvision.transforms import v2


PROMPT = "a wooden door slamming shut"


def deterministic(shape: tuple[int, ...], *, scale: float, phase: float = 0) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    values = torch.arange(count, dtype=torch.float32)
    return torch.sin(values * scale + phase).reshape(shape)


def cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    # Clone on the source device before crossing to CPU. PyTorch/MPS otherwise
    # loses the storage offset of some chunk/slice views during the transfer.
    return tensor.detach().clone().to(device="cpu", dtype=torch.float32).contiguous()


def export_preprocessing(traces: dict[str, torch.Tensor]) -> None:
    height, width = 241, 319
    values = torch.arange(height * width * 3, dtype=torch.int64).reshape(height, width, 3)
    rgb = ((values * 37 + 17) % 256).to(torch.uint8)
    chw = rgb.permute(2, 0, 1)
    clip = v2.Resize((384, 384), interpolation=v2.InterpolationMode.BICUBIC)(chw)
    clip = v2.ToDtype(torch.float32, scale=True)(clip)
    clip = v2.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )(clip)
    sync = v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC)(chw)
    sync = v2.CenterCrop(224)(sync)
    sync = v2.ToDtype(torch.float32, scale=True)(sync)
    sync = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(sync)
    traces["preprocess_rgb_hwc_u8"] = rgb.contiguous()
    traces["preprocess_clip_nhwc"] = clip.permute(1, 2, 0).unsqueeze(0).contiguous()
    traces["preprocess_sync_chw"] = sync.contiguous()


def export_clip(root: Path, device: torch.device, traces: dict[str, torch.Tensor]) -> None:
    model = open_clip.create_model("ViT-H-14-378-quickgelu", pretrained=None)
    state = load_file(str(root / "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"))
    state.pop("logit_bias", None)
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device=device, dtype=torch.float16)
    tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    token_ids = tokenizer([PROMPT])

    def encode_hidden(tokens: torch.Tensor) -> torch.Tensor:
        hidden = model.token_embedding(tokens).to(torch.float16)
        hidden = hidden + model.positional_embedding.to(torch.float16)
        hidden = model.transformer(hidden, attn_mask=model.attn_mask)
        return torch.nn.functional.normalize(model.ln_final(hidden), dim=-1)

    image = deterministic((1, 384, 384, 3), scale=0.000_13, phase=0.2)
    image = image.permute(0, 3, 1, 2).to(device=device, dtype=torch.float16)
    with torch.inference_mode():
        text = encode_hidden(token_ids.to(device))
        visual = model.encode_image(image, normalize=True)
    traces["clip_token_ids"] = token_ids.to(torch.int32).contiguous()
    traces["clip_text_hidden"] = cpu_float(text)
    traces["clip_image_nhwc"] = cpu_float(image.permute(0, 2, 3, 1))
    traces["clip_image_features"] = cpu_float(visual)


def export_network(root: Path, device: torch.device, traces: dict[str, torch.Tensor]) -> None:
    from mmaudio.model.networks import get_my_mmaudio

    model = get_my_mmaudio("large_44k_v2")
    state = load_file(str(root / "mmaudio_large_44k_v2_fp16.safetensors"))
    model.load_weights(state)
    model.update_seq_lengths(44, 8, 16)
    model = model.eval().to(device=device, dtype=torch.float16)
    latent = deterministic((1, 44, 40), scale=0.013, phase=0.1)
    clip = deterministic((1, 8, 1024), scale=0.003, phase=0.2)
    sync = deterministic((1, 16, 768), scale=0.002, phase=0.3)
    text = deterministic((1, 77, 1024), scale=0.001, phase=0.4)
    timestep = torch.tensor([0.25], dtype=torch.float32)
    with torch.inference_mode():
        projected = model.preprocess_conditions(
            clip.to(device=device, dtype=torch.float16),
            sync.to(device=device, dtype=torch.float16),
            text.to(device=device, dtype=torch.float16),
        )
        latent_device = latent.to(device=device, dtype=torch.float16)
        audio_projected = model.audio_input_proj(latent_device)
        timestep_device = timestep.to(device=device, dtype=torch.float16)
        global_base = model.global_cond_mlp(projected.clip_f_c + projected.text_f_c)
        global_condition = model.t_embed(timestep_device).unsqueeze(1) + global_base.unsqueeze(1)
        extended_condition = global_condition + projected.sync_f
        block = model.joint_blocks[0]
        latent_qkv, _ = block.latent_block.pre_attention(
            audio_projected,
            extended_condition,
            model.latent_rot,
        )
        latent_hidden = audio_projected
        clip_hidden = projected.clip_f
        text_hidden = projected.text_f
        block_outputs: dict[str, torch.Tensor] = {}
        for index, joint_block in enumerate(model.joint_blocks):
            latent_hidden, clip_hidden, text_hidden = joint_block(
                latent_hidden,
                clip_hidden,
                text_hidden,
                global_condition,
                extended_condition,
                model.latent_rot,
                model.clip_rot,
            )
            block_outputs[f"network_joint_{index}_latent"] = cpu_float(latent_hidden)
            block_outputs[f"network_joint_{index}_clip"] = cpu_float(clip_hidden)
            block_outputs[f"network_joint_{index}_text"] = cpu_float(text_hidden)
        for index, fused_block in enumerate(model.fused_blocks):
            latent_hidden = fused_block(latent_hidden, extended_condition, model.latent_rot)
            block_outputs[f"network_fused_{index}_latent"] = cpu_float(latent_hidden)
        final_modulation = model.final_layer.adaLN_modulation(global_condition)
        final_shift, final_scale = final_modulation.chunk(2, dim=-1)
        final_normalized = model.final_layer.norm(latent_hidden)
        final_modulated = final_normalized * (1 + final_scale) + final_shift
        block_outputs["network_final_modulation_raw"] = cpu_float(final_modulation)
        block_outputs["network_final_shift"] = cpu_float(final_shift)
        block_outputs["network_final_scale"] = cpu_float(final_scale)
        block_outputs["network_final_normalized"] = cpu_float(final_normalized)
        block_outputs["network_final_modulated"] = cpu_float(final_modulated)
        block_outputs["network_final_from_blocks"] = cpu_float(
            model.final_layer(latent_hidden, global_condition)
        )
        flow = model.predict_flow(
            latent_device,
            timestep_device,
            projected,
        )
    traces["network_latent_ntc"] = latent.contiguous()
    traces["network_clip_ntc"] = clip.contiguous()
    traces["network_sync_ntc"] = sync.contiguous()
    traces["network_text_ntc"] = text.contiguous()
    traces["network_timestep"] = timestep.contiguous()
    traces["network_final_conv_weight_oik"] = cpu_float(state["final_layer.conv.weight"])
    traces["network_final_conv_bias"] = cpu_float(state["final_layer.conv.bias"])
    traces["network_final_adaln_weight"] = cpu_float(state["final_layer.adaLN_modulation.1.weight"])
    traces["network_final_adaln_bias"] = cpu_float(state["final_layer.adaLN_modulation.1.bias"])
    traces["network_projected_clip"] = cpu_float(projected.clip_f)
    traces["network_projected_sync"] = cpu_float(projected.sync_f)
    traces["network_projected_text"] = cpu_float(projected.text_f)
    traces["network_clip_global"] = cpu_float(projected.clip_f_c)
    traces["network_text_global"] = cpu_float(projected.text_f_c)
    traces["network_audio_projected"] = cpu_float(audio_projected)
    traces["network_timestep_embedding"] = cpu_float(model.t_embed(timestep_device))
    traces["network_global_condition"] = cpu_float(global_condition)
    traces["network_extended_condition"] = cpu_float(extended_condition)
    traces["network_block0_latent_query"] = cpu_float(latent_qkv[0])
    traces["network_block0_latent_key"] = cpu_float(latent_qkv[1])
    traces["network_block0_latent_value"] = cpu_float(latent_qkv[2])
    traces.update(block_outputs)
    traces["network_flow_ntc"] = cpu_float(flow)


def export_vae(root: Path, device: torch.device, traces: dict[str, torch.Tensor]) -> torch.Tensor:
    from mmaudio.ext.autoencoder.vae import get_my_vae

    model = get_my_vae("44k")
    incompatible = model.load_state_dict(
        load_file(str(root / "mmaudio_vae_44k_fp16.safetensors")),
        strict=False,
    )
    if incompatible.unexpected_keys or any(not key.startswith("encoder.") for key in incompatible.missing_keys):
        raise RuntimeError(f"Unexpected VAE checkpoint inventory: {incompatible}")
    model.remove_weight_norm()
    model = model.eval().to(device=device, dtype=torch.float16)
    raw_input_weight = load_file(str(root / "mmaudio_vae_44k_fp16.safetensors"))["decoder.conv_in.weight"]
    traces["vae_input_weight_raw_oik"] = cpu_float(raw_input_weight)
    traces["vae_input_weight_effective_oik"] = cpu_float(model.decoder.conv_in.weight)
    latent_ntc = deterministic((1, 4, 40), scale=0.017, phase=0.5)
    with torch.inference_mode():
        latent_nct = latent_ntc.transpose(1, 2).to(device=device, dtype=torch.float16)
        decoder = model.decoder
        hidden = decoder.conv_in(latent_nct)
        traces["vae_input_projection_ntc"] = cpu_float(hidden.transpose(1, 2))
        hidden = decoder.mid.block_1(hidden)
        traces["vae_middle_first_ntc"] = cpu_float(hidden.transpose(1, 2))
        hidden = decoder.mid.attn_1(hidden)
        traces["vae_middle_attention_ntc"] = cpu_float(hidden.transpose(1, 2))
        hidden = decoder.mid.block_2(hidden).clamp(-decoder.clip_act, decoder.clip_act)
        traces["vae_middle_second_ntc"] = cpu_float(hidden.transpose(1, 2))
        for trace_index, level_index in enumerate(reversed(range(decoder.num_layers))):
            for block_index in range(decoder.num_res_blocks + 1):
                hidden = decoder.up[level_index].block[block_index](hidden)
                if len(decoder.up[level_index].attn) > 0:
                    hidden = decoder.up[level_index].attn[block_index](hidden)
                hidden = hidden.clamp(-decoder.clip_act, decoder.clip_act)
            if level_index in decoder.down_layers:
                hidden = decoder.up[level_index].upsample(hidden)
            traces[f"vae_up_{trace_index}_ntc"] = cpu_float(hidden.transpose(1, 2))
        hidden = torch.nn.functional.silu(hidden) / 0.596
        traces["vae_pre_output_ntc"] = cpu_float(hidden.transpose(1, 2))
        raw = decoder.conv_out(hidden, gain=(decoder.learnable_gain + 1))
        traces["vae_raw_spectrogram_ntc"] = cpu_float(raw.transpose(1, 2))
        spectrogram = model.unnormalize(raw)
    traces["vae_latent_ntc"] = latent_ntc.contiguous()
    traces["vae_spectrogram_ntc"] = cpu_float(spectrogram.transpose(1, 2))
    return spectrogram


def export_bigvgan(
    root: Path,
    device: torch.device,
    spectrogram: torch.Tensor,
    traces: dict[str, torch.Tensor],
) -> None:
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN, load_hparams_from_json

    config = load_hparams_from_json(root / "bigvgan" / "config.json")
    model = BigVGAN(config, use_cuda_kernel=False)
    checkpoint = torch.load(
        root / "bigvgan" / "bigvgan_generator.pt",
        map_location="cpu",
        weights_only=True,
    )["generator"]
    model.load_state_dict(checkpoint, strict=True)
    model = model.eval().to(device=device, dtype=torch.float16)
    with torch.inference_mode():
        waveform = model(spectrogram.to(device=device, dtype=torch.float16))
    traces["bigvgan_spectrogram_nct"] = cpu_float(spectrogram)
    traces["bigvgan_waveform_nct"] = cpu_float(waveform)


def export_synchformer(root: Path, device: torch.device, traces: dict[str, torch.Tensor]) -> None:
    from mmaudio.ext.synchformer import Synchformer

    model = Synchformer()
    model.load_state_dict(
        load_file(str(root / "mmaudio_synchformer_fp16.safetensors")),
        strict=True,
    )
    model = model.eval().to(device=device, dtype=torch.float32)
    frames_tchw = deterministic((16, 3, 224, 224), scale=0.000_011, phase=0.6)
    with torch.inference_mode():
        output = model(frames_tchw.unsqueeze(0).unsqueeze(0).to(device))
    traces["synchformer_frames_tchw"] = frames_tchw.contiguous()
    traces["synchformer_features_ntc"] = cpu_float(output.reshape(1, 8, 768))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--mmaudio-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--components",
        default="preprocess,clip,network,vae,bigvgan,synchformer",
        help="Comma-separated trace groups.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    root = args.model_root.resolve()
    source = args.mmaudio_source.resolve()
    sys.path.insert(0, str(source))
    requested = set(args.components.split(","))
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    traces: dict[str, torch.Tensor] = {}
    if "preprocess" in requested:
        export_preprocessing(traces)
    if "clip" in requested:
        export_clip(root, device, traces)
    if "network" in requested:
        export_network(root, device, traces)
    spectrogram = export_vae(root, device, traces) if {"vae", "bigvgan"} & requested else None
    if "bigvgan" in requested:
        assert spectrogram is not None
        export_bigvgan(root, device, spectrogram, traces)
    if "synchformer" in requested:
        export_synchformer(root, device, traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(traces, str(args.output), metadata={
        "prompt": PROMPT,
        "mmaudio_revision": "974010a026c731054592d8f777218bd9d85a6c24",
        "device": str(device),
    })
    print(json.dumps({"output": str(args.output), "tensors": sorted(traces), "device": str(device)}))


if __name__ == "__main__":
    main()

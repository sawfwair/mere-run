#!/usr/bin/env python3
"""Export a tiny real-weight DreamX AR causal-transformer fixture."""

import argparse
import json
import sys
from pathlib import Path


def upstream_tensor(key, value):
    if key == "patch_embedding_proj.weight":
        return "patch_embedding.weight", value.reshape(3072, 48, 1, 2, 2)
    exact = {
        "patch_embedding_proj.bias": "patch_embedding.bias",
        "text_embedding_0.weight": "text_embedding.0.weight",
        "text_embedding_0.bias": "text_embedding.0.bias",
        "text_embedding_1.weight": "text_embedding.2.weight",
        "text_embedding_1.bias": "text_embedding.2.bias",
        "time_embedding_0.weight": "time_embedding.0.weight",
        "time_embedding_0.bias": "time_embedding.0.bias",
        "time_embedding_1.weight": "time_embedding.2.weight",
        "time_embedding_1.bias": "time_embedding.2.bias",
        "time_projection.weight": "time_projection.1.weight",
        "time_projection.bias": "time_projection.1.bias",
    }
    if key in exact:
        return exact[key], value
    if ".ffn.fc1." in key:
        return key.replace(".ffn.fc1.", ".ffn.0."), value
    if ".ffn.fc2." in key:
        return key.replace(".ffn.fc2.", ".ffn.2."), value
    return key, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamx-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--camera-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import save_file

    sys.path.insert(0, str(args.dreamx_root))
    from wan.modules.causal_camera_model_2_2_prope_infinity import CausalWanModel
    import wan.modules.model_2_2 as wan_model
    from wan.modules.model_2_2 import rope_params

    def cpu_flash_attention(q, k, v, k_lens=None, softmax_scale=None, **_):
        """Match DreamX's flash-attention contract with portable PyTorch SDPA."""
        if k_lens is not None:
            assert q.shape[0] == 1
            key_length = int(k_lens[0])
            k = k[:, :key_length]
            v = v[:, :key_length]
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=softmax_scale,
        )
        return output.transpose(1, 2).to(v.dtype)

    wan_model.flash_attention = cpu_flash_attention

    with torch.device("meta"):
        model = CausalWanModel(
            model_type="ti2v",
            in_dim=48,
            dim=3072,
            ffn_dim=14336,
            freq_dim=256,
            text_dim=4096,
            out_dim=48,
            num_heads=24,
            num_layers=30,
            local_attn_size=12,
            sink_size=3,
            cross_attn_norm=True,
            add_control_adapter=True,
            cam_method="prope",
            attn_compress=4,
        )
    state = {}
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        for key in source.keys():
            mapped_key, value = upstream_tensor(key, source.get_tensor(key))
            state[mapped_key] = value
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    del state
    head_dimension = 128
    model.freqs = torch.cat([
        rope_params(1024, head_dimension - 4 * (head_dimension // 6)),
        rope_params(1024, 2 * (head_dimension // 6)),
        rope_params(1024, 2 * (head_dimension // 6)),
    ], dim=1)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm) and module.elementwise_affine:
            module.weight.data = module.weight.data.float()
            module.bias.data = module.bias.data.float()
    model.eval()

    camera_json = json.loads(args.camera_fixture.read_text())
    camera_frames = camera_json["view_matrices"]["shape"][1]
    views = torch.tensor(camera_json["view_matrices"]["values"], dtype=torch.bfloat16).reshape(
        1, camera_frames, 4, 4
    )[:, :3]
    intrinsics = torch.tensor(camera_json["intrinsics"]["values"], dtype=torch.bfloat16).reshape(
        1, camera_frames, 3, 3
    )[:, :3]
    latent_values = torch.arange(48 * 3 * 2 * 4, dtype=torch.float32).reshape(48, 3, 2, 4)
    latent = (torch.sin(latent_values / 79) * 0.25).to(torch.bfloat16)
    context_values = torch.arange(4 * 4096, dtype=torch.float32).reshape(4, 4096)
    context = (torch.cos(context_values / 113) * 0.125).to(torch.bfloat16)
    timesteps = torch.full((1, 6), 937.5, dtype=torch.float32)
    timesteps[:, :2] = 0
    cache_tokens = 12 * 2
    caches = [{
        "k": torch.zeros((1, cache_tokens, 24, 128), dtype=torch.bfloat16),
        "v": torch.zeros((1, cache_tokens, 24, 128), dtype=torch.bfloat16),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
    } for _ in range(30)]
    cross_caches = [{"is_init": False} for _ in range(30)]
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(
            x=[latent],
            t=timesteps,
            context=[context],
            seq_len=6,
            y_camera={"viewmats": views, "K": intrinsics},
            kv_cache=caches,
            crossattn_cache=cross_caches,
            current_start=0,
        )[0]
        recompute_input = latent + (torch.cos(latent_values / 53) * 0.01).to(torch.bfloat16)
        recompute_timesteps = torch.full((1, 6), 833.333333, dtype=torch.float32)
        recompute_timesteps[:, :2] = 0
        recompute_output = model(
            x=[recompute_input],
            t=recompute_timesteps,
            context=[context],
            seq_len=6,
            y_camera={"viewmats": views, "K": intrinsics},
            kv_cache=caches,
            crossattn_cache=cross_caches,
            current_start=0,
        )[0]
        second_values = latent_values + 48 * 3 * 2 * 4
        second_input = (torch.sin(second_values / 71) * 0.25).to(torch.bfloat16)
        second_timesteps = torch.full((1, 6), 937.5, dtype=torch.float32)
        second_views = views.clone()
        second_views[:, :, 0, 3] += torch.tensor(
            [0.025, 0.05, 0.075], dtype=torch.bfloat16
        ).reshape(1, 3)
        second_output = model(
            x=[second_input],
            t=second_timesteps,
            context=[context],
            seq_len=6,
            y_camera={"viewmats": second_views, "K": intrinsics},
            kv_cache=caches,
            crossattn_cache=cross_caches,
            current_start=6,
        )[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({
        "input": latent.contiguous(),
        "timesteps": timesteps.contiguous(),
        "context": context.contiguous(),
        "view_matrices": views.contiguous(),
        "intrinsics": intrinsics.contiguous(),
        "output": output.float().contiguous(),
        "recompute_input": recompute_input.contiguous(),
        "recompute_timesteps": recompute_timesteps.contiguous(),
        "recompute_output": recompute_output.float().contiguous(),
        "second_input": second_input.contiguous(),
        "second_timesteps": second_timesteps.contiguous(),
        "second_view_matrices": second_views.contiguous(),
        "second_output": second_output.float().contiguous(),
        "block0_cache_key": caches[0]["k"][:, :12].float().contiguous(),
        "block0_cache_value": caches[0]["v"][:, :12].float().contiguous(),
    }, args.output, metadata={
        "source_revision": "AMAP-ML/DreamX-World@f2bf6bf",
        "checkpoint_revision": "67487c4a61466bb7166d30b7187dd465e0ac9f6c",
        "compute_dtype": "bfloat16",
        "local_attention_frames": "12",
        "sink_frames": "3",
    })
    print(args.output)


if __name__ == "__main__":
    main()

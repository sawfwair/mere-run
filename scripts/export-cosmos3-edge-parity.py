#!/usr/bin/env python3
"""Export deterministic Cosmos3-Edge parity fixtures from NVIDIA upstream.

Run this script from the root of the pinned NVIDIA Cosmos framework checkout:

  python /path/to/export-cosmos3-edge-parity.py \
    --checkpoint-root /path/to/Cosmos3-Edge/snapshot \
    --output /path/to/output

The tiny MoT layer fixture intentionally exercises the Nemotron-specific Edge
attention contract without copying model weights into the Mere source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_all_seq,
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, MoTDecoderLayer
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.vision_siglip2 import (
    PatchMerger,
    Siglip2VisionTransformer,
    patch_merging_by_param,
)
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig


EXPECTED_UPSTREAM_REVISION = "ed8287fd7477113f8ac4f6b84290514d55cf0cdc"
EXPECTED_MODEL_REVISION = "6f58f6b4c91288838e60b6bcb2cc45d997e961de"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-revision-mismatch", action="store_true")
    return parser.parse_args()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def checkpoint_revision(root: Path) -> str:
    if root.name:
        return root.name
    return ""


def deterministic_parameter_(parameter: torch.Tensor, index: int) -> None:
    values = torch.arange(parameter.numel(), dtype=torch.float32, device=parameter.device)
    values = torch.sin(values * 0.013 + (index + 1) * 0.17) * 0.1
    parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))


def tiny_layer_fixture(output: Path) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Nemotron3DenseVLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        num_key_value_heads=1,
        layer_norm_epsilon=1e-5,
        rope_theta=100_000_000.0,
        mrope_section=[2, 1, 1],
        torch_dtype="float32",
    )
    layer_types = LayerTypes("nemotron_dense")
    layer = MoTDecoderLayer(
        config,
        layer_idx=0,
        layer_types=layer_types,
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for index, (_, parameter) in enumerate(layer.named_parameters()):
            deterministic_parameter_(parameter, index)

    packed_input = (
        torch.sin(torch.arange(5 * 16, device=device, dtype=torch.float32) * 0.071 + 0.2)
        .reshape(5, 16)
        .contiguous()
    )
    position_ids = torch.tensor(
        [
            [0.0, 1.0, 2.0, 15_003.0, 15_003.0],
            [0.0, 1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    pack = sequence_pack_from_packed_sequence(
        packed_sequence=packed_input,
        attn_modes=["causal", "full"],
        split_lens=[3, 2],
        sample_lens=[5],
        packed_und_token_indexes=torch.tensor([0, 1, 2], device=device),
        packed_gen_token_indexes=torch.tensor([3, 4], device=device),
    )
    rotary = MultiModalRotaryEmbedding(config, device=device).to(device)
    with torch.inference_mode():
        cosine, sine = rotary(packed_input, position_ids.unsqueeze(1))
        cosine_pack = from_all_seq(cosine[0], pack)
        sine_pack = from_all_seq(sine[0], pack)
        attention_mask = SplitInfo(
            split_lens=[3, 2],
            attn_modes=["causal", "full"],
            sample_lens=[5],
            actual_len=5,
        )
        normalized = from_und_gen_splits(
            layer.input_layernorm(get_und_seq(pack)),
            layer.input_layernorm_moe_gen(get_gen_seq(pack)),
            pack,
        )
        attended, _ = layer.self_attn(
            normalized,
            attention_mask,
            (cosine_pack, sine_pack),
        )
        residual_und = get_und_seq(pack) + get_und_seq(attended)
        residual_gen = get_gen_seq(pack) + get_gen_seq(attended)
        post_norm_und = layer.post_attention_layernorm(residual_und)
        post_norm_gen = layer.post_attention_layernorm_moe_gen(residual_gen)
        mlp_und = layer.mlp(post_norm_und)
        mlp_gen = layer.mlp_moe_gen(post_norm_gen)
        result, _, _ = layer(
            pack,
            attention_mask,
            (cosine_pack, sine_pack),
        )

    checkpoint_name_map = {
        "self_attn.q_proj.weight": "layers.0.self_attn.to_q.weight",
        "self_attn.k_proj.weight": "layers.0.self_attn.to_k.weight",
        "self_attn.v_proj.weight": "layers.0.self_attn.to_v.weight",
        "self_attn.o_proj.weight": "layers.0.self_attn.to_out.weight",
        "self_attn.q_proj_moe_gen.weight": "layers.0.self_attn.add_q_proj.weight",
        "self_attn.k_proj_moe_gen.weight": "layers.0.self_attn.add_k_proj.weight",
        "self_attn.v_proj_moe_gen.weight": "layers.0.self_attn.add_v_proj.weight",
        "self_attn.o_proj_moe_gen.weight": "layers.0.self_attn.to_add_out.weight",
        "self_attn.q_norm_moe_gen.weight": "layers.0.self_attn.norm_added_q.weight",
        "self_attn.k_norm_moe_gen.weight": "layers.0.self_attn.norm_added_k.weight",
        "self_attn.k_norm_und_for_gen.weight": "layers.0.self_attn.k_norm_und_for_gen.weight",
        "mlp.up_proj.weight": "layers.0.mlp.up_proj.weight",
        "mlp.down_proj.weight": "layers.0.mlp.down_proj.weight",
        "mlp_moe_gen.up_proj.weight": "layers.0.mlp_moe_gen.up_proj.weight",
        "mlp_moe_gen.down_proj.weight": "layers.0.mlp_moe_gen.down_proj.weight",
        "input_layernorm.weight": "layers.0.input_layernorm.weight",
        "input_layernorm_moe_gen.weight": "layers.0.input_layernorm_moe_gen.weight",
        "post_attention_layernorm.weight": "layers.0.post_attention_layernorm.weight",
        "post_attention_layernorm_moe_gen.weight": "layers.0.post_attention_layernorm_moe_gen.weight",
    }
    tensors = {
        "input.understanding": get_und_seq(pack).detach().cpu(),
        "input.generation": get_gen_seq(pack).detach().cpu(),
        "input.position_ids": position_ids.detach().cpu(),
        "rotary.cosine": cosine[0].detach().cpu(),
        "rotary.sine": sine[0].detach().cpu(),
        "expected.normalized_understanding": get_und_seq(normalized).detach().cpu(),
        "expected.normalized_generation": get_gen_seq(normalized).detach().cpu(),
        "expected.attention_understanding": get_und_seq(attended).detach().cpu(),
        "expected.attention_generation": get_gen_seq(attended).detach().cpu(),
        "expected.residual_understanding": residual_und.detach().cpu(),
        "expected.residual_generation": residual_gen.detach().cpu(),
        "expected.post_norm_understanding": post_norm_und.detach().cpu(),
        "expected.post_norm_generation": post_norm_gen.detach().cpu(),
        "expected.mlp_understanding": mlp_und.detach().cpu(),
        "expected.mlp_generation": mlp_gen.detach().cpu(),
        "expected.understanding": get_und_seq(result).detach().cpu(),
        "expected.generation": get_gen_seq(result).detach().cpu(),
    }
    state = layer.state_dict()
    for source, target in checkpoint_name_map.items():
        tensors[target] = state[source].detach().cpu()
    assert len(checkpoint_name_map) == len(state), (
        f"Fixture mapping covers {len(checkpoint_name_map)} of {len(state)} layer tensors"
    )
    save_file(
        tensors,
        output / "tiny-mot-layer.safetensors",
        metadata={
            "nvidia_framework_revision": EXPECTED_UPSTREAM_REVISION,
            "cosmos3_edge_revision": EXPECTED_MODEL_REVISION,
            "dtype": "float32",
        },
    )


def tiny_reasoner_vision_fixture(output: Path) -> None:
    """Pin the packed SigLIP2, bilinear positions, 2x2 merger, and projector."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_config = Siglip2VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_channels=3,
        num_patches=4,
        patch_size=2,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    projector_config = SimpleNamespace(
        spatial_merge_size=2,
        input_hidden_size=8,
        merger_intermedia=20,
        out_hidden_size=16,
    )
    visual = Siglip2VisionTransformer(vision_config).to(
        device=device,
        dtype=torch.float32,
    )
    projector = PatchMerger(projector_config).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for index, (_, parameter) in enumerate(
            list(visual.named_parameters()) + list(projector.named_parameters())
        ):
            deterministic_parameter_(parameter, index)

    patches = (
        torch.cos(torch.arange(16 * 12, device=device, dtype=torch.float32) * 0.037 + 0.1)
        .reshape(16, 12)
        .contiguous()
    )
    grid = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    with torch.inference_mode():
        visual_output = visual(patches, grid_thw=grid)
        merged, merged_grid = patch_merging_by_param(
            visual_output,
            grid,
            merge_size=2,
        )
        projected = projector(merged.view(-1, 4, 8))

    tensors = {
        "input.patches": patches.detach().cpu(),
        "input.grid_thw": grid.detach().cpu(),
        "expected.visual": visual_output.detach().cpu(),
        "expected.projected": projected.detach().cpu(),
        "expected.merged_grid_thw": merged_grid.detach().cpu(),
    }
    for key, value in visual.state_dict().items():
        tensors[f"visual.{key}"] = value.detach().cpu()
    for key, value in projector.state_dict().items():
        tensors[f"projector.{key}"] = value.detach().cpu()
    save_file(
        tensors,
        output / "tiny-reasoner-vision.safetensors",
        metadata={
            "nvidia_framework_revision": EXPECTED_UPSTREAM_REVISION,
            "cosmos3_edge_revision": EXPECTED_MODEL_REVISION,
            "dtype": "float32",
        },
    )


def safetensors_inventory(path: Path) -> dict[str, dict[str, object]]:
    with safe_open(path, framework="pt", device="cpu") as archive:
        return {
            key: {
                "shape": list(archive.get_slice(key).get_shape()),
                "dtype": archive.get_slice(key).get_dtype(),
            }
            for key in archive.keys()
        }


def checkpoint_inventory(root: Path, output: Path) -> None:
    transformer_index = json.loads(
        (root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    transformer_inventory: dict[str, dict[str, object]] = {}
    for shard_name in sorted(set(transformer_index["weight_map"].values())):
        transformer_inventory.update(
            safetensors_inventory(root / "transformer" / shard_name)
        )
    vae_inventory = safetensors_inventory(
        root / "vae" / "diffusion_pytorch_model.safetensors"
    )
    payload = {
        "schema": "mere.cosmos3-edge.upstream-inventory.v1",
        "nvidia_framework_revision": EXPECTED_UPSTREAM_REVISION,
        "cosmos3_edge_revision": EXPECTED_MODEL_REVISION,
        "transformer": transformer_inventory,
        "vae": vae_inventory,
    }
    (output / "checkpoint-inventory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    upstream_revision = git_revision()
    model_revision = checkpoint_revision(args.checkpoint_root)
    if not args.allow_revision_mismatch:
        assert upstream_revision == EXPECTED_UPSTREAM_REVISION, upstream_revision
        assert model_revision == EXPECTED_MODEL_REVISION, model_revision
    args.output.mkdir(parents=True, exist_ok=True)
    tiny_layer_fixture(args.output)
    tiny_reasoner_vision_fixture(args.output)
    checkpoint_inventory(args.checkpoint_root, args.output)
    receipt = {
        "schema": "mere.cosmos3-edge.upstream-fixtures.v1",
        "nvidia_framework_revision": upstream_revision,
        "cosmos3_edge_revision": model_revision,
        "files": {
            name: sha256(args.output / name)
            for name in [
                "tiny-mot-layer.safetensors",
                "tiny-reasoner-vision.safetensors",
                "checkpoint-inventory.json",
            ]
        },
    }
    (args.output / "RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

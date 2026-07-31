#!/usr/bin/env python3
"""Convert the pinned Inkling-Small checkpoint to the managed MLX artifact.

This script deliberately separates conversion from publication. Run the
conversion, validate the generated config and inventory, then upload the
directory with the Hugging Face CLI after a successful smoke test.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm.models.inkling as mlx_vlm_inkling
import mlx_vlm.models.inkling.language as mlx_vlm_inkling_language
import mlx_vlm.utils as mlx_vlm_utils
from huggingface_hub import snapshot_download
from mlx_vlm.convert import convert
from mlx_vlm.models.inkling.config import AudioConfig, TextConfig, VisionConfig
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear


BASE_REPO_ID = "thinkingmachines/Inkling-Small"
BASE_REVISION = "b2d4f225a02032c5d154bff748ab5a00c5ca26e4"
MLX_VERSION = "0.32.0"
MLX_VLM_VERSION = "0.6.7"
QUANTIZATION = {"bits": 2, "group_size": 128, "mode": "affine"}
QUANTIZATION_SCOPE = "routed_experts"
EXPERT_QUANTIZATION_CHUNK_SIZE = 16
CUDA_VALIDATION = {
    "validated_at": "2026-07-31",
    "device": "NVIDIA H200",
    "mlx_active_bytes": 84_530_593_368,
    "mlx_peak_bytes": 84_766_898_240,
    "greedy_checks": {
        "2_plus_2": "4",
        "all_but_9": "9",
        "swift_is_even": "correct",
    },
}
SOURCE_PATTERNS = [
    "*.json",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
    "LICENSE*",
    "model-*.safetensors",
]


def patch_mlx_vlm_inkling_small_config() -> None:
    """Map the released Small config onto mlx-vlm 0.6.7's field names.

    Inkling-Small publishes ``dense_intermediate_size`` for its two dense
    layers and ``intermediate_size`` for its routed experts. mlx-vlm 0.6.7's
    Inkling dataclass instead calls those fields ``intermediate_size`` and
    ``moe_intermediate_size``. Without this compatibility mapping it builds
    placeholder modules with 2,048-wide dense and 3,072-wide expert paths
    before loading the real tensors.
    """
    mlx_vlm_utils.MODEL_REMAPPING["inkling_mm_model"] = "inkling"
    mlx_vlm_inkling.TextConfig = TextConfig
    mlx_vlm_inkling.VisionConfig = VisionConfig
    mlx_vlm_inkling.AudioConfig = AudioConfig
    original = TextConfig.from_dict.__func__
    original_vision = VisionConfig.from_dict.__func__
    original_audio = AudioConfig.from_dict.__func__

    def from_official_config(
        cls: type[TextConfig],
        params: dict[str, object],
    ) -> TextConfig:
        mapped = dict(params)
        dense_size = mapped.get("dense_intermediate_size")
        expert_size = mapped.get("moe_intermediate_size", mapped.get("intermediate_size"))
        if dense_size is not None:
            mapped["intermediate_size"] = dense_size
        if expert_size is not None:
            mapped["moe_intermediate_size"] = expert_size
        return original(cls, mapped)

    TextConfig.from_dict = classmethod(from_official_config)

    def from_official_vision_config(
        cls: type[VisionConfig],
        params: dict[str, object],
    ) -> VisionConfig:
        mapped = dict(params)
        if "decoder_dmodel" in mapped:
            mapped["text_hidden_size"] = mapped["decoder_dmodel"]
        if "n_channels" in mapped:
            mapped["num_channels"] = mapped["n_channels"]
        return original_vision(cls, mapped)

    def from_official_audio_config(
        cls: type[AudioConfig],
        params: dict[str, object],
    ) -> AudioConfig:
        mapped = dict(params)
        if "decoder_dmodel" in mapped:
            mapped["text_hidden_size"] = mapped["decoder_dmodel"]
        return original_audio(cls, mapped)

    VisionConfig.from_dict = classmethod(from_official_vision_config)
    AudioConfig.from_dict = classmethod(from_official_audio_config)

    original_switch_quantization = SwitchLinear.to_quantized

    def chunked_switch_quantization(
        layer: SwitchLinear,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ) -> QuantizedSwitchLinear:
        """Avoid MLX's >=2^31-element expert-quantization kernel boundary."""
        if layer.weight.size < 1 << 31:
            return original_switch_quantization(layer, group_size, bits, mode)

        quantized_chunks = [
            mx.quantize(
                layer.weight[start : start + EXPERT_QUANTIZATION_CHUNK_SIZE],
                group_size,
                bits,
                mode=mode,
            )
            for start in range(0, layer.num_experts, EXPERT_QUANTIZATION_CHUNK_SIZE)
        ]
        quantized = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(quantized)
        quantized.weight = mx.concatenate([chunk[0] for chunk in quantized_chunks])
        quantized.scales = mx.concatenate([chunk[1] for chunk in quantized_chunks])
        quantized.biases = (
            mx.concatenate([chunk[2] for chunk in quantized_chunks])
            if len(quantized_chunks[0]) == 3
            else None
        )
        if "bias" in layer:
            quantized.bias = layer.bias
        quantized.group_size = group_size
        quantized.bits = bits
        quantized.mode = mode
        quantized.freeze()
        return quantized

    SwitchLinear.to_quantized = chunked_switch_quantization

    def portable_banded_additive_mask(
        rel: mx.array,
        proj: mx.array,
        q_offset: int,
        sequence_length: int,
        sliding: int,
        rel_extent: int,
    ) -> mx.array:
        """Use the released tensor fallback when CUDA has no Metal backend."""
        batch, query_length, heads, _ = rel.shape
        dtype = rel.dtype
        relative_logits = (rel @ proj).transpose(0, 2, 1, 3)
        query_positions = mx.arange(query_length) + q_offset
        key_positions = mx.arange(sequence_length)
        distance = query_positions[:, None] - key_positions[None, :]
        gather_indices = mx.broadcast_to(
            mx.clip(distance, 0, rel_extent - 1)[None, None],
            (batch, heads, query_length, sequence_length),
        )
        mask = mx.take_along_axis(relative_logits, gather_indices, axis=-1)
        mask = mx.where(
            (distance >= rel_extent)[None, None],
            mx.array(0.0, dtype),
            mask,
        )
        excluded = distance < 0
        if sliding > 0:
            excluded = excluded | (distance >= sliding)
        return mx.where(
            excluded[None, None],
            mx.array(-1e30, dtype),
            mask,
        ).astype(dtype)

    mlx_vlm_inkling_language.banded_additive_mask = portable_banded_additive_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_version(package: str, expected: str) -> None:
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise RuntimeError(f"{package} {expected} is required; found {actual}")


def routed_expert_quant_predicate(
    path: str,
    module: nn.Module,
) -> bool | dict[str, object]:
    """Quantize only the 256 routed MoE experts; preserve everything else."""
    if not isinstance(module, SwitchLinear):
        return False
    if ".switch_mlp." not in path:
        return False
    return dict(QUANTIZATION)


def fetch_source() -> Path:
    return Path(
        snapshot_download(
            repo_id=BASE_REPO_ID,
            revision=BASE_REVISION,
            allow_patterns=SOURCE_PATTERNS,
            max_workers=2,
        )
    )


def write_runtime_compatibility_config(output: Path) -> None:
    """Persist the released mlx-vlm dispatch name and both MLP-width names."""
    config_path = output / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["model_type"] = "inkling"
    text = config["text_config"]
    dense_size = text["dense_intermediate_size"]
    expert_size = text.get("moe_intermediate_size", text["intermediate_size"])
    text["intermediate_size"] = dense_size
    text["moe_intermediate_size"] = expert_size
    vision = config.get("vision_config", {})
    if "decoder_dmodel" in vision:
        vision["text_hidden_size"] = vision["decoder_dmodel"]
    if "n_channels" in vision:
        vision["num_channels"] = vision["n_channels"]
    audio = config.get("audio_config", {})
    if "decoder_dmodel" in audio:
        audio["text_hidden_size"] = audio["decoder_dmodel"]
    for key in ("quantization", "quantization_config"):
        quantization = config.get(key)
        if not isinstance(quantization, dict):
            raise RuntimeError(f"converted config is missing {key}")
        quantization["scope"] = QUANTIZATION_SCOPE
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer_path = output / "tokenizer_config.json"
    with tokenizer_path.open(encoding="utf-8") as handle:
        tokenizer = json.load(handle)
    tokenizer["eos_token"] = "<|content_model_end_sampling|>"
    tokenizer["pad_token"] = "<|endoftext|>"
    tokenizer_path.write_text(
        json.dumps(tokenizer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    special_tokens_path = output / "special_tokens_map.json"
    with special_tokens_path.open(encoding="utf-8") as handle:
        special_tokens = json.load(handle)
    special_tokens["eos_token"] = "<|content_model_end_sampling|>"
    special_tokens["pad_token"] = "<|endoftext|>"
    special_tokens_path.write_text(
        json.dumps(special_tokens, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_reference_files(source: Path, output: Path) -> None:
    for pattern in ("LICENSE*", "chat_template.jinja"):
        for path in source.glob(pattern):
            if path.is_file():
                shutil.copy2(path, output / path.name)


def validate_output(output: Path) -> dict[str, object]:
    config_path = output / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise RuntimeError(f"unexpected quantization config: {quantization!r}")
    expected_metadata = {**QUANTIZATION, "scope": QUANTIZATION_SCOPE}
    for key, expected in expected_metadata.items():
        if quantization.get(key) != expected:
            raise RuntimeError(
                f"unexpected quantization {key}: {quantization.get(key)!r}; "
                f"expected {expected!r}"
            )
    per_layer = {
        key: value
        for key, value in quantization.items()
        if isinstance(value, dict)
    }
    if len(per_layer) != 120:
        raise RuntimeError(
            f"expected 120 routed expert projections in quantization config; "
            f"found {len(per_layer)}"
        )
    for key, value in per_layer.items():
        if ".switch_mlp." not in key or value != QUANTIZATION:
            raise RuntimeError(f"unexpected quantized module {key}: {value!r}")
    if config.get("model_type") != "inkling":
        raise RuntimeError(f"unexpected model_type: {config.get('model_type')!r}")

    safetensors = sorted(output.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError("conversion produced no safetensors shards")

    index_path = output / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map", {})

    text = config["text_config"]
    dense_layer = 0
    sparse_layer = text["dense_mlp_idx"]
    hidden_size = text["hidden_size"]
    packed_hidden_size = hidden_size * QUANTIZATION["bits"] // 32
    shape_expectations = {
        f"language_model.model.layers.{dense_layer}.mlp.gate_proj.weight": [
            text["dense_intermediate_size"],
            hidden_size,
        ],
        f"language_model.model.layers.{sparse_layer}.mlp.switch_mlp.gate_proj.weight": [
            text["n_routed_experts"],
            text["moe_intermediate_size"],
            packed_hidden_size,
        ],
        f"language_model.model.layers.{sparse_layer}.mlp.shared_experts.gate_proj.weight": [
            text["n_shared_experts"],
            text["moe_intermediate_size"],
            hidden_size,
        ],
        f"language_model.model.layers.{sparse_layer}.self_attn.q_proj.weight": [
            text["num_attention_heads"] * text["head_dim"],
            hidden_size,
        ],
    }
    for key, expected_shape in shape_expectations.items():
        shard_name = weight_map.get(key)
        if shard_name is None:
            raise RuntimeError(f"converted index is missing {key}")
        tensors = mx.load(str(output / shard_name))
        actual_shape = list(tensors[key].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"unexpected {key} shape: {actual_shape}; expected {expected_shape}"
            )

        scales_key = key.removesuffix(".weight") + ".scales"
        should_be_quantized = ".switch_mlp." in key
        if (scales_key in weight_map) != should_be_quantized:
            raise RuntimeError(
                f"unexpected quantization state for {key}: "
                f"scales_present={scales_key in weight_map}"
            )

    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    return {
        "artifact_bytes": total_bytes,
        "safetensors_shards": len(safetensors),
    }


def write_provenance(output: Path, inventory: dict[str, object]) -> None:
    provenance = {
        "schema_version": 1,
        "base_repo_id": BASE_REPO_ID,
        "base_revision": BASE_REVISION,
        "mlx_version": MLX_VERSION,
        "mlx_vlm_version": MLX_VLM_VERSION,
        "config_compatibility": "mlx-vlm 0.6.7 dispatch plus official Inkling-Small MLP width aliases",
        "quantization": {**QUANTIZATION, "scope": QUANTIZATION_SCOPE},
        "cuda_validation": CUDA_VALIDATION,
        "converted_at": datetime.now(timezone.utc).isoformat(),
        **inventory,
    }
    (output / "conversion.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme = f"""---
license: apache-2.0
language: en
library_name: mlx
pipeline_tag: image-text-to-text
base_model: {BASE_REPO_ID}
tags:
- mlx
- inkling
- 2-bit
- mixed-precision
---

# Inkling-Small MLX mixed 2-bit

Mixed-precision MLX conversion of
[{BASE_REPO_ID}](https://huggingface.co/{BASE_REPO_ID}) at revision
`{BASE_REVISION}`. The conversion uses MLX `{MLX_VERSION}` and mlx-vlm
`{MLX_VLM_VERSION}`. See `conversion.json` for the machine-readable provenance
and artifact inventory.

Only the 256 routed experts in each sparse MoE layer use affine 2-bit,
group-128 weights. Attention, embeddings, routers, shared experts, dense MLPs,
normalization, vision, and audio weights remain at the released BF16
precision. This conservative split targets 128 GB unified-memory Macs while
protecting the non-routed paths that blanket 2-bit quantization damaged.

## Validation status

The conversion gate checks the affine 2-bit/group-128 routed-expert metadata,
all 120 quantized expert projections, every shard, and representative BF16 and
packed tensor shapes.

On 2026-07-31, a full CUDA load and greedy generation smoke with MLX
`{MLX_VERSION}` and mlx-vlm `{MLX_VLM_VERSION}` used 84,530,593,368 bytes of
active MLX memory and peaked at 84,766,898,240 bytes on an NVIDIA H200. The
model correctly answered `2 + 2` with `4`, the "all but 9" sheep question with
`9`, and produced a correct Swift `isEven` function. This proves CUDA loading
and token quality for the generated artifact; Apple Silicon runtime validation
is tracked separately by mere.run.

mlx-vlm `{MLX_VLM_VERSION}` also required the compatibility shims in the
conversion script for its Inkling config exports and CUDA mask fallback; the
managed mere.run lane uses its own native Swift/MLX loader.

```bash
mere.run text chat --model text-chat-inkling-small --prompt "Who are you?"
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def finalize_artifact(output: Path) -> dict[str, object]:
    write_runtime_compatibility_config(output)
    inventory = validate_output(output)
    for _ in range(3):
        write_provenance(output, inventory)
        final_inventory = validate_output(output)
        if final_inventory == inventory:
            return final_inventory
        inventory = final_inventory
    raise RuntimeError("artifact inventory did not stabilize after writing provenance")


def main() -> None:
    args = parse_args()
    require_version("mlx", MLX_VERSION)
    require_version("mlx-vlm", MLX_VLM_VERSION)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output}")

    patch_mlx_vlm_inkling_small_config()
    source = fetch_source()
    convert(
        hf_path=str(source),
        mlx_path=args.output,
        quantize=True,
        q_group_size=QUANTIZATION["group_size"],
        q_bits=QUANTIZATION["bits"],
        q_mode=QUANTIZATION["mode"],
        quant_predicate=routed_expert_quant_predicate,
    )
    copy_reference_files(source, args.output)
    inventory = finalize_artifact(args.output)
    print(json.dumps(inventory, sort_keys=True))


if __name__ == "__main__":
    main()

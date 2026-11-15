#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["coremltools==9.0", "numpy==2.3.5", "safetensors==0.8.0", "torch==2.7.0", "scikit-learn==1.9.1"]
# ///
"""Convert SOV's original Whisper encoder and stacked-frame projector to Core ML.

No remote Python code is executed. Input weights must be from the pinned native
checkpoint. The decoder and DFlash remain MLX workloads. Conversion is offline
release tooling, never an inference dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = "505d8fae243924c0d8acdc622f7f0abab344eba3"
REPOSITORY = "salesforce-sovereign-intelligence/SOV-Transcribe-1"


def channel_layer_norm(x, weight, bias, eps=1e-5, scale=1.0, layout="centered"):
    """Normalize channels without changing token order or the affine bias convention."""
    # Scaling x and epsilon together preserves LayerNorm algebraically while
    # reducing the range of the squared FP16 intermediates.
    x = x / scale
    eps = eps / (scale * scale)
    if layout == "instance":
        import torch.nn.functional as F
        batch, channels, height, width = x.shape
        spatial = x.permute(0, 2, 3, 1).reshape(batch, height * width, 1, channels)
        normalized = F.instance_norm(spatial, eps=eps).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return normalized * weight[None, :, None, None] + bias[None, :, None, None]
    centered = x - x.mean(dim=1, keepdim=True)
    variance = (x.square().mean(dim=1, keepdim=True) - x.mean(dim=1, keepdim=True).square()).clamp_min(0.0) if layout == "moments" else centered.square().mean(dim=1, keepdim=True)
    normalized = centered * (variance + eps).rsqrt()
    return normalized * weight[None, :, None, None] + bias[None, :, None, None]


def attention_kernel(q, k, v, head_group_size, query_chunk_size, layout, key_chunk_size=1500, valid_frames=None):
    """Full bidirectional attention, with exact query tiling and unchanged key context.

    Inputs use [batch, heads, head dimension, frames]. Query tiles are independent;
    every tile still attends to all keys. Channel-oriented scores place the key
    softmax on Core ML's channel axis. Output is [batch, channels, 1, frames].
    """
    import torch
    import torch.nn.functional as F
    batch, heads, dimensions, frames = q.shape
    key_bias = None
    if valid_frames is not None and valid_frames < frames:
        key_bias = torch.cat([torch.zeros(valid_frames, dtype=q.dtype, device=q.device),
            torch.full((frames-valid_frames,), -float("inf"), dtype=q.dtype, device=q.device)])
    if layout in {"einsum", "streaming", "packed", "packed-streaming", "direct", "direct-streaming", "batch", "batch-streaming"}:
        # Keep each head in ANE's [batch, channels, 1, frames] layout.
        # Equations follow the public SplitHeadsQ implementation in argmaxtools.
        outputs = []
        group = head_group_size if layout.startswith(("packed", "batch")) else 1
        direct_key = layout.startswith(("direct", "batch"))
        equation = "bchq,bchk->bkhq" if direct_key else "bchq,bkhc->bkhq"
        for head in range(0, heads, group):
            if layout.startswith("batch"):
                query, key, value = [tensor[:, head:head + group].reshape(-1, dimensions, 1, frames) for tensor in [q, k, v]]
            else:
                query = q[:, head:head + group].transpose(1, 2)
                key = k[:, head:head + group].transpose(1, 2) if direct_key else k[:, head:head + group].permute(0, 3, 1, 2)
                value = v[:, head:head + group].transpose(1, 2)
            pieces = []
            for start in range(0, frames, query_chunk_size):
                if layout.endswith("streaming"):
                    maximum = denominator = numerator = None
                    for key_start in range(0, frames, key_chunk_size):
                        key_piece = key[..., key_start:key_start + key_chunk_size] if direct_key else key[:, key_start:key_start + key_chunk_size]
                        scores = torch.einsum(equation, query[..., start:start + query_chunk_size], key_piece)
                        scores = scores * dimensions ** -0.5
                        if key_bias is not None:
                            scores = scores + key_bias[None, key_start:key_start + key_chunk_size, None, None]
                        local_maximum = scores.amax(dim=1, keepdim=True)
                        next_maximum = local_maximum if maximum is None else torch.maximum(maximum, local_maximum)
                        weights = (scores - next_maximum).exp()
                        weighted_value = torch.einsum("bkhq,bchk->bchq", weights, value[..., key_start:key_start + key_chunk_size])
                        weight_sum = weights.sum(dim=1, keepdim=True)
                        if maximum is None:
                            numerator, denominator = weighted_value, weight_sum
                        else:
                            correction = (maximum - next_maximum).exp()
                            numerator = numerator * correction + weighted_value
                            denominator = denominator * correction + weight_sum
                        maximum = next_maximum
                    pieces.append(numerator / denominator)
                    continue
                scores = torch.einsum(equation, query[..., start:start + query_chunk_size], key)
                scores = scores * dimensions ** -0.5
                if key_bias is not None:
                    scores = scores + key_bias[None, :, None, None]
                probabilities = scores.softmax(dim=1)
                pieces.append(torch.einsum("bkhq,bchk->bchq", probabilities, value))
            merged = torch.cat(pieces, dim=-1)
            outputs.append(merged.permute(0, 2, 1, 3).reshape(batch, -1, 1, frames))
        return torch.cat(outputs, dim=1)
    if layout == "sdpa":
        attended = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2),
            attn_mask=key_bias[None, None, None, :] if key_bias is not None else None)
        return attended.transpose(-1, -2).reshape(batch, heads * dimensions, 1, frames)
    outputs = []
    for head in range(0, heads, head_group_size):
        query = q[:, head:head + head_group_size] * dimensions ** -0.5
        key = k[:, head:head + head_group_size]
        value = v[:, head:head + head_group_size]
        pieces = []
        for start in range(0, frames, query_chunk_size):
            query_piece = query[..., start:start + query_chunk_size]
            if layout == "channel":
                scores = torch.matmul(key.transpose(-1, -2), query_piece).transpose(1, 2)
                if key_bias is not None:
                    scores = scores + key_bias[None, :, None, None]
                probabilities = F.softmax(scores, dim=1).transpose(1, 2)
                pieces.append(torch.matmul(value, probabilities))
            else:
                scores = torch.matmul(query_piece.transpose(-1, -2), key)
                if key_bias is not None:
                    scores = scores + key_bias
                pieces.append(torch.matmul(F.softmax(scores, dim=-1), value.transpose(-1, -2)).transpose(-1, -2))
        outputs.append(torch.cat(pieces, dim=-1))
    return torch.cat(outputs, dim=1).reshape(batch, heads * dimensions, 1, frames)


def feedforward_kernel(x, first_weight, first_bias, second_weight, second_bias, token_chunk, channel_chunk, schedule="tokens"):
    """Fuseable pointwise FFN tiles with complete hidden-channel reduction."""
    import torch
    import torch.nn.functional as F
    if schedule == "channels":
        total = None
        for channel in range(0, first_weight.shape[0], channel_chunk):
            stop = channel + channel_chunk
            pieces = []
            for start in range(0, x.shape[-1], token_chunk):
                hidden = F.gelu(F.conv2d(x[..., start:start + token_chunk], first_weight[channel:stop, :, None, None], first_bias[channel:stop]))
                pieces.append(F.conv2d(hidden, second_weight[:, channel:stop, None, None]))
            partial = torch.cat(pieces, dim=-1)
            total = partial if total is None else total + partial
        return total + second_bias[None, :, None, None]
    pieces = []
    for start in range(0, x.shape[-1], token_chunk):
        part = x[..., start:start + token_chunk]
        total = None
        for channel in range(0, first_weight.shape[0], channel_chunk):
            stop = channel + channel_chunk
            hidden = F.gelu(F.conv2d(part, first_weight[channel:stop, :, None, None], first_bias[channel:stop]))
            output = F.conv2d(hidden, second_weight[:, channel:stop, None, None])
            total = output if total is None else total + output
        pieces.append(total + second_bias[None, :, None, None])
    return torch.cat(pieces, dim=-1)


def build_graph(root, head_group_size=4):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text())
    # All audio/projector tensors in the pinned release live in this one shard.
    selected = {v for k, v in index["weight_map"].items() if k.startswith(("model.audio_tower.", "model.embed_audio."))}
    expected = {"model-00016-of-00016.safetensors": "438cd592fe6fc231d4f10dad756cfed5ab60964ea0021155ca136792885f6b5d"}
    if selected != set(expected):
        raise ValueError("Audio tensor shard layout differs from the pinned checkpoint")
    for name, digest in expected.items():
        with (root / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError(f"Source checkpoint checksum mismatch: {name}")
    weights = {}
    for shard in sorted(set(index["weight_map"].values())):
        names = [k for k, v in index["weight_map"].items() if v == shard and
                 k.startswith(("model.audio_tower.", "model.embed_audio."))]
        if not names:
            continue
        with safe_open(root / shard, framework="pt", device="cpu") as source:
            for name in names:
                weights[name.removeprefix("model.")] = source.get_tensor(name).float()

    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            self.names = {}
            self.head_group_size = head_group_size
            self.stage = "full"
            self.channel_norm = False
            self.spatial_width = 1500
            self.attention_layout = "qk"
            self.fuse_qkv = False
            self.layer_count = 32
            self.query_chunk_size = 1500
            self.mlp_chunk_size = 1500
            self.linear_operator = False
            self.layer_index = 0
            self.normalization_scale = 1.0
            self.key_chunk_size = 1500
            self.mlp_channel_chunk_size = 5120
            self.collect_activation_max = False
            self.activation_max = {}
            self.activation_min = {}
            self.activation_upper = {}
            self.linear_input_scales = {}
            self.collect_output_ranges = False
            self.output_min = {}
            self.output_max = {}
            self.normalization_layout = "centered"
            self.encoder_frames = 1500
            self.split_ffn_layers = set()
            self.mlp_schedule = "tokens"
            self.qkv_head_chunk_size = 0
            for i, (name, value) in enumerate(weights.items()):
                key = f"w{i}"
                self.register_buffer(key, value)
                self.names[name] = key

        def weight(self, name):
            return getattr(self, self.names[name])

        def linear(self, x, name, bias=True):
            if name in self.linear_input_scales:
                x = x / getattr(self, self.linear_input_scales[name])[None, :, None, None]
            if self.collect_activation_max:
                maximum = x.detach().abs().amax(dim=(0, 2, 3))
                self.activation_max[name] = torch.maximum(self.activation_max[name], maximum) if name in self.activation_max else maximum
                lower, upper = x.detach().amin(dim=(0, 2, 3)), x.detach().amax(dim=(0, 2, 3))
                self.activation_min[name] = torch.minimum(self.activation_min[name], lower) if name in self.activation_min else lower
                self.activation_upper[name] = torch.maximum(self.activation_upper[name], upper) if name in self.activation_upper else upper
            if self.linear_operator:
                sequence = x.permute(0, 2, 3, 1)
                b = self.weight(name + ".bias") if bias else None
                result = F.linear(sequence, self.weight(name + ".weight"), b).permute(0, 3, 1, 2)
            else:
                weight = self.weight(name + ".weight")[:, :, None, None]
                b = self.weight(name + ".bias") if bias else None
                channels = x.shape[1]
                frames = x.shape[2] * x.shape[3]
                width = self.spatial_width if frames % self.spatial_width == 0 else frames
                tiled = x.reshape(1, channels, frames // width, width)
                result = F.conv2d(tiled, weight, b).reshape(1, weight.shape[0], 1, frames)
            if self.collect_output_ranges:
                lower, upper = result.detach().amin(dim=(0, 2, 3)), result.detach().amax(dim=(0, 2, 3))
                self.output_min[name] = torch.minimum(self.output_min[name], lower) if name in self.output_min else lower
                self.output_max[name] = torch.maximum(self.output_max[name], upper) if name in self.output_max else upper
            return result

        def norm(self, x, name):
            if self.channel_norm:
                return channel_layer_norm(x, self.weight(name + ".weight"), self.weight(name + ".bias"), scale=self.normalization_scale, layout=self.normalization_layout)
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (1280,), self.weight(name + ".weight"), self.weight(name + ".bias"), 1e-5)
            return x.permute(0, 3, 1, 2)

        def stem(self, mel):
            x = F.gelu(F.conv2d(mel, self.weight("audio_tower.conv1.weight")[:, :, None, :],
                               self.weight("audio_tower.conv1.bias"), padding=(0, 1)))
            x = F.gelu(F.conv2d(x, self.weight("audio_tower.conv2.weight")[:, :, None, :],
                               self.weight("audio_tower.conv2.bias"), stride=(1, 2), padding=(0, 1)))
            x = x + self.weight("audio_tower.embed_positions.weight").T[None, :, None, :]
            return self.pad_frames(x)

        def pad_frames(self, x):
            return F.pad(x, (0, self.encoder_frames-x.shape[-1])) if self.encoder_frames > x.shape[-1] else x

        def attention(self, x, i):
            p = f"audio_tower.layers.{i}"
            y = self.norm(x, p + ".self_attn_layer_norm")
            if self.qkv_head_chunk_size:
                outputs = []
                for head in range(0, 20, self.qkv_head_chunk_size):
                    start, end = head*64, min(head+self.qkv_head_chunk_size, 20)*64
                    weight = torch.cat([self.weight(p + f".self_attn.{key}_proj.weight")[start:end] for key in ["q", "k", "v"]])
                    bias = torch.cat([self.weight(p + ".self_attn.q_proj.bias")[start:end],
                        torch.zeros(end-start), self.weight(p + ".self_attn.v_proj.bias")[start:end]])
                    qkv = F.conv2d(y, weight[:, :, None, None], bias)
                    q, k, v = qkv.split(end-start, dim=1)
                    outputs.append(attention_kernel(*[value.reshape(1, (end-start)//64, 64, self.encoder_frames) for value in [q, k, v]],
                        1, self.query_chunk_size, self.attention_layout, self.key_chunk_size, valid_frames=1500))
                return self.linear(torch.cat(outputs, dim=1), p + ".self_attn.out_proj")
            if self.fuse_qkv:
                qkv_weight = torch.cat([self.weight(p + f".self_attn.{key}_proj.weight") for key in ["q", "k", "v"]])[:, :, None, None]
                qkv_bias = torch.cat([self.weight(p + ".self_attn.q_proj.bias"), torch.zeros_like(self.weight(p + ".self_attn.q_proj.bias")), self.weight(p + ".self_attn.v_proj.bias")])
                width = self.spatial_width if self.encoder_frames % self.spatial_width == 0 else self.encoder_frames
                qkv = F.conv2d(y.reshape(1, 1280, self.encoder_frames // width, width), qkv_weight, qkv_bias).reshape(1, 3840, 1, self.encoder_frames)
                q, k, v = qkv[:, :1280], qkv[:, 1280:2560], qkv[:, 2560:]
            else:
                q = self.linear(y, p + ".self_attn.q_proj")
                k = self.linear(y, p + ".self_attn.k_proj", False)
                v = self.linear(y, p + ".self_attn.v_proj")
            grouped_q = q.reshape(1, 20, 64, self.encoder_frames)
            grouped_k = k.reshape(1, 20, 64, self.encoder_frames)
            grouped_v = v.reshape(1, 20, 64, self.encoder_frames)
            merged = attention_kernel(grouped_q, grouped_k, grouped_v, self.head_group_size,
                                      self.query_chunk_size, self.attention_layout, self.key_chunk_size, valid_frames=1500)
            return self.linear(merged, p + ".self_attn.out_proj")

        def feedforward(self, x, i):
            p = f"audio_tower.layers.{i}"
            y = self.norm(x, p + ".final_layer_norm")
            if self.mlp_channel_chunk_size < 5120:
                return feedforward_kernel(y, self.weight(p + ".fc1.weight"), self.weight(p + ".fc1.bias"),
                                          self.weight(p + ".fc2.weight"), self.weight(p + ".fc2.bias"),
                                          self.mlp_chunk_size, self.mlp_channel_chunk_size, self.mlp_schedule)
            pieces = []
            for start in range(0, self.encoder_frames, self.mlp_chunk_size):
                part = y[..., start:start + self.mlp_chunk_size]
                if i in self.split_ffn_layers:
                    regular = self.linear(F.gelu(self.linear(part, p + ".fc1_regular")), p + ".fc2_regular")
                    outlier = self.linear(F.gelu(self.linear(part, p + ".fc1_outlier")), p + ".fc2_outlier")
                    pieces.append(regular + outlier)
                else:
                    pieces.append(self.linear(F.gelu(self.linear(part, p + ".fc1")), p + ".fc2"))
            return torch.cat(pieces, dim=-1)

        def block(self, x, i):
            x = x + self.attention(x, i)
            return x + self.feedforward(x, i)

        def project(self, x):
            x = x[..., :1500]
            x = self.norm(x, "audio_tower.layer_norm")
            # Consecutive frames, each with 1280 channels, stack in source order.
            x = x.permute(0, 3, 2, 1).reshape(1, 375, 1, 5120).permute(0, 3, 2, 1)
            x = self.linear(F.gelu(self.linear(x, "embed_audio.linear_1", False)), "embed_audio.linear_2", False)
            return x.permute(0, 3, 2, 1).reshape(1, 375, 2816)

        def forward(self, x):
            if self.stage == "stem":
                return self.stem(x)
            if self.stage == "attention":
                return self.attention(self.pad_frames(x), self.layer_index)[..., :1500]
            if self.stage == "norm":
                return self.norm(x, f"audio_tower.layers.{self.layer_index}.final_layer_norm")
            if self.stage == "mlp":
                return self.feedforward(self.pad_frames(x), self.layer_index)[..., :1500]
            if self.stage == "block":
                return self.block(self.pad_frames(x), self.layer_index)[..., :1500]
            if self.stage == "projector":
                return self.project(x)
            x = self.stem(x)
            for i in range(self.layer_count):
                x = self.block(x, i)
            return self.project(x)

    return Graph().eval()


def conversion_precision(sensitive_fp32=False):
    import coremltools as ct
    if sensitive_fp32:
        return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in {"layer_norm", "softmax"})
    return ct.precision.FLOAT16


def palettize_model(model, bits, group_size=0):
    """Keep activations FP16 and encode convolution weights with learned LUTs."""
    import coremltools.optimize.coreml as optimization
    config = optimization.OpPalettizerConfig(
        mode="kmeans", nbits=bits,
        granularity="per_grouped_channel" if group_size else "per_tensor",
        group_size=group_size or 32,
    )
    return optimization.palettize_weights(model, optimization.OptimizationConfig(
        op_type_configs={"conv": config},
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--head-group-size", type=int, choices=[1, 4], default=None, help="Heads per attention group (default: 4)")
    parser.add_argument("--optimization", choices=["baseline", "feedforward", "tiled", "palettized"], default="tiled", help="Graph preset; palettized selects the qualified 8-bit per-head layout")
    parser.add_argument("--attention-layout", choices=["qk", "channel", "einsum", "streaming"], help="Experimental override; requires full qualification")
    parser.add_argument("--key-chunk-frames", type=int, choices=[250, 500, 750, 1500, 1536], default=1500, help="Experimental online-softmax key tile length; requires streaming attention")
    parser.add_argument("--feedforward-channel-chunk", type=int, choices=[512, 1024, 2048, 5120], default=5120, help="Experimental FFN hidden-channel tile length")
    parser.add_argument("--feedforward-chunk-frames", type=int, choices=[128, 500, 1500], help="Experimental FFN token tile length")
    parser.add_argument("--activation-calibration", type=Path, help="Experimental calibrated W8/input-A8 MLP quantization; requires independent calibration receipts")
    parser.add_argument("--ffn-outlier-count", type=int, choices=[0, 64, 128, 256, 512, 1024], default=0, help="Experimental number of FFN neurons retained in FP16 per layer")
    parser.add_argument("--query-chunk-frames", type=int, choices=[128, 256, 500, 512, 1500], help="Experimental query tile length override")
    parser.add_argument("--encoder-frames", type=int, choices=[1500, 1536], default=1500, help="Experimental masked padding to align ANE tile dimensions")
    parser.add_argument("--channel-normalization", action="store_true", help="Experimental channel reduction instead of layer_norm")
    parser.add_argument("--normalization-scale", type=float, choices=[1.0, 8.0], help="Experimental channel LayerNorm input scaling with compensated epsilon")
    parser.add_argument("--sensitive-fp32", action="store_true", help="Experimental FP32 layer_norm and softmax operators")
    parser.add_argument("--palettization-bits", type=int, choices=[6, 8], help="Experimental K-means convolution weight palettes")
    parser.add_argument("--palette-group-size", type=int, choices=[0, 8, 16, 32], default=0, help="Channels per palette; 0 uses one palette per weight tensor")
    parser.add_argument("--reference", type=Path, help="Native safetensors mel dump for independent FP32 reference")
    parser.add_argument("--reference-only", action="store_true", help="Write the FP32 reference without converting again")
    args = parser.parse_args()
    if args.reference_only and not args.reference:
        parser.error("--reference-only requires --reference")
    if args.key_chunk_frames != 1500 and args.attention_layout != "streaming":
        parser.error("--key-chunk-frames requires --attention-layout streaming")
    if args.activation_calibration and (args.feedforward_channel_chunk != 5120 or args.palettization_bits or args.optimization == "palettized"):
        parser.error("--activation-calibration requires unsplit hidden channels and no palettization")
    if args.ffn_outlier_count and not args.activation_calibration:
        parser.error("--ffn-outlier-count requires --activation-calibration")
    if args.normalization_scale is not None and not args.channel_normalization and args.optimization != "palettized":
        parser.error("--normalization-scale requires --channel-normalization")
    if args.palette_group_size and not args.palettization_bits and args.optimization != "palettized":
        parser.error("--palette-group-size requires --palettization-bits")
    import coremltools as ct
    import numpy as np
    import torch
    torch.set_num_threads(8)
    config = json.loads((args.checkpoint / "config.json").read_text())
    if config["audio_config"]["d_model"] != 1280 or config["audio_config"]["encoder_layers"] != 32:
        raise ValueError("Expected the original SOV-Transcribe-1 audio configuration")
    args.output.mkdir(parents=True, exist_ok=True)
    head_group_size = args.head_group_size if args.head_group_size is not None else 4
    graph = build_graph(args.checkpoint, head_group_size)
    if args.optimization in ("feedforward", "tiled"):
        graph.mlp_chunk_size = 128
    if args.optimization == "tiled":
        graph.channel_norm = False
        graph.attention_layout = "qk"
        graph.query_chunk_size = 128
        graph.mlp_chunk_size = 128
    if args.optimization == "palettized":
        graph.channel_norm = True
        graph.normalization_scale = 8.0
        graph.attention_layout = "einsum"
        graph.query_chunk_size = 500
        graph.mlp_chunk_size = 128
        args.palettization_bits = args.palettization_bits or 8
    if args.attention_layout:
        graph.attention_layout = args.attention_layout
    if graph.attention_layout in {"einsum", "streaming"}:
        graph.head_group_size = 1
    graph.key_chunk_size = args.key_chunk_frames
    graph.mlp_channel_chunk_size = args.feedforward_channel_chunk
    graph.encoder_frames = args.encoder_frames
    if args.feedforward_chunk_frames:
        graph.mlp_chunk_size = args.feedforward_chunk_frames
    if args.query_chunk_frames:
        graph.query_chunk_size = args.query_chunk_frames
    if args.channel_normalization:
        graph.channel_norm = True
    if args.normalization_scale is not None:
        graph.normalization_scale = args.normalization_scale
    if args.reference:
        from safetensors.numpy import load_file, save_file
        mel = load_file(args.reference)["mel"]
        with torch.inference_mode():
            projected = np.concatenate([graph(torch.from_numpy(window[None, :, None, :])).numpy() for window in mel], axis=1)
        save_file({"audio_embeddings": np.ascontiguousarray(projected)}, args.output / "torch-reference.safetensors")
    if args.reference_only:
        return
    quantization = None
    if args.activation_calibration:
        from safetensors.numpy import load_file
        from profile_sov_ane_kernels import smooth_graph, split_ffn
        maxima = load_file(args.activation_calibration / "activation-max.safetensors")
        ranges = load_file(args.activation_calibration / "activation-ranges.safetensors")
        scales = smooth_graph(graph, maxima, 0.5, "fc1" if args.ffn_outlier_count else "mlp", range(32))
        if args.ffn_outlier_count:
            split_ffn(graph, maxima, ranges, args.ffn_outlier_count, range(32), scales)
    example = torch.zeros(1, 128, 1, 3000)
    with torch.inference_mode():
        traced = torch.jit.trace(graph, example, check_trace=False)
    model = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       inputs=[ct.TensorType(name="mel", shape=example.shape, dtype=np.float16)],
                       outputs=[ct.TensorType(name="audio_embeddings", dtype=np.float16)],
                       compute_precision=conversion_precision(args.sensitive_fp32), compute_units=ct.ComputeUnit.CPU_AND_NE,
                       skip_model_load=True)
    if args.palettization_bits:
        model = palettize_model(model, args.palettization_bits, args.palette_group_size)
    if args.activation_calibration:
        from profile_sov_ane_kernels import quantize_inputs
        model, quantization = quantize_inputs(model, graph, ranges, scales, "regular" if args.ffn_outlier_count else "mlp")
        quantization["smoothAlpha"] = 0.5
        quantization["outlierNeuronsPerLayer"] = args.ffn_outlier_count
        quantization["calibrationFiles"] = {name: hashlib.sha256((args.activation_calibration / name).read_bytes()).hexdigest()
            for name in ["calibration.json", "activation-max.safetensors", "activation-ranges.safetensors"]}
    package = args.output / "encoder.mlpackage"
    model.save(str(package))
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(args.output)], check=True)
    files = []
    for path in sorted((args.output / "encoder.mlmodelc").rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(args.output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = dict(schemaVersion=1, sourceRepository=REPOSITORY, sourceRevision=REVISION,
                    inputShape=[1,128,1,3000], outputShape=[1,375,2816], files=files,
                    headGroupSize=graph.head_group_size, optimization=args.optimization,
                    queryChunkFrames=graph.query_chunk_size, feedforwardChunkFrames=graph.mlp_chunk_size,
                    keyChunkFrames=graph.key_chunk_size, feedforwardChannelChunk=graph.mlp_channel_chunk_size,
                    encoderFrames=graph.encoder_frames, validEncoderFrames=1500,
                    channelFirstNormalization=graph.channel_norm, attentionLayout=graph.attention_layout,
                    palettizationBits=args.palettization_bits, paletteGroupSize=args.palette_group_size,
                    sensitiveFP32=args.sensitive_fp32,
                    normalizationScale=graph.normalization_scale,
                    activationQuantization=quantization,
                    coremltools=ct.__version__, torch=torch.__version__,
                    xcode=subprocess.check_output(["xcodebuild", "-version"], text=True).strip())
    (args.output / "sov-coreml.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
